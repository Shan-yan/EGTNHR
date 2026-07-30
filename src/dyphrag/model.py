"""Reference DyPH-RAG model and baseline-compatible modes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .baselines import capabilities_for
from .contracts import EvidenceItem, EvidencePolarity, EvidenceSource, PatientDiseaseExample
from .evidence import EvidenceAggregator, PolarityClassifier
from .features import hashed_vector
from .hypergraph import (
    HeterogeneousHypergraphTransformer,
    HypergraphUpdater,
    NumericalEncoder,
    SimpleHypergraphEncoder,
    add_cohort_hyperedge,
    build_base_hypergraph,
)
from .retrieval import RetrievalSystem
from .router import DynamicRouter


@dataclass
class DyPHOutput:
    logits: torch.Tensor
    probability: torch.Tensor
    abstention_probability: torch.Tensor
    prediction: str
    loss: torch.Tensor | None
    evidence: tuple[EvidenceItem, ...]
    trace: dict[str, Any]


class DyPHRAGModel(nn.Module):
    def __init__(self, config: dict[str, Any], retrieval: RetrievalSystem) -> None:
        super().__init__()
        self.config = config
        self.retrieval = retrieval
        hidden = int(config["model"]["hidden_dim"])
        self.model_mode = str(config["model"]["mode"])
        self.capabilities = capabilities_for(self.model_mode)
        self.patient_encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.disease_encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.context_encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.sequence_gru = nn.GRU(hidden, hidden, batch_first=True)
        sequence_layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True)
        self.sequence_transformer = nn.TransformerEncoder(sequence_layer, num_layers=1)
        self.prior_head = nn.Linear(hidden * 3, 2)
        fixed_k = int(config["retrieval"]["fixed_top_k"]) if config["retrieval"]["fixed_top_k"] is not None else None
        self.router = DynamicRouter(
            hidden,
            config["router"]["mode"],
            int(config["retrieval"]["max_total_k"]),
            fixed_k,
            float(config["router"].get("source_gate_threshold", 0.5)),
            float(config["router"].get("retrieval_gate_threshold", 0.5)),
            float(config["router"]["stop_threshold"]),
        )
        self.numeric_encoder = NumericalEncoder(hidden, config["hypergraph"]["numeric_encoder"])
        graph_encoder = config["hypergraph"]["encoder"]
        self.graph_encoder: nn.Module
        if graph_encoder == "heterogeneous_transformer":
            self.graph_encoder = HeterogeneousHypergraphTransformer(hidden)
        else:
            self.graph_encoder = SimpleHypergraphEncoder(hidden, graph_encoder)
        update_operator = config["hypergraph"]["update_operator"] if self.capabilities.dynamic_hypergraph else "static"
        self.updater = HypergraphUpdater(
            update_operator,
            hidden,
            float(config["hypergraph"].get("deactivation_threshold", 0.05)),
        )
        self.evidence_aggregator = EvidenceAggregator(hidden, config["evidence"]["reliability_weighting"])
        self.polarity_classifier = PolarityClassifier(hidden)
        self.iteration_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.classifier = nn.Sequential(nn.Linear(hidden * 6 + 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        # These are fitted on validation data after model selection, never by
        # the training objective on the same observations as the classifier.
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer(
            "abstention_threshold",
            torch.tensor(float(config["evidence"]["abstention_threshold"])),
        )

    def set_posthoc_calibration(self, temperature: float, abstention_threshold: float) -> None:
        self.temperature.fill_(max(float(temperature), 0.05))
        self.abstention_threshold.fill_(min(max(float(abstention_threshold), 0.0), 1.0))

    def _patient_disease_context(self, example: PatientDiseaseExample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if self.model_mode in {"ehr_gru", "ehr_transformer"}:
            sequence = torch.stack(
                [
                    hashed_vector([event.concept_id, event.event_type], self.patient_encoder[0].in_features, device)
                    for event in example.prior_events
                ]
                or [torch.zeros(self.patient_encoder[0].in_features, device=device)]
            ).unsqueeze(0)
            if self.model_mode == "ehr_gru":
                patient = self.sequence_gru(sequence)[1][-1, 0]
            else:
                patient = self.sequence_transformer(sequence).mean(dim=1).squeeze(0)
        else:
            patient = hashed_vector(
                [event.concept_id for event in example.prior_events]
                + [f"{key}:{round(float(value), 2)}" for key, value in sorted(example.context.items())],
                self.patient_encoder[0].in_features,
                device,
            )
        disease_tokens = [example.target_disease] if self.config["retrieval"]["disease_conditioned"] else ["target_disease"]
        context_tokens = [
            f"{key}:{round(float(value), 4)}"
            for key, value in sorted(example.context.items())
        ] or ["context:missing"]
        return (
            self.patient_encoder(patient),
            self.disease_encoder(hashed_vector(disease_tokens, patient.numel(), device)),
            self.context_encoder(hashed_vector(context_tokens, patient.numel(), device)),
        )

    def forward(self, example: PatientDiseaseExample, label: int | None = None) -> DyPHOutput:
        # The outcome is passed only to the loss below; retrieval/encoding always
        # receives an explicitly masked query object.
        example = replace(example, label=2)
        hp, hd, context = self._patient_disease_context(example)
        prior_logits = self.prior_head(torch.cat([hp, hd, hp * hd]))
        prior_probability = torch.softmax(prior_logits, dim=-1)[1]
        uncertainty = float(1.0 - torch.max(torch.softmax(prior_logits, dim=-1)).detach())
        delta_t = 0.0
        if example.prior_events and self.config["retrieval"]["prediction_time_conditioned"]:
            delta_t = (example.cutoff_time - max(event.timestamp for event in example.prior_events)).total_seconds() / 86400.0
        graph = build_base_hypergraph(
            example,
            hp.numel(),
            self.numeric_encoder,
            bool(self.config["hypergraph"]["numeric_state"]),
            bool(self.config["hypergraph"]["visit_only"]),
            bool(self.config["hypergraph"]["temporal_encoding"]),
            bool(self.config["retrieval"]["disease_conditioned"]),
            hp,
            hd,
        )
        initial_graph_state = self.graph_encoder(graph) if self.capabilities.use_hypergraph else hp
        query_patient = 0.5 * (hp + initial_graph_state)
        # Required disease/cutoff query: [h_p; h_d; h_p ⊙ h_d; delta_t; context].
        normalized_delta = torch.tensor([delta_t / 365.0], device=hp.device)

        def make_query(patient_state: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    patient_state,
                    hd,
                    patient_state * hd,
                    normalized_delta,
                    context,
                ]
            )

        query = make_query(query_patient)
        collected: dict[str, EvidenceItem] = {}
        router_traces = []
        max_iterations = int(self.config["retrieval"]["max_iterations"]) if self.capabilities.dynamic_retrieval else 1
        evidence_quality = 0.0
        last_routed = None
        iteration_logits: list[torch.Tensor] = []
        route_supervision: list[dict[str, Any]] = []
        routed_for_next = None
        for iteration in range(max_iterations):
            routed = routed_for_next or self.router(query, uncertainty, evidence_quality)
            routed_for_next = None
            last_routed = routed
            source_enabled = [
                bool(self.config["retrieval"]["self_enabled"]) and self.capabilities.use_retrieval,
                bool(self.config["retrieval"]["peer_enabled"]) and self.capabilities.use_retrieval,
                bool(self.config["retrieval"]["medical_enabled"]) and self.capabilities.use_retrieval,
                bool(self.config["retrieval"]["prior_enabled"]),
            ]
            mask = torch.tensor(source_enabled, dtype=routed.weights.dtype, device=hp.device)
            routed.weights = routed.weights * mask
            if float(routed.weights.sum().detach()) > 0:
                routed.weights = routed.weights / routed.weights.sum()
            routed.source_gate_probabilities = routed.source_gate_probabilities * mask[:3]
            effective_retrieval_probability = (
                routed.retrieval_probability
                if bool(self.config["retrieval"]["prior_enabled"])
                else 1.0
            )
            routed.budgets = self.router.allocate_budgets(
                routed.weights,
                effective_retrieval_probability,
                routed.source_gate_probabilities,
            )
            if not self.capabilities.use_retrieval:
                routed.budgets = {"self": 0, "peer": 0, "medical": 0}
            routed.continue_sources = {
                name: routed.budgets[name] > 0 and source_enabled[index]
                for index, name in enumerate(("self", "peer", "medical"))
            }
            routed.prior_only = sum(routed.budgets.values()) == 0
            evidence = self.retrieval.retrieve(example, routed.budgets, iteration) if self.capabilities.use_retrieval else ()
            source_indices = {EvidenceSource.SELF: 0, EvidenceSource.PEER: 1, EvidenceSource.MEDICAL: 2}
            evidence = tuple(
                replace(item, applicability=item.applicability * float(routed.weights[source_indices[item.source]].detach()))
                for item in evidence
            )
            graph_is_static = self.config["hypergraph"]["update_operator"] == "static"
            if (
                self.capabilities.use_retrieval
                and self.config["hypergraph"]["cohort"]
                and (not graph_is_static or iteration == 0)
            ):
                peer_members: dict[str, tuple[str, tuple[float, ...], str]] = {}
                for item in evidence:
                    if item.source != EvidenceSource.PEER or item.patient_id is None:
                        continue
                    embedding = tuple(float(value) for value in item.metadata.get("peer_embedding", ()))
                    snapshot_cutoff = str(item.metadata.get("peer_snapshot_cutoff", ""))
                    if embedding and snapshot_cutoff:
                        peer_members[item.patient_id] = (
                            item.patient_id,
                            embedding,
                            snapshot_cutoff,
                        )
                add_cohort_hyperedge(
                    graph,
                    list(peer_members.values()),
                    self.retrieval.peer_retriever.corpus_hash,
                )
            collected.update((item.evidence_id, item) for item in evidence)
            graph = self.updater.update(graph, evidence, iteration)
            evidence_quality = sum(max(item.score, 0.0) for item in collected.values()) / max(len(collected), 1)
            graph_state = self.graph_encoder(graph) if self.capabilities.use_hypergraph else hp
            interim_logits = self.iteration_head(torch.cat([graph_state, hd]))
            iteration_logits.append(interim_logits)
            interim_distribution = torch.softmax(interim_logits, dim=-1)
            entropy = -(interim_distribution * interim_distribution.clamp_min(1e-8).log()).sum() / torch.log(
                torch.tensor(2.0, device=hp.device)
            )
            uncertainty = float(entropy.detach())
            provisional_evidence = tuple(collected.values())
            if self.capabilities.evidence_polarity and self.config["evidence"]["polarity_classifier"]:
                provisional_evidence = tuple(
                    replace(
                        item,
                        polarity=self.polarity_classifier(
                            item,
                            hd,
                            use_metadata_label=False,
                        )[0],
                    )
                    for item in provisional_evidence
                )
            provisional_repr = self.evidence_aggregator(
                provisional_evidence,
                hp.device,
                hd,
                bool(self.config["retrieval"]["frozen"]),
                bool(self.config["evidence"]["include_refute"]),
                bool(self.config["evidence"]["include_differential"]),
            )
            learned_sufficiency = float(provisional_repr.sufficiency.detach())
            next_query = make_query(graph_state)
            post_routed = self.router(
                next_query,
                uncertainty,
                evidence_quality,
                learned_sufficiency,
            )
            router_traces.append(
                {
                    "iteration": iteration,
                    **routed.diagnostics(),
                    "post_retrieval_stop_probability": post_routed.stopping_probability,
                    "post_retrieval_sufficiency_probability": post_routed.evidence_sufficiency_probability,
                    "measured_evidence_sufficiency": learned_sufficiency,
                    "evidence_count": len(collected),
                    "interim_probability": float(interim_distribution[1].detach()),
                    "uncertainty": uncertainty,
                }
            )
            sufficient = learned_sufficiency >= float(
                self.config["router"].get("evidence_sufficiency_threshold", 0.65)
            )
            source_targets = torch.tensor(
                [
                    float(any(item.source == source for item in evidence))
                    for source in (
                        EvidenceSource.SELF,
                        EvidenceSource.PEER,
                        EvidenceSource.MEDICAL,
                    )
                ],
                device=hp.device,
            )
            route_supervision.append(
                {
                    "route": routed,
                    "post_route": post_routed,
                    "source_targets": (
                        source_targets
                        if bool(source_targets.sum())
                        else torch.tensor(source_enabled[:3], device=hp.device, dtype=torch.float32)
                        * float(uncertainty >= 0.25 and not sufficient)
                    ),
                    "retrieve_target": float(uncertainty >= 0.25 and not sufficient),
                    "sufficiency_target": float(sufficient),
                    "stop_target": float(sufficient or iteration == max_iterations - 1),
                }
            )
            if (
                not self.capabilities.dynamic_retrieval
                or not self.config["retrieval"]["iterative"]
                or post_routed.stopping_probability >= float(self.config["router"]["stop_threshold"])
                or post_routed.prior_only
                or sufficient
            ):
                last_routed = post_routed
                break
            query = next_query
            routed_for_next = post_routed
        evidence_tuple = tuple(
            sorted(
                collected.values(),
                key=lambda item: item.score * item.reliability * item.applicability,
                reverse=True,
            )
        )
        polarity_losses: list[torch.Tensor] = []
        if self.capabilities.evidence_polarity and self.config["evidence"]["polarity_classifier"]:
            classified = []
            for item in evidence_tuple:
                predicted, polarity_logits = self.polarity_classifier(item, hd, use_metadata_label=False)
                classified.append(replace(item, polarity=predicted))
                expected = item.metadata.get("polarity_label")
                if expected is not None:
                    target_index = list(EvidencePolarity).index(EvidencePolarity(expected))
                    polarity_losses.append(F.cross_entropy(polarity_logits.unsqueeze(0), torch.tensor([target_index], device=hp.device)))
            evidence_tuple = tuple(classified)
        elif not self.capabilities.evidence_polarity:
            evidence_tuple = tuple(replace(item, polarity=EvidencePolarity.SUPPORT) for item in evidence_tuple)
        representations = self.evidence_aggregator(
            evidence_tuple,
            hp.device,
            hd,
            bool(self.config["retrieval"]["frozen"]),
            bool(self.config["evidence"]["include_refute"]),
            bool(self.config["evidence"]["include_differential"]),
        )
        graph_repr = self.graph_encoder(graph) if self.capabilities.use_hypergraph else hp
        final_input = torch.cat(
            [
                graph_repr,
                hp,
                hd,
                representations.support,
                representations.refute,
                representations.differential,
                (
                    prior_probability
                    * (
                        last_routed.weights[3]
                        if last_routed is not None and self.config["retrieval"]["prior_enabled"]
                        else float(bool(self.config["retrieval"]["prior_enabled"]))
                    )
                ).reshape(1),
                representations.sufficiency.reshape(1),
            ]
        )
        logits = self.classifier(final_input)
        calibrated_temperature = self.temperature.clamp_min(0.05)
        raw_probability = torch.softmax(logits, dim=-1)[1]
        probability = torch.softmax(logits / calibrated_temperature, dim=-1)[1]
        abstention = 1.0 - representations.sufficiency
        abstention_enabled = bool(self.config["evidence"]["abstention"]) and self.capabilities.use_retrieval
        if abstention_enabled and float(abstention.detach()) > float(self.abstention_threshold):
            prediction = "insufficient_evidence"
        else:
            prediction = "yes" if float(probability.detach()) >= 0.5 else "no"
        loss = None
        losses: dict[str, torch.Tensor] = {}
        if label is not None and label in (0, 1):
            target = torch.tensor([label], device=hp.device)
            losses["classification"] = F.cross_entropy(logits.unsqueeze(0), target)
            if iteration_logits:
                losses["classification"] = losses["classification"] + 0.1 * torch.stack(
                    [F.cross_entropy(value.unsqueeze(0), target) for value in iteration_logits]
                ).mean()
            routing_terms: list[torch.Tensor] = []
            retrieval_cost_terms: list[torch.Tensor] = []
            for supervision in route_supervision:
                route = supervision["route"]
                post_route = supervision["post_route"]
                routing_terms.extend(
                    [
                        F.binary_cross_entropy(
                            route.retrieval_score,
                            torch.tensor(supervision["retrieve_target"], device=hp.device),
                        ),
                        F.binary_cross_entropy(
                            route.source_gate_probabilities,
                            supervision["source_targets"],
                        ),
                        F.binary_cross_entropy(
                            post_route.evidence_sufficiency_score,
                            torch.tensor(supervision["sufficiency_target"], device=hp.device),
                        ),
                        F.binary_cross_entropy(
                            post_route.stopping_score,
                            torch.tensor(supervision["stop_target"], device=hp.device),
                        ),
                    ]
                )
                retrieval_cost_terms.append(route.retrieval_score)
            losses["routing"] = (
                torch.stack(routing_terms).mean()
                if routing_terms
                else torch.zeros((), device=hp.device)
            )
            losses["retrieval_cost"] = (
                torch.stack(retrieval_cost_terms).mean()
                if retrieval_cost_terms
                else torch.zeros((), device=hp.device)
            )
            source_scores = []
            for source in (EvidenceSource.SELF, EvidenceSource.PEER, EvidenceSource.MEDICAL, None):
                scores = (
                    [max(item.score, 0.0) for item in evidence_tuple if item.source == source]
                    if source is not None
                    else [float(prior_probability.detach()) if self.config["retrieval"]["prior_enabled"] else 0.0]
                )
                source_scores.append(sum(scores) / max(len(scores), 1))
            source_target = torch.tensor(source_scores, device=hp.device).softmax(dim=0)
            routed_weights = last_routed.weights if last_routed is not None else torch.full((4,), 0.25, device=hp.device)
            losses["retrieval"] = F.mse_loss(routed_weights, source_target)
            margin_direction = 1.0 if label == 1 else -1.0
            evidence_margin = representations.support_strength - representations.refute_strength
            losses["support_refute"] = F.relu(torch.tensor(0.2, device=hp.device) - margin_direction * evidence_margin)
            losses["consistency"] = (
                probability - representations.evidence_probability.detach()
            ) ** 2 if self.config["evidence"]["consistency_loss"] else torch.zeros((), device=hp.device)
            losses["calibration"] = (raw_probability - float(label)) ** 2
            losses["selective"] = (
                (1.0 - abstention.squeeze()) * losses["classification"] + 0.05 * abstention.squeeze()
                if abstention_enabled
                else torch.zeros((), device=hp.device)
            )
            losses["polarity"] = torch.stack(polarity_losses).mean() if polarity_losses else torch.zeros((), device=hp.device)
            weights = self.config["losses"]
            loss = sum(losses[name] * float(weights[name]) for name in losses)
        elif label == 2:
            loss = -torch.log(abstention.squeeze().clamp_min(1e-6))
        trace = {
            "router": router_traces,
            "iterations": len(router_traces),
            "evidence_count": len(evidence_tuple),
            "evidence": [item.safe_trace() for item in evidence_tuple],
            "contradiction_rate": representations.contradiction_rate,
            "active_hyperedges": len(graph.active_edges),
            "total_hyperedges": len(graph.hyperedges),
            "prior_used": bool(self.config["retrieval"]["prior_enabled"]),
            "model_family": self.capabilities.family,
            "query_schema": ["h_patient", "h_disease", "interaction", "delta_t", "context"],
            "active_node_types": sorted({node.node_type for node in graph.nodes.values()}),
            "active_hyperedge_types": sorted({edge.edge_type for edge in graph.active_edges}),
            "calibrated_temperature": float(calibrated_temperature.detach()),
            "calibrated_abstention_threshold": float(self.abstention_threshold),
            "raw_probability": float(raw_probability.detach()),
            "calibrated_probability": float(probability.detach()),
            "target_label_masked": True,
        }
        return DyPHOutput(logits, probability, abstention, prediction, loss, evidence_tuple, trace)
