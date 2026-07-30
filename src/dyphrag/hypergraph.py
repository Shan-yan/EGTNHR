"""Typed temporal patient hypergraph and retrieval-driven update operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Literal, Sequence

import torch
from torch import nn

from .contracts import ClinicalEvent, EvidenceItem, PatientDiseaseExample, parse_time, stable_hash
from .features import hashed_vector


NODE_TYPES = {
    "patient",
    "disease",
    "diagnosis",
    "medication",
    "procedure",
    "lab/vital",
    "time",
    "visit",
    "context",
    "external evidence",
    "patient_state",
    # Internal value nodes retain the complete numeric measurement record.
    "value",
}
HYPEREDGE_TYPES = {
    "visit", "numeric_state", "treatment", "chronic_context", "temporal_transition", "cohort", "evidence"
}


@dataclass
class HyperNode:
    node_id: str
    node_type: str
    features: torch.Tensor
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hyperedge:
    edge_id: str
    edge_type: str
    node_ids: tuple[str, ...]
    weight: float = 1.0
    active: bool = True
    immutable_raw: bool = False
    derived: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientHypergraph:
    nodes: dict[str, HyperNode] = field(default_factory=dict)
    hyperedges: dict[str, Hyperedge] = field(default_factory=dict)
    cutoff_time: datetime | None = None

    def add_node(self, node: HyperNode) -> None:
        if node.node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {node.node_type}")
        self.nodes.setdefault(node.node_id, node)

    def add_hyperedge(self, edge: Hyperedge) -> None:
        if edge.edge_type not in HYPEREDGE_TYPES:
            raise ValueError(f"Unknown hyperedge type: {edge.edge_type}")
        missing = set(edge.node_ids) - self.nodes.keys()
        if missing:
            raise ValueError(f"Hyperedge references missing nodes: {sorted(missing)}")
        self.hyperedges[edge.edge_id] = edge

    def deactivate(self, edge_id: str) -> None:
        edge = self.hyperedges[edge_id]
        if edge.immutable_raw:
            raise ValueError("Immutable raw EHR hyperedges can never be deactivated")
        edge.active = False

    @property
    def active_edges(self) -> tuple[Hyperedge, ...]:
        return tuple(edge for edge in self.hyperedges.values() if edge.active)


class NumericalEncoder(nn.Module):
    def __init__(self, output_dim: int, mode: str = "continuous_mlp") -> None:
        super().__init__()
        self.mode = mode
        input_dims = {
            "continuous_mlp": 6,
            "disease_conditioned": 6,
            "fourier": 12,
            "spline": 11,
            "monotonic": 6,
            "bucketing": 6,
        }
        if mode not in input_dims:
            raise ValueError(f"Unknown numerical encoder: {mode}")
        self.projection = nn.Sequential(nn.Linear(input_dims[mode], output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
        self.disease_film = (
            nn.Linear(output_dim, output_dim * 2)
            if mode == "disease_conditioned"
            else None
        )

    def _basis(self, values: torch.Tensor) -> torch.Tensor:
        if self.mode in {"continuous_mlp", "disease_conditioned", "monotonic"}:
            basis = values
        elif self.mode == "bucketing":
            basis = torch.stack(
                [
                    torch.bucketize(value, torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=values.device)).float() / 5.0
                    for value in values
                ]
            )
        elif self.mode == "fourier":
            basis = torch.cat([torch.sin(values), torch.cos(values)], dim=-1)
        else:
            knots = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0], device=values.device)
            basis = torch.relu(values[0] - knots)
            basis = torch.cat([basis, values[1:]])
        output = self.projection(basis)
        return torch.nn.functional.softplus(output) if self.mode == "monotonic" else output

    def forward(
        self,
        event: ClinicalEvent,
        device: torch.device,
        temporal: bool = True,
        disease_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        measurement = event.numeric
        if measurement is None:
            return torch.zeros(self.projection[-1].out_features, device=device)
        values = torch.tensor(
            [
                measurement.normalized_value,
                (measurement.slope or 0.0) if temporal else 0.0,
                measurement.baseline_deviation or 0.0,
                ((measurement.time_since_prior_hours or 0.0) / 720.0) if temporal else 0.0,
                float(measurement.missing),
                float(measurement.observed),
            ],
            dtype=torch.float32,
            device=device,
        )
        encoded = self._basis(values)
        if self.mode == "disease_conditioned":
            if disease_embedding is None:
                raise ValueError("disease_conditioned numerical encoding requires h_d")
            gamma, beta = self.disease_film(disease_embedding.to(device)).chunk(2)
            encoded = encoded * (1.0 + torch.tanh(gamma)) + beta
        return encoded


def build_base_hypergraph(
    example: PatientDiseaseExample,
    hidden_dim: int,
    numeric_encoder: NumericalEncoder,
    include_numeric: bool = True,
    visit_only: bool = False,
    include_temporal: bool = True,
    disease_conditioned: bool = True,
    patient_embedding: torch.Tensor | None = None,
    disease_embedding: torch.Tensor | None = None,
) -> PatientHypergraph:
    device = next(numeric_encoder.parameters()).device
    graph = PatientHypergraph(cutoff_time=example.cutoff_time)
    patient_node = "patient:current"
    disease_token = example.target_disease if disease_conditioned else "target_disease:masked"
    disease_node = f"disease:{stable_hash(disease_token)[:12]}"
    graph.add_node(
        HyperNode(
            patient_node,
            "patient",
            patient_embedding if patient_embedding is not None else hashed_vector(["patient"], hidden_dim, device),
        )
    )
    graph.add_node(
        HyperNode(
            disease_node,
            "disease",
            disease_embedding
            if disease_embedding is not None
            else hashed_vector([disease_token], hidden_dim, device),
        )
    )
    context_nodes = []
    for key, value in sorted(example.context.items()):
        context_id = f"context:{stable_hash(key)[:12]}"
        graph.add_node(
            HyperNode(
                context_id,
                "context",
                hashed_vector([key, f"{float(value):.4f}"], hidden_dim, device),
                {"context_type": key, "value": float(value)},
            )
        )
        context_nodes.append(context_id)
    if context_nodes and not visit_only:
        graph.add_hyperedge(
            Hyperedge(
                "context:baseline",
                "chronic_context",
                tuple((patient_node, disease_node, *context_nodes)),
                immutable_raw=True,
            )
        )
    visits: dict[str, list[str]] = {}
    visit_events: dict[str, list[tuple[ClinicalEvent, str, str]]] = {}
    for index, event in enumerate(example.prior_events):
        event_ref = stable_hash(event.event_id)[:12]
        node_type = {
            "diagnosis": "diagnosis",
            "medication": "medication",
            "procedure": "procedure",
            "lab": "lab/vital",
            "vital": "lab/vital",
            "context": "context",
        }[event.event_type]
        node_id = f"{node_type}:{event_ref}"
        feature = hashed_vector([event.concept_id, event.event_type], hidden_dim, device)
        if include_numeric and event.numeric is not None:
            feature = feature + numeric_encoder(event, device, include_temporal, disease_embedding)
        graph.add_node(
            HyperNode(
                node_id,
                node_type,
                feature,
                {
                    "timestamp": event.timestamp.isoformat(),
                    "concept_id": event.concept_id,
                    "event_id_hash": stable_hash(event.event_id)[:16],
                },
            )
        )
        time_id = f"time:{index}"
        time_tokens = [event.timestamp.isoformat()] if include_temporal else ["time:masked"]
        graph.add_node(HyperNode(time_id, "time", hashed_vector(time_tokens, hidden_dim, device)))
        visit_key = event.visit_id or f"unassigned-{index}"
        visits.setdefault(visit_key, []).extend([node_id, time_id])
        visit_events.setdefault(visit_key, []).append((event, node_id, time_id))
        if include_numeric and event.numeric is not None and not visit_only:
            value_id = f"value:{event_ref}"
            graph.add_node(
                HyperNode(
                    value_id,
                    "value",
                    numeric_encoder(event, device, include_temporal, disease_embedding),
                    {
                        "concept_id": event.numeric.concept_id,
                        "raw_value": event.numeric.raw_value,
                        "normalized_value": event.numeric.normalized_value,
                        "unit": event.numeric.unit,
                        "canonical_unit": event.numeric.canonical_unit,
                        "reference_low": event.numeric.reference_low,
                        "reference_high": event.numeric.reference_high,
                        "abnormal_direction": event.numeric.abnormal_direction,
                        "context": event.numeric.context,
                        "measurement_condition": event.numeric.measurement_condition or event.numeric.context,
                        "time": (event.numeric.measurement_time or event.timestamp).isoformat(),
                        "timestamp": event.timestamp.isoformat(),
                        "time_since_prior_hours": event.numeric.time_since_prior_hours,
                        "slope": event.numeric.slope,
                        "baseline_deviation": event.numeric.baseline_deviation,
                        "missing": event.numeric.missing,
                        "observed": event.numeric.observed,
                    },
                )
            )
            visits[visit_key].append(value_id)
        if event.event_type == "context" and not visit_only:
            graph.add_hyperedge(Hyperedge(f"context:{event_ref}", "chronic_context", (patient_node, disease_node, node_id), immutable_raw=True))
    visit_states: dict[str, tuple[str, datetime]] = {}
    for visit_id, members in visits.items():
        visit_node = f"visit:{stable_hash(visit_id)[:12]}"
        graph.add_node(
            HyperNode(
                visit_node,
                "visit",
                hashed_vector([visit_id], hidden_dim, device),
                {"visit_ref": stable_hash(visit_id)[:16]},
            )
        )
        records = visit_events[visit_id]
        state_id = f"patient_state:{stable_hash(visit_id)[:12]}"
        state_members = [
            graph.nodes[node_id].features.to(device)
            for _, node_id, _ in records
        ]
        state_features = (
            torch.stack(state_members).mean(dim=0)
            if state_members
            else graph.nodes[patient_node].features
        )
        state_time = max(event.timestamp for event, _, _ in records)
        graph.add_node(
            HyperNode(
                state_id,
                "patient_state",
                state_features,
                {
                    "visit_ref": stable_hash(visit_id)[:16],
                    "timestamp": state_time.isoformat(),
                },
            )
        )
        visit_states[visit_id] = (state_id, state_time)
        graph.add_hyperedge(
            Hyperedge(
                f"visit:{stable_hash(visit_id)[:12]}",
                "visit",
                tuple(dict.fromkeys((patient_node, disease_node, visit_node, state_id, *members))),
                immutable_raw=True,
            )
        )
        if include_numeric and not visit_only:
            numeric_members = [
                node_id
                for event, node_id, _ in records
                if event.numeric is not None
            ]
            value_members = [
                f"value:{stable_hash(event.event_id)[:12]}"
                for event, _, _ in records
                if event.numeric is not None
            ]
            medication_members = [
                node_id
                for event, node_id, _ in records
                if event.event_type == "medication"
            ]
            context_members_for_state = [
                node_id
                for event, node_id, _ in records
                if event.event_type == "context"
            ]
            time_members = [time_id for event, _, time_id in records if event.numeric is not None]
            if numeric_members:
                graph.add_hyperedge(
                    Hyperedge(
                        f"numeric_state:{stable_hash(visit_id)[:12]}",
                        "numeric_state",
                        tuple(
                            dict.fromkeys(
                                (
                                    patient_node,
                                    disease_node,
                                    visit_node,
                                    state_id,
                                    *numeric_members,
                                    *value_members,
                                    *medication_members,
                                    *context_nodes,
                                    *context_members_for_state,
                                    *time_members,
                                )
                            )
                        ),
                        immutable_raw=True,
                        attributes={
                            "schema": "numeric values + medication + context + time",
                            "measurement_count": len(numeric_members),
                        },
                    )
                )
    if not visit_only:
        ordered_visits = sorted(visit_states.items(), key=lambda item: item[1][1])
        prior_state = patient_node
        prior_time: datetime | None = None
        for visit_id, (state_id, state_time) in ordered_visits:
            if include_temporal and prior_state != patient_node:
                graph.add_hyperedge(
                    Hyperedge(
                        f"transition:{stable_hash(visit_id)[:12]}",
                        "temporal_transition",
                        (patient_node, disease_node, prior_state, state_id),
                        immutable_raw=True,
                        attributes={
                            "from_time": prior_time.isoformat() if prior_time else None,
                            "to_time": state_time.isoformat(),
                            "delta_hours": (
                                (state_time - prior_time).total_seconds() / 3600.0
                                if prior_time is not None
                                else None
                            ),
                        },
                    )
                )
            for event, node_id, time_id in visit_events[visit_id]:
                if event.event_type not in {"medication", "procedure"}:
                    continue
                event_ref = stable_hash(event.event_id)[:12]
                graph.add_hyperedge(
                    Hyperedge(
                        f"treatment:{event_ref}",
                        "treatment",
                        (patient_node, disease_node, prior_state, node_id, state_id, time_id),
                        immutable_raw=True,
                        attributes={
                            "pre_state": prior_state,
                            "post_state": state_id,
                            "treatment_concept": event.concept_id,
                            "timestamp": event.timestamp.isoformat(),
                        },
                    )
                )
            prior_state = state_id
            prior_time = state_time
    return graph


def add_cohort_hyperedge(
    graph: PatientHypergraph,
    cohort_members: Sequence[tuple[str, Sequence[float], str]],
    cohort_hash: str,
) -> None:
    """Add train-peer nodes using their cutoff-safe snapshot embeddings."""
    if not cohort_members:
        return
    patient_node = "patient:current"
    nodes = [patient_node]
    snapshot_cutoffs: dict[str, str] = {}
    for ref, embedding, snapshot_cutoff in cohort_members:
        node_id = f"patient:peer:{stable_hash(ref)[:12]}"
        feature = torch.as_tensor(
            embedding,
            dtype=graph.nodes[patient_node].features.dtype,
            device=graph.nodes[patient_node].features.device,
        )
        if feature.numel() != graph.nodes[patient_node].features.numel():
            raise ValueError("Peer snapshot embedding dimension does not match graph hidden dimension")
        graph.add_node(
            HyperNode(
                node_id,
                "patient",
                feature,
                {
                    "peer_ref": stable_hash(ref)[:16],
                    "snapshot_cutoff": snapshot_cutoff,
                    "train_only": True,
                },
            )
        )
        nodes.append(node_id)
        snapshot_cutoffs[node_id] = snapshot_cutoff
    graph.add_hyperedge(
        Hyperedge(
            f"cohort:{cohort_hash[:12]}",
            "cohort",
            tuple(nodes),
            derived=True,
            attributes={
                "corpus_hash": cohort_hash,
                "snapshot_cutoffs": snapshot_cutoffs,
                "train_only": True,
            },
        )
    )


class HypergraphUpdater:
    def __init__(self, operator: str, hidden_dim: int, deactivation_threshold: float = 0.05) -> None:
        allowed = {"add_only", "reweight_only", "add_reweight", "add_deactivate_reweight", "static"}
        if operator not in allowed:
            raise ValueError(f"Unknown graph update operator: {operator}")
        self.operator = operator
        self.hidden_dim = hidden_dim
        self.deactivation_threshold = deactivation_threshold

    @staticmethod
    def _aligned_nodes(graph: PatientHypergraph, item: EvidenceItem) -> tuple[tuple[str, ...], float]:
        concepts = {
            str(value).strip().lower()
            for value in item.metadata.get("concepts", ())
            if str(value).strip()
        }
        if not concepts and item.provenance_span:
            concepts = {
                value.strip().lower()
                for value in item.provenance_span.split(";")
                if value.strip()
            }
        matches = tuple(
            node.node_id
            for node in graph.nodes.values()
            if str(node.attributes.get("concept_id", "")).strip().lower() in concepts
        )
        return matches, min(len(matches) / max(len(concepts), 1), 1.0) if concepts else 0.0

    def update(self, graph: PatientHypergraph, evidence: Iterable[EvidenceItem], iteration: int) -> PatientHypergraph:
        if self.operator == "static":
            return graph
        can_add = self.operator in {"add_only", "add_reweight", "add_deactivate_reweight"}
        can_reweight = self.operator in {"reweight_only", "add_reweight", "add_deactivate_reweight"}
        for item in evidence:
            aligned_nodes, alignment_confidence = self._aligned_nodes(graph, item)
            confidence = max(
                float(item.score) * float(item.reliability) * float(item.applicability),
                0.0,
            )
            evidence_ref = stable_hash(item.evidence_id)[:12]
            edge_id = f"evidence:{evidence_ref}"
            if edge_id in graph.hyperedges and can_reweight:
                edge = graph.hyperedges[edge_id]
                edge.weight = max(0.5 * edge.weight + 0.5 * confidence, 0.0)
                edge.active = True
            elif can_add and edge_id not in graph.hyperedges:
                node_id = f"evidence:{evidence_ref}"
                graph.add_node(
                    HyperNode(
                        node_id,
                        "external evidence",
                        hashed_vector([item.text], self.hidden_dim),
                        {
                            "source": item.source.value,
                            "source_ref": stable_hash(item.source_id)[:16],
                            "provenance": item.provenance_span,
                            "concepts": tuple(item.metadata.get("concepts", ())),
                        },
                    )
                )
                disease_node = next(
                    node.node_id for node in graph.nodes.values() if node.node_type == "disease"
                )
                graph.add_hyperedge(
                    Hyperedge(
                        edge_id,
                        "evidence",
                        tuple(dict.fromkeys(("patient:current", disease_node, *aligned_nodes, node_id))),
                        weight=confidence,
                        derived=True,
                        attributes={
                            "polarity": item.polarity.value,
                            "source": item.source.value,
                            "reliability": item.reliability,
                            "applicability": item.applicability,
                            "iteration_added": iteration,
                            "alignment_confidence": alignment_confidence,
                            "aligned_node_count": len(aligned_nodes),
                            "confidence": confidence,
                        },
                    )
                )
        if self.operator == "add_deactivate_reweight":
            for edge in graph.hyperedges.values():
                if (
                    edge.derived
                    and edge.edge_type == "evidence"
                    and edge.weight < self.deactivation_threshold
                ):
                    graph.deactivate(edge.edge_id)
        return graph


class HeterogeneousHypergraphTransformer(nn.Module):
    """Typed temporal hyperedge self-attention followed by patient cross-attention."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_types = sorted(HYPEREDGE_TYPES)
        self.node_types = sorted(NODE_TYPES)
        self.edge_type_embedding = nn.Embedding(len(self.edge_types), hidden_dim)
        self.node_type_embedding = nn.Embedding(len(self.node_types), hidden_dim)
        self.temporal_projection = nn.Linear(4, hidden_dim)
        self.edge_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.patient_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))

    def _temporal_features(
        self,
        graph: PatientHypergraph,
        edge: Hyperedge,
        device: torch.device,
    ) -> torch.Tensor:
        timestamps = []
        for node_id in edge.node_ids:
            value = graph.nodes[node_id].attributes.get("timestamp")
            if value:
                timestamps.append(parse_time(value))
        if not timestamps or graph.cutoff_time is None:
            delta_days = 0.0
        else:
            delta_days = max(
                (graph.cutoff_time - max(timestamps)).total_seconds() / 86400.0,
                0.0,
            )
        scaled = torch.tensor(delta_days / 365.0, device=device)
        return torch.stack(
            [scaled, torch.log1p(scaled), torch.sin(scaled), torch.cos(scaled)]
        )

    def forward(self, graph: PatientHypergraph) -> torch.Tensor:
        patient = graph.nodes["patient:current"].features
        edge_states: list[torch.Tensor] = []
        for edge in graph.active_edges:
            member_features = torch.stack(
                [
                    graph.nodes[node_id].features.to(patient.device)
                    + self.node_type_embedding.weight[
                        self.node_types.index(graph.nodes[node_id].node_type)
                    ]
                    for node_id in edge.node_ids
                ]
            )
            pooled = (
                member_features.mean(dim=0)
                + self.edge_type_embedding.weight[self.edge_types.index(edge.edge_type)]
                + self.temporal_projection(self._temporal_features(graph, edge, patient.device))
            )
            edge_states.append(pooled * edge.weight)
        if not edge_states:
            return patient
        stacked = torch.stack(edge_states).unsqueeze(0)
        contextual, _ = self.edge_attention(stacked, stacked, stacked, need_weights=False)
        contextual = self.edge_norm(contextual + stacked)
        patient_query = patient.reshape(1, 1, -1)
        aggregate, _ = self.patient_attention(
            patient_query,
            contextual,
            contextual,
            need_weights=False,
        )
        aggregate = aggregate.reshape(-1)
        return self.output(torch.cat([patient, aggregate]))


class SimpleHypergraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, mode: Literal["hgnn", "hgat", "clique", "star"]) -> None:
        super().__init__()
        self.mode = mode
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, graph: PatientHypergraph) -> torch.Tensor:
        patient = graph.nodes["patient:current"].features
        if self.mode == "hgnn":
            edge_states = [
                torch.stack([graph.nodes[node].features.to(patient.device) for node in edge.node_ids]).mean(dim=0)
                for edge in graph.active_edges
            ]
            aggregate = torch.stack(edge_states).mean(dim=0) if edge_states else patient
        elif self.mode == "hgat":
            edge_states = [
                torch.stack([graph.nodes[node].features.to(patient.device) for node in edge.node_ids]).mean(dim=0)
                for edge in graph.active_edges
            ]
            if edge_states:
                stacked = torch.stack(edge_states)
                aggregate = (torch.softmax(self.attention(stacked).squeeze(-1), dim=0)[:, None] * stacked).sum(dim=0)
            else:
                aggregate = patient
        elif self.mode == "clique":
            neighbors = [
                graph.nodes[node].features.to(patient.device)
                for edge in graph.active_edges
                if "patient:current" in edge.node_ids
                for node in edge.node_ids
                if node != "patient:current"
            ]
            aggregate = torch.stack(neighbors).mean(dim=0) if neighbors else patient
        else:  # star expansion: each hyperedge is represented by one center state.
            centers = [
                torch.stack([graph.nodes[node].features.to(patient.device) for node in edge.node_ids]).sum(dim=0)
                / max(len(edge.node_ids), 1)
                for edge in graph.active_edges
            ]
            aggregate = torch.stack(centers).mean(dim=0) if centers else patient
        return torch.tanh(self.projection(0.5 * patient + 0.5 * aggregate))
