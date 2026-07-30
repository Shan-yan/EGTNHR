from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import timedelta

import torch

from src.dyphrag.config import load_config
from src.dyphrag.data import synthetic_bundle
from src.dyphrag.hypergraph import (
    HypergraphUpdater,
    NumericalEncoder,
    add_cohort_hyperedge,
    build_base_hypergraph,
)
from src.dyphrag.model import DyPHRAGModel
from src.dyphrag.retrieval import (
    MedicalKnowledgeRetriever,
    PeerPatientRetriever,
    RetrievalSystem,
    SelfHistoryRetriever,
    default_medical_corpus,
)


class ComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(["experiment=full_dyphrag", "dataset=synthetic", "seed=42"])
        self.bundle = synthetic_bundle(train_size=8, validation_size=4, test_size=4)

    def test_peer_bank_rejects_held_out_examples(self) -> None:
        with self.assertRaisesRegex(Exception, "training"):
            PeerPatientRetriever(self.bundle.validation)

    def test_retrieval_is_cutoff_safe_and_train_only(self) -> None:
        system = RetrievalSystem(self.bundle.train, self.config, 42)
        evidence = system.retrieve(self.bundle.test[0], {"self": 2, "peer": 2, "medical": 2}, 0)
        self.assertTrue(evidence)
        self.assertTrue(all(item.timestamp <= self.bundle.test[0].cutoff_time for item in evidence))
        self.assertTrue(all(item.split == "train" for item in evidence if item.source.value == "peer"))

    def test_medical_polarity_is_relative_to_target_disease(self) -> None:
        query = replace(self.bundle.test[0], target_disease="disease_alpha")
        evidence = MedicalKnowledgeRetriever(default_medical_corpus()).retrieve(query, top_k=99).evidence
        beta_support = next(item for item in evidence if item.source_id == "synthetic-guideline-beta")
        self.assertEqual(beta_support.polarity.value, "differential")
        self.assertEqual(beta_support.differential_disease, "disease_beta")

    def test_raw_hyperedges_are_immutable(self) -> None:
        encoder = NumericalEncoder(32)
        graph = build_base_hypergraph(self.bundle.test[0], 32, encoder)
        raw = next(edge for edge in graph.hyperedges.values() if edge.immutable_raw)
        with self.assertRaisesRegex(ValueError, "Immutable"):
            graph.deactivate(raw.edge_id)

    def test_reweight_only_never_changes_raw_ehr_edges(self) -> None:
        encoder = NumericalEncoder(32)
        graph = build_base_hypergraph(self.bundle.test[0], 32, encoder)
        before = {edge.edge_id: edge.weight for edge in graph.hyperedges.values() if edge.immutable_raw}
        evidence = RetrievalSystem(self.bundle.train, self.config, 42).retrieve(
            self.bundle.test[0], {"self": 1, "peer": 1, "medical": 1}, 0
        )
        HypergraphUpdater("reweight_only", 32).update(graph, evidence, 0)
        after = {edge.edge_id: edge.weight for edge in graph.hyperedges.values() if edge.immutable_raw}
        self.assertEqual(before, after)

    def test_all_numeric_encoders(self) -> None:
        for mode in ("continuous_mlp", "fourier", "spline", "monotonic", "bucketing"):
            encoder = NumericalEncoder(32, mode)
            graph = build_base_hypergraph(self.bundle.test[0], 32, encoder)
            self.assertTrue(graph.hyperedges, mode)

    def test_disease_conditioned_numeric_encoder_uses_target_embedding(self) -> None:
        encoder = NumericalEncoder(32, "disease_conditioned")
        event = next(event for event in self.bundle.test[0].prior_events if event.numeric is not None)
        first = encoder(event, torch.device("cpu"), disease_embedding=torch.zeros(32))
        second = encoder(event, torch.device("cpu"), disease_embedding=torch.ones(32))
        self.assertFalse(torch.allclose(first, second))

    def test_peer_coarse_index_cannot_see_future_snapshot(self) -> None:
        base = self.bundle.train[0]
        future_event = replace(
            base.events[0],
            event_id="future-index-event",
            timestamp=base.cutoff_time + timedelta(days=1),
        )
        later = replace(
            base,
            cutoff_time=base.cutoff_time + timedelta(days=2),
            events=(*base.events, future_event),
        )
        query = replace(self.bundle.validation[0], cutoff_time=base.cutoff_time)
        base_retriever = PeerPatientRetriever((base,))
        future_retriever = PeerPatientRetriever((base, later))
        base_snapshot = base_retriever.eligible_snapshots(query)[0]
        future_safe_snapshot = future_retriever.eligible_snapshots(query)[0]
        self.assertEqual(base_snapshot.events, future_safe_snapshot.events)
        self.assertEqual(base_snapshot.vector, future_safe_snapshot.vector)
        self.assertNotIn("future-index-event", {event.event_id for event in future_safe_snapshot.events})

    def test_cohort_nodes_use_peer_snapshot_features(self) -> None:
        graph = build_base_hypergraph(self.bundle.test[0], 32, NumericalEncoder(32))
        peer_embedding = tuple(float(index) / 32.0 for index in range(32))
        add_cohort_hyperedge(
            graph,
            [("peer-a", peer_embedding, self.bundle.train[0].cutoff_time.isoformat())],
            "cohort-hash",
        )
        peer = next(
            node for node in graph.nodes.values()
            if node.node_type == "patient" and node.node_id != "patient:current"
        )
        self.assertTrue(torch.allclose(peer.features, torch.tensor(peer_embedding)))
        self.assertFalse(torch.allclose(peer.features, graph.nodes["patient:current"].features))

    def test_state_transition_and_treatment_edges_have_state_semantics(self) -> None:
        graph = build_base_hypergraph(self.bundle.test[0], 32, NumericalEncoder(32))
        treatments = [edge for edge in graph.hyperedges.values() if edge.edge_type == "treatment"]
        self.assertTrue(all("pre_state" in edge.attributes and "post_state" in edge.attributes for edge in treatments))
        self.assertIn("patient_state", {node.node_type for node in graph.nodes.values()})

    def test_self_retriever_filters_future_events(self) -> None:
        from dataclasses import replace
        from datetime import timedelta

        query = self.bundle.test[0]
        future = replace(query.events[0], event_id="future", timestamp=query.cutoff_time + timedelta(seconds=1))
        result = SelfHistoryRetriever().retrieve(replace(query, events=(*query.events, future)), top_k=99)
        self.assertNotIn("self:future", {item.evidence_id for item in result.evidence})

    def test_full_model_forward_has_no_prior_citation(self) -> None:
        model = DyPHRAGModel(self.config, RetrievalSystem(self.bundle.train, self.config, 42))
        output = model(self.bundle.test[0], self.bundle.test[0].label)
        self.assertEqual(output.logits.shape, torch.Size([2]))
        self.assertIsNotNone(output.loss)
        self.assertNotIn("prior", {item.source.value for item in output.evidence})
        self.assertIn(output.prediction, {"yes", "no", "insufficient_evidence"})
        self.assertNotIn("temperature", dict(model.named_parameters()))
        self.assertIn("temperature", dict(model.named_buffers()))


if __name__ == "__main__":
    unittest.main()
