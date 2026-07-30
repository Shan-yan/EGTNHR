from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import timedelta

import torch

from src.dyphrag.baselines import MODEL_REGISTRY, MedicalPredictor
from src.dyphrag.config import CONFIG_ROOT, load_config
from src.dyphrag.data import synthetic_bundle
from src.dyphrag.hypergraph import NODE_TYPES, NumericalEncoder, build_base_hypergraph
from src.dyphrag.inference import predict
from src.dyphrag.leakage import LeakageError, audit_examples
from src.dyphrag.model import DyPHRAGModel
from src.dyphrag.retrieval import PeerPatientRetriever, RetrievalSystem, SelfHistoryRetriever
from src.dyphrag.router import DynamicRouter


class SchemaAndRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = synthetic_bundle(train_size=8, validation_size=4, test_size=4)
        self.config = load_config(["experiment=full_dyphrag", "dataset=synthetic"])

    def test_required_node_schema_and_numeric_audit_fields(self) -> None:
        required = {
            "patient", "disease", "diagnosis", "medication", "procedure",
            "lab/vital", "time", "visit", "context", "external evidence",
        }
        self.assertTrue(required <= NODE_TYPES)
        graph = build_base_hypergraph(self.bundle.test[0], 32, NumericalEncoder(32))
        numeric = next(node for node in graph.nodes.values() if node.node_type == "value")
        self.assertTrue(
            {
                "raw_value", "normalized_value", "unit", "canonical_unit", "reference_low",
                "reference_high", "abnormal_direction", "measurement_condition", "time",
                "slope", "baseline_deviation", "missing", "observed",
            }
            <= numeric.attributes.keys()
        )
        self.assertIn("numeric_state", {edge.edge_type for edge in graph.hyperedges.values()})

    def test_self_history_exposes_all_required_granularities(self) -> None:
        result = SelfHistoryRetriever().retrieve(self.bundle.test[0], top_k=100)
        unit_types = {item.source_type for item in result.evidence}
        self.assertTrue(
            {"visit", "event", "episode", "numeric_trend_window", "patient-state-hyperedge"}
            <= unit_types
        )

    def test_peer_index_is_unique_patient_level(self) -> None:
        duplicated = (*self.bundle.train, *self.bundle.train)
        retriever = PeerPatientRetriever(duplicated)
        self.assertEqual(len(retriever.bank), len({item.patient_id for item in self.bundle.train}))
        result = retriever.retrieve(self.bundle.test[0], top_k=20, candidate_k=20)
        self.assertTrue(all(item.patient_id != self.bundle.test[0].patient_id for item in result.evidence))

    def test_router_budget_is_bounded_and_source_specific(self) -> None:
        for mode in ("learned_soft", "sparse_gumbel", "uniform", "heuristic"):
            router = DynamicRouter(32, mode, max_total_k=9)
            output = router(torch.zeros(32 * 4 + 1))
            self.assertLessEqual(sum(output.budgets.values()), 9, mode)
            self.assertGreaterEqual(sum(output.budgets.values()), 0, mode)
            self.assertEqual(set(output.continue_sources), {"self", "peer", "medical"})

    def test_router_can_choose_prior_only_and_full_retrieval(self) -> None:
        router = DynamicRouter(32, "learned_soft", max_total_k=9)
        query = torch.zeros(32 * 4 + 1)
        with torch.no_grad():
            router.retrieval_gate_head[-1].weight.zero_()
            router.retrieval_gate_head[-1].bias.fill_(-20)
        prior_only = router(query)
        self.assertTrue(prior_only.prior_only)
        self.assertEqual(sum(prior_only.budgets.values()), 0)
        with torch.no_grad():
            router.retrieval_gate_head[-1].bias.fill_(20)
            router.source_gate_head[-1].weight.zero_()
            router.source_gate_head[-1].bias.fill_(20)
        retrieved = router(query)
        self.assertFalse(retrieved.prior_only)
        self.assertEqual(sum(retrieved.budgets.values()), 9)

    def test_uniform_predictor_registry_and_query_schema(self) -> None:
        requested = {
            "ehr_gru", "ehr_transformer", "static_heterogeneous_graph", "visit_hypergraph",
            "graphcare_adapted", "kare_adapted", "static_vector_rag",
            "disease_conditioned_retrieval", "dynamic_no_polarity", "full_dyphrag",
        }
        self.assertTrue(requested <= MODEL_REGISTRY.keys())
        model = DyPHRAGModel(self.config, RetrievalSystem(self.bundle.train, self.config, 42))
        self.assertIsInstance(model, MedicalPredictor)
        output = model(self.bundle.test[0])
        self.assertEqual(
            output.trace["query_schema"],
            ["h_patient", "h_disease", "interaction", "delta_t", "context"],
        )
        result = predict(model, self.bundle.test[0])
        self.assertIn(result.answer, {"yes", "no", "insufficient_evidence"})
        self.assertTrue(0.0 <= result.calibrated_probability <= 1.0)
        self.assertNotIn("prior", {item["source"] for item in result.evidence})
        medical = [item for item in result.evidence if item["source"] == "medical"]
        self.assertTrue(all(item["citation_id"] and item["provenance_locator"] for item in medical))

    def test_exact_requested_ablation_configs_exist(self) -> None:
        names = {
            "ablation_no_self", "ablation_no_peer", "ablation_no_medical", "ablation_no_prior",
            "ablation_uniform_router", "ablation_no_disease_conditioning",
            "ablation_no_time_conditioning", "ablation_fixed_topk", "ablation_single_shot",
            "ablation_frozen_retriever", "ablation_random_retrieval",
            "ablation_no_numeric_hyperedges", "ablation_no_cohort_hyperedges",
            "ablation_visit_only", "ablation_static_hypergraph", "ablation_add_only",
            "ablation_reweight_only", "ablation_no_deactivation",
            "ablation_ordinary_graph_expansion", "ablation_no_temporal_encoding",
            "ablation_numeric_bucketing", "ablation_no_refute", "ablation_no_differential",
            "ablation_no_polarity_classifier", "ablation_no_consistency_loss",
            "ablation_no_source_reliability", "ablation_no_abstention",
            "ablation_no_iterative_query", "ablation_shuffled_evidence_negative_control",
        }
        existing = {path.stem for path in (CONFIG_ROOT / "experiment").glob("*.yaml")}
        self.assertTrue(names <= existing)

    def test_late_availability_timestamp_fails(self) -> None:
        example = self.bundle.test[0]
        event = replace(
            example.events[0],
            metadata={"available_time": (example.cutoff_time + timedelta(seconds=1)).isoformat()},
        )
        with self.assertRaisesRegex(LeakageError, "available"):
            audit_examples((replace(example, events=(event,)),), self.bundle.manifest)


if __name__ == "__main__":
    unittest.main()
