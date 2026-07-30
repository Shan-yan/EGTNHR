from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import timedelta

from src.dyphrag.contracts import EvidenceItem, EvidenceSource, SplitManifest
from src.dyphrag.data import synthetic_bundle
from src.dyphrag.leakage import LeakageError, audit_examples, audit_retrieval


class LeakageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = synthetic_bundle(train_size=8, validation_size=4, test_size=4)

    def test_synthetic_manifest_is_patient_disjoint(self) -> None:
        audit_examples((*self.bundle.train, *self.bundle.validation, *self.bundle.test), self.bundle.manifest)

    def test_duplicate_patient_split_fails_loudly(self) -> None:
        manifest = SplitManifest("synthetic", "1", ("same",), ("same",), ("other",), "documented")
        with self.assertRaisesRegex(LeakageError, "more than one split"):
            audit_examples((), manifest)

    def test_future_input_fails_loudly(self) -> None:
        example = self.bundle.test[0]
        future = replace(example.events[0], timestamp=example.cutoff_time + timedelta(seconds=1))
        with self.assertRaisesRegex(LeakageError, "after"):
            audit_examples((replace(example, events=(future,)),), self.bundle.manifest)

    def test_validation_peer_fails_loudly(self) -> None:
        query = self.bundle.test[0]
        evidence = EvidenceItem(
            "bad", EvidenceSource.PEER, "bad", query.cutoff_time, "safe synthetic evidence", 1.0,
            patient_id="held-out", split="validation",
        )
        with self.assertRaisesRegex(LeakageError, "training split"):
            audit_retrieval(query, (evidence,))

    def test_target_patient_peer_fails_loudly(self) -> None:
        query = self.bundle.test[0]
        evidence = EvidenceItem(
            "bad", EvidenceSource.PEER, "bad", query.cutoff_time, "safe synthetic evidence", 1.0,
            patient_id=query.patient_id, split="train",
        )
        with self.assertRaisesRegex(LeakageError, "own peer"):
            audit_retrieval(query, (evidence,))

    def test_target_label_metadata_fails_loudly(self) -> None:
        query = self.bundle.test[0]
        event = replace(query.events[0], metadata={"target_label": 1})
        with self.assertRaisesRegex(LeakageError, "Forbidden"):
            audit_examples((replace(query, events=(event,)),), self.bundle.manifest)

    def test_discharge_diagnosis_fails_loudly(self) -> None:
        query = self.bundle.test[0]
        event = replace(query.events[0], metadata={"at_discharge": True})
        with self.assertRaisesRegex(LeakageError, "discharge diagnosis"):
            audit_examples((replace(query, events=(event,)),), self.bundle.manifest)


if __name__ == "__main__":
    unittest.main()
