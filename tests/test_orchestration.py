from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.dyphrag.runtime import write_json
from src.orchestrate import ALL_EXPERIMENTS, _experiments, aggregate


class OrchestrationTests(unittest.TestCase):
    def test_all_matrix_has_no_compatibility_alias_duplicates(self) -> None:
        self.assertEqual(_experiments("all"), ALL_EXPERIMENTS)
        self.assertEqual(len(ALL_EXPERIMENTS), len(set(ALL_EXPERIMENTS)))

    def test_aggregate_reports_multi_seed_mean_and_std(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dyphrag-aggregate-") as directory:
            root = Path(directory)
            state = {}
            for seed, auroc in ((42, 0.6), (43, 0.8)):
                run_dir = root / "runs" / f"full_dyphrag_seed{seed}"
                run_dir.mkdir(parents=True)
                write_json(
                    run_dir / "summary.json",
                    {
                        "status": "succeeded",
                        "metrics": {"auroc": auroc, "auprc": auroc - 0.1, "brier": 0.2, "ece": 0.1},
                        "efficiency": {"gpu_hours": 1.0, "peak_vram_gib": 2.0, "training_seconds": 3.0},
                    },
                )
                state[f"full_dyphrag_seed{seed}"] = {
                    "experiment": "full_dyphrag",
                    "seed": seed,
                    "run_dir": str(run_dir),
                }
            aggregate(root, state)
            statistics_rows = json.loads((root / "aggregate_stats.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(statistics_rows[0]["auroc_mean"], 0.7)
            self.assertGreater(statistics_rows[0]["auroc_std"], 0.0)
            with (root / "aggregate_stats.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            self.assertTrue((root / "aggregate.md").is_file())
            self.assertTrue((root / "reproducibility_bundle.tar.gz").is_file())


if __name__ == "__main__":
    unittest.main()
