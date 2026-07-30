from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.dyphrag.config import PROJECT_ROOT, load_config
from src.train import ControlledInterruption, run_training


class InterruptionResumeTests(unittest.TestCase):
    def _config(self, output: Path, interrupt_after: int) -> dict:
        return load_config(
            [
                "experiment=full_dyphrag",
                "dataset=synthetic",
                "seed=123",
                f"output_dir={output}",
                "data.train_size=4",
                "data.validation_size=4",
                "data.test_size=4",
                "training.epochs=1",
                "evaluation.bootstrap_samples=0",
                "runtime.device=cpu",
                "runtime.progress=false",
                "tracking.enabled=false",
                f"runtime.interrupt_after_steps={interrupt_after}",
            ]
        )

    def test_interrupted_run_resumes_to_identical_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dyphrag-resume-", dir=PROJECT_ROOT / "results") as directory:
            root = Path(directory)
            resumed = root / "resumed"
            reference = root / "reference"
            with self.assertRaises(ControlledInterruption):
                run_training(self._config(resumed, 2))
            interrupted = torch.load(resumed / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(interrupted["state"]["global_step"], 2)

            resumed_summary = run_training(self._config(resumed, 0))
            reference_summary = run_training(self._config(reference, 0))
            self.assertEqual(resumed_summary["status"], "succeeded")
            self.assertEqual(reference_summary["status"], "succeeded")

            resumed_state = torch.load(resumed / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
            reference_state = torch.load(reference / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed_state["state"], reference_state["state"])
            for name, tensor in resumed_state["model"].items():
                self.assertTrue(torch.equal(tensor, reference_state["model"][name]), name)


if __name__ == "__main__":
    unittest.main()
