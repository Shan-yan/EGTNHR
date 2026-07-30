"""Filesystem configuration shared by the local KARE entry points.

The original release hard-coded the authors' ``/shared/eng/...`` paths.  This
module keeps generated artifacts under one configurable directory instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True)
class KAREPaths:
    project_root: Path
    data_root: Path
    mimic3_root: Path | None
    mimic4_root: Path | None
    model_cache: Path

    @property
    def ehr(self) -> Path:
        return self.data_root / "ehr_data"

    @property
    def indexing(self) -> Path:
        return self.data_root / "indexing"

    @property
    def patient_context(self) -> Path:
        return self.data_root / "patient_context"

    @property
    def finetune(self) -> Path:
        return self.data_root / "llm_finetune_data"

    @property
    def outputs(self) -> Path:
        return self.data_root / "outputs"

    def create_output_dirs(self) -> None:
        for path in (
            self.data_root,
            self.ehr,
            self.indexing,
            self.patient_context / "base_context",
            self.patient_context / "similar_patient",
            self.patient_context / "augmented_context",
            self.finetune,
            self.outputs,
            self.model_cache,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_paths() -> KAREPaths:
    data_root = _env_path("KARE_DATA_DIR", PROJECT_ROOT / "data")
    mimic3 = os.environ.get("MIMIC3_ROOT")
    mimic4 = os.environ.get("MIMIC4_ROOT")
    return KAREPaths(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        mimic3_root=Path(mimic3).expanduser().resolve() if mimic3 else None,
        mimic4_root=Path(mimic4).expanduser().resolve() if mimic4 else None,
        model_cache=_env_path("KARE_MODEL_CACHE", data_root / "models"),
    )
