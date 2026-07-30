"""Uniform predictor capabilities for DyPH-RAG and all requested controls.

The original GraphCare/KARE/PyHealth files are intentionally not imported or
modified here.  These are cutoff-safe adaptations over the common DyPH-RAG
sample contract, so comparisons share the same splits, labels and evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contracts import PatientDiseaseExample


@runtime_checkable
class MedicalPredictor(Protocol):
    """The single forward contract used by training, baselines and ablations."""

    def forward(self, example: PatientDiseaseExample, label: int | None = None): ...


@dataclass(frozen=True)
class ModelCapabilities:
    canonical_name: str
    family: str
    sequence_encoder: str | None = None
    use_hypergraph: bool = True
    use_retrieval: bool = True
    dynamic_retrieval: bool = True
    dynamic_hypergraph: bool = True
    evidence_polarity: bool = True
    description: str = ""


MODEL_REGISTRY: dict[str, ModelCapabilities] = {
    "ehr_gru": ModelCapabilities(
        "ehr_only_gru", "ehr_only", "gru", False, False, False, False, False,
        "EHR-only GRU over cutoff-safe events.",
    ),
    "ehr_transformer": ModelCapabilities(
        "ehr_only_transformer", "ehr_only", "transformer", False, False, False, False, False,
        "EHR-only Transformer over cutoff-safe events.",
    ),
    "static_heterogeneous_graph": ModelCapabilities(
        "static_heterogeneous_graph", "graph", None, True, False, False, False, False,
        "Static heterogeneous graph control.",
    ),
    "visit_hypergraph": ModelCapabilities(
        "visit_only_hypergraph", "hypergraph", None, True, False, False, False, False,
        "Visit-only static hypergraph control.",
    ),
    "graphcare_adapted": ModelCapabilities(
        "graphcare_like", "graphcare", None, True, True, False, False, False,
        "Cutoff-safe GraphCare-like graph/knowledge adaptation; not a paper reproduction.",
    ),
    "kare_adapted": ModelCapabilities(
        "kare_like_static_community_retrieval", "kare", None, True, True, False, False, False,
        "Cutoff-safe KARE-like static/community retrieval adaptation; not a paper reproduction.",
    ),
    "static_vector_rag": ModelCapabilities(
        "static_vector_rag", "rag", None, True, True, False, False, False,
        "Single-shot static vector RAG.",
    ),
    "disease_conditioned_retrieval": ModelCapabilities(
        "disease_conditioned_retrieval_only", "retrieval", None, True, True, False, False, False,
        "Disease-conditioned, single-shot retrieval control.",
    ),
    "dynamic_no_polarity": ModelCapabilities(
        "dynamic_hypergraph_without_evidence_polarity", "dyphrag", None, True, True, True, True, False,
        "Dynamic retrieval/hypergraph without polarity separation.",
    ),
    "full_dyphrag": ModelCapabilities(
        "full_dyphrag", "dyphrag", None, True, True, True, True, True,
        "Full disease- and cutoff-conditioned DyPH-RAG.",
    ),
    "dyphrag_mimic_core": ModelCapabilities(
        "dyphrag_mimic_core", "dyphrag", None, True, True, True, True, True,
        "Full core model with unavailable sources explicitly disabled.",
    ),
}


def capabilities_for(mode: str) -> ModelCapabilities:
    try:
        return MODEL_REGISTRY[mode]
    except KeyError as error:
        raise ValueError(f"Unknown model mode {mode!r}; choose one of {sorted(MODEL_REGISTRY)}") from error
