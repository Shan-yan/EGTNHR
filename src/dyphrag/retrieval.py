"""Cutoff-aware self, train-only peer and versioned medical retrievers."""

from __future__ import annotations

import random
import json
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .contracts import (
    ClinicalEvent,
    EvidenceItem,
    EvidencePolarity,
    EvidenceSource,
    PatientDiseaseExample,
    stable_hash,
    parse_time,
)
from .features import cosine, hashed_vector, temporal_decay
from .leakage import LeakageError, audit_retrieval


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[EvidenceItem, ...]
    candidate_count: int
    corpus_hash: str


def _query_tokens(example: PatientDiseaseExample, disease_conditioned: bool = True) -> list[str]:
    values = [event.concept_id for event in example.prior_events]
    values.extend(event.event_type for event in example.prior_events)
    if disease_conditioned:
        values.append(example.target_disease)
    values.extend(f"context:{key}:{round(float(value), 2)}" for key, value in sorted(example.context.items()))
    return values


def _profile_tokens(events: Sequence[ClinicalEvent], context: dict[str, float]) -> list[str]:
    """Peer index features never contain task labels or target-disease fields."""
    values = [event.concept_id for event in events]
    values.extend(event.event_type for event in events)
    values.extend(f"context:{key}:{round(float(value), 2)}" for key, value in sorted(context.items()))
    return values


def _event_polarity(event: ClinicalEvent, disease: str) -> EvidencePolarity:
    concept = event.concept_id.lower()
    disease_suffix = disease.rsplit("_", 1)[-1].lower()
    if concept.startswith("risk_") and disease_suffix in concept:
        return EvidencePolarity.SUPPORT
    if concept.startswith("protective_") and disease_suffix in concept:
        return EvidencePolarity.REFUTE
    if concept.startswith("risk_"):
        return EvidencePolarity.DIFFERENTIAL
    return EvidencePolarity.NEUTRAL


def _population_applicability(population: str, context: dict[str, float]) -> float:
    description = population.lower()
    score = 1.0
    if "adult" in description and "age_normalized" in context:
        score *= 1.0 if float(context["age_normalized"]) * 90.0 >= 18.0 else 0.25
    if any(token in description for token in ("pediatric", "child", "adolescent")) and "age_normalized" in context:
        score *= 1.0 if float(context["age_normalized"]) * 90.0 < 18.0 else 0.25
    if "female" in description and "sex_female" in context:
        score *= 1.0 if float(context["sex_female"]) >= 0.5 else 0.5
    if "male" in description and "sex_female" in context:
        score *= 1.0 if float(context["sex_female"]) < 0.5 else 0.5
    return max(min(score, 1.0), 0.0)


class SelfHistoryRetriever:
    def __init__(self, dimension: int = 32, half_life_hours: float = 24.0 * 180.0) -> None:
        self.dimension = dimension
        self.half_life_hours = half_life_hours

    def retrieve(
        self,
        query: PatientDiseaseExample,
        top_k: int,
        disease_conditioned: bool = True,
        time_conditioned: bool = True,
        random_control: bool = False,
        seed: int = 0,
    ) -> RetrievalResult:
        disease_vector = hashed_vector(
            [query.target_disease if disease_conditioned else "target_disease:masked"],
            self.dimension,
        )
        scored: list[EvidenceItem] = []

        def add_candidate(
            candidate_id: str,
            events: Sequence[ClinicalEvent],
            unit_type: str,
            score_bonus: float = 0.0,
        ) -> None:
            if not events:
                return
            latest = max(event.timestamp for event in events)
            concepts = [event.concept_id for event in events]
            semantic = cosine(hashed_vector(concepts, self.dimension), disease_vector)
            elapsed = (query.cutoff_time - latest).total_seconds() / 3600.0
            temporal = temporal_decay(elapsed, self.half_life_hours) if time_conditioned else 1.0
            numerical = max((abs(event.numeric.baseline_deviation or 0.0) for event in events if event.numeric), default=0.0)
            conditioned = (
                1.0
                if disease_conditioned
                and any(query.target_disease.rsplit("_", 1)[-1] in event.concept_id.lower() for event in events)
                else 0.0
            )
            event_polarities = [
                _event_polarity(event, query.target_disease) if disease_conditioned else EvidencePolarity.NEUTRAL
                for event in events
            ]
            polarity = next(
                (value for value in (EvidencePolarity.SUPPORT, EvidencePolarity.REFUTE, EvidencePolarity.DIFFERENTIAL) if value in event_polarities),
                EvidencePolarity.NEUTRAL,
            )
            score = 0.25 * semantic + 0.25 * temporal + 0.20 * min(numerical, 1.0) + 0.25 * conditioned + score_bonus
            scored.append(
                EvidenceItem(
                    evidence_id=f"self:{unit_type}:{candidate_id}",
                    source=EvidenceSource.SELF,
                    source_id=candidate_id,
                    timestamp=latest,
                    text=" | ".join((event.text or event.concept_id) for event in events),
                    score=score,
                    polarity=polarity,
                    patient_id=query.patient_id,
                    split=query.split,
                    source_type=unit_type,
                    provenance_span=";".join(event.event_id for event in events),
                    metadata={
                        "event_types": sorted({event.event_type for event in events}),
                        "unit_type": unit_type,
                        "event_count": len(events),
                        "concepts": tuple(sorted(set(concepts))),
                        **(
                            {"polarity_label": polarity.value}
                            if all(bool(event.metadata.get("synthetic")) for event in events)
                            else {}
                        ),
                    },
                )
            )

        prior = tuple(query.prior_events)
        for event in prior:
            add_candidate(event.event_id, (event,), "event")
        visits: dict[str, list[ClinicalEvent]] = defaultdict(list)
        for event in prior:
            visits[event.visit_id or f"unassigned:{event.event_id}"].append(event)
        for visit_id, events in visits.items():
            add_candidate(visit_id, events, "visit", 0.03)
            numeric_events = [event for event in events if event.numeric is not None]
            if numeric_events:
                add_candidate(visit_id, numeric_events, "patient-state-hyperedge", 0.04)
        by_concept: dict[str, list[ClinicalEvent]] = defaultdict(list)
        for event in prior:
            by_concept[event.concept_id].append(event)
        for concept, events in by_concept.items():
            ordered = sorted(events, key=lambda event: event.timestamp)
            if len(ordered) >= 2:
                add_candidate(concept, ordered, "episode", 0.02)
                if any(event.numeric is not None for event in ordered):
                    add_candidate(concept, ordered, "numeric_trend_window", 0.05)
        if random_control:
            random.Random(seed).shuffle(scored)
            selected = scored[: max(top_k, 0)]
        else:
            selected = sorted(scored, key=lambda item: item.score, reverse=True)[: max(top_k, 0)]
        result = tuple(selected)
        audit_retrieval(query, result)
        return RetrievalResult(result, len(scored), stable_hash([x.event_id for x in query.prior_events]))


@dataclass(frozen=True)
class PeerProfile:
    patient_id: str
    events: tuple[ClinicalEvent, ...]
    context: dict[str, float]


@dataclass(frozen=True)
class PeerSnapshot:
    """One train-patient representation observable at a particular cutoff."""

    patient_id: str
    cutoff_time: datetime
    events: tuple[ClinicalEvent, ...]
    context: dict[str, float]
    vector: tuple[float, ...]


class PeerPatientRetriever:
    """Cutoff-safe coarse patient retrieval followed by event reranking.

    Every coarse vector is a materialized historical snapshot.  At query time
    only the latest snapshot not later than the query cutoff is eligible, so
    future information cannot affect either retrieval stage.
    """

    def __init__(self, train_examples: Sequence[PatientDiseaseExample], dimension: int = 32) -> None:
        if any(example.split != "train" for example in train_examples):
            raise LeakageError("Peer bank construction accepts training examples only")
        self.dimension = dimension
        grouped_events: dict[str, dict[str, ClinicalEvent]] = defaultdict(dict)
        examples_by_patient: dict[str, list[PatientDiseaseExample]] = defaultdict(list)
        for example in train_examples:
            examples_by_patient[example.patient_id].append(example)
            for event in example.prior_events:
                grouped_events[example.patient_id][event.event_id] = event
        self.bank = tuple(
            PeerProfile(
                patient_id=patient_id,
                events=tuple(sorted(events.values(), key=lambda event: (event.timestamp, event.event_id))),
                context=dict(
                    sorted(
                        max(
                            examples_by_patient[patient_id],
                            key=lambda value: value.cutoff_time,
                        ).context.items()
                    )
                ),
            )
            for patient_id, events in sorted(grouped_events.items())
        )
        snapshots: dict[tuple[str, datetime], PeerSnapshot] = {}
        for profile in self.bank:
            patient_examples = sorted(
                examples_by_patient[profile.patient_id],
                key=lambda value: value.cutoff_time,
            )
            for example in patient_examples:
                visible_events = tuple(event for event in profile.events if event.timestamp <= example.cutoff_time)
                vector = hashed_vector(_profile_tokens(visible_events, dict(example.context)), dimension)
                snapshot = PeerSnapshot(
                        patient_id=profile.patient_id,
                        cutoff_time=example.cutoff_time,
                        events=visible_events,
                        context=dict(example.context),
                        vector=tuple(float(value) for value in vector.tolist()),
                    )
                snapshots.setdefault((profile.patient_id, example.cutoff_time), snapshot)
        self.snapshots = tuple(
            sorted(snapshots.values(), key=lambda value: (value.patient_id, value.cutoff_time))
        )
        # Kept for index export compatibility; these are snapshot vectors, not
        # full-trajectory patient vectors.
        self.vectors = tuple(torch.tensor(snapshot.vector) for snapshot in self.snapshots)
        self.corpus_hash = stable_hash(
            [
                {
                    "patient": snapshot.patient_id,
                    "snapshot_cutoff": snapshot.cutoff_time.isoformat(),
                    "events": [
                        (event.event_id, event.timestamp.isoformat(), event.concept_id, event.unit_type)
                        for event in snapshot.events
                    ],
                }
                for snapshot in self.snapshots
            ]
        )

    def eligible_snapshots(self, query: PatientDiseaseExample) -> tuple[PeerSnapshot, ...]:
        """Return at most one cutoff-safe snapshot per train patient."""
        latest: dict[str, PeerSnapshot] = {}
        for snapshot in self.snapshots:
            if snapshot.patient_id == query.patient_id or snapshot.cutoff_time > query.cutoff_time:
                continue
            previous = latest.get(snapshot.patient_id)
            if previous is None or snapshot.cutoff_time > previous.cutoff_time:
                latest[snapshot.patient_id] = snapshot
        return tuple(latest[key] for key in sorted(latest))

    def retrieve(
        self,
        query: PatientDiseaseExample,
        top_k: int,
        candidate_k: int,
        mask_target_labels: bool = True,
        disease_conditioned: bool = True,
        random_control: bool = False,
        seed: int = 0,
    ) -> RetrievalResult:
        query_vector = hashed_vector(_query_tokens(query, disease_conditioned), self.dimension)
        candidates = [
            (cosine(query_vector, torch.tensor(snapshot.vector)), snapshot)
            for snapshot in self.eligible_snapshots(query)
        ]
        if random_control:
            random.Random(seed).shuffle(candidates)
        else:
            candidates.sort(key=lambda pair: pair[0], reverse=True)
        fine: list[EvidenceItem] = []
        target_token = query.target_disease.lower()
        for patient_score, snapshot in candidates[:candidate_k]:
            for event in snapshot.events:
                text = event.text or event.concept_id
                if mask_target_labels and target_token in text.lower():
                    text = "[TARGET_DISEASE_MASKED]"
                disease_token = query.target_disease if disease_conditioned else "target_disease:masked"
                event_score = 0.65 * patient_score + 0.35 * cosine(
                    hashed_vector([event.concept_id], self.dimension), hashed_vector([disease_token], self.dimension)
                )
                polarity = (
                    _event_polarity(event, query.target_disease)
                    if disease_conditioned
                    else EvidencePolarity.NEUTRAL
                )
                fine.append(
                    EvidenceItem(
                        evidence_id=f"peer:{snapshot.patient_id}:{snapshot.cutoff_time.isoformat()}:{event.event_id}",
                        source=EvidenceSource.PEER,
                        source_id=event.event_id,
                        timestamp=event.timestamp,
                        text=text,
                        score=event_score,
                        polarity=polarity,
                        patient_id=snapshot.patient_id,
                        split="train",
                        source_type=event.unit_type,
                        applicability=max(min((patient_score + 1.0) / 2.0, 1.0), 0.0),
                        provenance_span=event.concept_id,
                        metadata={
                            "masked_target": mask_target_labels,
                            "event_type": event.event_type,
                            "retrieval_stage": "coarse_patient_then_fine_event",
                            "patient_score": patient_score,
                            "peer_snapshot_cutoff": snapshot.cutoff_time.isoformat(),
                            "peer_embedding": snapshot.vector,
                            "concepts": (event.concept_id,),
                            **(
                                {"polarity_label": polarity.value}
                                if bool(event.metadata.get("synthetic"))
                                else {}
                            ),
                        },
                    )
                )
        result = tuple(sorted(fine, key=lambda item: item.score, reverse=True)[: max(top_k, 0)])
        audit_retrieval(query, result)
        return RetrievalResult(result, len(candidates), self.corpus_hash)


@dataclass(frozen=True)
class MedicalDocument:
    source_id: str
    text: str
    concepts: tuple[str, ...]
    publication_date: datetime
    source_type: str
    version: str
    population: str
    evidence_grade: str
    polarity: EvidencePolarity
    provenance_span: str
    license: str
    target_diseases: tuple[str, ...]
    reliability: float = 0.8
    supersedes: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    graph_neighbors: tuple[str, ...] = ()
    differential_disease: str | None = None


def default_medical_corpus() -> tuple[MedicalDocument, ...]:
    date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return (
        MedicalDocument("synthetic-guideline-alpha", "risk alpha is associated with disease alpha", ("risk_alpha", "disease_alpha"), date, "guideline", "1", "synthetic adults", "synthetic-A", EvidencePolarity.SUPPORT, "section-alpha", "CC0", ("disease_alpha",)),
        MedicalDocument("synthetic-guideline-alpha-refute", "protective alpha lowers evidence for disease alpha", ("protective_alpha", "disease_alpha"), date, "guideline", "1", "synthetic adults", "synthetic-B", EvidencePolarity.REFUTE, "section-alpha-refute", "CC0", ("disease_alpha",)),
        MedicalDocument("synthetic-guideline-beta", "risk beta is associated with disease beta", ("risk_beta", "disease_beta"), date, "ontology", "1", "synthetic adults", "synthetic-A", EvidencePolarity.SUPPORT, "node-beta", "CC0", ("disease_beta",)),
        MedicalDocument("synthetic-guideline-beta-refute", "protective beta lowers evidence for disease beta", ("protective_beta", "disease_beta"), date, "curated_database", "1", "synthetic adults", "synthetic-B", EvidencePolarity.REFUTE, "entry-beta-refute", "CC0", ("disease_beta",)),
    )


def load_medical_corpus(path: Path) -> tuple[MedicalDocument, ...]:
    """Load versioned external evidence without copying it into run logs."""
    required = {
        "source_id", "text", "concepts", "publication_date", "source_type", "version", "population",
        "evidence_grade", "polarity", "provenance_span", "license",
        "target_diseases",
    }
    documents: list[MedicalDocument] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            missing = required - value.keys()
            if missing:
                raise ValueError(f"Medical corpus line {line_number} lacks required metadata: {sorted(missing)}")
            if value.get("source_type") == "schema_example" or str(value["source_id"]).startswith("SCHEMA_ONLY"):
                raise ValueError(
                    f"Medical corpus line {line_number} is a schema placeholder, not admissible evidence"
                )
            documents.append(
                MedicalDocument(
                    source_id=str(value["source_id"]),
                    text=str(value["text"]),
                    concepts=tuple(map(str, value["concepts"])),
                    publication_date=parse_time(value["publication_date"]),
                    source_type=str(value["source_type"]),
                    version=str(value["version"]),
                    population=str(value["population"]),
                    evidence_grade=str(value["evidence_grade"]),
                    polarity=EvidencePolarity(value["polarity"]),
                    provenance_span=str(value["provenance_span"]),
                    license=str(value["license"]),
                    target_diseases=tuple(map(str, value["target_diseases"])),
                    reliability=float(value.get("reliability", 0.8)),
                    supersedes=tuple(map(str, value.get("supersedes", []))),
                    contradicts=tuple(map(str, value.get("contradicts", []))),
                    graph_neighbors=tuple(map(str, value.get("graph_neighbors", []))),
                    differential_disease=(
                        str(value["differential_disease"])
                        if value.get("differential_disease") is not None
                        else None
                    ),
                )
            )
    if not documents:
        raise ValueError("Medical corpus is empty")
    return tuple(documents)


class MedicalKnowledgeRetriever:
    def __init__(self, documents: Iterable[MedicalDocument], dimension: int = 32, retrieval_mode: str = "hybrid") -> None:
        self.documents = tuple(documents)
        self.dimension = dimension
        self.retrieval_mode = retrieval_mode
        self.corpus_hash = stable_hash([doc.source_id + ":" + doc.version for doc in self.documents])

    def retrieve(
        self,
        query: PatientDiseaseExample,
        top_k: int,
        disease_conditioned: bool = True,
        random_control: bool = False,
        seed: int = 0,
    ) -> RetrievalResult:
        query_terms = [event.concept_id for event in query.prior_events]
        if disease_conditioned:
            query_terms.insert(0, query.target_disease)
        query_vector = hashed_vector(query_terms, self.dimension)
        items: list[EvidenceItem] = []
        for doc in self.documents:
            if doc.publication_date > query.cutoff_time:
                continue
            vector_score = cosine(query_vector, hashed_vector((*doc.concepts, doc.text), self.dimension))
            graph_score = len(set(query_terms) & set((*doc.concepts, *doc.graph_neighbors))) / max(len(doc.concepts), 1)
            scores = {"vector": vector_score, "graph": graph_score, "hybrid": 0.65 * vector_score + 0.35 * graph_score}
            if self.retrieval_mode not in scores:
                raise ValueError(f"Unknown medical retrieval mode: {self.retrieval_mode}")
            target_matches = query.target_disease in doc.target_diseases
            if target_matches:
                query_polarity = doc.polarity
                differential_disease = doc.differential_disease
            elif doc.polarity == EvidencePolarity.SUPPORT and doc.target_diseases:
                query_polarity = EvidencePolarity.DIFFERENTIAL
                differential_disease = doc.target_diseases[0]
            else:
                query_polarity = EvidencePolarity.NEUTRAL
                differential_disease = None
            items.append(
                EvidenceItem(
                    evidence_id=f"medical:{doc.source_id}:{doc.version}",
                    source=EvidenceSource.MEDICAL,
                    source_id=doc.source_id,
                    timestamp=doc.publication_date,
                    text=doc.text,
                    score=scores[self.retrieval_mode],
                    polarity=query_polarity,
                    source_type=doc.source_type,
                    version=doc.version,
                    population=doc.population,
                    evidence_grade=doc.evidence_grade,
                    reliability=doc.reliability,
                    applicability=_population_applicability(doc.population, dict(query.context)),
                    provenance_span=doc.provenance_span,
                    license=doc.license,
                    supersedes=doc.supersedes,
                    contradicts=doc.contradicts,
                    differential_disease=differential_disease,
                    metadata={
                        "polarity_label": query_polarity.value,
                        "population_applicability": _population_applicability(doc.population, dict(query.context)),
                        "concepts": doc.concepts,
                        "target_diseases": doc.target_diseases,
                        "differential_disease": differential_disease,
                    },
                )
            )
        if random_control:
            random.Random(seed).shuffle(items)
            selected = items[: max(top_k, 0)]
        else:
            selected = sorted(items, key=lambda item: item.score, reverse=True)[: max(top_k, 0)]
        result = tuple(selected)
        audit_retrieval(query, result)
        return RetrievalResult(result, len(items), self.corpus_hash)


class RetrievalSystem:
    def __init__(self, train_examples: Sequence[PatientDiseaseExample], config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = seed
        dimension = int(config["model"]["hidden_dim"])
        self.self_retriever = SelfHistoryRetriever(dimension)
        self.peer_retriever = PeerPatientRetriever(train_examples, dimension)
        corpus_path = config["retrieval"].get("medical_corpus") or os.environ.get("DYPHRAG_MEDICAL_CORPUS")
        if corpus_path:
            documents = load_medical_corpus(Path(corpus_path).expanduser())
        elif config["dataset"] == "synthetic":
            documents = default_medical_corpus()
        elif config["retrieval"]["medical_enabled"]:
            raise FileNotFoundError("Set DYPHRAG_MEDICAL_CORPUS to a licensed, versioned medical JSONL corpus")
        else:
            documents = ()
        self.medical_retriever = MedicalKnowledgeRetriever(documents, dimension, config["retrieval"]["medical_mode"])
        self.corpus_hash = stable_hash(
            {
                "peer": self.peer_retriever.corpus_hash,
                "medical": self.medical_retriever.corpus_hash,
                "medical_mode": config["retrieval"]["medical_mode"],
            }
        )

    def retrieve(self, query: PatientDiseaseExample, budgets: dict[str, int], iteration: int) -> tuple[EvidenceItem, ...]:
        cfg = self.config["retrieval"]
        results: list[EvidenceItem] = []
        if cfg["self_enabled"]:
            results.extend(
                self.self_retriever.retrieve(
                    query,
                    budgets["self"],
                    cfg["disease_conditioned"],
                    cfg["prediction_time_conditioned"],
                    bool(cfg["random_control"]),
                    self.seed + iteration,
                ).evidence
            )
        if cfg["peer_enabled"]:
            results.extend(
                self.peer_retriever.retrieve(
                    query,
                    budgets["peer"],
                    int(cfg["peer_candidate_k"]),
                    bool(cfg["mask_peer_target"]),
                    bool(cfg["disease_conditioned"]),
                    bool(cfg["random_control"]),
                    self.seed + iteration,
                ).evidence
            )
        if cfg["medical_enabled"]:
            results.extend(
                self.medical_retriever.retrieve(
                    query,
                    budgets["medical"],
                    cfg["disease_conditioned"],
                    bool(cfg["random_control"]),
                    self.seed + iteration,
                ).evidence
            )
        if cfg.get("shuffle_across_patients", False):
            donors = list(results)
            random.Random(self.seed + iteration).shuffle(donors)
            results = [
                replace(
                    item,
                    evidence_id=f"negative-control:{item.evidence_id}",
                    text=donor.text,
                    polarity=donor.polarity,
                    provenance_span="",
                    metadata={
                        **donor.metadata,
                        "negative_control": "evidence_content_permuted_across_retrieved_items",
                        "original_source": item.source.value,
                    },
                )
                for item, donor in zip(results, donors)
            ]
        audit_retrieval(query, results)
        return tuple(results)
