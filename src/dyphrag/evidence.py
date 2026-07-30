"""Support/refute/differential evidence classification and aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .contracts import EvidenceItem, EvidencePolarity, EvidenceSource
from .features import hashed_vector


@dataclass
class EvidenceRepresentations:
    support: torch.Tensor
    refute: torch.Tensor
    differential: torch.Tensor
    neutral: torch.Tensor
    sufficiency: torch.Tensor
    contradiction_rate: float
    support_strength: torch.Tensor
    refute_strength: torch.Tensor
    evidence_probability: torch.Tensor


class PolarityClassifier(nn.Module):
    """Classify evidence polarity relative to the queried disease."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.source_embedding = nn.Embedding(len(EvidenceSource), hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        item: EvidenceItem,
        disease_query: torch.Tensor,
        use_metadata_label: bool = True,
    ) -> tuple[EvidencePolarity, torch.Tensor]:
        device = disease_query.device
        vector = hashed_vector(
            [item.text, *map(str, item.metadata.get("concepts", ()))],
            self.hidden_dim,
            device,
        )
        source_index = list(EvidenceSource).index(item.source)
        source = self.source_embedding(torch.tensor(source_index, device=device))
        logits = self.classifier(
            torch.cat([vector, disease_query, vector * disease_query, source])
        )
        if use_metadata_label:
            return item.polarity, logits
        polarity = list(EvidencePolarity)[int(torch.argmax(logits).detach())]
        return polarity, logits


class EvidenceAggregator(nn.Module):
    def __init__(self, hidden_dim: int, reliability_weighting: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reliability_weighting = reliability_weighting
        self.reranker = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.sufficiency_head = nn.Sequential(nn.Linear(hidden_dim * 3 + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    @staticmethod
    def _contradiction_rate(items: list[EvidenceItem]) -> float:
        if not items:
            return 0.0
        identifiers = {item.source_id for item in items}
        explicit_pairs = {
            tuple(sorted((item.source_id, target)))
            for item in items
            for target in item.contradicts
            if target in identifiers
        }
        def concepts(item: EvidenceItem) -> set[str]:
            values = item.metadata.get("concepts", ())
            if isinstance(values, str):
                values = (values,)
            result = {str(value).strip().lower() for value in values if str(value).strip()}
            if not result and item.provenance_span:
                result.update(
                    value.strip().lower()
                    for value in item.provenance_span.split(";")
                    if value.strip()
                )
            return result

        support_concepts = {
            concept
            for item in items
            if item.polarity == EvidencePolarity.SUPPORT
            for concept in concepts(item)
        }
        refute_concepts = {
            concept
            for item in items
            if item.polarity == EvidencePolarity.REFUTE
            for concept in concepts(item)
        }
        polarity_conflicts = len((support_concepts & refute_concepts) - {""})
        return min((len(explicit_pairs) + polarity_conflicts) / max(len(items), 1), 1.0)

    def _pool(self, items: list[EvidenceItem], device: torch.device, query: torch.Tensor, frozen: bool) -> torch.Tensor:
        if not items:
            return torch.zeros(self.hidden_dim, device=device)
        vectors: list[torch.Tensor] = []
        fixed_weights = []
        for item in items:
            vectors.append(hashed_vector([item.text], self.hidden_dim, device))
            reliability = item.reliability * item.applicability if self.reliability_weighting else 1.0
            fixed_weights.append(max(float(item.score) * reliability, 1e-4))
        stacked = torch.stack(vectors)
        query_batch = query.unsqueeze(0).expand_as(stacked)
        learned_relevance = torch.sigmoid(self.reranker(stacked, query_batch).squeeze(-1))
        if frozen:
            learned_relevance = learned_relevance.detach()
        weight_tensor = torch.tensor(fixed_weights, device=device) * learned_relevance
        weight_tensor /= weight_tensor.sum()
        return stacked.mul(weight_tensor[:, None]).sum(dim=0)

    def forward(
        self,
        items: Iterable[EvidenceItem],
        device: torch.device,
        query: torch.Tensor,
        frozen_retriever: bool = False,
        include_refute: bool = True,
        include_differential: bool = True,
    ) -> EvidenceRepresentations:
        pools = {polarity: [] for polarity in EvidencePolarity}
        all_items = list(items)
        for item in all_items:
            pools[item.polarity].append(item)
        support = self._pool(pools[EvidencePolarity.SUPPORT], device, query, frozen_retriever)
        refute = self._pool(pools[EvidencePolarity.REFUTE], device, query, frozen_retriever) if include_refute else torch.zeros_like(support)
        differential = self._pool(pools[EvidencePolarity.DIFFERENTIAL], device, query, frozen_retriever) if include_differential else torch.zeros_like(support)
        neutral = self._pool(pools[EvidencePolarity.NEUTRAL], device, query, frozen_retriever)
        counts = torch.tensor(
            [len(pools[EvidencePolarity.SUPPORT]), len(pools[EvidencePolarity.REFUTE]), len(pools[EvidencePolarity.DIFFERENTIAL])],
            dtype=torch.float32,
            device=device,
        )
        learned_sufficiency = torch.sigmoid(self.sufficiency_head(torch.cat([support, refute, differential, counts])))
        coverage = (counts.sum() / 3.0).clamp(0.0, 1.0)
        sufficiency = learned_sufficiency * coverage
        def strength(polarity: EvidencePolarity) -> torch.Tensor:
            values = [
                max(float(item.score), 0.0)
                * (item.reliability * item.applicability if self.reliability_weighting else 1.0)
                for item in pools[polarity]
            ]
            return torch.tensor(sum(values), dtype=torch.float32, device=device)

        support_strength = strength(EvidencePolarity.SUPPORT)
        refute_strength = strength(EvidencePolarity.REFUTE)
        evidence_probability = torch.sigmoid(support_strength - refute_strength)
        contradiction_rate = self._contradiction_rate(all_items)
        return EvidenceRepresentations(
            support,
            refute,
            differential,
            neutral,
            sufficiency,
            contradiction_rate,
            support_strength,
            refute_strength,
            evidence_probability,
        )
