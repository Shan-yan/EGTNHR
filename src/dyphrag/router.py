"""Disease- and prediction-time-conditioned 3+1 source router."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RouterOutput:
    weights: torch.Tensor
    budgets: dict[str, int]
    continue_sources: dict[str, bool]
    needs_iteration: bool
    stopping_probability: float
    stopping_score: torch.Tensor
    retrieval_probability: float
    retrieval_score: torch.Tensor
    source_gate_probabilities: torch.Tensor
    evidence_sufficiency_probability: float
    evidence_sufficiency_score: torch.Tensor
    prior_only: bool

    def diagnostics(self) -> dict[str, object]:
        names = ("self", "peer", "medical", "prior")
        return {
            "weights": {name: round(float(weight), 6) for name, weight in zip(names, self.weights.detach().cpu())},
            "budgets": self.budgets,
            "continue_sources": self.continue_sources,
            "needs_iteration": self.needs_iteration,
            "stopping_probability": round(self.stopping_probability, 6),
            "retrieval_probability": round(self.retrieval_probability, 6),
            "source_gate_probabilities": {
                name: round(float(value), 6)
                for name, value in zip(names[:3], self.source_gate_probabilities.detach().cpu())
            },
            "evidence_sufficiency_probability": round(self.evidence_sufficiency_probability, 6),
            "prior_only": self.prior_only,
        }


class DynamicRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mode: str,
        max_total_k: int,
        fixed_top_k: int | None = None,
        source_gate_threshold: float = 0.5,
        retrieval_gate_threshold: float = 0.5,
        stop_threshold: float = 0.65,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.max_total_k = max_total_k
        self.fixed_top_k = fixed_top_k
        self.source_gate_threshold = source_gate_threshold
        self.retrieval_gate_threshold = retrieval_gate_threshold
        self.stop_threshold = stop_threshold
        query_dim = hidden_dim * 4 + 1
        self.source_head = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 4))
        decision_dim = query_dim + 3
        self.source_gate_head = nn.Sequential(nn.Linear(decision_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 3))
        self.retrieval_gate_head = nn.Sequential(nn.Linear(decision_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.sufficiency_head = nn.Sequential(nn.Linear(decision_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.stop_head = nn.Sequential(nn.Linear(decision_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        # Retrieval-positive initialization prevents an untrained router from
        # collapsing into the absorbing prior-only path before it has observed
        # evidence utility. The gates remain fully learnable.
        with torch.no_grad():
            self.source_gate_head[-1].bias.fill_(1.5)
            self.retrieval_gate_head[-1].bias.fill_(1.5)
            self.sufficiency_head[-1].bias.fill_(-1.0)
            self.stop_head[-1].bias.fill_(-1.0)

    def allocate_budgets(
        self,
        weights: torch.Tensor,
        retrieval_probability: float = 1.0,
        source_gates: torch.Tensor | None = None,
    ) -> dict[str, int]:
        names = ("self", "peer", "medical")
        gates = (
            torch.ones(3, device=weights.device, dtype=weights.dtype)
            if source_gates is None
            else source_gates.detach().clamp(0.0, 1.0)
        )
        active = gates >= self.source_gate_threshold
        if retrieval_probability < self.retrieval_gate_threshold or self.max_total_k <= 0:
            return {name: 0 for name in names}
        if self.fixed_top_k is not None:
            return {
                name: max(int(self.fixed_top_k), 0) if bool(active[index]) else 0
                for index, name in enumerate(names)
            }
        total_k = max(1, min(self.max_total_k, int(round(self.max_total_k * retrieval_probability))))
        retrieval_weights = weights[:3].detach().clamp_min(0) * gates * active
        if float(retrieval_weights.sum()) <= 0:
            best = int(torch.argmax(weights[:3].detach()))
            retrieval_weights[best] = 1.0
        shares = retrieval_weights / retrieval_weights.sum() * total_k
        floors = torch.floor(shares).to(dtype=torch.int64)
        remaining = total_k - int(floors.sum())
        if remaining > 0:
            remainders = shares - floors
            for index in torch.topk(remainders, k=min(remaining, len(names))).indices.tolist():
                floors[index] += 1
        return {name: int(floors[index]) for index, name in enumerate(names)}

    def forward(
        self,
        query: torch.Tensor,
        uncertainty: float = 1.0,
        evidence_quality: float = 0.0,
        evidence_sufficiency: float = 0.0,
    ) -> RouterOutput:
        logits = self.source_head(query)
        decision_input = torch.cat(
            [
                query,
                torch.tensor(
                    [uncertainty, evidence_quality, evidence_sufficiency],
                    device=query.device,
                    dtype=query.dtype,
                ),
            ]
        )
        if self.mode == "uniform":
            weights = torch.full_like(logits, 0.25)
            source_gate_scores = torch.ones(3, device=query.device, dtype=query.dtype)
            retrieval_score = torch.ones((), device=query.device, dtype=query.dtype)
        elif self.mode == "heuristic":
            raw = torch.tensor([uncertainty, uncertainty, evidence_quality + 0.1, 1.0 - uncertainty + 0.1], device=query.device)
            weights = raw.clamp_min(1e-5) / raw.clamp_min(1e-5).sum()
            source_gate_scores = torch.tensor(
                [uncertainty, uncertainty, max(uncertainty, 1.0 - evidence_sufficiency)],
                device=query.device,
                dtype=query.dtype,
            ).clamp(0.0, 1.0)
            retrieval_score = torch.tensor(
                max(uncertainty, 1.0 - evidence_sufficiency),
                device=query.device,
                dtype=query.dtype,
            ).clamp(0.0, 1.0)
        elif self.mode == "sparse_gumbel":
            weights = (
                F.gumbel_softmax(logits, tau=0.7, hard=False, dim=-1)
                if self.training
                else F.softmax(logits, dim=-1)
            )
            keep = torch.topk(weights, k=2).indices
            mask = torch.zeros_like(weights).scatter_(0, keep, 1.0)
            weights = weights * mask
            weights = weights / weights.sum().clamp_min(1e-8)
            source_gate_scores = torch.sigmoid(self.source_gate_head(decision_input)) * mask[:3]
            retrieval_score = torch.sigmoid(self.retrieval_gate_head(decision_input)).squeeze()
        elif self.mode == "learned_soft":
            weights = F.softmax(logits, dim=-1)
            source_gate_scores = torch.sigmoid(self.source_gate_head(decision_input))
            retrieval_score = torch.sigmoid(self.retrieval_gate_head(decision_input)).squeeze()
        else:
            raise ValueError(f"Unsupported router mode: {self.mode}")
        if self.mode == "heuristic":
            sufficiency_score = torch.tensor(
                evidence_sufficiency,
                device=query.device,
                dtype=query.dtype,
            ).clamp(0.0, 1.0)
            stop_score = torch.maximum(
                sufficiency_score,
                torch.tensor(1.0 - uncertainty, device=query.device, dtype=query.dtype),
            ).clamp(0.0, 1.0)
        else:
            sufficiency_score = torch.sigmoid(self.sufficiency_head(decision_input)).squeeze()
            stop_score = torch.sigmoid(self.stop_head(decision_input)).squeeze()
        retrieval_probability = float(retrieval_score.detach())
        budgets = self.allocate_budgets(weights, retrieval_probability, source_gate_scores)
        continue_sources = {
            name: budgets[name] > 0
            for index, name in enumerate(("self", "peer", "medical"))
        }
        stop = float(stop_score.detach())
        sufficiency = float(sufficiency_score.detach())
        prior_only = sum(budgets.values()) == 0
        return RouterOutput(
            weights,
            budgets,
            continue_sources,
            not prior_only and stop < self.stop_threshold,
            stop,
            stop_score,
            retrieval_probability,
            retrieval_score,
            source_gate_scores,
            sufficiency,
            sufficiency_score,
            prior_only,
        )
