"""Locality-weighted Generalized Cross-Entropy for reading order prediction.

Computes loss on antisymmetric pairwise precedence scores and decodes optimal sequence.
"""

from __future__ import annotations

import math
import torch


def decode_order(S: torch.Tensor) -> torch.Tensor:
    """Decode reading order sequence from pairwise score matrix S.
    
    P(i precedes j) = sigmoid(S[i, j]) if i < j else 1 - sigmoid(S[j, i])
    Votes[j] = total elements preceding j.
    Argsort ascending yields rank sequence (first to last).
    """
    sc = torch.sigmoid(S)
    votes = sc.triu(1).sum(0) + (1.0 - sc.t()).tril(-1).sum(0)
    return torch.argsort(votes)


def pairwise_scores(Q: torch.Tensor, ro_q: torch.nn.Module, ro_k: torch.nn.Module) -> torch.Tensor:
    """Project queries Q into antisymmetric precedence matrix S [B, N, N]."""
    A, K = ro_q(Q), ro_k(Q)
    M = A @ K.transpose(-1, -2)
    return (M - M.transpose(-1, -2)) / math.sqrt(A.shape[-1])


def locality_gce(S: torch.Tensor, order: torch.Tensor, q: float = 0.7, tau: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Locality-weighted Generalized Cross-Entropy loss over pairwise scores."""
    m = order.numel()
    if m < 2:
        return S.sum() * 0.0
    P = (order.unsqueeze(1) < order.unsqueeze(0)).float()
    p = torch.sigmoid(S)
    pc = torch.where(P > 0.5, p, 1 - p).clamp(eps, 1)
    gce = (1 - pc.pow(q)) / q
    pos = order.argsort().argsort().float()
    dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
    W = torch.exp(-dist / tau)
    mask = torch.triu(torch.ones(m, m, dtype=torch.bool, device=S.device), diagonal=1)
    return (gce * W)[mask].sum() / W[mask].sum().clamp_min(eps)

