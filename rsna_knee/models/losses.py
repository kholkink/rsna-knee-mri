from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SoftTargetBCE(nn.Module):
    """BCE-with-logits against soft targets with per-(study,label) weights; NaN targets are ignored."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        valid = torch.isfinite(targets)
        t = torch.where(valid, targets, torch.zeros_like(targets))
        if self.ls > 0:
            t = t * (1 - self.ls) + 0.5 * self.ls
        w = torch.ones_like(t) if weights is None else weights
        w = torch.where(valid, w, torch.zeros_like(w))
        loss = F.binary_cross_entropy_with_logits(logits.float(), t.float(), reduction="none")
        return (loss * w).sum() / w.sum().clamp(min=1e-6)


class QueuePairwiseLoss(nn.Module):
    """AUC surrogate: the current logits are ranked against a FIFO queue of recent (detached) logits of the same label.

    Only confident cells form pairs (target <= lo is a negative, >= hi a positive): mid-range pseudo-labels and NaN
    never take part, so label noise is not amplified. A current positive z against a queued negative q costs
    softplus((q - z) / tau), a current negative against a queued positive softplus((z - q) / tau). The queue gives rare
    findings pairs even with 4 studies per step. Labels are averaged equally, like the macro AUC they stand for.
    """

    def __init__(self, n_labels: int, size: int = 512, lo: float = 0.15, hi: float = 0.85, tau: float = 1.0):
        super().__init__()
        self.lo, self.hi, self.tau = lo, hi, tau
        self.register_buffer("q_logits", torch.zeros(size, n_labels))
        self.register_buffer("q_targets", torch.full((size, n_labels), float("nan")))
        self.ptr = 0

    @torch.no_grad()
    def _enqueue(self, z: torch.Tensor, t: torch.Tensor) -> None:
        n, size = z.shape[0], self.q_logits.shape[0]
        idx = (self.ptr + torch.arange(n, device=z.device)) % size
        self.q_logits[idx], self.q_targets[idx] = z, t
        self.ptr = int((self.ptr + n) % size)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        z, t = logits.float(), targets.float()
        pos, neg = t >= self.hi, t <= self.lo  # NaN compares False both ways
        q_pos, q_neg = self.q_targets >= self.hi, self.q_targets <= self.lo
        d = (z[:, None, :] - self.q_logits[None, :, :]) / self.tau  # (B, Q, L)
        m_pn = pos[:, None, :] & q_neg[None, :, :]
        m_np = neg[:, None, :] & q_pos[None, :, :]
        num = (F.softplus(-d) * m_pn + F.softplus(d) * m_np).sum((0, 1))
        den = (m_pn | m_np).sum((0, 1))
        has = den > 0
        loss = (num[has] / den[has]).mean() if bool(has.any()) else z.sum() * 0.0
        self._enqueue(z.detach(), t.detach())
        return loss
