"""Hierarchical study model: slice encoder -> series transformer -> study transformer -> label queries.

Input is a *flat* list of series (any number per study), each with N 2.5D slices:
    images           (M, N, 3, H, W) uint8 or float          M = total series in the batch
    slice_pos        (M, N)   normalised slice coordinate in [0,1]
    plane_ids        (M,)     sagittal / coronal / axial / unknown
    seq_ids          (M,)     (weighting, fat-sat) id
    series_batch_idx (M,)     which study each series belongs to (sorted ascending)
Output: logits (B, L). Twelve learned label queries cross-attend over all slice tokens of the study
(query2label / DETR style), so focal findings (fracture, small tears) can be read from one slice.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import N_LABELS, N_PLANES, N_SEQ_IDS
from .backbone import create_backbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SlicePosEncoding(nn.Module):
    def __init__(self, d: int, scale: float = 100.0):
        super().__init__()
        self.d, self.scale = d, scale
        self.proj = nn.Linear(d, d)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:  # (M,N) -> (M,N,d)
        half = self.d // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=pos.device, dtype=torch.float32) / half)
        ang = pos.float().unsqueeze(-1) * self.scale * freq
        pe = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if pe.shape[-1] < self.d:
            pe = F.pad(pe, (0, self.d - pe.shape[-1]))
        return self.proj(pe.to(self.proj.weight.dtype))


def _per_slice(e: torch.Tensor) -> torch.Tensor:
    """Embedding of per-series ids (M,d) -> (M,1,d); per-window ids (M,N,d) stay as they are."""
    return e if e.ndim == 3 else e[:, None]


def _encoder(d: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class SpatialFindingBranch(nn.Module):
    """Residual branch that keeps the slice feature MAP instead of its average.

    Stage 1 (inside a slice): each finding owns a query that soft-selects cells of the h x w map (a meniscal tear is
    2-4 cells of 144; global average pooling dilutes it ~50x). A learned per-finding gate mixes the selected cells
    with the plain average, so diffuse findings (effusion, OA) can fall back to it.
    Stage 2 (across the study): each finding owns a second query that soft-selects slices of all series.
    The output projection is zero-initialised: at step 0 the branch adds nothing to the base model.
    """

    def __init__(self, feat_dim: int, n_labels: int, dim: int = 256, grid: int = 12, dropout: float = 0.1):
        super().__init__()
        self.dim, self.n_labels, self.grid = dim, n_labels, grid
        self.cell_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim))
        self.cell_pos = nn.Parameter(torch.zeros(1, dim, grid, grid))
        self.cell_q = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.gate = nn.Parameter(torch.zeros(n_labels))          # sigmoid(0) = 0.5: half local, half average
        self.slice_pos = SlicePosEncoding(dim)
        self.plane_emb = nn.Embedding(N_PLANES, dim)
        self.seq_emb = nn.Embedding(N_SEQ_IDS, dim)
        self.slice_norm = nn.LayerNorm(dim)
        self.slice_q = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out_w = nn.Parameter(torch.zeros(n_labels, dim))    # zero init -> neutral at start
        self.out_b = nn.Parameter(torch.zeros(n_labels))
        nn.init.trunc_normal_(self.cell_pos, std=0.02)

    def pool_cells(self, cells: torch.Tensor, hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """cells (X, h*w, C) -> per-finding slice vectors (X, L, dim) and the cell attention (X, L, h*w)."""
        x = self.cell_proj(cells)
        pos = self.cell_pos if hw == (self.grid, self.grid) else F.interpolate(self.cell_pos, size=hw, mode="bilinear", align_corners=False)
        x = x + pos.flatten(2).transpose(1, 2).to(x.dtype)
        att = torch.einsum("xcd,ld->xlc", x, self.cell_q.to(x.dtype)) / math.sqrt(self.dim)
        att = att.float().softmax(-1).to(x.dtype)
        local = torch.einsum("xlc,xcd->xld", att, x)
        g = torch.sigmoid(self.gate).view(1, -1, 1).to(x.dtype)
        return g * local + (1.0 - g) * x.mean(1, keepdim=True), att

    def forward(self, cells: torch.Tensor, hw: tuple[int, int], M: int, N: int, slice_pos: torch.Tensor, plane_ids: torch.Tensor,
                seq_ids: torch.Tensor, series_batch_idx: torch.Tensor, slot: torch.Tensor, B: int, S_max: int,
                slice_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z, cell_att = self.pool_cells(cells, hw)                                    # (M*N, L, dim)
        z = z.view(M, N, self.n_labels, self.dim)
        ctx = self.slice_pos(slice_pos) + _per_slice(self.plane_emb(plane_ids)) + _per_slice(self.seq_emb(seq_ids))
        z = self.slice_norm(z + ctx.unsqueeze(2).to(z.dtype))
        tok = z.new_zeros((B, S_max, N, self.n_labels, self.dim))
        valid = torch.zeros((B, S_max, N), dtype=torch.bool, device=z.device)
        tok[series_batch_idx, slot] = z
        valid[series_batch_idx, slot] = slice_valid.bool() if slice_valid is not None else True
        tok, valid = tok.view(B, S_max * N, self.n_labels, self.dim), valid.view(B, S_max * N)
        att = torch.einsum("btld,ld->blt", tok, self.slice_q.to(tok.dtype)) / math.sqrt(self.dim)
        att = att.float().masked_fill(~valid[:, None, :], float("-inf")).softmax(-1).to(tok.dtype)
        pooled = self.drop(torch.einsum("blt,btld->bld", att, tok))
        logits = torch.einsum("bld,ld->bl", pooled, self.out_w.to(pooled.dtype)) + self.out_b.to(pooled.dtype)
        return {"logits": logits, "slice_att": att, "cell_att": cell_att.view(M, N, self.n_labels, -1)}


class StudyModel(nn.Module):
    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained: bool = True,
        checkpoint_path: str = "",
        feature_dim: int = 0,
        d_model: int = 512,
        n_heads: int = 8,
        series_layers: int = 2,
        study_layers: int = 2,
        n_labels: int = N_LABELS,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        grad_checkpoint: bool = False,
        backbone_chunk: int = 0,
        channels_last: bool = False,
        spatial_head: bool = False,
        spatial_dim: int = 256,
        pool: str = "attn",
    ):
        super().__init__()
        self.n_labels = n_labels
        self.pool = pool  # "attn": label queries attend over windows | "meanmax": mean + max over windows -> linear (no attention at all)
        self.backbone_chunk = backbone_chunk
        self.channels_last = channels_last
        if feature_dim > 0:
            self.backbone, feat_dim = None, feature_dim
        else:
            self.backbone, feat_dim = create_backbone(backbone, pretrained, checkpoint_path, drop_path, grad_checkpoint)
            if channels_last:
                self.backbone = self.backbone.to(memory_format=torch.channels_last)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d_model), nn.Dropout(dropout))
        self.pos = SlicePosEncoding(d_model)
        self.plane_emb = nn.Embedding(N_PLANES, d_model)
        self.seq_emb = nn.Embedding(N_SEQ_IDS, d_model)
        self.series_enc = _encoder(d_model, n_heads, series_layers, dropout) if series_layers > 0 else None
        self.study_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.study_enc = _encoder(d_model, n_heads, study_layers, dropout) if study_layers > 0 else None
        self.label_queries = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.q_norm = nn.LayerNorm(d_model)
        self.head_w = nn.Parameter(torch.randn(n_labels, d_model) * (1.0 / math.sqrt(d_model)))
        self.head_b = nn.Parameter(torch.zeros(n_labels))
        self.global_head = nn.Linear(d_model, n_labels)
        self.pool_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * feat_dim, n_labels)) if pool == "meanmax" else None
        self.spatial = SpatialFindingBranch(feat_dim, n_labels, spatial_dim, dropout=dropout) if (spatial_head and feature_dim == 0) else None
        nn.init.trunc_normal_(self.study_token, std=0.02)

    # ------------------------------------------------------------------ slice encoding
    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int]]:
        """-> pooled (X,F), cells (X,h*w,F) or None, (h,w)."""
        if self.spatial is None:
            return self.backbone(x), None, (0, 0)
        fm = self.backbone.forward_features(x)
        if fm.ndim == 4:                                   # CNN: (X,F,h,w)
            hw = (fm.shape[2], fm.shape[3])
            pooled = self.backbone.forward_head(fm, pre_logits=True) if hasattr(self.backbone, "forward_head") else fm.mean((2, 3))
            return pooled, fm.flatten(2).transpose(1, 2), hw
        n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))  # ViT: (X, prefix + h*w, F)
        cells = fm[:, n_prefix:]
        side = int(round(math.sqrt(cells.shape[1])))
        pooled = self.backbone.forward_head(fm, pre_logits=True) if hasattr(self.backbone, "forward_head") else fm[:, 0]
        return pooled, cells, (side, cells.shape[1] // max(side, 1))

    def encode_slices(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int]]:
        """(M,N,3,H,W) -> pooled (M,N,F), cells (M*N,h*w,F) or None, (h,w)."""
        M, N = images.shape[:2]
        x = images.flatten(0, 1)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        x = (x - self.mean) / self.std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.backbone_chunk and x.shape[0] > self.backbone_chunk and not self.training:
            parts = [self._features(x[i:i + self.backbone_chunk]) for i in range(0, x.shape[0], self.backbone_chunk)]
            pooled = torch.cat([p[0] for p in parts])
            cells = torch.cat([p[1] for p in parts]) if parts[0][1] is not None else None
            hw = parts[0][2]
        else:
            pooled, cells, hw = self._features(x)
        return pooled.view(M, N, -1), cells, hw

    # ------------------------------------------------------------------ forward
    def forward(self, images: torch.Tensor, slice_pos: torch.Tensor, plane_ids: torch.Tensor, seq_ids: torch.Tensor,
                series_batch_idx: torch.Tensor, batch_size: int, return_attn: bool = False,
                slice_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B = int(batch_size)
        d = self.study_token.shape[-1]
        M = images.shape[0]
        if M > 0:
            cells, hw = None, (0, 0)
            if self.backbone is not None:
                feats, cells, hw = self.encode_slices(images)
            else:
                feats = images
            N = feats.shape[1]
            if self.pool_head is not None:  # leaders' recipe: no attention, no position / plane embeddings, just pooling
                sv = slice_valid.bool() if slice_valid is not None else torch.ones((M, N), dtype=torch.bool, device=feats.device)
                f, F_ = feats.float(), feats.shape[-1]
                w = sv.unsqueeze(-1).to(f.dtype)
                idx = series_batch_idx.view(M, 1).expand(M, F_)
                total = f.new_zeros((B, F_)).index_add_(0, series_batch_idx, (f * w).sum(1))
                count = f.new_zeros((B, 1)).index_add_(0, series_batch_idx, w.sum(1))
                mx = f.new_full((B, F_), -1e4).scatter_reduce(0, idx, f.masked_fill(~sv.unsqueeze(-1), -1e4).amax(1), reduce="amax")
                mx = torch.where(count > 0, mx, torch.zeros_like(mx))
                return {"logits": self.pool_head(torch.cat([total / count.clamp_min(1.0), mx], -1)).float()}
            x = self.proj(feats) + self.pos(slice_pos) + _per_slice(self.plane_emb(plane_ids)) + _per_slice(self.seq_emb(seq_ids))
            sv = slice_valid.bool() if slice_valid is not None else torch.ones((M, N), dtype=torch.bool, device=x.device)
            if self.series_enc is not None:
                x = self.series_enc(x, src_key_padding_mask=~sv)
            # scatter series into (B, S_max, N, d)
            counts = torch.bincount(series_batch_idx, minlength=B)
            S_max = int(counts.max().item())
            first = torch.cumsum(counts, 0) - counts
            slot = torch.arange(M, device=x.device) - first[series_batch_idx]
            tokens = x.new_zeros((B, S_max, N, d))
            valid = torch.zeros((B, S_max, N), dtype=torch.bool, device=x.device)
            tokens[series_batch_idx, slot] = x
            valid[series_batch_idx, slot] = sv
            tokens = tokens.view(B, S_max * N, d)
            valid = valid.view(B, S_max * N)
            spatial_out = (self.spatial(cells, hw, M, N, slice_pos, plane_ids, seq_ids, series_batch_idx, slot, B, S_max, sv)
                           if (self.spatial is not None and cells is not None) else None)
        else:
            spatial_out = None
            tokens = self.study_token.new_zeros((B, 0, d))
            valid = torch.zeros((B, 0), dtype=torch.bool, device=tokens.device)
        tokens = torch.cat([self.study_token.expand(B, -1, -1).to(tokens.dtype), tokens], 1)
        valid = torch.cat([torch.ones((B, 1), dtype=torch.bool, device=valid.device), valid], 1)
        pad = ~valid
        if self.study_enc is not None:
            tokens = self.study_enc(tokens, src_key_padding_mask=pad)
        q = self.label_queries.unsqueeze(0).expand(B, -1, -1).to(tokens.dtype)
        out, attn = self.cross(q, tokens, tokens, key_padding_mask=pad, need_weights=return_attn, average_attn_weights=True)
        out = self.q_norm(out)
        logits = torch.einsum("bld,ld->bl", out, self.head_w.to(out.dtype)) + self.head_b.to(out.dtype)
        logits = logits + self.global_head(tokens[:, 0])
        if spatial_out is not None:
            logits = logits + spatial_out["logits"].to(logits.dtype)
        res = {"logits": logits.float()}
        if return_attn and spatial_out is not None:
            res["cell_att"], res["slice_att"] = spatial_out["cell_att"].float(), spatial_out["slice_att"].float()
        if return_attn and attn is not None:
            res["attn"] = attn.float()  # (B, L, 1 + S_max*N)
        return res

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
        bb = set(id(p) for p in self.backbone.parameters()) if self.backbone is not None else set()
        groups = {"bb_decay": [], "bb_no": [], "hd_decay": [], "hd_no": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            no_decay = (p.ndim <= 1 or n.endswith(".bias") or "emb" in n or "token" in n or "label_queries" in n
                        or n.endswith(("cell_q", "slice_q", "cell_pos", "gate")))
            key = ("bb_" if id(p) in bb else "hd_") + ("no" if no_decay else "decay")
            groups[key].append(p)
        return [
            {"params": groups["bb_decay"], "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": groups["bb_no"], "lr": lr_backbone, "weight_decay": 0.0},
            {"params": groups["hd_decay"], "lr": lr_head, "weight_decay": weight_decay},
            {"params": groups["hd_no"], "lr": lr_head, "weight_decay": 0.0},
        ]
