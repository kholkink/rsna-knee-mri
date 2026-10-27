"""timm backbone factory (CNN or ViT) returning pooled slice embeddings.

Licence notes for a prize-eligible solution (check the competition rules):
  * tf_efficientnetv2_* (Apache-2.0), convnext_*.fb_in22k (MIT), eva02_* (MIT), vit_*_dinov2 (Apache-2.0)
  * convnextv2_* (CC-BY-NC), dinov3 (custom licence) - strong but review before using in a winning entry
"""
from __future__ import annotations

import logging

import timm
from torch import nn

log = logging.getLogger(__name__)

RECOMMENDED = {
    "fast": "tf_efficientnetv2_s.in21k_ft_in1k",
    "balanced": "convnext_small.fb_in22k_ft_in1k_384",
    "strong": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    "dino": "vit_base_patch14_dinov2.lvd142m",
}


def create_backbone(name: str, pretrained: bool = True, checkpoint_path: str = "", drop_path: float = 0.1,
                    grad_checkpoint: bool = False) -> tuple[nn.Module, int]:
    kwargs: dict = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if checkpoint_path:
        kwargs["pretrained"] = True
        kwargs["pretrained_cfg_overlay"] = dict(file=checkpoint_path)
    try:
        m = timm.create_model(name, drop_path_rate=drop_path, **kwargs)
    except TypeError:
        m = timm.create_model(name, **kwargs)
    if grad_checkpoint:
        try:
            m.set_grad_checkpointing(True)
        except Exception as e:  # pragma: no cover
            log.warning("grad checkpointing unsupported for %s: %s", name, e)
    return m, int(m.num_features)
