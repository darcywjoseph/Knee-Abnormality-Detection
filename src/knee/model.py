"""The 2.5D model, shared by the dry run and the diagnostics.

The training and submission notebooks keep their own copies of these classes because a Kaggle
kernel cannot import from the repo. Keep the three classes here identical to those copies;
`knee.infer.dryrun` exercises the submission one.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from knee.common import TARGETS

__all__ = ["AttnPool", "LabelAttnHead", "Model2p5D"]


class AttnPool(nn.Module):
    """Gated attention over the slices of one slot (Ilse et al., attention MIL).

    One gate serves every slot and every label, so this pooling is label-agnostic and
    order-blind. See `LabelAttnHead` for the per-label alternative.

    Args:
        dim: Slice embedding width.
        hidden: Width of the gate MLP.
    """

    def __init__(self, dim, hidden):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x):
        """Pools (B, S, D) slice embeddings into (B, D).

        Args:
            x: Slice embeddings of one slot per row.

        Returns:
            The pooled embedding and the (B, S) attention weights.
        """
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))   # (B, S, 1)
        a = a.softmax(dim=1)
        return (a * x).sum(dim=1), a.squeeze(-1)


class LabelAttnHead(nn.Module):
    """Per-label attention over every slice embedding in the study (attention-MIL).

    Each label owns one softmax over the K*S slice embeddings, so a finding visible on two
    slices of one slot is kept rather than averaged down by a gate shared across labels.
    Slices of absent slots are masked out of every softmax.

    Args:
        dim: Slice embedding width.
        hidden: Width of the attention MLP.
        n_slots: Number of cache slots K.
        n_out: Number of labels.
        slot_emb: Add a learned per-slot vector to each slice embedding before attention.
        dropout: Dropout inside the attention MLP.
    """

    def __init__(self, dim, hidden, n_slots, n_out, slot_emb, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, dim)) if slot_emb else None
        self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))
        self.cls_w = nn.Parameter(torch.empty(n_out, dim))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        nn.init.trunc_normal_(self.cls_w, std=0.02)

    def forward(self, f, present):
        """Pools (B, K, S, D) slice embeddings into (B, n_out) logits.

        Args:
            f: Slice embeddings, one per (slot, slice).
            present: (B, K) mask, 1 where the slot exists in the study.

        Returns:
            Logits of shape (B, n_out).
        """
        b, k, s, d = f.shape
        if self.slot_emb is not None:
            f = f + self.slot_emb[None, :, None, :]
        h = self.norm(f).reshape(b, k * s, d)
        a = self.att(h)                                                # (B, K*S, n_out)
        keep = (present > 0.5)[:, :, None].expand(b, k, s).reshape(b, k * s, 1)
        a = a.masked_fill(~keep, -1e4).softmax(dim=1)
        pooled = torch.einsum("bnl,bnd->bld", a, h)                    # (B, n_out, D)
        return (pooled * self.cls_w).sum(-1) + self.cls_b


class Model2p5D(nn.Module):
    """2.5D study classifier: one encoder per slice, pooled per slot, one logit per label.

    Args:
        cfg: The training CFG dict; reads BACKBONE, PRETRAINED, VIT_*, GRAD_CHECKPOINT,
            N_SLOTS, HEAD, HEAD_HIDDEN, SLOT_EMB, ATTN_DIM and DROPOUT.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_slots = cfg["N_SLOTS"]
        self.is_vit = "vit" in cfg["BACKBONE"]
        kw = {"img_size": cfg["VIT_IMG"]} if self.is_vit else {}
        self.backbone = timm.create_model(
            cfg["BACKBONE"], pretrained=cfg["PRETRAINED"], num_classes=0, in_chans=3, **kw)
        if cfg["GRAD_CHECKPOINT"]:
            self.backbone.set_grad_checkpointing(True)
        dim = self.backbone.num_features
        if self.is_vit:
            # DINOv2 recipe: freeze everything but the last VIT_UNFREEZE blocks (+ final
            # norm); feature = [CLS ; mean of patch tokens]. VIT_UNFREEZE >= depth means a
            # genuinely full fine-tune, with patch_embed, cls_token and pos_embed trainable
            # too, so that a ViT can be compared fairly against a fully trained convnext.
            n_blocks = len(self.backbone.blocks)
            if cfg["VIT_UNFREEZE"] < n_blocks:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                for blk in self.backbone.blocks[-cfg["VIT_UNFREEZE"]:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True
            else:
                print(f"ViT: FULL fine-tune ({n_blocks} blocks, nothing frozen)")
            dim = dim * 2
        self.head_kind = cfg["HEAD"]
        if self.head_kind == "label_attn":
            self.head = LabelAttnHead(dim, cfg["HEAD_HIDDEN"], self.n_slots, len(TARGETS),
                                      cfg["SLOT_EMB"], cfg["DROPOUT"])
        else:
            self.pool = AttnPool(dim, cfg["ATTN_DIM"])
            self.drop = nn.Dropout(cfg["DROPOUT"])
            self.head = nn.Linear(dim * self.n_slots, len(TARGETS))

    def forward(self, x, present=None):
        """Scores a batch of studies.

        Args:
            x: uint8 or float images of shape (B, slots, S, 3, H, W).
            present: (B, slots) mask, 1 where the slot exists in the study.

        Returns:
            Logits of shape (B, 12).
        """
        b, k, s = x.shape[:3]
        flat = x.reshape(b * k * s, *x.shape[3:])
        if self.is_vit:
            t = self.backbone.forward_features(flat)                   # (N, 1+P, D)
            npre = self.backbone.num_prefix_tokens
            f = torch.cat([t[:, 0], t[:, npre:].mean(dim=1)], dim=1)   # (N, 2D)
        else:
            f = self.backbone(flat)                                    # (B*k*S, D)
        if self.head_kind == "label_attn":
            if present is None:
                present = torch.ones(b, k, device=f.device)
            return self.head(f.reshape(b, k, s, -1), present)
        f = f.reshape(b * k, s, -1)
        emb, _ = self.pool(f)                                          # (B*k, D)
        emb = emb.reshape(b, k, -1)
        if present is not None:
            # A missing slot is all zeros in the cache; zero its embedding too, so the head
            # sees "absent" rather than "the embedding of a black volume".
            emb = emb * present.unsqueeze(-1)
        return self.head(self.drop(emb.reshape(b, -1)))
