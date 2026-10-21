"""
Cost-sensitive objectness head for RT-DETR — Direction 3 (calibration lever).

The two-level diagnosis (``critical-class-miss-not-confusion``) showed the
critical thin defects are *missed below threshold*, not misclassified: the
defect is found by some query but its confidence sits under any usable global
threshold. Two mechanisms explain it:

1. RT-DETR's varifocal ``gt_score`` == predicted IoU, so thin defects (tiny
   IoU) get a systematically low confidence *target*; and
2. DETR has no calibrated *background* score — the binary "is this a defect"
   gate is usually read off ``max_c(sigmoid(logits))``, which the B-口径
   experiment showed cannot separate defect from clean fabric (cross-domain
   FPR ~0.99).

This module adds a dedicated **objectness head**: a single ``Linear(hd, 1)``
applied to the decoder's final hidden state, producing a per-query
"this is a real object" logit that is independent of the 18-way class logits.
It is trained with a **cost-sensitive** BCE so that queries matched to a
critical class (hole / broken_warp / broken_spandex) are weighted up, making
the calibration *not miss* high-cost defects — the concrete instantiation of
the D03 (calibration) and D04 (miss) dimensions of the seven-dim reward.

Two-stage training keeps the result stable:

* **Stage 1 — linear probe** (``--probe-only``): freeze every existing
  parameter, train only the objectness head. A convex-ish linear probe on the
  frozen 256-dim decoder features cannot damage the detector and directly
  answers "can a calibrated objectness beat max-class-confidence".
* **Stage 2 — joint fine-tune** (optional): unfreeze and train with the
  cost-sensitive weight at low LR.

Usage::

    from backend.training.objectness import (
        ObjectnessRTDETRDetectionLoss,
        install_objectness_head,
        set_default_objectness,
    )
    import ultralytics.models.utils.loss as _detr_loss

    _detr_loss.RTDETRDetectionLoss = ObjectnessRTDETRDetectionLoss
    model = YOLO("runs/rtdetr/tianchi18_full/weights/best.pt")
    install_objectness_head(model)
    model.train(...)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import RTDETRDetectionLoss
import ultralytics.nn.modules.head as _head
import ultralytics.nn.modules.transformer as _transformer
from ultralytics.nn.modules.utils import inverse_sigmoid
import ultralytics.nn.tasks as _tasks

# Module-level defaults (overridable via ``set_default_objectness``).
_DEFAULT_OBJECTNESS_GAIN = 1.0
_DEFAULT_CRITICAL_IDS: Sequence[int] = (0, 9, 15)  # hole, broken_warp, broken_spandex
_DEFAULT_W_CRIT = 3.0
_DEFAULT_W_BG = 0.1

#: Stage-1 linear-probe switch consumed by ``ObjectnessRTDETRTrainer``.
_PROBE_ONLY = False


def set_probe_only(probe_only: bool) -> None:
    """Set whether the objectness trainer freezes everything but the head.

    Args:
        probe_only: When True, ``ObjectnessRTDETRTrainer`` freezes every existing
            parameter and trains only the freshly attached objectness head
            (Stage 1 linear probe).
    """
    global _PROBE_ONLY
    _PROBE_ONLY = bool(probe_only)


def set_default_objectness(
    gain: Optional[float] = None,
    critical_ids: Optional[Sequence[int]] = None,
    w_crit: Optional[float] = None,
    w_bg: Optional[float] = None,
) -> None:
    """Set module-level defaults consumed by ``ObjectnessRTDETRDetectionLoss``.

    Args:
        gain: Weight of the objectness loss term in the total loss.
        critical_ids: Class ids whose matched queries get up-weighted.
        w_crit: Weight for critical-class matched queries (default 3.0).
        w_bg: Weight for background queries, to balance the ~290:10 ratio
            (default 0.1).
    """
    global \
        _DEFAULT_OBJECTNESS_GAIN, \
        _DEFAULT_CRITICAL_IDS, \
        _DEFAULT_W_CRIT, \
        _DEFAULT_W_BG
    if gain is not None:
        _DEFAULT_OBJECTNESS_GAIN = gain
    if critical_ids is not None:
        _DEFAULT_CRITICAL_IDS = tuple(critical_ids)
    if w_crit is not None:
        _DEFAULT_W_CRIT = w_crit
    if w_bg is not None:
        _DEFAULT_W_BG = w_bg


# ---------------------------------------------------------------------------
# Model-side patches (applied once, globally, by ``install_objectness_head``).
# ---------------------------------------------------------------------------


def _patched_decoder_forward(
    self,
    embed,
    refer_bbox,
    feats,
    shapes,
    bbox_head,
    score_head,
    pos_mlp,
    attn_mask=None,
    padding_mask=None,
    obj_head=None,
):
    """Mirror ``DeformableTransformerDecoder.forward`` and add an objectness head."""
    output = embed
    dec_bboxes = []
    dec_cls = []
    last_refined_bbox = None
    refer_bbox = refer_bbox.sigmoid()
    for i, layer in enumerate(self.layers):
        output = layer(
            output,
            refer_bbox,
            feats,
            shapes,
            padding_mask,
            attn_mask,
            pos_mlp(refer_bbox),
        )
        bbox = bbox_head[i](output)
        refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
        if self.training:
            dec_cls.append(score_head[i](output))
            if i == 0:
                dec_bboxes.append(refined_bbox)
            else:
                dec_bboxes.append(
                    torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox))
                )
        elif i == self.eval_idx:
            dec_cls.append(score_head[i](output))
            dec_bboxes.append(refined_bbox)
            break
        last_refined_bbox = refined_bbox
        refer_bbox = refined_bbox.detach() if self.training else refined_bbox

    dec_obj = obj_head(output) if obj_head is not None else None
    return torch.stack(dec_bboxes), torch.stack(dec_cls), dec_obj


def _patched_head_forward(self, x, batch=None):
    """Mirror ``RTDETRDecoder.forward`` and thread the objectness logits out."""
    from ultralytics.models.utils.ops import get_cdn_group

    feats, shapes = self._get_encoder_input(x)
    dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
        batch,
        self.nc,
        self.num_queries,
        self.denoising_class_embed.weight,
        self.num_denoising,
        self.label_noise_ratio,
        self.box_noise_scale,
        self.training,
    )
    embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(
        feats, shapes, dn_embed, dn_bbox
    )
    dec_bboxes, dec_scores, dec_obj = self.decoder(
        embed,
        refer_bbox,
        feats,
        shapes,
        self.dec_bbox_head,
        self.dec_score_head,
        self.query_pos_head,
        attn_mask=attn_mask,
        obj_head=getattr(self, "dec_obj_head", None),
    )
    if self.training and dn_meta is None:
        dec_bboxes = dec_bboxes + 0 * self.denoising_class_embed.weight.sum()
    x = dec_bboxes, dec_scores, dec_obj, enc_bboxes, enc_scores, dn_meta
    if self.training:
        return x
    y = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
    return y if self.export else (y, x)


def _patched_model_loss(self, batch, preds=None):
    """Mirror ``RTDETRDetectionModel.loss`` and thread the objectness logits out."""
    if not hasattr(self, "criterion"):
        self.criterion = self.init_criterion()

    img = batch["img"]
    bs = img.shape[0]
    batch_idx = batch["batch_idx"]
    gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
    targets = {
        "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
        "bboxes": batch["bboxes"].to(device=img.device),
        "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
        "gt_groups": gt_groups,
    }

    if preds is None:
        preds = self.predict(img, batch=targets)
    x = preds if self.training else preds[1]
    dec_bboxes, dec_scores, dec_obj, enc_bboxes, enc_scores, dn_meta = x
    if dn_meta is None:
        dn_bboxes, dn_scores = None, None
    else:
        dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
        dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
        # dec_obj is (bs, nq_total, 1) — 3-dim, only the final layer — so split dim 1.
        _, dec_obj = torch.split(dec_obj, dn_meta["dn_num_split"], dim=1)

    dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
    dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

    loss = self.criterion(
        (dec_bboxes, dec_scores, dec_obj),
        targets,
        dn_bboxes=dn_bboxes,
        dn_scores=dn_scores,
        dn_meta=dn_meta,
    )
    return sum(loss.values()), torch.as_tensor(
        [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]],
        device=img.device,
    )


_PATCHED = False


def _apply_patches() -> None:
    """Swap the three ultralytics methods for objectness-aware versions (once)."""
    global _PATCHED
    if _PATCHED:
        return
    _transformer.DeformableTransformerDecoder.forward = _patched_decoder_forward
    _head.RTDETRDecoder.forward = _patched_head_forward
    _tasks.RTDETRDetectionModel.loss = _patched_model_loss
    _PATCHED = True


def attach_rtdetr_objectness_head(rtdetr) -> nn.Linear:
    """Attach a fresh objectness head to a bare ``RTDETRDetectionModel``.

    The patches are always (re)applied — they are runtime-only and never survive
    a ``YOLO(weights)`` reload.

    Args:
        rtdetr: An ``RTDETRDetectionModel`` whose ``.model[-1]`` is the decoder.

    Returns:
        The attached ``nn.Linear(hd, 1)`` objectness head.
    """
    _apply_patches()
    decoder = rtdetr.model[-1]
    if hasattr(decoder, "dec_obj_head"):
        return decoder.dec_obj_head
    ref = next(decoder.parameters())
    decoder.dec_obj_head = nn.Linear(
        decoder.hidden_dim, 1, device=ref.device, dtype=ref.dtype
    )
    return decoder.dec_obj_head


def install_objectness_head(model) -> nn.Linear:
    """Attach a fresh objectness head to a ``YOLO`` instance's decoder.

    Args:
        model: A ``ultralytics.YOLO`` instance whose ``.model.model[-1]`` is the
            ``RTDETRDecoder``.

    Returns:
        The attached ``nn.Linear(hd, 1)`` objectness head.
    """
    return attach_rtdetr_objectness_head(model.model)


class ObjectnessRTDETRTrainer(RTDETRTrainer):
    """RT-DETR trainer that attaches the objectness head to the built model.

    ``YOLO.train()`` rebuilds the model via ``get_model()``, so a head attached
    to the pre-train ``YOLO`` instance is lost. Overriding ``get_model`` attaches
    the head to the model the trainer will actually optimize.

    Probe-only mode also overrides ``build_optimizer`` to (re-)freeze the model
    *immediately before* the optimizer is built: ultralytics' ``_setup_train``
    "Freeze layers" block re-enables ``requires_grad`` on any layer it does not
    itself freeze (``engine/trainer.py``), which undoes a freeze applied earlier
    in ``get_model``. Re-freezing at optimizer-build time leaves only the
    objectness head trainable.
    """

    @staticmethod
    def _freeze_for_probe(model) -> None:
        """Freeze every parameter except the objectness head (probe-only)."""
        if not _PROBE_ONLY:
            return
        head = model.model[-1].dec_obj_head
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the RT-DETR model, then attach (and optionally isolate) the head."""
        model = super().get_model(cfg, weights, verbose)
        attach_rtdetr_objectness_head(model)
        self._freeze_for_probe(model)
        return model

    def build_optimizer(
        self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5
    ):
        """Re-apply the probe freeze right before constructing the optimizer."""
        self._freeze_for_probe(model)
        return super().build_optimizer(model, name, lr, momentum, decay, iterations)


def load_objectness_model(weights: str, device: str = "cpu"):
    """Load a YOLO checkpoint and restore the objectness head and patches.

    ultralytics' ``BaseModel.load`` intersects the checkpoint state dict with
    the freshly built architecture, which drops the ``dec_obj_head`` keys (they
    are not part of the standard RT-DETR yaml). This helper re-attaches the head
    and restores its weights from the checkpoint so the objectness gate can be
    evaluated from a saved ``best.pt``.

    Args:
        weights: Path to a ``*.pt`` checkpoint saved by the objectness trainer.
        device: Device string (``"cpu"`` or ``"0"``).

    Returns:
        A ``(model, head)`` tuple: the loaded ``YOLO`` instance and the restored
        ``nn.Linear`` head.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    head = install_objectness_head(model)

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    # Strip any module path prefix (e.g. "model.28.dec_obj_head.") down to the
    # bare "weight"/"bias" keys the nn.Linear expects.
    obj_sd = {}
    for k, v in sd.items():
        if "dec_obj_head" in k:
            obj_sd[k.split("dec_obj_head.", 1)[-1]] = v
    if obj_sd:
        head.load_state_dict(obj_sd)

    if device and str(device) != "cpu":
        # Normalize a bare GPU index ("0") into a torch.device; nn.Module.to()
        # rejects the raw digit string.
        dev = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
        model.model.to(dev)
    return model, head


# ---------------------------------------------------------------------------
# Loss-side: cost-sensitive objectness term.
# ---------------------------------------------------------------------------


class ObjectnessRTDETRDetectionLoss(RTDETRDetectionLoss):
    """RT-DETR detection loss with an added cost-sensitive objectness term.

    Inherits the class + box + denoising losses unchanged and appends
    ``loss_obj`` (a weighted BCE over "matched vs background"), so the objectness
    head learns a calibrated foreground/background score that does not miss
    critical classes.
    """

    def __init__(
        self,
        nc: int = 80,
        objectness_gain: Optional[float] = None,
        critical_ids: Optional[Sequence[int]] = None,
        w_crit: Optional[float] = None,
        w_bg: Optional[float] = None,
        **kwargs,
    ):
        """Build the upstream loss and record the cost-sensitive objectness config."""
        super().__init__(nc=nc, **kwargs)
        self.objectness_gain = (
            objectness_gain if objectness_gain is not None else _DEFAULT_OBJECTNESS_GAIN
        )
        self.critical_ids = set(
            critical_ids if critical_ids is not None else _DEFAULT_CRITICAL_IDS
        )
        self.w_crit = w_crit if w_crit is not None else _DEFAULT_W_CRIT
        self.w_bg = w_bg if w_bg is not None else _DEFAULT_W_BG

    def _objectness_loss(
        self, pred_obj: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Cost-sensitive BCE between objectness logits and matched/background targets.

        Args:
            pred_obj: Objectness logits of shape (bs, nq, 1).
            targets: Class-per-query tensor of shape (bs, nq) where ``self.nc``
                marks background and a class id marks a matched query.

        Returns:
            A scalar objectness loss (already scaled by ``objectness_gain``).
        """
        bs, nq = pred_obj.shape[:2]
        obj_target = (targets < self.nc).float().unsqueeze(-1)
        matched = targets < self.nc

        weight = torch.ones((bs, nq, 1), device=pred_obj.device)
        weight[~matched.unsqueeze(-1)] = self.w_bg
        for cid in self.critical_ids:
            weight[(targets == cid).unsqueeze(-1)] = self.w_crit

        bce = F.binary_cross_entropy_with_logits(pred_obj, obj_target, reduction="none")
        return self.objectness_gain * (bce * weight).sum() / nq

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """Compute detection losses, then append the objectness term on the main layer."""
        pred_bboxes, pred_scores, pred_obj = preds
        total_loss = super().forward(
            (pred_bboxes, pred_scores), batch, dn_bboxes, dn_scores, dn_meta
        )

        if pred_obj is None:
            total_loss["loss_obj"] = torch.tensor(0.0, device=pred_bboxes.device)
            return total_loss

        # Re-derive the matched/background mask for the main (last) layer. The
        # matcher is deterministic, so this mirrors ``DETRLoss._get_loss``.
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        match_indices = self.matcher(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups
        )
        idx, gt_idx = self._get_index(match_indices)
        bs, nq = pred_scores[-1].shape[:2]
        targets = torch.full(
            (bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype
        )
        targets[idx] = gt_cls[gt_idx]

        total_loss["loss_obj"] = self._objectness_loss(pred_obj, targets)
        return total_loss
