"""Adaptive cls/reg loss weighting for RT-DETR -- Uncertainty Weighting and GradNorm.

Motivation (docs/plan_stage2_adaptive.md)
-----------------------------------------
All seven manual reward/penalty interventions failed on the strong baseline
(recall-precision zero-sum; mean -0.015..-0.029). The *base* loss weighting
itself was never isolated: ultralytics DETRLoss defaults are
``class x1, bbox(L1) x5, giou x2`` -- a reg-heavy prior for a metric
(mean of mAP@0.1/0.3/0.5) that is classification-dominated at the two low
IoU thresholds. This module lets the model *learn* the cls/reg balance:

- ``uw``       -- Kendall et al., CVPR 2018 (arXiv:1705.07115). Two learnable
                 log-variances; per-step analytic update
                 ``lv_i -= lr * (1 - wu_i * L_i)`` with ``wu_i = exp(-lv_i)``
                 (stationary ``wu_i = 1/L_i``), then a scale-preserving
                 re-normalisation so the applied weights keep the TOTAL loss
                 equal to the uniform control -- the only variable is the
                 cls:reg ratio (= L_reg:L_cls at stationarity). Init lv=0 ->
                 weights exactly 1:1 -> identical to the uniform control at
                 step 0. No extra backward passes.
- ``gradnorm`` -- Chen et al., ICML 2018 (arXiv:1711.02257). Per-step
                 ``autograd.grad`` of each group w.r.t. the shared trunk
                 (HybridEncoder) gives gradient norms G_i; multiplicative
                 update ``w_i *= (Gbar/G_i) * (r_i/rbar)^alpha`` with inverse
                 training rate r_i = L_i / L_i(0). ~2x backward cost.

Robustness (2026-10-20 post-mortem, two bugs found and fixed):
1. ``w = exp(+lv)`` sign error -> positive feedback runaway to the clip
   ceiling. Correct is ``exp(-lv)`` (high loss -> lv up -> w down).
2. Using the normalised weight in the update term -> both arms pin at the
   clip ceiling (equal weights, exact 1:1, arm degenerates to the uniform
   control). The term must use the unnormalised ``wu = exp(-lv)``.
Also: at ~1.8 GT instances/batch the reg group loss is exactly 0 on many
batches; the naive update reads that as "boost this arm" -- EMA smoothing
plus the MIN_L freeze make degenerate batches inert.

The criterion keeps plain-Python weights (no nn.Parameter), because the
trainer builds its optimizer *before* ``init_criterion`` (engine/trainer.py:
optimizer at line ~280, criterion rebuild at line ~1027 inside _setup_train)
-- new criterion parameters would never be optimized. The analytic updates
sidestep the optimizer entirely.

Grouping: cls group = every loss key containing "class" (main + aux + dn);
reg group = every key containing "bbox" or "giou". Same keys are returned
with weighted values so results.csv logging (giou/cls/l1) stays intact --
same injection semantics as backend/training/boxweighted.py.

Attach AFTER _setup_train (on_train_start), replacing trainer.model.criterion
-- same pattern as train_rtdetr_reward.py's attach_reward_criterion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from ultralytics.models.utils.loss import RTDETRDetectionLoss

W_CLIP = (0.05, 10.0)  # hard bounds on learned weights
LV_CLIP = (-3.0, 3.0)  # log-var bounds for UW -> w_u in [0.050, 20.1]; wide
# enough that realistic stationary points (ln L, with
# L~1-15 at these loss scales) never pin -- pinning
# both arms at the same bound silently degenerates
# the arm to the uniform control (2026-10-20 bug #2)
EMA_DECAY = 0.99  # per-step loss smoothing for the weight updates
WARMUP_STEPS = 20  # cumulative-mean warmup before EMA kicks in
MIN_L = 1e-4  # arms with smoothed loss below this are frozen
DEBUG = bool(__import__("os").environ.get("ADAPTIVE_DEBUG"))
DEBUG_EVERY = int(__import__("os").environ.get("ADAPTIVE_DEBUG_EVERY", "50"))


class AdaptiveRTDETRDetectionLoss(RTDETRDetectionLoss):
    """RT-DETR loss with learnable cls/reg group weights (UW or GradNorm)."""

    def __init__(
        self,
        nc: int = 80,
        method: str = "uw",
        uw_lr: float = 1e-3,
        gradnorm_alpha: float = 1.0,
        gradnorm_lr: float = 1e-3,
        trunk_params: Optional[list] = None,
        log_every: int = 50,
        **kwargs,
    ):
        super().__init__(nc=nc, **kwargs)
        assert method in ("uw", "gradnorm"), method
        self.method = method
        self.uw_lr = uw_lr
        self.gradnorm_alpha = gradnorm_alpha
        self.gradnorm_lr = gradnorm_lr
        self.trunk_params = list(trunk_params) if trunk_params else None
        self.log_every = log_every

        # plain-python state (updated analytically / multiplicatively)
        self.lv = {"cls": 0.0, "reg": 0.0}  # UW log-variances
        self.w = {"cls": 1.0, "reg": 1.0}  # current weights (both methods)
        self.ema = {"cls": None, "reg": None}  # smoothed losses for updates
        self.warm = 0  # warmup step counter
        self.L0: Optional[dict] = None  # GradNorm initial losses
        self.step = 0
        self.weight_log: list[dict] = []

    # ---------------------------------------------------------------- helpers
    def _group_losses(
        self, comp: dict
    ) -> tuple[torch.Tensor, torch.Tensor, list, list]:
        cls_keys = [k for k in comp if "class" in k]
        reg_keys = [k for k in comp if ("bbox" in k or "giou" in k)]
        assert len(cls_keys) + len(reg_keys) == len(comp), (
            f"unpartitioned loss keys: {sorted(comp)}"
        )
        L_cls = comp[cls_keys[0]]
        for k in cls_keys[1:]:
            L_cls = L_cls + comp[k]
        L_reg = comp[reg_keys[0]]
        for k in reg_keys[1:]:
            L_reg = L_reg + comp[k]
        return L_cls, L_reg, cls_keys, reg_keys

    def _grad_norm(self, loss: torch.Tensor, w: float) -> Optional[float]:
        """||grad_{trunk} (w * loss)||_2 -- GradNorm's shared-layer approximation."""
        if not self.trunk_params:
            return None
        grads = torch.autograd.grad(
            w * loss, self.trunk_params, retain_graph=True, allow_unused=True
        )
        sq = sum((g**2).sum() for g in grads if g is not None)
        return float(sq.sqrt().item())

    def _smooth(self, key: str, v: float) -> float:
        """Cumulative mean for the first WARMUP_STEPS steps, then EMA.

        Zero-loss batches (no positive reg matches -- common at ~1.8 GT
        instances/batch) must not be read as "this arm is weak": EMA dilution
        plus the MIN_L freeze below make degenerate batches inert.
        """
        self.warm += 1
        if self.ema[key] is None:
            if v <= 0:
                return 0.0
            self.ema[key] = v
        elif self.warm <= WARMUP_STEPS:
            self.ema[key] += (v - self.ema[key]) / self.warm
        else:
            self.ema[key] = EMA_DECAY * self.ema[key] + (1 - EMA_DECAY) * v
        return self.ema[key]

    def _update_weights(self, L_cls: torch.Tensor, L_reg: torch.Tensor) -> None:
        lc = self._smooth("cls", float(L_cls.detach()))
        lr = self._smooth("reg", float(L_reg.detach()))
        if DEBUG and (self.step % DEBUG_EVERY == 0 or self.step < 5):
            print(
                f"[dbg] step={self.step} L_cls={float(L_cls.detach()):.4f} "
                f"L_reg={float(L_reg.detach()):.4f} ema=({lc:.4f},{lr:.4f}) "
                f"lv=({self.lv['cls']:.4f},{self.lv['reg']:.4f}) "
                f"wu=({float(torch.exp(-torch.tensor(self.lv['cls']))):.4f},"
                f"{float(torch.exp(-torch.tensor(self.lv['reg']))):.4f}) "
                f"w=({self.w['cls']:.4f},{self.w['reg']:.4f})",
                flush=True,
            )
        if self.method == "uw":
            # Kendall et al. 2018: F = sum_i (w_i L_i + lv_i), w_i = exp(-lv_i).
            # dF/dlv_i = 1 - w_i L_i  ->  lv_i -= lr * (1 - w_i L_i), stationary
            # w_i = 1/L_i (each arm contributes exactly 1 to the total).
            # CRITICAL (2026-10-20 post-mortem, second bug): the update term must
            # use the UNNORMALISED w_u = exp(-lv). Using the scale-normalised
            # weight here biases lv past its stationary point until BOTH arms
            # pin at the clip ceiling -- equal weights, exact 1:1, zero
            # rebalancing (the arm silently degenerates to the uniform control).
            for k, L in (("cls", lc), ("reg", lr)):
                if L < MIN_L:
                    continue  # degenerate arm: freeze (do not "boost" zero loss)
                wu = float(torch.exp(-torch.tensor(self.lv[k])))
                self.lv[k] -= self.uw_lr * (1.0 - wu * L)
                self.lv[k] = float(min(max(self.lv[k], LV_CLIP[0]), LV_CLIP[1]))
            # scale-preserving re-normalisation: the applied weights keep the
            # TOTAL loss identical to the uniform control (w_cls*L_cls +
            # w_reg*L_reg = L_cls + L_reg), so the ONLY experimental variable
            # is the cls:reg ratio (= L_reg:L_cls at stationarity). No lr
            # inflation, no loss-scale confound.
            wu_c = float(torch.exp(-torch.tensor(self.lv["cls"])))
            wu_r = float(torch.exp(-torch.tensor(self.lv["reg"])))
            denom = wu_c * lc + wu_r * lr
            scale = (lc + lr) / denom if denom > 1e-9 else 1.0
            self.w["cls"] = float(min(max(wu_c * scale, W_CLIP[0]), W_CLIP[1]))
            self.w["reg"] = float(min(max(wu_r * scale, W_CLIP[0]), W_CLIP[1]))
        else:  # gradnorm
            if lc + lr <= 0:
                return
            if self.L0 is None:
                self.L0 = {"cls": max(lc, 1e-6), "reg": max(lr, 1e-6)}
            G_cls = self._grad_norm(L_cls, self.w["cls"])
            G_reg = self._grad_norm(L_reg, self.w["reg"])
            if G_cls is None or G_reg is None or G_cls <= 0 or G_reg <= 0:
                return
            r_cls, r_reg = lc / self.L0["cls"], lr / self.L0["reg"]
            r_bar = 0.5 * (r_cls + r_reg)
            G_bar = 0.5 * (G_cls + G_reg)
            a = self.gradnorm_alpha
            self.w["cls"] *= (G_bar / G_cls) * (r_cls / max(r_bar, 1e-9)) ** a
            self.w["reg"] *= (G_bar / G_reg) * (r_reg / max(r_bar, 1e-9)) ** a
            s = self.w["cls"] + self.w["reg"]
            self.w["cls"] *= 2.0 / s
            self.w["reg"] *= 2.0 / s
            self.w["cls"] = float(min(max(self.w["cls"], W_CLIP[0]), W_CLIP[1]))
            self.w["reg"] = float(min(max(self.w["reg"], W_CLIP[0]), W_CLIP[1]))

    def _log(self, lc: float, lr: float) -> None:
        self.step += 1
        if self.step % self.log_every:
            return
        row = {
            "step": self.step,
            "w_cls": round(self.w["cls"], 5),
            "w_reg": round(self.w["reg"], 5),
            "ratio": round(self.w["cls"] / max(self.w["reg"], 1e-9), 4),
            "L_cls_ema": round(lc, 5),
            "L_reg_ema": round(lr, 5),
        }
        self.weight_log.append(row)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict,
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        comp = super().forward(
            preds, batch, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        L_cls, L_reg, cls_keys, reg_keys = self._group_losses(comp)

        self._update_weights(L_cls, L_reg)  # weights advance AFTER this batch's
        self._log(self.ema["cls"] or 0.0, self.ema["reg"] or 0.0)  # losses (SGD order)

        out = {}
        for k in cls_keys:
            out[k] = self.w["cls"] * comp[k]
        for k in reg_keys:
            out[k] = self.w["reg"] * comp[k]
        return out

    # ---------------------------------------------------------------- persist
    def dump_weight_log(self, save_dir: str) -> Path:
        path = Path(save_dir) / "adaptive_weights.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "method": self.method,
                    "final_w": {k: round(v, 5) for k, v in self.w.items()},
                    "final_lv": {k: round(v, 5) for k, v in self.lv.items()},
                    "n_steps": self.step,
                    "trajectory": self.weight_log,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path
