"""copypaste_det.py -- detection-task Copy-Paste augmentation for ultralytics.

WHY THIS EXISTS
---------------
ultralytics' native CopyPaste transform silently no-ops on detection datasets:

    def __call__(self, labels):
        if len(labels["instances"].segments) == 0 or self.p == 0:
            return labels          # <-- detection labels have no segments -> always skipped

So `copy_paste=0.5` on a pure detection dataset changes NOTHING (verified by source
inspection of ultralytics 8.4.91, ultralytics/data/augment.py, 2026-10-21).

WHAT THIS MODULE DOES
---------------------
Monkey-patches CopyPaste.__call__ so that detection labels (no segments) take a
cross-image paste path:

  * donor = a random other training image (raw, un-augmented, via
    dataset.get_image_and_label)
  * donor instances are sampled with probability proportional to 1/class_freq
    (long-tail classes get pasted more often -- the point of the augmentation)
  * each pasted instance: pixel-level bbox crop, scale jitter 0.5-1.5x,
    random horizontal flip, pasted at a random position with IoU < 0.10 against
    all existing GT boxes (no occlusion ambiguity)
  * pasted boxes/clses are appended to labels["instances"] (normalized xywh)

SEGMENTATION labels keep the original ultralytics behavior untouched.

PIPELINE POSITION
-----------------
v8_transforms builds pre_transform = Compose([Mosaic, CopyPaste, affine]) when
copy_paste_mode == "flip", so our transform sees the mosaic-composited image
(2*imgsz square) with normalized instances. bbox conversions are done with the
live image shape, so this is position-independent.

USAGE
-----
    from backend.training.copypaste_det import install_detection_copypaste
    install_detection_copypaste()   # before model.train(...)
    model.train(..., copy_paste=0.5, copy_paste_mode="flip")

DEBUG
-----
CP_DEBUG_DIR=/path  -> save the first 8 pasted images as PNG (visual smoke test).
CP_DEBUG_N=16       -> how many to save (default 8).
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import random

import cv2
import numpy as np

_DEBUG_DIR = os.environ.get("CP_DEBUG_DIR")
_DEBUG_N = int(os.environ.get("CP_DEBUG_N", "8"))
_debug_count = 0
# 类分布统计 (CP_STATS_DIR 设置时启用): 记录每次粘贴的类 ID, 用于机制诊断
# (1/类频次加权是否真正提升长尾类实例占比)。不影响训练 (env 未设置时为零开销)。
_STATS_DIR = os.environ.get("CP_STATS_DIR")
_stats_cls: list[int] = []

_ORIG_CALL = None


def _class_freq(dataset) -> np.ndarray | None:
    """Count GT boxes per class from dataset.labels (no image loading)."""
    freq = np.zeros(max(getattr(dataset, "nc", 0), 1), dtype=np.int64)
    labels = getattr(dataset, "labels", None)
    if not labels:
        return None
    for lab in labels:
        cls = lab.get("cls") if isinstance(lab, dict) else None
        if cls is None:
            continue
        arr = np.asarray(cls).reshape(-1).astype(int)
        arr = arr[(arr >= 0) & (arr < len(freq))]
        if arr.size:
            freq[arr] += 1
    if freq.sum() == 0:
        return None
    return freq


def _save_debug(img: np.ndarray, inst, cls: np.ndarray) -> None:
    global _debug_count
    if not _DEBUG_DIR or _debug_count >= _DEBUG_N:
        return
    Path(_DEBUG_DIR).mkdir(parents=True, exist_ok=True)
    vis = img.copy()
    h, w = vis.shape[:2]
    for b, c in zip(inst.bboxes, np.asarray(cls).reshape(-1), strict=False):
        x, y, bw, bh = [float(v) for v in b]
        x1, y1, x2, y2 = (
            int((x - bw / 2) * w),
            int((y - bh / 2) * h),
            int((x + bw / 2) * w),
            int((y + bh / 2) * h),
        )
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            vis,
            str(int(c)),
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    cv2.imwrite(
        str(Path(_DEBUG_DIR) / f"cp_debug_{_debug_count:03d}.jpg"),
        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
    )
    _debug_count += 1


def _paste_detection(self, labels: dict) -> dict:
    """Cross-image copy-paste for detection labels (the patched path)."""
    global _debug_count
    from ultralytics.utils.metrics import bbox_ioa

    if self.p <= 0:
        return labels
    dataset = self.dataset
    n = len(dataset)
    if n < 2:
        return labels

    # --- sample a donor image that actually has GT (<=10 tries) ---
    donor = None
    for _ in range(10):
        di = random.randrange(n)
        try:
            cand = dataset.get_image_and_label(di)
        except Exception:
            continue
        if len(cand["instances"]) > 0:
            donor = cand
            break
    if donor is None:
        return labels

    img = labels["img"]  # HWC uint8 (mosaic output at this stage)
    H, W = img.shape[:2]
    inst = labels["instances"]
    inst.convert_bbox("xyxy")
    inst.denormalize(W, H)

    dimg = donor["img"]
    dh, dw = dimg.shape[:2]
    dinst = deepcopy(donor["instances"])
    dinst.convert_bbox("xyxy")
    dinst.denormalize(dw, dh)

    # --- 1/class_freq weighted instance selection from the donor ---
    donor_cls = (
        np.asarray(donor.get("cls", np.zeros((len(dinst), 1)))).reshape(-1).astype(int)
    )
    if getattr(self, "_freq", None) is None:
        self._freq = _class_freq(dataset)
    if self._freq is not None and len(dinst) > 0:
        valid = (donor_cls >= 0) & (donor_cls < len(self._freq))
        if valid.any():
            inv = np.zeros(len(dinst), dtype=float)
            inv[valid] = 1.0 / np.maximum(self._freq[donor_cls[valid]], 1.0)
            nz = int((inv > 0).sum())
            if inv.sum() > 0 and nz > 0:
                # replace=False 要求 p 的非零项 >= size, 否则 numpy 抛
                # "Fewer non-zero entries in p than size" (2026-10-21 冒烟实测)
                probs = inv / inv.sum()
                k = min(random.randint(1, 3), nz)
                sel = np.random.choice(len(dinst), size=k, replace=False, p=probs)
            else:
                sel = np.arange(min(random.randint(1, 3), len(dinst)))
        else:
            sel = np.arange(min(random.randint(1, 3), len(dinst)))
    else:
        sel = np.arange(min(random.randint(1, 3), len(dinst)))

    new_boxes: list[list[float]] = []
    new_cls: list[float] = []
    for j in sel:
        x1, y1, x2, y2 = [int(round(v)) for v in dinst.bboxes[j]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(dw, x2), min(dh, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        patch = dimg[y1:y2, x1:x2].copy()
        s = random.uniform(0.5, 1.5)
        nh, nw = max(2, int(patch.shape[0] * s)), max(2, int(patch.shape[1] * s))
        if nh >= H or nw >= W:
            continue
        patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if random.random() < 0.5:
            patch = patch[:, ::-1]
        # find a free slot: IoA < 0.10 vs existing GT (<=10 tries)
        placed = False
        for _ in range(10):
            tx = random.randrange(0, W - nw)
            ty = random.randrange(0, H - nh)
            cand_box = np.array([[tx, ty, tx + nw, ty + nh]], dtype=float)
            if (
                len(inst.bboxes) == 0
                or float(bbox_ioa(cand_box, inst.bboxes).max()) < 0.10
            ):
                img[ty : ty + nh, tx : tx + nw] = patch
                new_boxes.append([tx, ty, tx + nw, ty + nh])
                new_cls.append(float(donor_cls[j]))
                placed = True
                break
        if not placed:
            continue

    if new_boxes:
        if _STATS_DIR:
            _stats_cls.extend(int(c) for c in new_cls)
        # dtype 保持: 真实管道 instances.bboxes/cls 为 float32; 若用 dtype=float(float64)
        # vstack 会提升为 float64 -> torch.from_numpy 出 Double -> RT-DETR get_cdn_group
        # padding_bbox(Float) <- dn_bbox(Double) 崩溃 (2026-10-21 冒烟实测)
        bd = inst.bboxes.dtype
        inst.update(np.vstack([inst.bboxes, np.array(new_boxes, dtype=bd)]))
        main_cls = np.asarray(labels.get("cls", np.zeros((0, 1)))).reshape(-1, 1)
        cd = main_cls.dtype if main_cls.size else np.float32
        labels["cls"] = np.vstack(
            [main_cls, np.array(new_cls, dtype=cd).reshape(-1, 1)]
        )
        inst.normalize(W, H)
        inst.convert_bbox("xywh")
        labels["img"] = img
        if _DEBUG_DIR and _debug_count < _DEBUG_N:
            _save_debug(img, inst, labels["cls"])
    return labels


def flush_stats(dataset_nc: int = 18) -> None:
    """把粘贴类分布写到 CP_STATS_DIR/pasted_class_hist.json (诊断用)。"""
    if not _STATS_DIR or not _stats_cls:
        return
    from collections import Counter
    import json

    Path(_STATS_DIR).mkdir(parents=True, exist_ok=True)
    cnt = Counter(_stats_cls)
    with open(Path(_STATS_DIR) / "pasted_class_hist.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_pasted": len(_stats_cls),
                "hist": {str(k): v for k, v in sorted(cnt.items())},
            },
            f,
        )


def install_detection_copypaste() -> None:
    """Patch ultralytics CopyPaste so detection datasets get a working paste path.

    Segmentation datasets (labels with segments) keep the original behavior.
    Idempotent: safe to call multiple times.
    """
    global _ORIG_CALL
    from ultralytics.data.augment import CopyPaste

    if getattr(CopyPaste, "_detect_patched", False):
        return

    _ORIG_CALL = CopyPaste.__call__

    def __call__(self, labels: dict) -> dict:
        if self.p <= 0:
            return labels
        # segmentation path -> untouched original
        if len(labels["instances"].segments) > 0:
            return _ORIG_CALL(self, labels)
        # detection path -> our cross-image paste (probability gate same as base)
        if random.uniform(0, 1) > self.p:
            return labels
        return _paste_detection(self, labels)

    CopyPaste.__call__ = __call__
    CopyPaste._detect_patched = True
