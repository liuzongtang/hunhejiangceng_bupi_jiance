#!/usr/bin/env python
"""eval_multIou_yolo.py -- official-pipeline mAP at IoU 0.1/0.3/0.5 for YOLO detectors.

Handoff section 9 (S1) item 3 asks for a structural copy of ``scripts/eval_multIou.py``
with ``RTDETRValidator`` swapped for the ultralytics ``DetectionValidator`` and nothing
else changed. This file is that copy: it runs the unmodified ultralytics YOLO val
pipeline (same dataloader, same letterbox, same conf=0.001 / max_det=300, same official
``ap_per_class``) and only swaps the IoU matching vector from ``linspace(0.5, 0.95, 10)``
to ``[0.1, 0.3, 0.5]``, so the resulting mAP@0.1/0.3/0.5 are directly comparable with
both the official contest metric and ``scripts/eval_multIou.py`` output.

Difference 1 -- rect letterbox (behavioural, deliberate)
--------------------------------------------------------
``RTDETRValidator`` hard-codes square letterbox: ``models/rtdetr/val.py`` forces
``rect=False`` and silently ignores an externally supplied ``rect=True`` (that is
handoff section 11 F5). ``DetectionValidator`` does not. ``YOLO.val()`` in
``engine/model.py`` injects ``custom = {"rect": True}`` into the validator args, so the
validator keeps the rectangular letterbox the detector was trained with. This script
therefore never passes ``rect=...`` and lets the validator's own default stand.

Consequence for cross-script comparison: on 2.4:1 fabric images square 1280 spends
about 59% of its pixels on padding, rectangular letterbox does not. The effective pixel
scale of the two pipelines therefore differs, so only end-to-end mAP is comparable
between ``eval_multIou.py`` (RT-DETR, rect off) and this script (YOLO, rect on) -- never
raw detection counts, candidate-set sizes or P/R at a fixed conf.

Difference 2 -- sanity contract (behavioural)
---------------------------------------------
The RT-DETR script pins mAP@0.5 to one specific run's ``results.csv`` value. The YOLO
contract is the one handoff section 9 item 4 pre-registers: this script's mAP@0.5 must
reproduce *that run's* ``results.csv`` ``metrics/mAP50(B)`` to within +-0.004
(section 11 F6). Instead of a hard-coded constant, the CSV is auto-located next to the
checkpoint (``<run>/results.csv`` beside ``<run>/weights/best.pt``), the row with the
best ``fitness`` is selected, and the delta is printed with its PASS/FAIL against
+-0.004. If no CSV is found the contract is printed as a reminder only -- no number is
invented, and no gate is ever hard-coded.

APIs verified against the installed ultralytics 8.4.91
------------------------------------------------------
* ``DetectionValidator.__init__`` sets ``self.iouv = torch.linspace(0.5, 0.95, 10)`` and
  ``self.niou = self.iouv.numel()``, the same contract ``RTDETRValidator`` has, so the
  same two-line IoU patch applies unchanged.
* ``self.metrics`` is a ``DetMetrics`` (created in ``__init__``) whose ``.box`` is a
  ``Metric`` exposing ``all_ap`` / ``ap_class_index`` / ``p`` / ``r`` / ``f1``, and
  ``validator.metrics.nt_per_class`` is a full-length bincount indexed by class id,
  exactly as in the RT-DETR path.
* ``BaseValidator.__call__`` runs ``get_stats()`` (which calls ``DetMetrics.process()``,
  the call that actually fills ``metrics.box`` via ``ap_per_class``) and only then fires
  ``on_val_end`` -- so the ``_capture`` hook always sees populated arrays. ``get_stats``
  also calls ``clear_stats()``, but ``DetectionValidator.print_results`` does not
  re-process, so the captured arrays stay valid.
* ``add_callback("on_val_end", fn)`` is inherited from ``BaseValidator``; the callback
  receives the validator as its single argument.
* ``Metric.map75`` indexes ``all_ap[:, 5]`` and does raise ``IndexError`` on a 3-column
  ``all_ap``, but 8.4.91's ``Metric.mean_results()`` returns ``[mp, mr, map50, map]``
  and ``DetMetrics.summary()`` reports mAP50 / mAP50-95, so the guard is inert on this
  release. It is kept for parity with ``eval_multIou.py`` (whose target release did read
  it) and as insurance against a release that reads it again.
* ``YOLO.val()`` injects ``custom = {"rect": True}`` and
  ``DetectionValidator.build_dataset`` forwards ``args.rect`` to ``build_yolo_dataset``;
  ``RTDETRValidator.build_dataset`` instead hard-codes ``rect=False``.

Usage:
    python scripts/eval_multIou_yolo.py \
        --weights runs/yolo/yolo11l_mosaic1280/weights/best.pt \
        --data data/tianchi/data.yaml \
        --out runs/multi_iou/yolo11l_mosaic1280.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
from typing import Dict, Optional

import numpy as np
import torch

IOUS = (0.1, 0.3, 0.5)

# Pre-registered sanity tolerance for mAP@0.5 vs the run's results.csv mAP50
# (handoff section 9 item 4; section 11 F6 escalates a larger deviation to a
# pipeline-divergence stop). This is a tolerance, not a baseline target: no
# mAP value on this machine is hard-coded anywhere in this file.
SANITY_TOL = 0.004

# ---------------------------------------------------------------------------
# Patch 1: swap the IoU matching vector. Everything else in the official
# pipeline (dataloader, letterbox policy, conf=0.001, max_det=300,
# ap_per_class) stays untouched so numbers remain official-grade.
#
# Unlike the RT-DETR validator, DetectionValidator does not force square
# letterbox, and YOLO.val() injects rect=True. We deliberately do not touch
# that: the rect policy is left to the validator's own default (see module
# docstring, "Difference 1").
# ---------------------------------------------------------------------------
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import Metric

_orig_init = DetectionValidator.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.iouv = torch.tensor(IOUS)
    self.niou = (
        self.iouv.numel()
    )  # keep niou in sync: empty-pred fast path in update_metrics uses it


DetectionValidator.__init__ = _patched_init

# Patch 2: Metric.map75 indexes all_ap[:, 5], so calling it on a 3-column all_ap raises
# IndexError. Guarded here verbatim from eval_multIou.py for parity. Verified on
# 8.4.91: Metric.mean_results() returns [mp, mr, map50, map] and DetMetrics.summary()
# reports mAP50 / mAP50-95, so nothing on this release's val path actually reads map75
# and the guard is inert -- it is kept because the parent script needed it for the
# older 10-value summary() and because a release that reads map75 again would crash
# the run otherwise.
Metric.map75 = property(
    lambda self: (
        float(self.all_ap[:, 5].mean())
        if getattr(self.all_ap, "shape", (0, 0))[1] > 5
        else float("nan")
    )
    if len(self.all_ap)
    else 0.0
)

CAPTURE: dict = {}


def _capture(validator):
    m = validator.metrics.box
    CAPTURE.clear()
    CAPTURE.update(
        iouv=[float(x) for x in validator.iouv],
        all_ap=np.asarray(m.all_ap).copy(),  # (n_classes_present, len(IOUS))
        ap_class_index=list(np.asarray(m.ap_class_index)),
        p=np.asarray(m.p).copy(),
        r=np.asarray(m.r).copy(),
        f1=np.asarray(m.f1).copy(),
        nt_per_class=np.asarray(validator.metrics.nt_per_class).copy(),
    )


_orig_init_reg = DetectionValidator.__init__


def _patched_init_reg(self, *args, **kwargs):
    _orig_init_reg(self, *args, **kwargs)
    self.add_callback("on_val_end", _capture)


DetectionValidator.__init__ = _patched_init_reg


def _find_results_csv(weights: Path) -> Optional[Path]:
    """Locate the training ``results.csv`` that belongs to ``weights``.

    Args:
        weights: Path to the ``.pt`` checkpoint, conventionally
            ``<run>/weights/best.pt``.

    Returns:
        The matching ``results.csv`` path, or ``None`` when this run has no CSV
        next to it (e.g. an exported or copied checkpoint).
    """
    weights = weights.resolve()
    candidates = (
        weights.parent.parent / "results.csv",  # <run>/weights/best.pt -> <run>/results.csv
        weights.parent / "results.csv",  # checkpoint saved directly in <run>/
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _read_results_csv(csv_path: Path) -> Optional[Dict[str, object]]:
    """Read the best-fitness row of an ultralytics training ``results.csv``.

    Args:
        csv_path: Path to ``results.csv``.

    Returns:
        Dict with ``epoch``, ``map50``, ``map5095`` and ``path``, or ``None``
        when the expected metric columns are absent (an older or foreign CSV).
    """
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeDecodeError):
        return None
    if not rows:
        return None

    header = list(rows[0].keys())

    def pick(*needles: str) -> Optional[str]:
        for needle in needles:
            for col in header:
                if needle in col:
                    return col
        return None

    col_map50 = pick("mAP50(B)", "mAP50")
    col_map5095 = pick("mAP50-95(B)", "mAP50-95")
    col_fitness = pick("fitness")
    if col_map50 is None:
        return None

    def as_float(row: dict, col: Optional[str]) -> float:
        if col is None:
            return float("nan")
        try:
            return float(row[col])
        except (TypeError, ValueError):
            return float("nan")

    def key(row: dict) -> float:
        fit = as_float(row, col_fitness)
        if fit == fit:  # not NaN
            return fit
        secondary = as_float(row, col_map5095)
        return secondary if secondary == secondary else as_float(row, col_map50)

    best = max(rows, key=key)
    epoch_raw = best.get("epoch", "?")
    try:
        epoch: object = int(float(str(epoch_raw)))
    except (TypeError, ValueError):
        epoch = epoch_raw

    return {
        "epoch": epoch,
        "map50": as_float(best, col_map50),
        "map5095": as_float(best, col_map5095),
        "path": str(csv_path.resolve()),
    }


def main() -> int:
    """Run the YOLO val pipeline and report mAP at IoU 0.1/0.3/0.5.

    Returns:
        Process exit code: 0 on success, 2 when the ``on_val_end`` capture did
        not fire (the validator never reached the callback).
    """
    ap = argparse.ArgumentParser(description="Official-pipeline mAP@IoU 0.1/0.3/0.5")
    ap.add_argument("--weights", required=True, help="path to .pt checkpoint")
    ap.add_argument(
        "--data", default="data/tianchi/data.yaml", help="dataset yaml (val split)"
    )
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None, help="run name for runs/val artifacts")
    ap.add_argument("--out", default=None, help="where to write the JSON report")
    args = ap.parse_args()

    import ultralytics
    from ultralytics import YOLO

    weights_path = Path(args.weights)

    model = YOLO(args.weights)
    # No rect=... on purpose: YOLO.val() supplies rect=True itself and the
    # validator's own default must stand (module docstring, Difference 1).
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        save=False,
        name=args.name or f"multIou_{weights_path.parent.parent.name}",
    )

    if not CAPTURE:
        print("ERROR: on_val_end capture failed", file=sys.stderr)
        return 2

    all_ap = CAPTURE["all_ap"]  # (n_present, 3)
    idx = CAPTURE["ap_class_index"]
    nt = CAPTURE["nt_per_class"]
    names = model.names  # dict id -> name (from checkpoint)

    map_at = {f"mAP@{t}": float(all_ap[:, i].mean()) for i, t in enumerate(IOUS)}

    per_class = []
    for row, c in enumerate(idx):
        per_class.append(
            {
                "id": int(c),
                "name": names[int(c)],
                "n_inst": int(nt[c]),
                "AP@0.1": round(float(all_ap[row, 0]), 4),
                "AP@0.3": round(float(all_ap[row, 1]), 4),
                "AP@0.5": round(float(all_ap[row, 2]), 4),
                "P@maxF1": round(float(CAPTURE["p"][row]), 4),
                "R@maxF1": round(float(CAPTURE["r"][row]), 4),
            }
        )

    # Sanity contract: this run's own results.csv mAP50, +-0.004 (section 9
    # item 4 / section 11 F6). Never a hard-coded baseline.
    csv_path = _find_results_csv(weights_path)
    csv_stats = _read_results_csv(csv_path) if csv_path is not None else None

    report = {
        "weights": str(weights_path.resolve()),
        "data": str(Path(args.data).resolve()),
        "imgsz": args.imgsz,
        "ious": list(IOUS),
        "val_conf": 0.001,
        "max_det": 300,
        "n_classes_present": len(idx),
        "n_instances_total": int(nt.sum()),
        "mAP": {k: round(v, 4) for k, v in map_at.items()},
        "note": (
            "sanity: mAP@0.5 must reproduce this run's results.csv mAP50 within "
            f"+-{SANITY_TOL}; YOLO val keeps its own rect default (rect=True, not forced square)"
        ),
        "per_class": per_class,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        },
    }

    print("\n================ multi-IoU evaluation ================")
    print(f"weights : {report['weights']}")
    print(
        f"data    : {report['data']}  (imgsz={args.imgsz}, conf=0.001, max_det=300)"
    )
    print(
        f"instances: {report['n_instances_total']} across {report['n_classes_present']} classes"
    )
    for k, v in report["mAP"].items():
        print(f"  {k:10s} = {v:.4f}")
    print(
        f"  sanity  : mAP@0.5 must match this run's results.csv mAP50 (+-{SANITY_TOL})"
    )
    if csv_stats is not None:
        delta = map_at["mAP@0.5"] - float(csv_stats["map50"])
        verdict = "PASS" if abs(delta) <= SANITY_TOL else "FAIL"
        print(
            f"            results.csv (best fitness, epoch {csv_stats['epoch']}) "
            f"mAP50 = {float(csv_stats['map50']):.4f} | delta = {delta:+.4f} | "
            f"+-{SANITY_TOL} -> {verdict}"
        )
        print(f"            source: {csv_stats['path']}")
    else:
        print(
            "            no results.csv found next to the checkpoint; "
            "compare by hand before trusting AP@0.1/@0.3"
        )
    print("\n  class            n_inst   AP@0.1  AP@0.3  AP@0.5   P@F1   R@F1")
    for r in per_class:
        print(
            f"  {r['name']:<16s} {r['n_inst']:>6d}  {r['AP@0.1']:>6.3f} {r['AP@0.3']:>6.3f} "
            f"{r['AP@0.5']:>6.3f} {r['P@maxF1']:>6.3f} {r['R@maxF1']:>6.3f}"
        )
    crit = {"hole", "broken_warp", "broken_spandex"}
    print("\n  critical classes (hole/broken_warp/broken_spandex):")
    for r in per_class:
        if r["name"] in crit:
            print(
                f"    {r['name']:<16s} AP@0.1={r['AP@0.1']:.3f}  AP@0.3={r['AP@0.3']:.3f}  "
                f"AP@0.5={r['AP@0.5']:.3f}  R@maxF1={r['R@maxF1']:.3f}"
            )
    print("=====================================================\n")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report written to {out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
