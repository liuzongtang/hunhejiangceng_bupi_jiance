"""Stage 1.1-C — High-resolution training run (1536 rect, no mosaic, workers=0).

Single intended change vs baseline tianchi18_full (imgsz=1280 square letterbox,
60 epochs, batch 4, lr 5e-4, optimizer auto->MuSGD, workers=0, amp=False,
seed 42):
  - content resolution 1280x523 -> 1536x628 (~1.2x linear) via rect=True
    (rectangular batches keep the native 2446x1000 aspect 2.45; a square 1536
    letterbox would waste 59% of tokens on gray padding: 3.1 s/it vs ~1.3 s/it)
  - amp False -> True for speed (~3x); verified healthy at e1 or falls back
Mosaic deliberately stays OFF in this run (same augmentation as the baseline):
  with workers=0, mosaic's 4x image decode becomes the serial bottleneck.
  Resolution is validated FIRST; mosaic is added in a follow-up run once the
  gain is confirmed. (Note: ultralytics also force-disables mosaic when rect=True.)

Machine-specific pitfalls (hit during 2026-10-15 setup, all worked around here):
  1. workers>=2 -> DataLoader workers die with WinError 1455 (page file/commit
     exhausted: each spawned worker re-loads torch CUDA DLLs). Only workers=0
     is stable on this machine under current memory pressure (desktop apps
     hold ~15GB VRAM / significant RAM). The 18.5h baseline ran with workers=0.
  2. MuSGD optimizer -> "Couldn't open shared file mapping" (same 1455). Use AdamW.
  3. WDDM VRAM pressure -> EMA deepcopy OOM ("22GB free but 2MB allocation fails").
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True must be set BEFORE torch
     is imported (i.e. at module top, before the ultralytics import).
  4. Relative project= path gets mangled by the parentheses in "维度模型(1)"
     (save_dir becomes Desktop\\(1)\\...). Always pass ABSOLUTE project paths.
  5. Always log to a FILE, not a pipe: a crashed main process leaves orphaned
     workers holding the pipe open, which hides the traceback.
  6. amp/optimizer pitfall (2026-10-15): amp=True + AdamW collapsed TWICE
     (square and rect 1536): cls loss low (0.9-1.1 vs baseline 9.6) but
     giou/l1 stuck near max (2.3-2.4 vs baseline 1.2->0.8) and val mAP = 0 at
     e1 (baseline 0.126) — box branch never establishes query-GT matching.
     Confound: baseline = MuSGD + amp=False, both collapsed runs = AdamW +
     amp=True. Protocol: try MuSGD + amp=True first (fast ~28min/ep); if e1
     collapses -> amp guilty -> amp=False fp32 (~97min/ep at 1536, batch 4,
     WDDM spill; consider warm-start from baseline best.pt to cut epochs).
     Always diff args.yaml against the baseline before launching.
  7. rect=True is essential for speed: native aspect is 2.45, square 1536
     letterbox wastes 59% of tokens on gray padding (3.1 s/it fp32 vs ~1.3).
     Content resolution is IDENTICAL (1536x628) so the experiment variable
     (resolution) is unchanged. ultralytics force-disables mosaic when rect.
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

from ultralytics import RTDETR

EPOCHS = 60  # equal-epoch comparison vs the 60-epoch 1280 baseline (resolution is the only intended variable)
WORKERS = int(
    os.environ.get("HI1536_WORKERS", "0")
)  # 0 = stable; set 4 only if desktop apps are closed (frees commit/VRAM)
AMP = (
    os.environ.get("HI1536_AMP", "true").lower() == "true"
)  # default ON for speed; set HI1536_AMP=false for the baseline-matching no-amp run
PROJECT = str(Path(__file__).resolve().parent.parent / "runs" / "rtdetr")  # absolute
RESUME = len(sys.argv) > 1 and sys.argv[1] == "resume"


def main():
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data="data/tianchi/data.yaml",
        project=PROJECT,
        name="tianchi18_hi1536",
        imgsz=1536,
        epochs=EPOCHS,
        patience=20,
        batch=4,
        workers=WORKERS,
        device=0,
        rect=True,  # rectangular batches: content resolution identical to square-1536 (1536x628) but 2.45x fewer tokens (native 2446x1000 aspect 2.45 -> square letterbox wastes 59% on gray padding: 3.1s/it -> ~1.3s/it)
        mosaic=0.0,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        lr0=0.0005,
        lrf=0.01,
        optimizer="auto",  # MuSGD (RT-DETR default; crashes with 1455 under memory pressure — retry on the now-clean system; AdamW fallback if it crashes again)
        amp=AMP,  # see pitfall 6/7 below; AMP env switch for the speed test
        seed=42,  # match baseline
        save_period=1,  # match baseline (per-epoch snapshot)
        val=True,
        plots=True,
        exist_ok=True,
        resume=RESUME,
    )
    print("training complete")


if __name__ == "__main__":
    main()
