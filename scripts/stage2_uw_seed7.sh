#!/bin/bash
# Stage 2A-D3: UW 4th seed (verdict UNSTABLE at 1/3 after the zero-sum clause).
# Identical protocol to the 3-seed chain (docs/plan_stage2_adaptive.md):
#   mosaic1280 best.pt start, 5ep, batch 4, imgsz 1280 (square), lr0 1e-4,
#   amp=False, mosaic 0.0, deterministic. Only variable = UW loss weighting.
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
STRONG=runs/rtdetr/tianchi18_mosaic1280/weights/best.pt
PROJ=runs/ablation_adaptive
LOG=logs/stage2_uw_seed7.log

echo "=== seed 7 uw train start $(date) ===" >> "$LOG"
$PY scripts/train_adaptive_loss.py --method uw --seed 7 \
  --weights "$STRONG" --epochs 5 --batch 4 --lr0 0.0001 \
  --name "uw_seed7" --project "$PROJ" >> "$LOG" 2>&1
echo "=== seed 7 uw train exit=$? $(date) ===" >> "$LOG"

$PY scripts/eval_multIou.py --weights "$PROJ/uw_seed7/weights/best.pt" \
  --name "uw_seed7" --out "runs/multi_iou/uw_seed7.json" >> "$LOG" 2>&1
echo "=== seed 7 uw eval exit=$? $(date) ===" >> "$LOG"
echo "=== SEED7_ALL_DONE $(date) ===" >> "$LOG"
