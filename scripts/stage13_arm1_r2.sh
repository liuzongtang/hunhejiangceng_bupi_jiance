#!/bin/bash
# 阶段1.3 第一臂: R2 box 严重度加权 (强基线重测)
# 用法: bash scripts/stage13_arm1_r2.sh <对照权重路径>
# 对照权重 = 当前最强模型 (v4 best.pt 或 mosaic1280 best.pt, 由官方口径评定)
# 口径与历史七负实验完全一致: 1280方形, amp=False, MuSGD默认, lr0=1e-4, 5ep, seed=42
# 唯一变量 = 起点权重 (弱基线0.541 -> 强基线)
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
WEIGHTS="${1:?用法: stage13_arm1_r2.sh <对照权重路径>}"
PROJ=runs/ablation_strong
LOG=logs/stage13_arm1_r2.log

echo "=== stage13 arm1 R2 start $(date) weights=$WEIGHTS ===" >> "$LOG"

# 干预臂: severity box 加权 (critical 2.0x / secondary 1.5x)
$PY scripts/train_rtdetr_boxweighted.py \
  --weights "$WEIGHTS" --preset severity --epochs 5 \
  --project "$PROJ" --name strong_r2_severity >> "$LOG" 2>&1
echo "=== severity exit=$? $(date) ===" >> "$LOG"

# 对照臂: uniform (同配置同seed, 唯一差异=干预项)
$PY scripts/train_rtdetr_boxweighted.py \
  --weights "$WEIGHTS" --preset none --epochs 5 \
  --project "$PROJ" --name strong_r2_uniform >> "$LOG" 2>&1
echo "=== uniform exit=$? $(date) ===" >> "$LOG"

# 官方口径双臂评测
$PY scripts/eval_multIou.py --weights "$PROJ/strong_r2_severity/weights/best.pt" \
  --name strong_r2_severity_official --out runs/multi_iou/strong_r2_severity.json >> "$LOG" 2>&1
echo "=== eval severity exit=$? $(date) ===" >> "$LOG"
$PY scripts/eval_multIou.py --weights "$PROJ/strong_r2_uniform/weights/best.pt" \
  --name strong_r2_uniform_official --out runs/multi_iou/strong_r2_uniform.json >> "$LOG" 2>&1
echo "=== eval uniform exit=$? $(date) ===" >> "$LOG"

echo "=== ARM1_R2_ALL_DONE $(date) ===" >> "$LOG"
