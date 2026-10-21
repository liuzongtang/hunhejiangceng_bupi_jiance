#!/bin/bash
# Seed 42 补跑（原 seed 42 被并行 GPU 进程的 SIGTERM 事故杀死，见 memory 2026-10-20）。
# 在 stage2_uw_seeds.sh 主链（seed 123/2024）完成后单独顺序执行，之后接 E3 集成评估。
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
STRONG=runs/rtdetr/tianchi18_mosaic1280/weights/best.pt
PROJ=runs/ablation_adaptive
LOG=logs/stage2_uw.log

echo "=== seed 42 uw RERUN start $(date) ===" >> "$LOG"
$PY scripts/train_adaptive_loss.py --method uw --seed 42 \
  --weights "$STRONG" --epochs 5 --batch 4 --lr0 0.0001 \
  --name "uw_seed42" --project "$PROJ" >> "$LOG" 2>&1
echo "=== seed 42 uw train exit=$? $(date) ===" >> "$LOG"

$PY scripts/eval_multIou.py --weights "$PROJ/uw_seed42/weights/best.pt" \
  --name "uw_seed42" --out "runs/multi_iou/uw_seed42.json" >> "$LOG" 2>&1
echo "=== seed 42 uw eval exit=$? $(date) ===" >> "$LOG"

# E3: 3 成员集成不确定性（既有均匀对照，混合协议标注）
$PY scripts/eval_ensemble_uncertainty.py \
  --weights "$PROJ/../ablation_strong/strong_r2_uniform/weights/best.pt" \
            "$PROJ/../ablation_strong/l1_seed123_ctl/weights/best.pt" \
            "$PROJ/../ablation_strong/l1_seed2024_ctl/weights/best.pt" \
  --out "runs/calibration/ensemble_ctl3.json" >> "$LOG" 2>&1
echo "=== E3 ensemble_ctl3 exit=$? $(date) ===" >> "$LOG"

echo "=== STAGE2_UW_SEED42_RERUN_DONE $(date) ===" >> "$LOG"
