#!/bin/bash
# Layer-1 奖励臂多种子重复实验 (阶段1.3 收尾)
# 目的: 臂4 (seed 42) 出现唯一"零和被打破"信号 (mean +0.0007 / crit +0.053)
#       但未过 +0.005 线 —— 用 seed 123/2024 重复, 判"真信号 vs 噪声"。
# 配置与臂4逐项一致: mosaic1280 best.pt 起点, beta=0.5 +misses, 10ep,
#   batch 4, lr0=1e-4, amp=False, 唯一变量=seed (42 -> 123/2024)。
# 每种子 A/B: 干预 (beta=0.5) vs 同种子均匀对照 (beta=0)。
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
STRONG=runs/rtdetr/tianchi18_mosaic1280/weights/best.pt
PROJ=runs/ablation_strong
LOG=logs/l1_seeds.log

echo "=== l1 seeds chain start $(date) strong=$STRONG ===" >> "$LOG"

for SEED in 123 2024; do
  echo "=== seed $SEED reward start $(date) ===" >> "$LOG"
  $PY scripts/train_rtdetr_reward.py --weights "$STRONG" \
    --beta 0.5 --include-misses --epochs 10 --batch 4 --seed "$SEED" \
    --project "$PROJ" --name "l1_seed${SEED}_reward" >> "$LOG" 2>&1
  echo "=== seed $SEED reward exit=$? $(date) ===" >> "$LOG"

  echo "=== seed $SEED control start $(date) ===" >> "$LOG"
  $PY scripts/train_rtdetr_reward.py --weights "$STRONG" \
    --beta 0 --epochs 10 --batch 4 --seed "$SEED" \
    --project "$PROJ" --name "l1_seed${SEED}_ctl" >> "$LOG" 2>&1
  echo "=== seed $SEED control exit=$? $(date) ===" >> "$LOG"

  $PY scripts/eval_multIou.py --weights "$PROJ/l1_seed${SEED}_reward/weights/best.pt" \
    --name "l1_seed${SEED}_reward" --out "runs/multi_iou/l1_seed${SEED}_reward.json" >> "$LOG" 2>&1
  $PY scripts/eval_multIou.py --weights "$PROJ/l1_seed${SEED}_ctl/weights/best.pt" \
    --name "l1_seed${SEED}_ctl" --out "runs/multi_iou/l1_seed${SEED}_ctl.json" >> "$LOG" 2>&1
  echo "=== seed $SEED evals exit=$? $(date) ===" >> "$LOG"

  $PY scripts/l1_seed_aggregate.py >> "$LOG" 2>&1
done

echo "=== L1_SEEDS_ALL_DONE $(date) ===" >> "$LOG"
