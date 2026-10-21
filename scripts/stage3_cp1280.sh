#!/usr/bin/env bash
# 阶段3 Copy-Paste 臂: 60ep 从头训 + 官方口径评测 (链, GPU 顺序执行)
# 判定门 (实验前登记于 docs/plan_stage3_breakthrough.md §5, 事后不移动):
#   转正: mean > 0.6349 + 0.005 = 0.6399 且 crit_R 不降
#   零和: mean <= 0.6399 但 crit_R 升 >0.02 且 crit_P 降 >0.02 (数据层零和)
#   阴性: 其余
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
LOG=logs/stage3_cp1280.log

echo "=== cp1280 train start $(date) ===" >> "$LOG"
$PY scripts/train_copypaste.py >> "$LOG" 2>&1
echo "=== cp1280 train exit=$? $(date) ===" >> "$LOG"

$PY scripts/eval_multIou.py \
  --weights runs/rtdetr/tianchi18_cp1280/weights/best.pt \
  --name tianchi18_cp1280_official \
  --out runs/multi_iou/tianchi18_cp1280.json >> "$LOG" 2>&1
echo "=== cp1280 eval exit=$? $(date) ===" >> "$LOG"

echo "=== CP1280_ALL_DONE $(date) ===" >> "$LOG"
