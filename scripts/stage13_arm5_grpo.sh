#!/bin/bash
# 阶段1.3 臂5: Layer-3 GRPO (修 std_reward 地板后, 强基线重测)
# 仅当臂1-4全负时由 stage13_arms_234.sh 调用
#
# 修复内容 (协议 §5 "先修 advantage-eps 地板"):
#   --advantage-eps 1e-2 (历史默认) -> 1e-4
#   证据 runs/rtdetr-grpo/noise_diag/noise_flatness.csv: 奖励面在策略实际
#   采样噪声 (sigma~0.01-0.1) 下 std_reward 仅 2e-3~1.7e-2, 历史训练实测
#   std_reward=1e-5~2e-3, 全部低于 1e-2 地板 -> advantage 被压小 10~1000 倍
#   -> REINFORCE 失效, log_std 塌至 -4.78, 均值梯度劣化模型。
#   地板降到 1e-4 后, std~1e-3 的组 advantage 恢复 O(1) 量级 (真信号),
#   std<1.1e-4 的近退化组仍被压制 (防纯噪声爆炸), clip ±5 兜底。
#
# 其余配置与历史七负 GRPO 臂逐项一致 (保证"唯一变量=起点权重+地板修复"):
#   beta_reward=0.5, beta_kl=0.01, num-groups=4, batch=8, lr=1e-4,
#   lr_log_std=0.1, log_std_init=-2.3, ref_log_std=-4.6, 1280, seed=42
# epoch: 历史3ep -> 5ep (与其余臂/共享对照等长, 公平A/B)
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
WEIGHTS="${1:?用法: stage13_arm5_grpo.sh <对照权重路径>}"
PROJ=runs/ablation_strong
LOG=logs/stage13_arm5_grpo.log

echo "=== stage13 arm5 GRPO start $(date) weights=$WEIGHTS ===" >> "$LOG"

$PY scripts/train_rtdetr_grpo.py --weights "$WEIGHTS" \
  --epochs 5 --batch 8 --lr 1e-4 \
  --beta-reward 0.5 --beta-kl 0.01 --num-groups 4 \
  --lr-log-std 0.1 --log-std-init -2.3 --ref-log-std -4.6 \
  --advantage-eps 1e-4 --advantage-clip 5.0 \
  --project "$PROJ" --name strong_l3_grpo >> "$LOG" 2>&1
echo "=== grpo train exit=$? $(date) ===" >> "$LOG"

$PY scripts/eval_multIou.py --weights "$PROJ/strong_l3_grpo/weights/best.pt" \
  --name strong_l3_grpo_official --out runs/multi_iou/strong_l3_grpo.json >> "$LOG" 2>&1
echo "=== grpo eval exit=$? $(date) ===" >> "$LOG"

# 判定 vs 共享均匀对照 (5ep)
$PY scripts/ablation_verdict.py runs/multi_iou/strong_r2_uniform.json \
  runs/multi_iou/strong_l3_grpo.json >> "$LOG" 2>&1

echo "=== ARM5_GRPO_ALL_DONE $(date) ===" >> "$LOG"
