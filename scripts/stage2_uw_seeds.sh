#!/bin/bash
# Stage 2A: Uncertainty Weighting 3-seed chain (docs/plan_stage2_adaptive.md)
# 目的: 检验"自动学习 cls/reg 损失权重"能否打破七负消融的召回-精度零和。
# 协议与强基线对照 strong_r2_uniform (官方口径 mean 0.6158) 逐项一致:
#   mosaic1280 best.pt 起点, 5ep, batch 4, imgsz 1280 (方形), lr0 1e-4,
#   amp=False (fp16 BN 污染教训), mosaic 0.0, deterministic, workers 0。
#   唯一变量 = UW 损失加权 (uw_lr 1e-3, lv 截断 [-3,+3], EMA 平滑 + 尺度保持归一化)。
#   2026-10-20 修复两个实现 bug 后重启: ① w=exp(+lv) 符号错误(正反馈 runaway);
#   ② 更新项误用归一化权重(双 lv 钉住截断→精确 1:1 退化为对照)。修复版更新项用
#   未归一化 wu=exp(-lv)(Kendall 稳态 wu=1/L), 应用权重尺度保持(总损失=默认)。
#   全量数据诊断验证: 900 步 w=(1.91,0.73) 单调收敛向稳态比 L_reg/L_cls≈3.2, 无钉住。
# 预承诺判定 (见计划书 1.4): 转正 = 种子平均 mean > 0.6208 且 crit_R 不降;
#    crit_R 升>0.02 且 crit_P 降>0.02 判零和复现, 不计转正。
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
STRONG=runs/rtdetr/tianchi18_mosaic1280/weights/best.pt
PROJ=runs/ablation_adaptive
LOG=logs/stage2_uw.log

echo "=== stage2 UW chain start $(date) strong=$STRONG ===" >> "$LOG"

for SEED in 42 123 2024; do
  echo "=== seed $SEED uw train start $(date) ===" >> "$LOG"
  $PY scripts/train_adaptive_loss.py --method uw --seed "$SEED" \
    --weights "$STRONG" --epochs 5 --batch 4 --lr0 0.0001 \
    --name "uw_seed${SEED}" --project "$PROJ" >> "$LOG" 2>&1
  echo "=== seed $SEED uw train exit=$? $(date) ===" >> "$LOG"

  $PY scripts/eval_multIou.py --weights "$PROJ/uw_seed${SEED}/weights/best.pt" \
    --name "uw_seed${SEED}" --out "runs/multi_iou/uw_seed${SEED}.json" >> "$LOG" 2>&1
  echo "=== seed $SEED uw eval exit=$? $(date) ===" >> "$LOG"
done

echo "=== STAGE2_UW_ALL_DONE $(date) ===" >> "$LOG"
