#!/bin/bash
# 阶段1.3 臂2-4 + 条件GRPO + 无mosaic对照臂 编排链
# 用户授权全自动推进 (2026-10-18 "跑完之后继续按照计划进行, 不用问我")
# 协议: docs/plan_stage1.3_rerun.md (判定规则预先承诺)
# 依赖: R2臂 (scripts/stage13_arm1_r2.sh, 由 stage13_orchestrator.sh 于 01:47 启动)
#   共享均匀对照 = R2臂的 strong_r2_uniform (1280方形 amp=False lr0=1e-4 5ep seed=42)
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
LOG=logs/stage13_arms_234.log
STRONG=runs/rtdetr/tianchi18_mosaic1280/weights/best.pt
PROJ=runs/ablation_strong
CTL=runs/multi_iou/strong_r2_uniform.json
ANY_POS=0   # 任一臂转正(含部分转正)则跳过GRPO

echo "=== arms234 start $(date) ===" >> "$LOG"

# ---------- 0. 等 R2 臂完成 ----------
for i in $(seq 1 300); do
  grep -aq "ARM1_R2_ALL_DONE" logs/stage13_arm1_r2.log 2>/dev/null && break
  sleep 300
done
if ! grep -aq "ARM1_R2_ALL_DONE" logs/stage13_arm1_r2.log; then
  echo "FATAL: R2 arm did not finish $(date)" >> "$LOG"; exit 1
fi
echo "=== R2 arm done $(date) ===" >> "$LOG"
if [ -f runs/multi_iou/strong_r2_severity.json ]; then
  $PY scripts/ablation_verdict.py "$CTL" runs/multi_iou/strong_r2_severity.json >> "$LOG" 2>&1
  [ $? -ne 1 ] && ANY_POS=1
else
  echo "WARN: strong_r2_severity.json missing, R2 verdict skipped" >> "$LOG"
fi

# ---------- 1. 臂2 Direction B 分类严重度加权 positive-only (5ep) ----------
echo "=== arm2 Direction B positive-only start $(date) ===" >> "$LOG"
$PY scripts/train_rtdetr_costsensitive.py --weights "$STRONG" \
  --preset severity --weight-mode positive --epochs 5 \
  --project "$PROJ" --name strong_b_pos >> "$LOG" 2>&1
echo "=== arm2 train exit=$? $(date) ===" >> "$LOG"
$PY scripts/eval_multIou.py --weights "$PROJ/strong_b_pos/weights/best.pt" \
  --name strong_b_pos_official --out runs/multi_iou/strong_b_pos.json >> "$LOG" 2>&1
echo "=== arm2 eval exit=$? $(date) ===" >> "$LOG"
$PY scripts/ablation_verdict.py "$CTL" runs/multi_iou/strong_b_pos.json >> "$LOG" 2>&1
[ $? -ne 1 ] && ANY_POS=1

# ---------- 2. 臂3 R3 NWD 损失几何 (5ep) ----------
echo "=== arm3 R3 NWD start $(date) ===" >> "$LOG"
$PY scripts/train_rtdetr_nwd.py --weights "$STRONG" \
  --preset nwd --epochs 5 \
  --project "$PROJ" --name strong_r3_nwd >> "$LOG" 2>&1
echo "=== arm3 train exit=$? $(date) ===" >> "$LOG"
$PY scripts/eval_multIou.py --weights "$PROJ/strong_r3_nwd/weights/best.pt" \
  --name strong_r3_nwd_official --out runs/multi_iou/strong_r3_nwd.json >> "$LOG" 2>&1
echo "=== arm3 eval exit=$? $(date) ===" >> "$LOG"
$PY scripts/ablation_verdict.py "$CTL" runs/multi_iou/strong_r3_nwd.json >> "$LOG" 2>&1
[ $? -ne 1 ] && ANY_POS=1

# ---------- 3. 臂4 Layer-1 可微奖励 β=0.5 + include-misses (10ep) ----------
#    自带 beta=0 对照 (10ep, 同脚本同配置, 与干预等长——公平A/B)
echo "=== arm4 Layer-1 reward start $(date) ===" >> "$LOG"
$PY scripts/train_rtdetr_reward.py --weights "$STRONG" \
  --beta 0.5 --include-misses --epochs 10 --batch 4 \
  --project "$PROJ" --name strong_l1_reward >> "$LOG" 2>&1
echo "=== arm4 train exit=$? $(date) ===" >> "$LOG"
$PY scripts/train_rtdetr_reward.py --weights "$STRONG" \
  --beta 0 --epochs 10 --batch 4 \
  --project "$PROJ" --name strong_l1_ctl >> "$LOG" 2>&1
echo "=== arm4 ctl exit=$? $(date) ===" >> "$LOG"
$PY scripts/eval_multIou.py --weights "$PROJ/strong_l1_reward/weights/best.pt" \
  --name strong_l1_reward_official --out runs/multi_iou/strong_l1_reward.json >> "$LOG" 2>&1
$PY scripts/eval_multIou.py --weights "$PROJ/strong_l1_ctl/weights/best.pt" \
  --name strong_l1_ctl_official --out runs/multi_iou/strong_l1_ctl.json >> "$LOG" 2>&1
echo "=== arm4 evals exit=$? $(date) ===" >> "$LOG"
$PY scripts/ablation_verdict.py runs/multi_iou/strong_l1_ctl.json runs/multi_iou/strong_l1_reward.json >> "$LOG" 2>&1
[ $? -ne 1 ] && ANY_POS=1

# ---------- 4. 条件 GRPO: 四臂全负才跑 ----------
if [ "$ANY_POS" -eq 0 ]; then
  echo "=== ALL FOUR NEGATIVE -> GRPO arm $(date) ===" >> "$LOG"
  bash scripts/stage13_arm5_grpo.sh "$STRONG" >> "$LOG" 2>&1
  echo "=== grpo arm exit=$? $(date) ===" >> "$LOG"
else
  echo "=== at least one arm positive, GRPO skipped $(date) ===" >> "$LOG"
fi

# ---------- 5. 因子分离解耦: 1280+rect 无mosaic 对照臂 (60ep, ~14h) ----------
echo "=== nomosaic control arm start $(date) ===" >> "$LOG"
rm -f logs/BN_POISONED.marker logs/bn_health_nomosaic1280.log
NLOG=logs/nomosaic1280_train.log
for i in 1 2 3; do
  echo "=== attempt $i start $(date) ===" >> "$NLOG"
  if [ "$i" -eq 1 ]; then
    $PY scripts/train_ctrl_nomosaic1280.py >> "$NLOG" 2>&1
  else
    $PY scripts/train_ctrl_nomosaic1280.py resume >> "$NLOG" 2>&1
  fi
  code=$?
  echo "=== attempt $i exit=$code $(date) ===" >> "$NLOG"
  if [ -f logs/BN_POISONED.marker ]; then echo "BN_POISONED_ABORT" >> "$LOG"; break; fi
  grep -aq "training complete" "$NLOG" && break
  sleep 10
done
$PY scripts/eval_multIou.py --weights runs/rtdetr/tianchi18_nomosaic1280/weights/best.pt \
  --name nomosaic1280_official --out runs/multi_iou/nomosaic1280_official.json >> "$LOG" 2>&1
echo "=== nomosaic eval exit=$? $(date) ===" >> "$LOG"

echo "=== ARMS234_ALL_DONE $(date) ===" >> "$LOG"
