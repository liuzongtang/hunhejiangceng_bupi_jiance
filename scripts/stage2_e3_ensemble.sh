#!/bin/bash
# Stage 2B-2 (E3): deep-ensemble uncertainty as an alarm gate.
# Chain runs AFTER the UW seed-7 chain completes (GPU serialization iron rule:
# never run inference concurrently with a training chain -- a SIGTERM'd
# inference process with in-flight CUDA kernels can kill the training job).
#
# Two ensembles, both same-start (mosaic1280 best.pt) different-seed members:
#   A) ctl3  : strong_r2_uniform(s42,5ep) + l1_seed123_ctl(s123,10ep)
#              + l1_seed2024_ctl(s2024,10ep)  -- pre-registered in plan 2B-2
#              (protocol caveat: member epochs 5/10/10, noted in the report)
#   B) uw4   : uw_seed42/123/2024/7           -- protocol-matched (all 5ep,
#              only variable = seed + UW weighting)
# Question: does disagreement (std_conf) gating beat score thresholding
# (mean_conf) at fixed TP recall -- i.e. is the Bayesian route cheaper in FPR?
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
LOG=logs/stage2_e3_ensemble.log

echo "=== E3 ensemble chain start $(date) ===" >> "$LOG"

$PY scripts/eval_ensemble_uncertainty.py \
  --weights runs/ablation_strong/strong_r2_uniform/weights/best.pt \
            runs/ablation_strong/l1_seed123_ctl/weights/best.pt \
            runs/ablation_strong/l1_seed2024_ctl/weights/best.pt \
  --out runs/calibration/ensemble_ctl3.json >> "$LOG" 2>&1
echo "=== ctl3 exit=$? $(date) ===" >> "$LOG"

$PY scripts/eval_ensemble_uncertainty.py \
  --weights runs/ablation_adaptive/uw_seed42/weights/best.pt \
            runs/ablation_adaptive/uw_seed123/weights/best.pt \
            runs/ablation_adaptive/uw_seed2024/weights/best.pt \
            runs/ablation_adaptive/uw_seed7/weights/best.pt \
  --out runs/calibration/ensemble_uw4.json >> "$LOG" 2>&1
echo "=== uw4 exit=$? $(date) ===" >> "$LOG"
echo "=== E3_ALL_DONE $(date) ===" >> "$LOG"
