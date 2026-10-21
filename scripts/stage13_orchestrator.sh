#!/bin/bash
# 阶段1.3 总编排: 等对照臂训练+评测完成 -> 选最强权重 -> 自动启动 R2 臂
# 用户授权全自动推进 (2026-10-18 "跑完之后继续按照计划进行, 不用问我")
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/master/anaconda3/python.exe
LOG=logs/stage13_orchestrator.log

echo "=== orchestrator start $(date) ===" >> "$LOG"

# 1. 等对照臂训练完成标志
for i in $(seq 1 120); do
  if grep -aq "training complete" logs/mosaic1280_train.log 2>/dev/null; then
    echo "training done $(date)" >> "$LOG"; break
  fi
  sleep 300
done

# 2. 等 aUrqMT 自动评测链产出 (mosaic1280 双口径 JSON)
for i in $(seq 1 60); do
  if [ -f runs/multi_iou/mosaic1280_official.json ] && [ -f runs/multi_iou/mosaic1280_at1536.json ]; then
    echo "evals done $(date)" >> "$LOG"; break
  fi
  sleep 120
done

# 3. 选最强权重 (同口径@1280 letterbox 比较 mean)
WEIGHTS=$($PY - << 'EOF'
import json, sys
def mean_of(p):
    try:
        d = json.load(open(p))
        m = d.get("mAP", {})
        if isinstance(m, dict):
            vals = list(m.values())
            return sum(vals)/len(vals) if vals else 0
        return 0
    except Exception:
        return 0
m4 = mean_of("runs/multi_iou/hi1536_v4_official.json")   # v4 @1280
mm = mean_of("runs/multi_iou/mosaic1280_official.json")  # 对照臂 @1280
print(f"v4@1280 mean={m4:.4f}  mosaic1280@1280 mean={mm:.4f}", file=sys.stderr)
if mm > m4:
    print("runs/rtdetr/tianchi18_mosaic1280/weights/best.pt")
else:
    print("runs/rtdetr/tianchi18_hi1536_v4/weights/best.pt")
EOF
)
echo "选定最强权重: $WEIGHTS" >> "$LOG"

# 4. 启动 R2 臂 (severity + uniform 对照 + 官方口径双臂评测)
bash scripts/stage13_arm1_r2.sh "$WEIGHTS" >> "$LOG" 2>&1
echo "=== orchestrator arm1 done $(date) ===" >> "$LOG"
