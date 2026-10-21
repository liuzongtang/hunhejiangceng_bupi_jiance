"""
阶段1.1b: 1536高分辨率重训 (bf16 修复 fp16-AMP 的 BN 污染)

背景 (2026-10-16 定位):
  hi1536 v1 (amp=True/fp16) 在 epoch9 训练时 39 个 neck/encoder BN 层
  running_var 被清零 -> eval 模式除以 sqrt(eps) 而非 sqrt(var) 放大
  100-400x -> 激活爆炸 -> epoch10 起 mAP=0, 早停 e28, 最佳 e8
  mAP50=0.246 (基线 0.541)。根因: fp16 autocast 下 neck 激活 ~200-300,
  平方 4e4-9e4 逼近 fp16 上限 65504, cuDNN BN 方差统计溢出归零。
  bf16 指数范围同 fp32 (max 3.4e38), 无此问题; 基线 amp=False 同样健康。

本机坑 (实测):
  1. workers 必须=0 (WinError 1455 页面文件不足)
  2. project 必须传绝对路径 (路径含中文括号被 save_dir 截断)
  3. MuSGD (optimizer="auto") + AdamW 会塌陷, 勿换
  4. ultralytics check_amp 用 atol=0.5 对比 fp32/bf16 输出会误判禁用 AMP,
     必须同时补丁 check_amp (本脚本已处理)
  5. fp32@1536 显存贴顶 24.2GB 触发 WDDM 抖动 9.9s/步, 不可用
"""

from pathlib import Path
import sys

import torch
import ultralytics.engine.trainer as TR

# ---- bf16 补丁: autocast 改 bf16 + 绕过 check_amp 误判 ----
import ultralytics.utils.torch_utils as TU


def _bf16_autocast(enabled=True, device="cuda"):
    return torch.amp.autocast(device, dtype=torch.bfloat16, enabled=bool(enabled))


TU.autocast = _bf16_autocast
TR.autocast = _bf16_autocast
TR.check_amp = lambda model: True  # bf16 vs fp32 atol=0.5 会误判, 手动放行

from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parent.parent
PROJECT = str(ROOT / "runs" / "rtdetr")
INIT = str(ROOT / "runs" / "rtdetr" / "tianchi18_full" / "weights" / "best.pt")
MARKER = ROOT / "logs" / "BN_POISONED.marker"
HEALTH = ROOT / "logs" / "bn_health_v2.log"
RESUME = len(sys.argv) > 1 and sys.argv[1] == "resume"


def bn_watchdog(trainer):
    """每 epoch 末检查 BN running_var 是否被清零 (fp16-AMP 污染特征)。
    同时查原始模型与 EMA; 发现即写标记并抛异常中止。"""
    zeroed_all = []
    for mdl in (trainer.model, getattr(trainer.ema, "ema", None)):
        if mdl is None:
            continue
        sd = mdl.state_dict()
        zeroed = [
            k for k, v in sd.items() if "running_var" in k and v.abs().max().item() == 0
        ]
        zeroed_all += zeroed
    with open(HEALTH, "a") as f:
        f.write(f"epoch {trainer.epoch}: running_var归零层数={len(zeroed_all)}\n")
    if zeroed_all:
        MARKER.write_text(
            f"epoch {trainer.epoch}: {len(zeroed_all)}层归零 {zeroed_all[:5]}\n"
        )
        raise RuntimeError(f"BN污染: {len(zeroed_all)}层 running_var 归零, 中止训练")


model = RTDETR(INIT)
model.add_callback("on_train_epoch_end", bn_watchdog)

model.train(
    data=str(ROOT / "data" / "tianchi" / "data.yaml"),
    project=PROJECT,
    name="tianchi18_hi1536_v2",
    imgsz=1536,
    epochs=40,
    patience=15,
    batch=4,
    workers=0,
    device=0,
    rect=True,
    mosaic=0.0,
    optimizer="auto",  # auto -> MuSGD
    amp=True,  # 已被补丁改为 bf16
    seed=42,
    lr0=0.0002,
    lrf=0.01,
    resume=RESUME,
    exist_ok=True,
)
print("training complete")
