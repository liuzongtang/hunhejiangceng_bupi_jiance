"""
阶段1.1c: 1536高分辨率重训 v3 (v2 失败后的保守策略)

v2 失败分析 (2026-10-16):
  从收敛基线(0.541)微调上 1536 rect, lr0=0.0002 + MuSGD(momentum 0.937):
  - epoch1 mAP 即从 0.541 跌至 0.457 (warmup 期就大跌)
  - e1-e5: 0.457->0.460->0.431->0.414->0.409 三连降, 损失平台 0.64
    (基线收敛值 0.552, 高 16%), val 损失同步上升
  - BN 全程 0 层污染 (bf16 修复经 5 epoch 验证有效)
  两个疑因: (a) lr 对"已收敛模型+新 letterbox 几何"过大;
  (b) 基线 BN running stats 适配方形1280分布, rect-1536 激活分布
      不同, 统计量失配 (momentum 0.03 适配慢).

v3 策略: 10x 降 lr + 延长 warmup, 让模型留在基线盆地里渐进适应.
  若 epoch10 仍无 mAP>0.52 的回升, 判定"微调上分辨率"路线失败,
  转 mosaic 轮或 HP 扫描 (按 docs/plan_stage1.3_rerun.md 依赖树).

本机坑 (实测, 勿再踩):
  1. workers 必须=0 (WinError 1455)
  2. project 必须绝对路径
  3. MuSGD 勿换 AdamW (塌陷)
  4. bf16 必须同时补丁 autocast + check_amp (否则静默跑 fp32)
  5. 用户桌面程序 (Bambu Studio/ToDesk/Edge) 抢 GPU 可使每轮从
     19min 恶化到 112min, util 2%; 训练前确认 GPU 空闲
  6. BN 哨兵每 epoch 检查 running_var 归零
"""

from pathlib import Path
import sys

import torch
import ultralytics.engine.trainer as TR
import ultralytics.utils.torch_utils as TU


def _bf16_autocast(enabled=True, device="cuda"):
    return torch.amp.autocast(device, dtype=torch.bfloat16, enabled=bool(enabled))


TU.autocast = _bf16_autocast
TR.autocast = _bf16_autocast
TR.check_amp = lambda model: True

from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parent.parent
PROJECT = str(ROOT / "runs" / "rtdetr")
INIT = str(ROOT / "runs" / "rtdetr" / "tianchi18_full" / "weights" / "best.pt")
MARKER = ROOT / "logs" / "BN_POISONED.marker"
HEALTH = ROOT / "logs" / "bn_health_v3.log"
RESUME = len(sys.argv) > 1 and sys.argv[1] == "resume"


def bn_watchdog(trainer):
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
    name="tianchi18_hi1536_v3",
    imgsz=1536,
    epochs=30,
    patience=10,
    batch=4,
    workers=0,
    device=0,
    rect=True,
    mosaic=0.0,
    optimizer="auto",
    amp=True,  # 已补丁为 bf16
    seed=42,
    lr0=0.0001,
    lrf=0.01,
    warmup_epochs=5.0,
    resume=RESUME,
    exist_ok=True,
)
print("training complete")
