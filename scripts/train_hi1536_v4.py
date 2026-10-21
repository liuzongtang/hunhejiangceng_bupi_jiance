"""
阶段1.1d: 1536高分辨率从头训练 v4 (微调路线二次失败后的转向)

失败链 (2026-10-16):
  v1: 从COCO, fp16-AMP -> epoch9 BN污染(39层running_var清零) -> 崩
  v2: 从收敛基线0.541微调, lr0=0.0002 -> e1-e5 mAP 0.46->0.41 三连降 -> 中止
  v3: 同v2但 lr0=0.0001+warmup5 -> e1-e6 mAP 0.50->0.40 同样签名 -> 中止
  几何诊断: 基线best.pt直接1536rect推理只掉0.015 (0.572->0.557官方mean)
  => "已收敛权重微调上新几何"在本机配置下稳定劣化, 与lr大小无关;
     唯一健康轨迹是 v1 的从COCO从头训 (e1=0.150->e8=0.246 单调升).

v4 策略 (执行用户阶段1原计划: 高分辨率+mosaic+更长训练):
  从 rtdetr-l.pt (COCO) 从头训, 1536 rect b4 bf16 MuSGD lr0=0.0005,
  mosaic=1.0 (基线1280是mosaic=0, 此为新增杠杆), 60ep patience=20.
  判据: e10 时 mAP50 应 >=0.35 且单调升 (v1同期0.30); 若再次劣化,
  则1536路线整体证伪, 回1280做mosaic/HP (plan_stage1.3_rerun.md 依赖树).

本机坑 (实测):
  1. workers=0 (WinError 1455)
  2. project 绝对路径
  3. MuSGD 勿换 AdamW
  4. bf16 需同时补丁 autocast + check_amp
  5. 用户桌面3D软件(Bambu Studio/ZBrush)抢GPU: 19min/epoch -> 160min/epoch
  6. BN哨兵每epoch查running_var
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
INIT = str(ROOT / "rtdetr-l.pt")
MARKER = ROOT / "logs" / "BN_POISONED.marker"
HEALTH = ROOT / "logs" / "bn_health_v4.log"
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
    name="tianchi18_hi1536_v4",
    imgsz=1536,
    epochs=60,
    patience=20,
    batch=4,
    workers=0,
    device=0,
    rect=True,
    mosaic=1.0,
    optimizer="auto",
    amp=True,  # 已补丁为 bf16
    seed=42,
    lr0=0.0005,
    lrf=0.01,
    resume=RESUME,
    exist_ok=True,
)
print("training complete")
