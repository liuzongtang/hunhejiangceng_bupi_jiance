"""
阶段1.2a: 1280+mosaic 对照臂 (分离"分辨率"与"mosaic"两因子)

目的 (2026-10-17):
  v4 (1536+mosaic, COCO从头训) 官方口径 mean mAP=0.6094, 超基线
  (1280, 无mosaic) 0.5719 达 +0.0375。但提升由两个变量共同驱动,
  需本对照臂分离: 1280+mosaic vs v4(1536+mosaic) vs 基线(1280无mosaic)。
  若 1280+mosaic ≈ v4 (0.61) => mosaic 是主贡献, 分辨率边际小;
  若 1280+mosaic ≈ 基线 (0.57) => 分辨率是主贡献。

配方与 v4 完全一致, 仅 imgsz 1536->1280 (单变量):
  COCO init, rect, bf16, MuSGD(auto), lr0=0.0005, lrf=0.01,
  mosaic=1.0 (e50关), 60ep, patience=20, seed=42, batch=4。

本机坑 (实测, 勿再踩):
  1. workers=0 (WinError 1455)
  2. project 绝对路径
  3. MuSGD 勿换 AdamW (塌陷)
  4. bf16 需同时补丁 autocast + check_amp (否则静默跑 fp32)
  5. 用户桌面3D软件抢GPU: 20min/epoch -> 160min/epoch
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
HEALTH = ROOT / "logs" / "bn_health_m1280.log"
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
    name="tianchi18_mosaic1280",
    imgsz=1280,
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
