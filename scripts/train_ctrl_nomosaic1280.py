"""
阶段1.2b: 1280+rect 无mosaic 对照臂 (解耦 mosaic 与 rect 两因子)

目的 (2026-10-18):
  因子分离现状:
    - 分辨率因子已干净分离: v4(1536+mosaic) vs mosaic1280(1280+mosaic),
      仅 imgsz 不同 -> 1280 全面占优 (官方口径@1280: 0.6349 vs 0.6077;
      @1536: 0.6106 vs 0.6094), 且每轮快 25%。
    - 但相对旧基线 (tianchi18_full: 1280方形无mosaic, 0.5719) 的 +0.063 增益中,
      mosaic 与 rect 两因子捆绑变更, 未解耦。
  本臂 = mosaic1280 配方单变量 mosaic=0 (rect 保持 True),
  若本臂 ≈ 基线 0.57 => mosaic 是主增益; 若本臂 ≈ 0.63 => rect 是主增益。

配方与 train_mosaic1280.py 逐行一致, 仅 mosaic=0.0:
  COCO init, rect, bf16, MuSGD(auto), lr0=0.0005, lrf=0.01,
  60ep, patience=20, seed=42, batch=4, workers=0。

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
HEALTH = ROOT / "logs" / "bn_health_nomosaic1280.log"
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
    name="tianchi18_nomosaic1280",
    imgsz=1280,
    epochs=60,
    patience=20,
    batch=4,
    workers=0,
    device=0,
    rect=True,
    mosaic=0.0,
    optimizer="auto",
    amp=True,  # 已补丁为 bf16
    seed=42,
    lr0=0.0005,
    lrf=0.01,
    resume=RESUME,
    exist_ok=True,
)
print("training complete")
