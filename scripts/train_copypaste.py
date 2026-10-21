"""
阶段3: 1280+rect+mosaic+Copy-Paste 臂 (数据层正交信息, 计划书 docs/plan_stage3_breakthrough.md)

目的 (2026-10-21):
  阶段1-2 五类干预 (静态奖惩/推理侧重加权/单调重校准/集成歧义闸门/自适应加权)
  全部无法稳定突破召回-精度零和 -> 瓶颈在特征级判别结构 (论文5.5边界声明)。
  突破必要条件 = 引入与既有分数正交的信息源。已证明有效的"便宜因子"路线
  (rect +0.036 / mosaic +0.027) 属数据/几何层, 本臂将其延伸到最后一个
  未测数据层因子: 类感知 Copy-Paste (长尾类实例跨图粘贴, 1/类频次加权)。

  判定门 (实验前登记于 plan_stage3_breakthrough.md §5, 事后不移动):
    转正: 官方口径 mean > 0.6349 + 0.005 = 0.6399 且 crit_R 不降
    零和: mean <= 0.6399 但 crit_R 升 >0.02 且 crit_P 降 >0.02 (数据层零和, 更强机理证据)
    阴性: 其余

配方与 scripts/train_mosaic1280.py 逐行一致, 唯一变量 = copy_paste=0.5:
  COCO init, rect, bf16, MuSGD(auto), lr0=0.0005, lrf=0.01,
  60ep, patience=20, seed=42, batch=4, workers=0。

本机坑 (实测, 勿再踩):
  1. workers=0 (WinError 1455)
  2. project 绝对路径
  3. MuSGD 勿换 AdamW (塌陷)
  4. bf16 需同时补丁 autocast + check_amp (否则静默跑 fp32 或被 check_amp 禁用)
  5. 用户桌面3D软件抢GPU: 20min/epoch -> 160min/epoch
  6. BN哨兵每epoch查running_var
  7. ultralytics 原生 copy_paste 对检测任务静默无效 (segments 为空直接 return),
     必须 install_detection_copypaste() (backend/training/copypaste_det.py)

冒烟模式: python scripts/train_copypaste.py smoke   (3ep + 可视化抽检)
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import ultralytics.engine.trainer as TR
import ultralytics.utils.torch_utils as TU


def _bf16_autocast(enabled=True, device="cuda"):
    return torch.amp.autocast(device, dtype=torch.bfloat16, enabled=bool(enabled))


TU.autocast = _bf16_autocast
TR.autocast = _bf16_autocast
TR.check_amp = lambda model: True

# Copy-Paste 检测版补丁 (必须在 RTDETR 构建 dataloader 前安装)
from backend.training.copypaste_det import install_detection_copypaste

install_detection_copypaste()

from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parent.parent
PROJECT = str(ROOT / "runs" / "rtdetr")
INIT = str(ROOT / "rtdetr-l.pt")
MARKER = ROOT / "logs" / "BN_POISONED.marker"
HEALTH = ROOT / "logs" / "bn_health_cp1280.log"
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
RESUME = len(sys.argv) > 1 and sys.argv[1] == "resume"
FRACTION = 0.05 if SMOKE else 1.0  # 冒烟只跑 5% 数据 (~100步/ep), 仅验证管道


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
    name="tianchi18_smoke_cp" if SMOKE else "tianchi18_cp1280",
    imgsz=1280,
    epochs=3 if SMOKE else 60,
    patience=20,
    batch=4,
    workers=0,
    device=0,
    rect=True,
    mosaic=1.0,
    optimizer="auto",
    copy_paste=0.5,
    copy_paste_mode="flip",
    fraction=FRACTION,
    amp=True,  # 已补丁为 bf16
    seed=42,
    lr0=0.0005,
    lrf=0.01,
    resume=RESUME,
    exist_ok=True,
)
print("training complete")
