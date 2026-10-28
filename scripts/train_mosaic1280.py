"""
闃舵1.2a: 1280+mosaic 瀵圭収鑷?(鍒嗙"鍒嗚鲸鐜?涓?mosaic"涓ゅ洜瀛?

鐩殑 (2026-10-17):
  v4 (1536+mosaic, COCO浠庡ご璁? 瀹樻柟鍙ｅ緞 mean mAP=0.6094, 瓒呭熀绾?
  (1280, 鏃爉osaic) 0.5719 杈?+0.0375銆備絾鎻愬崌鐢变袱涓彉閲忓叡鍚岄┍鍔?
  闇€鏈鐓ц噦鍒嗙: 1280+mosaic vs v4(1536+mosaic) vs 鍩虹嚎(1280鏃爉osaic)銆?
  鑻?1280+mosaic 鈮?v4 (0.61) => mosaic 鏄富璐＄尞, 鍒嗚鲸鐜囪竟闄呭皬;
  鑻?1280+mosaic 鈮?鍩虹嚎 (0.57) => 鍒嗚鲸鐜囨槸涓昏础鐚€?

閰嶆柟涓?v4 瀹屽叏涓€鑷? 浠?imgsz 1536->1280 (鍗曞彉閲?:
  COCO init, rect, bf16, MuSGD(auto), lr0=0.0005, lrf=0.01,
  mosaic=1.0 (e50鍏?, 60ep, patience=20, seed=42, batch=4銆?

鏈満鍧?(瀹炴祴, 鍕垮啀韪?:
  1. workers=0 (WinError 1455)
  2. project 缁濆璺緞
  3. MuSGD 鍕挎崲 AdamW (濉岄櫡)
  4. bf16 闇€鍚屾椂琛ヤ竵 autocast + check_amp (鍚﹀垯闈欓粯璺?fp32)
  5. 鐢ㄦ埛妗岄潰3D杞欢鎶PU: 20min/epoch -> 160min/epoch
  6. BN鍝ㄥ叺姣廵poch鏌unning_var
"""

from pathlib import Path
import sys
import os

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
        f.write(f"epoch {trainer.epoch}: running_var褰掗浂灞傛暟={len(zeroed_all)}\n")
    if zeroed_all:
        MARKER.write_text(
            f"epoch {trainer.epoch}: {len(zeroed_all)}灞傚綊闆?{zeroed_all[:5]}\n"
        )
        raise RuntimeError(f"BN姹℃煋: {len(zeroed_all)}灞?running_var 褰掗浂, 涓璁粌")


def main() -> int:
    """Run the recipe.

    The ``if __name__ == "__main__"`` guard at the bottom is load-bearing: with
    ``workers > 0`` Windows uses the *spawn* start method, so every dataloader
    worker re-imports this module. Without the guard each worker re-entered
    ``model.train()`` and Python aborted with "An attempt has been made to start
    a new process before the current process has finished its bootstrapping
    phase". That is why this project historically pinned ``workers=0``; the
    guard removes the need for that (and is why the smoke arm, which already had
    one, ran fine at workers=4).
    """
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
        # 2026-10-21: workers 0 -> TRAIN_WORKERS (default 4). On this machine
        # workers=0 measured 89.1 s/epoch vs 58.9 s/epoch at workers=4 on the
        # same 5% subset (GPU util ~70% mean, dipping to ~13% while the loader
        # starves the GPU). WinError 1455 did not reproduce with 48 GB free RAM.
        # workers is a throughput knob only -- seed, lr, optimizer, mosaic, rect
        # and batch are untouched, so the recipe stays comparable to the
        # historical 0.6349 run.
        workers=int(os.environ.get("TRAIN_WORKERS", "4")),
        device=0,
        rect=True,
        mosaic=1.0,
        optimizer="auto",
        amp=True,  # 宸茶ˉ涓佷负 bf16
        seed=42,
        lr0=0.0005,
        lrf=0.01,
        resume=RESUME,
        exist_ok=True,
    )
    print("training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

