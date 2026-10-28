"""
B1: family detector probe - swap an 18-class RT-DETR head to 3 families and fine-tune.

Hypothesis (handoff section 6)
    The 3 defect families (A_linear / B_blotch / C_point, bootstrap support 1.00) are
    an "easier sub-problem" than the 18-class task: their PR curve sits further from
    the optimum, so the family-level detector may have headroom that the 18-class
    model no longer has. A dedicated 3-class detector is the cheapest architecture
    change that tests this.

Pre-registered criterion (handoff section 6)
    Family-level AP@0.5 of the dedicated family detector is >= 0.02 above the
    "18 classes aggregated to families" pipeline -> the family sub-problem has
    headroom -> start B2; otherwise stage B is closed and the compute goes to C/M.

Failure branch (handoff section 6)
    If the dedicated detector does not beat the aggregated 18-class pipeline by
    +0.02 AP@0.5, the family-level sub-problem has no extra headroom on this
    representation: close stage B, do not start B2.

Protocol (from-scratch recipe of ``train_mosaic1280.py``)
    ``--init`` 18-class checkpoint | 20 epochs | batch 4 | imgsz 1280 | rect True |
    optimizer "auto" (MuSGD; never AdamW) | lr0 5e-4 | lrf 0.01 | mosaic 1.0
    (closed at ``--close-mosaic``) | workers 0 | seed 42 | bf16 patch + BN watchdog.

Head swap (verified against the installed ultralytics 8.4.91 source)
    The RT-DETR head ``ultralytics.nn.modules.head.RTDETRDecoder`` carries exactly
    three class-count-dependent modules (15 tensors): ``enc_score_head``
    (nn.Linear hd -> nc), ``dec_score_head`` (ModuleList of ndl nn.Linear hd -> nc)
    and ``denoising_class_embed`` (nn.Embedding nc -> hd). The box heads
    (``enc_bbox_head`` / ``dec_bbox_head``) and every backbone/transformer tensor
    are nc-independent. This script therefore rebuilds the three class modules for
    ``--nc``, copies the first ``min(nc_old, nc_new)`` rows (and the bias) from the
    checkpoint, re-initialises any genuinely new rows from the copied-row mean,
    and updates ``decoder.nc`` / ``model.nc`` / ``model.names`` / ``model.yaml``.
    Box heads stay byte-identical. A ``on_train_start`` guard then proves that the
    trainer's rebuilt decoder really got the swapped weights (Ultralytics rebuilds
    the model inside ``_setup_train``; a mismatch would silently train a random
    head at 1280 for 20 epochs).

    If the installed decoder does not expose that layout, the script falls back to
    option (b): build a fresh ``RTDETRDetectionModel`` from the checkpoint's yaml
    with the new ``nc`` and partially load the state dict (``strict=False``),
    printing the exact matched/skipped tensors either way.

    Note: copying the first ``min(nc_old, nc_new)`` rows follows the B1 spec; the
    18-class head is per-class sigmoid so there is no semantic class mapping from
    18 defects to 3 families. The copied rows are a warm start only - the head is
    retrained for 20 epochs.

Usage:
    # The handoff command
    python scripts/train_family_detector.py \
        --init runs/rtdetr/tianchi18_mosaic1280/weights/best.pt \
        --data data/tianchi_family3/data.yaml --epochs 20 --name b1_family3_s42

    # Verify the head swap without a checkpoint or dataset (CPU only)
    python scripts/train_family_detector.py --check-swap

    # Then the two evaluation protocols (handoff section 6 step 3):
    #   a) 3-family mAP on the official pipeline (nc=3)
    #   b) 18-class predictions aggregated to families via families.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# The bf16 patch must be applied before the trainer/RTDETR classes are imported
# (canonical order of scripts/train_mosaic1280.py; handoff section 14 pitfall 3).
import torch
import torch.nn as nn
import ultralytics.engine.trainer as TR
import ultralytics.utils.torch_utils as TU


def _bf16_autocast(enabled=True, device="cuda"):
    """bf16 autocast replacement for Ultralytics' fp16 autocast."""
    return torch.amp.autocast(device, dtype=torch.bfloat16, enabled=bool(enabled))


TU.autocast = _bf16_autocast
TR.autocast = _bf16_autocast
TR.check_amp = lambda model: True

from ultralytics import RTDETR
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.tasks import RTDETRDetectionModel

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT = str(ROOT / "runs" / "rtdetr-family")
MARKER = ROOT / "logs" / "BN_POISONED.marker"
HEALTH = ROOT / "logs" / "bn_health_family3.log"

#: Class-count-dependent tensor names of RTDETRDecoder in ultralytics 8.4.91.
HEAD_TENSOR_ATTRS = ("enc_score_head", "dec_score_head", "denoising_class_embed")


def _resize_linear(old: nn.Linear, out_features: int) -> nn.Linear:
    """
    Build a new ``nn.Linear`` with ``out_features`` rows, copying the overlap.

    The first ``min(old.out_features, out_features)`` weight/bias rows are copied
    verbatim (option (a) of the B1 spec); any genuinely new rows are initialised
    from the mean of the copied rows so they start in the same activation range.

    Args:
        old: Checkpoint classification head.
        out_features: Target number of classes.

    Returns:
        The resized linear layer on the same device/dtype as ``old``.
    """
    new = nn.Linear(
        old.in_features,
        out_features,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    n_copy = min(old.out_features, out_features)
    with torch.no_grad():
        new.weight[:n_copy].copy_(old.weight[:n_copy])
        if old.bias is not None and new.bias is not None:
            new.bias[:n_copy].copy_(old.bias[:n_copy])
        if out_features > n_copy:
            mean_w = old.weight[:n_copy].mean(dim=0, keepdim=True)
            new.weight[n_copy:].copy_(mean_w.expand(out_features - n_copy, -1))
            if old.bias is not None and new.bias is not None:
                mean_b = old.bias[:n_copy].mean()
                new.bias[n_copy:].copy_(mean_b.expand(out_features - n_copy))
    return new


def _resize_embedding(old: nn.Embedding, num_embeddings: int) -> nn.Embedding:
    """
    Build a new ``nn.Embedding`` with ``num_embeddings`` rows, copying the overlap.

    Args:
        old: Checkpoint denoising class embedding (``nc`` rows in 8.4.91).
        num_embeddings: Target number of classes.

    Returns:
        The resized embedding on the same device/dtype as ``old``.
    """
    new = nn.Embedding(
        num_embeddings,
        old.embedding_dim,
        padding_idx=None,
        max_norm=old.max_norm,
        norm_type=old.norm_type,
        scale_grad_by_freq=old.scale_grad_by_freq,
        sparse=old.sparse,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    n_copy = min(old.num_embeddings, num_embeddings)
    with torch.no_grad():
        new.weight[:n_copy].copy_(old.weight[:n_copy])
        if num_embeddings > n_copy:
            mean_w = old.weight[:n_copy].mean(dim=0, keepdim=True)
            new.weight[n_copy:].copy_(mean_w.expand(num_embeddings - n_copy, -1))
    return new


def _swap_via_yaml(
    model: RTDETRDetectionModel, nc_new: int, names: Dict[int, str]
) -> Tuple[RTDETRDetectionModel, dict]:
    """
    Fallback head swap: rebuild from the checkpoint yaml and partially load.

    Used only when the installed decoder does not expose the verified 8.4.91
    layout (``enc_score_head`` / ``dec_score_head`` / ``denoising_class_embed``).

    Args:
        model: Source (18-class) model.
        nc_new: Target class count.
        names: Class-id -> name mapping for the new head.

    Returns:
        Tuple ``(new_model, report)``; ``report`` carries matched/skipped counts.
    """
    print(
        "[family] WARNING: RTDETRDecoder layout not recognised; falling back to "
        "option (b): yaml rebuild + partial state_dict load (strict=False)"
    )
    cfg = dict(model.yaml)
    cfg["nc"] = nc_new
    cfg.setdefault("channels", 3)
    new_model = RTDETRDetectionModel(cfg=cfg, nc=nc_new, verbose=False)
    src = model.state_dict()
    dst = new_model.state_dict()
    matched = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    skipped = sorted(k for k in src if k not in matched)
    new_model.load_state_dict(matched, strict=False)
    new_model.nc = nc_new
    new_model.names = dict(names)
    new_model.yaml["nc"] = nc_new
    report = {
        "mechanism": "b_yaml_partial_load",
        "nc_old": int(getattr(model.model[-1], "nc", -1)),
        "nc_new": nc_new,
        "matched": len(matched),
        "skipped": len(skipped),
        "skipped_keys": skipped,
        "resized_keys": skipped,
    }
    return new_model, report


def swap_class_head(
    model: RTDETRDetectionModel,
    nc_new: int = 3,
    names: Optional[Dict[int, str]] = None,
) -> Tuple[RTDETRDetectionModel, dict]:
    """
    Swap the RT-DETR class head from ``nc_old`` to ``nc_new`` classes.

    Option (a) of the B1 spec, verified against ultralytics 8.4.91: rebuild
    ``enc_score_head``, every ``dec_score_head[i]`` and ``denoising_class_embed``
    for the new class count, copying the first ``min(nc_old, nc_new)`` rows and
    keeping the box heads intact. Falls back to option (b) if the decoder does not
    expose that layout.

    Args:
        model: Loaded 18-class ``RTDETRDetectionModel`` (``RTDETR(init).model``).
        nc_new: Target number of classes.
        names: Optional class-id -> name mapping (defaults to ``{i: str(i)}``).

    Returns:
        Tuple ``(model, report)``; ``model`` is the same object for option (a) and
        a freshly built model for option (b). ``report`` holds the mechanism and
        exact matched/skipped tensor counts.
    """
    if names is None:
        names = {i: str(i) for i in range(nc_new)}
    if len(names) != nc_new:
        raise ValueError(f"got {len(names)} names for nc={nc_new}")

    decoder = model.model[-1]
    if not isinstance(decoder, RTDETRDecoder) or not all(
        hasattr(decoder, attr) for attr in HEAD_TENSOR_ATTRS
    ):
        return _swap_via_yaml(model, nc_new, names)

    nc_old = int(decoder.nc)
    if nc_old == nc_new:
        print(f"[family] head swap skipped: nc already {nc_new}")
        decoder.nc = nc_new
        model.names = dict(names)
        return model, {
            "mechanism": "a_in_place",
            "nc_old": nc_old,
            "nc_new": nc_new,
            "matched": len(model.state_dict()),
            "skipped": 0,
            "resized_keys": [],
        }

    resized: List[str] = []
    with torch.no_grad():
        decoder.enc_score_head = _resize_linear(decoder.enc_score_head, nc_new)
        resized += ["enc_score_head.weight", "enc_score_head.bias"]

        for i, head in enumerate(decoder.dec_score_head):
            was_bias = head.bias is not None
            decoder.dec_score_head[i] = _resize_linear(head, nc_new)
            resized.append(f"dec_score_head.{i}.weight")
            if was_bias:
                resized.append(f"dec_score_head.{i}.bias")

        if hasattr(decoder, "denoising_class_embed"):
            old_embed = decoder.denoising_class_embed
            decoder.denoising_class_embed = _resize_embedding(old_embed, nc_new)
            resized.append("denoising_class_embed.weight")

    decoder.nc = nc_new
    model.nc = nc_new
    model.names = dict(names)
    model.yaml["nc"] = nc_new
    try:  # model.args is a dict on 8.4.x checkpoints; keep it consistent if present.
        model.args["nc"] = nc_new
    except (AttributeError, TypeError):
        pass

    total = len(model.state_dict())
    report = {
        "mechanism": "a_in_place",
        "decoder": type(decoder).__name__,
        "nc_old": nc_old,
        "nc_new": nc_new,
        "matched": total - len(resized),
        "skipped": len(resized),
        "resized_keys": resized,
    }
    return model, report


def _print_report(report: dict) -> None:
    """Print the head-swap mechanism and exact matched/skipped tensor counts."""
    print(
        f"[family] head swap ({report['mechanism']}) on "
        f"{report.get('decoder', 'RTDETRDetectionModel')}: "
        f"nc {report['nc_old']} -> {report['nc_new']}"
    )
    print(
        f"[family] matched tensors (carried over verbatim): {report['matched']} | "
        f"skipped/resized tensors: {report['skipped']}"
    )
    keys = report["resized_keys"]
    if keys:
        print(f"[family] resized keys ({len(keys)}): {', '.join(keys)}")
    if report["mechanism"] != "a_in_place":
        print(f"[family] skipped keys: {', '.join(keys[:20])}"
              + (" ..." if len(keys) > 20 else ""))


def _check_swap(nc_old: int, nc_new: int, names: Dict[int, str]) -> int:
    """
    Verify the head swap on a synthetic random model (no checkpoint, CPU only).

    Builds an ``nc_old``-class ``RTDETRDetectionModel`` from the bundled
    ``rtdetr-l.yaml``, swaps to ``nc_new``, and asserts that (1) every non-head
    tensor is untouched, (2) the copied rows equal the source rows, (3) the
    decoder emits ``nc_new`` class scores, and (4) the model runs a CPU forward.

    Args:
        nc_old: Synthetic source class count.
        nc_new: Target class count.
        names: Class-id -> name mapping.

    Returns:
        Process exit code (0 on success).
    """
    print(f"[family] --check-swap: building synthetic {nc_old}-class rtdetr-l on CPU")
    model = RTDETRDetectionModel("rtdetr-l.yaml", nc=nc_old, verbose=False)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    decoder = model.model[-1]
    old_enc = decoder.enc_score_head.weight.detach().clone()
    old_dec = [h.weight.detach().clone() for h in decoder.dec_score_head]

    swapped, report = swap_class_head(model, nc_new, names)
    _print_report(report)
    if report["mechanism"] != "a_in_place":
        print("[family] --check-swap: fallback path exercised; verifying shapes only")
    after = swapped.state_dict()

    # (1) every non-head tensor untouched
    changed = [
        k
        for k, v in before.items()
        if not any(k.endswith(rk) for rk in report["resized_keys"])
        and k in after
        and not torch.equal(v, after[k])
    ]
    if changed:
        raise RuntimeError(f"non-head tensors changed by swap: {changed[:5]}")
    # (2) copied rows / biases preserved
    n_copy = min(nc_old, nc_new)
    dec = swapped.model[-1]
    assert torch.equal(dec.enc_score_head.weight[:n_copy], old_enc[:n_copy]), (
        "enc_score_head rows not copied"
    )
    for i, head in enumerate(dec.dec_score_head):
        assert torch.equal(head.weight[:n_copy], old_dec[i][:n_copy]), (
            f"dec_score_head.{i} rows not copied"
        )
    assert dec.enc_score_head.out_features == nc_new
    assert all(h.out_features == nc_new for h in dec.dec_score_head)
    assert dec.nc == nc_new and swapped.nc == nc_new
    # (3) inference output carries nc_new class scores
    swapped.eval()
    with torch.no_grad():
        out = swapped(torch.zeros(1, 3, 320, 320))
    pred = out[0] if isinstance(out, (tuple, list)) else out
    print(
        f"[family] --check-swap OK: nc {nc_old} -> {nc_new}, "
        f"matched={report['matched']} skipped={report['skipped']}, "
        f"forward out={tuple(pred.shape)}"
    )
    if pred.shape[-1] != 6:
        raise RuntimeError(f"unexpected inference output shape {tuple(pred.shape)}")
    return 0


def _check_data_nc(data_path: str, nc: int) -> None:
    """Fail fast if an existing data.yaml declares a different ``nc``."""
    path = Path(data_path)
    if not path.exists():
        print(f"[family] data.yaml not found ({path}); expectations are checked at train start")
        return
    import yaml

    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    data_nc = cfg.get("nc")
    if data_nc is not None and int(data_nc) != int(nc):
        raise SystemExit(
            f"[family] data.yaml nc={data_nc} but --nc={nc}. The trainer rebuilds the "
            "decoder with data nc, which would silently drop the swapped head."
        )
    print(f"[family] data.yaml nc={data_nc} matches --nc={nc}")


def _bn_watchdog(trainer) -> None:
    """Abort training if bf16/fp16 poisons BN (running_var collapses to zero)."""
    zeroed_all: List[str] = []
    for mdl in (trainer.model, getattr(trainer.ema, "ema", None)):
        if mdl is None:
            continue
        sd = mdl.state_dict()
        zeroed_all += [
            k for k, v in sd.items() if "running_var" in k and v.abs().max().item() == 0
        ]
    HEALTH.parent.mkdir(parents=True, exist_ok=True)
    with HEALTH.open("a", encoding="utf-8") as fh:
        fh.write(f"epoch {trainer.epoch}: running_var zeroed layers={len(zeroed_all)}\n")
    if zeroed_all:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text(
            f"epoch {trainer.epoch}: {len(zeroed_all)} layers zeroed {zeroed_all[:5]}\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"BN poisoned: {len(zeroed_all)} running_var tensors collapsed to zero; aborting"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init",
        default="runs/rtdetr/tianchi18_mosaic1280/weights/best.pt",
        help="18-class RT-DETR checkpoint to start from (must exist to train)",
    )
    parser.add_argument(
        "--data",
        default="data/tianchi_family3/data.yaml",
        help="Family dataset yaml (nc=3; build with scripts/build_family_dataset.py)",
    )
    parser.add_argument("--nc", type=int, default=3, help="Target number of family classes")
    parser.add_argument(
        "--names",
        default="A_linear,B_blotch,C_point",
        help="Comma-separated class names for the new head",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--lr0", type=float, default=0.0005, help="From-scratch recipe LR (MuSGD + auto)"
    )
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction")
    parser.add_argument(
        "--optimizer",
        default="auto",
        help="From-scratch recipe uses MuSGD via optimizer='auto'; never switch to AdamW",
    )
    parser.add_argument(
        "--amp",
        type=lambda s: str(s).lower() in ("true", "1", "yes", "on"),
        default=True,
        help="Use the bf16 patch (autocast + check_amp); on by default per the recipe",
    )
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument(
        "--rect",
        type=lambda s: str(s).lower() in ("true", "1", "yes", "on"),
        default=True,
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Output project dir")
    parser.add_argument("--name", default="b1_family3_s42", help="Run name")
    parser.add_argument(
        "--check-swap",
        action="store_true",
        help="CPU-only synthetic verification of the head swap; no checkpoint/data needed",
    )
    parser.add_argument(
        "--check-nc-old",
        type=int,
        default=18,
        help="Source class count for --check-swap's synthetic model",
    )
    args = parser.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if len(names) != args.nc:
        parser.error(f"--names has {len(names)} entries but --nc={args.nc}")
    name_map = {i: n for i, n in enumerate(names)}

    if args.check_swap:
        return _check_swap(args.check_nc_old, args.nc, name_map)

    args.project = os.path.abspath(args.project)
    _check_data_nc(args.data, args.nc)

    if not Path(args.init).exists():
        print(
            f"[family] ERROR: --init checkpoint not found: {args.init}\n"
            "[family] This is expected on a machine without the trained 18-class best.pt; "
            "pass --init <path> once it exists. Nothing was trained."
        )
        return 2

    model = RTDETR(args.init)
    swapped, report = swap_class_head(model.model, args.nc, name_map)
    model.model = swapped
    # Ultralytics' Model.model setter copies the module's `nc` into
    # model.overrides; `nc` is not a valid train() argument, so get_cfg raises
    # "SyntaxError: 'nc' is not a valid YOLO argument" before epoch 1.
    for _stale in ("nc", "names"):
        model.overrides.pop(_stale, None)
    _print_report(report)
    print(f"[family] names: {name_map}")

    def _verify_swapped(trainer) -> None:
        # Ultralytics rebuilds the model in _setup_train; prove the rebuilt decoder
        # actually received the swapped weights before spending 20 epochs.
        dec = trainer.model.model[-1]
        if int(getattr(dec, "nc", -1)) != args.nc:
            raise RuntimeError(
                f"trainer decoder nc={getattr(dec, 'nc', None)} != --nc={args.nc}; the "
                "head swap was dropped during trainer setup (check data.yaml nc)"
            )
        src = swapped.state_dict()
        dst = trainer.model.state_dict()
        carried = sum(
            1 for k, v in src.items() if k in dst and dst[k].shape == v.shape
        )
        print(
            f"[family] trainer decoder nc={dec.nc}; carried {carried}/{len(src)} swapped "
            "tensors into the rebuilt model"
        )
        src_dec = swapped.model[-1]
        src_enc = src_dec.enc_score_head.weight
        dst_enc = dec.enc_score_head.weight
        n = min(src_enc.shape[0], dst_enc.shape[0])
        if not torch.equal(dst_enc[:n], src_enc[:n]):
            raise RuntimeError("rebuilt decoder did not receive the swapped class head")
        if not torch.equal(
            dec.dec_score_head[0].weight[:n], src_dec.dec_score_head[0].weight[:n]
        ):
            raise RuntimeError("rebuilt decoder did not receive the swapped dec_score_head")

    model.add_callback("on_train_start", _verify_swapped)
    model.add_callback("on_train_epoch_end", _bn_watchdog)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        amp=args.amp,  # patched to bf16
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        rect=args.rect,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
    )
    print("[family] training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
