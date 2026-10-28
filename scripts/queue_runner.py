#!/usr/bin/env python
"""queue_runner.py -- unattended, strictly serial GPU arm queue for stage 2.

Why this exists: the DSH sandbox kills every process it spawns when the tool call
ends, `schtasks` and WMI process creation are both denied, so an agent-launched
training run cannot survive the session. This runner is therefore meant to be
started by the *user* (double-click ``run_stage2.bat``), outside DSH: once it is
running, the whole queue is independent of any agent session.

Guarantees it enforces:

* **Iron law 1 (no GPU concurrency).** Before every arm it polls
  ``nvidia-smi --query-gpu=memory.used`` and refuses to start until the GPU is
  below ``gpu_free_mib``. A busy GPU is waited on for ``gpu_wait_min`` minutes.
* **Iron law 2 (pre-registration).** An arm whose ``PREREG.md`` is missing is
  refused, unless the arm sets ``"allow_missing_prereg": true`` (used for the
  baseline, whose criterion is the plan document itself).
* **Serial only.** One arm at a time; the next starts only after the previous
  process exits.
* **Stop switch.** Create ``logs/STOP`` at any time: the queue stops after the
  currently running arm finishes (it never kills a training run in flight, since
  a mid-epoch kill wastes the whole arm).
* **Resumable bookkeeping.** State lives in ``tools/queue_state.json``; arms
  already marked ``completed`` are skipped, so re-running the queue continues
  where it stopped.

Each arm may declare:

    {
      "name": "r3prime_mosaic1280_s42",
      "requires": ["data/tianchi/data.yaml"],
      "prereg": "runs/stage2/r3prime/PREREG.md",
      "gpu": true,
      "cmd": ["{py}", "scripts/train_mosaic1280.py"],
      "log": "logs/stage2_r3prime.log",
      "eval": {
        "cmd": ["{py}", "scripts/eval_multIou.py", "--weights", "...", "--out", "..."],
        "json": "runs/multi_iou/r3prime_official.json",
        "control_key": null
      },
      "notes": "human-readable purpose"
    }

``{py}`` expands to this interpreter, ``{repo}`` to the repository root.

Usage::

    python tools/queue_runner.py --queue tools/queue.json --dry-run
    python tools/queue_runner.py --queue tools/queue.json
    python tools/queue_runner.py --queue tools/queue.json --only r3prime_mosaic1280_s42
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parent.parent
REPO = WORKSPACE / "repo"
STATE = Path(__file__).resolve().parent / "queue_state.json"
STOP_FILE = REPO / "logs" / "STOP"


def now() -> str:
    """Local timestamp for the state file."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def expand(tokens: list[str], py: str) -> list[str]:
    """Substitute ``{py}`` / ``{repo}`` in an argv list."""
    return [t.replace("{py}", py).replace("{repo}", str(REPO)) for t in tokens]


def load_state(path: Path) -> dict:
    """Load the running state, tolerating a missing or corrupt file."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[queue] WARNING: {path} unreadable, starting a fresh state file")
    return {"arms": {}, "created": now()}


def save_state(path: Path, state: dict) -> None:
    """Persist the state atomically enough for a single-writer runner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def gpu_used_mib() -> int | None:
    """Currently used GPU memory in MiB, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def wait_for_gpu(free_mib: int, wait_min: int) -> bool:
    """Block until the GPU is idle enough (iron law 1). Returns False on timeout."""
    deadline = time.time() + wait_min * 60
    while True:
        used = gpu_used_mib()
        if used is None:
            print("[queue] WARNING: nvidia-smi unavailable; cannot enforce iron law 1")
            return True
        if used < free_mib:
            print(f"[queue] GPU OK: {used} MiB used (< {free_mib})")
            return True
        if time.time() > deadline:
            print(f"[queue] GPU still at {used} MiB after {wait_min} min; giving up on this arm")
            return False
        print(f"[queue] GPU busy ({used} MiB >= {free_mib}); waiting 30 s ...")
        time.sleep(30)


def run_cmd(cmd: list[str], log_path: Path, dry: bool) -> int:
    """Run one command, streaming into ``log_path``; return the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[queue] $ {' '.join(cmd)}")
    print(f"[queue]   log -> {log_path}")
    if dry:
        return 0
    with log_path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n===== {now()} $ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            # Ultralytics progress bars use box-drawing characters that the
            # GBK console code page cannot encode; a failed echo must never kill
            # the runner (it did once, taking a training arm with it).
            echo = "    " + line
            try:
                sys.stdout.write(echo)
                sys.stdout.flush()
            except UnicodeEncodeError:
                sys.stdout.write(echo.encode("gbk", "replace").decode("gbk", "replace"))
                sys.stdout.flush()
        proc.wait()
        fh.write(f"===== {now()} exit={proc.returncode}\n")
    return proc.returncode


def try_metrics(eval_cfg: dict, state: dict) -> dict:
    """Read the arm's official JSON and attach mean mAP (+ delta vs a control)."""
    out: dict = {}
    json_rel = eval_cfg.get("json") if eval_cfg else None
    if not json_rel:
        return out
    path = REPO / json_rel
    if not path.exists():
        out["metrics_error"] = f"{json_rel} not written"
        return out
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        out["metrics_error"] = f"{json_rel} unreadable: {exc}"
        return out
    m = rep.get("mAP") or {}
    values = [v for k, v in m.items() if k.startswith("mAP@")]
    if values:
        out["mean"] = round(sum(values) / len(values), 4)
    out["mAP50"] = m.get("mAP@0.5")
    out["per_class_n"] = len(rep.get("per_class") or [])
    out["official_json"] = json_rel

    ctrl_key = eval_cfg.get("control_key")
    if ctrl_key and ctrl_key in state.get("arms", {}):
        ctrl = state["arms"][ctrl_key].get("mean")
        if ctrl is not None and out.get("mean") is not None:
            delta = round(out["mean"] - ctrl, 4)
            out["control"] = ctrl_key
            out["delta_mean"] = delta
            out["verdict"] = (
                "ABOVE +0.005" if delta > 0.005 else ("WITHIN NOISE" if abs(delta) <= 0.0093 else "BELOW")
            )
    return out


def main() -> int:
    # Redirected console output on a Chinese Windows host is GBK; force UTF-8 so
    # a progress bar character can never abort an arm again.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default=str(Path(__file__).resolve().parent / "queue.json"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--only", default=None, help="run a single arm by name")
    ap.add_argument("--gpu-free-mib", type=int, default=2500)
    ap.add_argument("--gpu-wait-min", type=int, default=30)
    ap.add_argument("--python", default=sys.executable, help="interpreter for {py}")
    ap.add_argument("--state", default=str(STATE), help="state file (override for tests)")
    args = ap.parse_args()
    state_path = Path(args.state)
    if args.dry_run:
        # A dry run must never persist "completed" arms: the real run would then
        # skip every experiment. Keep rehearsal bookkeeping strictly separate.
        state_path = WORKSPACE / "scratch" / "queue_state_dryrun.json"

    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    arms = queue["arms"]
    if args.only:
        arms = [a for a in arms if a["name"] == args.only]
        if not arms:
            print(f"[queue] no arm named {args.only}")
            return 1

    state = load_state(state_path)
    state["updated"] = now()
    print(f"[queue] {len(arms)} arm(s), repo={REPO}")

    for arm in arms:
        name = arm["name"]
        rec = state["arms"].get(name, {})
        if rec.get("status") == "completed":
            print(f"[queue] skip {name}: already completed")
            continue
        if STOP_FILE.exists():
            print(f"[queue] STOP file present ({STOP_FILE}); halting before {name}")
            state["stopped"] = {"at": now(), "before": name}
            save_state(state_path, state)
            return 0

        missing = [p for p in arm.get("requires", []) if not (REPO / p).exists()]
        if missing and not args.dry_run:
            msg = f"missing required asset(s): {missing}"
            print(f"[queue] BLOCKED {name}: {msg}")
            state["arms"][name] = {**rec, "status": "blocked", "reason": msg, "at": now()}
            save_state(state_path, state)
            return 2
        if missing:
            print(f"[queue] dry-run: {name} would be BLOCKED on {missing} (plan shown anyway)")

        prereg = arm.get("prereg")
        if prereg and not (REPO / prereg).exists() and not arm.get("allow_missing_prereg"):
            msg = f"PREREG missing at {prereg} (iron law 2)"
            print(f"[queue] BLOCKED {name}: {msg}")
            state["arms"][name] = {**rec, "status": "blocked", "reason": msg, "at": now()}
            save_state(state_path, state)
            return 2

        if arm.get("gpu", True) and not wait_for_gpu(args.gpu_free_mib, args.gpu_wait_min):
            state["arms"][name] = {**rec, "status": "gpu_busy", "at": now()}
            save_state(state_path, state)
            return 3

        cmd = expand(arm["cmd"], args.python)
        # Resume support: a 15 h arm that loses its window should not restart from
        # zero. If the arm declares `resume_if`, that path is checked and
        # `resume_arg` is appended only when a previous run actually exists.
        resume_if = arm.get("resume_if")
        if resume_if and (REPO / resume_if).exists():
            resume_arg = arm.get("resume_arg", "resume")
            cmd = [*cmd, *resume_arg.split()]
            print(f"[queue] resume detected ({resume_if}); appending '{resume_arg}'")
        log_path = REPO / arm.get("log", f"logs/{name}.log")
        state["arms"][name] = {**rec, "status": "running", "started": now(), "log": str(log_path), "cmd": cmd}
        save_state(state_path, state)
        print(f"\n[queue] ==== {name} ====\n[queue] {arm.get('notes', '')}")
        rc = run_cmd(cmd, log_path, args.dry_run)
        rec = state["arms"][name]
        rec["finished"] = now()
        rec["exit_code"] = rc
        if rc != 0:
            rec["status"] = "failed"
            save_state(state_path, state)
            print(f"[queue] {name} FAILED (exit {rc}); stopping the queue. See {log_path}")
            return 4

        rec["status"] = "completed"
        if arm.get("eval"):
            ecfg = arm["eval"]
            ecmd = expand(ecfg["cmd"], args.python)
            erc = run_cmd(ecmd, log_path.with_name(log_path.stem + "_eval.log"), args.dry_run)
            rec["eval_exit"] = erc
            if erc != 0:
                rec["status"] = "completed_eval_failed"
            rec.update(try_metrics(ecfg, state))
        save_state(state_path, state)
        print(f"[queue] {name} done (exit 0)")
        if rec.get("mean") is not None:
            print(f"[queue] mean={rec['mean']} mAP50={rec.get('mAP50')} delta={rec.get('delta_mean')} -> {rec.get('verdict')}")

    state["updated"] = now()
    state["all_done"] = True
    save_state(state_path, state)
    print(f"\n[queue] queue finished. State: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

