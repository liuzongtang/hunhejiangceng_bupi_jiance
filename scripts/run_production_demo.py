#!/usr/bin/env python3
"""
End-to-End Production Line Simulation.

Demonstrates all 8 modules working together in a simulated fabric inspection line:
  1. Generate synthetic fabric images with random defects
  2. Run inference via FabricDefectDetector
  3. Store images via ImageStorage
  4. Submit detection reports to PostgreSQL
  5. Trigger alerts via AlertService → WebSocket broadcast
  6. Display real-time console + web dashboard

Usage:
    python scripts/run_production_demo.py --batches 10 --interval 2
    python scripts/run_production_demo.py --batches 50 --interval 0.5 --seed 123
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def generate_fabric_image(
    rng: np.random.RandomState, with_defect: bool = True
) -> np.ndarray:
    """
    Generate a synthetic fabric texture image (640x480 RGB).

    Simulates a woven fabric with optional defects.
    """
    h, w = 480, 640
    # Base fabric texture: diagonal weave pattern
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    base = (128 + 30 * np.sin(xx * 0.05 + yy * 0.02) + 20 * np.sin(yy * 0.08)).astype(
        np.uint8
    )

    # RGB from grayscale base with slight color variation
    img = np.stack([base, base, base], axis=-1).astype(np.uint8)
    img[:, :, 0] = np.clip(img[:, :, 0] + rng.randint(-5, 5, (h, w)), 0, 255).astype(
        np.uint8
    )
    img[:, :, 1] = np.clip(img[:, :, 1] + rng.randint(-3, 3, (h, w)), 0, 255).astype(
        np.uint8
    )

    if with_defect and rng.random() < 0.6:
        defect_type = rng.choice(
            ["broken_warp", "hole", "stain", "star_skip", "size_stain"]
        )
        x0, y0 = rng.randint(50, w - 150), rng.randint(50, h - 150)

        if defect_type == "broken_warp":
            # Thin dark line across the fabric
            for i in range(rng.randint(30, 120)):
                dx, dy = rng.randint(-2, 3), rng.randint(-1, 2)
                cx = np.clip(x0 + dx * i, 0, w - 1)
                cy = np.clip(y0 + dy * i, 0, h - 1)
                img[cy, cx] = [20, 20, 30]

        elif defect_type == "hole":
            # Dark circular region
            r = rng.randint(8, 25)
            yy_g, xx_g = np.ogrid[:h, :w]
            mask = (xx_g - x0) ** 2 + (yy_g - y0) ** 2 <= r**2
            img[mask] = [10, 10, 15]

        elif defect_type == "stain":
            # Irregular discolored patch
            r = rng.randint(15, 40)
            yy_g, xx_g = np.ogrid[:h, :w]
            mask = (xx_g - x0) ** 2 + (yy_g - y0) ** 2 <= r**2
            stain_color = rng.choice([[80, 60, 30], [100, 90, 40], [50, 40, 60]])
            img[mask] = np.clip(img[mask].astype(int) + stain_color, 0, 255).astype(
                np.uint8
            )

        elif defect_type == "star_skip":
            # Horizontal gap (skipped weft)
            for dy in range(y0, min(y0 + rng.randint(3, 8), h)):
                img[dy, x0 : x0 + rng.randint(30, 80)] = [180, 180, 190]

        elif defect_type == "size_stain":
            # Subtle color shift in a region
            r = rng.randint(30, 60)
            yy_g, xx_g = np.ogrid[:h, :w]
            mask = (xx_g - x0) ** 2 + (yy_g - y0) ** 2 <= r**2
            shift = rng.choice([-20, 20, -15])
            img[mask, rng.randint(0, 3)] = np.clip(
                img[mask, rng.randint(0, 3)].astype(int) + shift, 0, 255
            ).astype(np.uint8)

    return img


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Production Line Simulation"
    )
    parser.add_argument(
        "--batches", type=int, default=20, help="Number of batches to simulate"
    )
    parser.add_argument(
        "--interval", type=float, default=1.5, help="Seconds between batches"
    )
    parser.add_argument(
        "--images-per-batch", type=int, default=2, help="Images per batch"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup config
    from backend import config as cfg

    cfg._config = cfg.BackendConfig(environment="development", debug=False)
    import asyncio

    from backend.database import init_db

    async def _init():
        await init_db()

    asyncio.run(_init())

    # Initialize all components
    from backend.inference.detector import get_detector
    from backend.services.alert_service import get_alert_service
    from backend.storage import get_storage

    detector = get_detector(backend="dummy", seed=args.seed)
    storage = get_storage()
    alert_service = get_alert_service()
    rng = np.random.RandomState(args.seed)

    # Tracking
    total_images = 0
    total_defects = 0
    defect_counts = {}
    alert_counts = {"critical": 0, "major": 0, "medium": 0, "minor": 0}

    print()
    print("=" * 60)
    print("  Fabric Defect Detection — Production Line Simulation")
    print("=" * 60)
    print(f"  Batches:      {args.batches}")
    print(f"  Interval:     {args.interval}s")
    print(f"  Images/batch: {args.images_per_batch}")
    print("  Backend:      dummy")
    print(f"  Seed:         {args.seed}")
    print("=" * 60)
    print()
    print(
        f"{'Batch':<8} {'Images':<8} {'Defects':<8} {'Critical':<10} {'Major':<8} {'FPS':<10}"
    )
    print("-" * 60)

    start_time = time.time()

    for batch_idx in range(1, args.batches + 1):
        batch_start = time.time()

        # 1. Generate synthetic fabric images
        images = []
        for _ in range(args.images_per_batch):
            img = generate_fabric_image(rng, with_defect=rng.random() < 0.7)
            images.append(img)
        total_images += len(images)

        # 2. Run inference
        device_id = f"CAM-{rng.randint(1, 5):03d}"
        batch_id = f"B{time.strftime('%Y%m%d')}-{batch_idx:04d}"
        report = detector.build_report(device_id, batch_id, images)

        # 3. Store images
        for i, img in enumerate(images):
            import cv2

            _, buf = cv2.imencode(".jpg", img)
            storage.upload(
                f"images/{device_id}/{batch_id}/frame_{i:03d}.jpg",
                buf.tobytes(),
                "image/jpeg",
            )

        # 4. Submit detection report
        import contextlib

        import requests

        with contextlib.suppress(requests.ConnectionError):
            requests.post(
                "http://localhost:8000/api/v1/detection/report",
                json=report.model_dump(mode="json"),
                timeout=2,
            )

        # 5. Count defects + trigger alerts
        batch_defects = report.results.get("total_defects", 0)
        total_defects += batch_defects
        for d in report.results.get("defect_list", []):
            defect_counts[d["type"]] = defect_counts.get(d["type"], 0) + 1
            alert = alert_service.evaluate_defect(
                defect_type=d["type"],
                severity=d["severity"],
                device_id=device_id,
                batch_id=batch_id,
                bbox=d["bbox"],
                confidence=d["confidence"],
            )
            if alert:
                alert_counts[alert.severity] = alert_counts.get(alert.severity, 0) + 1

        # Console output
        batch_time = time.time() - batch_start
        fps = len(images) / batch_time if batch_time > 0 else 0
        print(
            f"{batch_idx:<8} {total_images:<8} {total_defects:<8} "
            f"{alert_counts['critical']:<10} {alert_counts['major']:<8} "
            f"{fps:<10.1f}"
        )

        # Simulate line speed
        if batch_idx < args.batches:
            time.sleep(args.interval)

    total_time = time.time() - start_time
    total_fps = total_images / total_time

    # Final summary
    print("-" * 60)
    print(f"\n{'=' * 60}")
    print("  Simulation Complete")
    print(f"{'=' * 60}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Total images:    {total_images}")
    print(f"  Total defects:   {total_defects}")
    print(f"  Overall FPS:     {total_fps:.1f}")
    print(f"  Detection rate:  {total_defects / total_images:.1f} defects/image")
    print("\n  Defect Distribution:")
    for dtype, count in sorted(defect_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(30, count * 3) + "-" * max(0, 30 - count * 3)
        print(f"    {dtype:18s}: {bar} {count}")
    print("\n  Alert Summary:")
    for sev in ["critical", "major", "medium", "minor"]:
        c = alert_counts.get(sev, 0)
        if c > 0:
            print(f"    {sev:10s}: {c}")
    print("\n  Images stored in: data/images/")
    print("  Dashboard: http://localhost:8000/dashboard")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
