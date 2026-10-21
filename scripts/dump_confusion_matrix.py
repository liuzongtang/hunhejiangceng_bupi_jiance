"""0.4c — Dump numeric confusion matrix of the 18-class baseline on val (957 images).

Uses the official ultralytics val pipeline (same as eval_multIou.py sanity path),
then reads metrics.confusion_matrix.matrix (18x18, rows=GT, cols=pred) and writes:
  runs/family_clustering/confusion_raw.csv      raw counts
  runs/family_clustering/confusion_row_norm.csv row-normalized (recall view)
  runs/family_clustering/confusion_top_pairs.json  top confusable GT->pred pairs (off-diagonal)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import RTDETR

CLASSES = [
    "hole",
    "stain",
    "three_silk",
    "knot",
    "flower_board",
    "hundred_feet",
    "hair_particle",
    "coarse_warp",
    "loose_warp",
    "broken_warp",
    "hanging_warp",
    "weft_shrink",
    "size_stain",
    "warping_knot",
    "star_skip",
    "broken_spandex",
    "dense_section",
    "weave_defect",
]
OUT = Path("runs/family_clustering")


def main():
    model = RTDETR("runs/rtdetr/tianchi18_full/weights/best.pt")
    res = model.val(
        data="data/tianchi/data.yaml",
        imgsz=1280,
        device=0,
        workers=4,
        plots=True,  # confusion matrix is ONLY updated when plots=True (ultralytics quirk)
        project="runs/family_clustering",
        name="confusion_val",
        exist_ok=True,
        verbose=False,
    )
    # NOTE: ultralytics matrix index order is [pred, gt]; last row/col = background
    # (FN / FP). Transpose to the standard [gt, pred] convention for the CSV.
    cm = np.asarray(res.confusion_matrix.matrix)
    print(f"raw matrix shape: {cm.shape}, total: {cm.sum()}")
    if cm.shape[0] > 18:
        cm = cm[:18, :18]  # drop background row/col
    cm = cm.T  # now rows = GT, cols = pred
    assert cm.shape == (18, 18), cm.shape

    df = pd.DataFrame(cm.astype(int), index=CLASSES, columns=CLASSES)
    df.to_csv(OUT / "confusion_raw.csv", encoding="utf-8")
    row_sum = cm.sum(1, keepdims=True)
    row_norm = np.divide(cm, np.where(row_sum == 0, 1, row_sum))
    pd.DataFrame(np.round(row_norm, 4), index=CLASSES, columns=CLASSES).to_csv(
        OUT / "confusion_row_norm.csv", encoding="utf-8"
    )

    # top off-diagonal confusion pairs (row=GT, col=pred), exclude diagonal
    pairs = []
    for i in range(18):
        for j in range(18):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {"gt": CLASSES[i], "pred": CLASSES[j], "count": int(cm[i, j])}
                )
    pairs.sort(key=lambda x: -x["count"])
    (OUT / "confusion_top_pairs.json").write_text(
        json.dumps(pairs[:40], indent=2), encoding="utf-8"
    )
    print("\ntop-15 confusable pairs (GT -> pred):")
    for p in pairs[:15]:
        print(f"  {p['gt']:15s} -> {p['pred']:15s} : {p['count']}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
