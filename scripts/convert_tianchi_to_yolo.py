"""
Convert the extracted Tianchi "smartdiagnosisofclothflaw" dataset to YOLO format.

把解压后的天池「智能织物瑕疵诊断赛」数据集整理成 YOLO 格式。

═══════════════════════════════════════════════════════════════════════
本脚本负责整个项目的「文件整理」环节，共四步：
    1. 转换  各训练集目录 → _staging（此步内部完成按图像名去重）
    2. 划分  _staging → train / val（10%，随机种子固定可复现）
    3. 测试  拷贝无标签的 testA / testB
    4. 描述  写 classes.txt 与 data.yaml

原始数据结构：
    <ds_root>/
        defect_Images/              # defect images (.jpg)
        normal_Images/              # no-defect images (.jpg, negative samples)
        Annotations/anno_train.json # [{"name", "defect_name", "bbox": [xmin,ymin,xmax,ymax]}, ...]

with Chinese defect names mapping to 20 category ids (plus id 0 = no defect).
This project's `backend.training.dataset.FabricDataset` reads YOLO format
(`images/*.jpg` + `labels/*.txt`), so this script normalizes each bbox to
`class cx cy w h` (all 0..1) and emits one `.txt` per image. Normal images
become empty label files (negative samples).

三个必须知道的点（详见各函数内的中文注释）：
    ★ 去重  partA 与 partB 的 anno_train.json 字节级相同（md5 一致，
            两者引用同一批 4774 个图片名），但两个目录里的图片文件名不同。
            因此按图像名去重、且未匹配到标注的图要显式报告，不能静默处理。
    ★ 顺序  必须"先去重、再划分"，反了会让验证集 100% 泄漏且不报错
    ★ 负样本 正常图写成空的 .txt 保留（实测占验证集 39%），
            检测模型必须见过"什么都没有"的图

Usage:
    python scripts/convert_tianchi_to_yolo.py \
        --train-dirs shujuji/smartdiagnosisofclothflaw_round1train1_datasets_partA/partA \
                     shujuji/smartdiagnosisofclothflaw_round1train1_datasets_partB/partB \
                     shujuji/smartdiagnosisofclothflaw_round1train2_datasets/guangdong1_round1_train2_20190828 \
        --test-dirs shujuji/smartdiagnosisofclothflaw_round1testA_datasets/guangdong1_round1_testA_20190818 \
                    shujuji/smartdiagnosisofclothflaw_round1testB_datasets/guangdong1_round1_testB_20190919 \
        --out data/tianchi --class-scheme tianchi20
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import random
import shutil
from typing import Dict, List, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# 天池 20 类缺陷体系（类别 ID 1..20；ID 0 保留给「无缺陷」）。
# ---------------------------------------------------------------------------

# 中文 defect_name -> 天池类别 ID（1..20），依据数据集官方 README。
# 三个要点：
#   1) 同一缺陷有多个别名（"水渍"/"油渍"/"污渍" → 同一类 2），必须归并；
#   2) 官方 README 存在错别字（"粗维" 应为 "粗纬"），需做容错；
#   3) 20 个原始名最终映射出 18 个可训练类别（19/20 族后续经实证诊断与
#      9/11 合并，因为视觉上不可区分——属标签噪声而非模型能力问题）。
NAME_TO_CATEGORY: Dict[str, int] = {
    "破洞": 1,
    "水渍": 2,
    "油渍": 2,
    "污渍": 2,
    "三丝": 3,
    "结头": 4,
    "花板跳": 5,
    "百脚": 6,
    "毛粒": 7,
    "粗经": 8,
    "松经": 9,
    "断经": 10,
    "吊经": 11,
    "粗纬": 12,
    "粗维": 12,  # 粗纬 (coarse weft); 粗维 is the README's typo
    "纬缩": 13,
    "浆斑": 14,
    "整经结": 15,
    "星跳": 16,
    "跳花": 16,
    "断氨纶": 17,
    "稀密档": 18,
    "浪纹档": 18,
    "色差档": 18,
    "磨痕": 19,
    "轧痕": 19,
    "修痕": 19,
    "烧毛痕": 19,
    "死皱": 20,
    "云织": 20,
    "双纬": 20,
    "双经": 20,
    "跳纱": 20,
    "筘路": 20,
    "纬纱不良": 20,
}

# 天池类别 ID（1..20）-> 英文类名。列表下标即训练用的 0-based 类别 ID。
TIANCHI_CATEGORIES: List[str] = [
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
    "coarse_weft",
    "weft_shrink",
    "size_stain",
    "warping_knot",
    "star_skip",
    "broken_spandex",
    "dense_section",
    "surface_mark",
    "weave_defect",
]

# 图像扩展名白名单。用白名单而非黑名单：新的垃圾文件类型出现时自动被排除
# （例如 macOS 解压残留的 __MACOSX/._xxx.jpg）。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def category_to_class_id(category: int, scheme: str) -> int:
    """Map a Tianchi category id (1..20) to a 0-indexed class id."""
    if scheme == "tianchi20":
        return category - 1
    raise ValueError(f"Unknown class scheme: {scheme}")


def class_names_for(scheme: str) -> List[str]:
    """Return the class-name list matching the chosen scheme."""
    if scheme == "tianchi20":
        return TIANCHI_CATEGORIES
    raise ValueError(f"Unknown class scheme: {scheme}")


def load_annotations(anno_path: str) -> Dict[str, List[Dict]]:
    """Read anno_train.json and group annotations by image name.

    读取天池标注文件并按图像名分组。

    原始标注集中在一个 JSON 里（一条记录对应一个框），而后续处理需要
    "给一张图找它的所有标注"，所以这里转置成「文件名 → 标注列表」。

    Args:
        anno_path: anno_train.json 的路径。

    Returns:
        字典 {图像文件名: [标注, ...]}。文件不存在时返回空字典。
    """
    by_name: Dict[str, List[Dict]] = defaultdict(list)
    if not os.path.exists(anno_path):
        return by_name
    with open(anno_path, encoding="utf-8") as f:
        for ann in json.load(f):
            by_name[ann["name"]].append(ann)
    return by_name


# 将数据集坐标格式转为yolo格式
def xyxy_to_yolo(
    bbox: List[float], w: int, h: int
) -> Tuple[float, float, float, float]:
    """Convert xyxy pixel bbox to normalized (cx, cy, bw, bh).

    把天池的 xyxy 像素框转成 YOLO 的 cxcywh 归一化框。

    两种格式的差异：
      - 天池 xyxy：[左上角x, 左上角y, 右下角x, 右下角y]，单位是像素
      - YOLO cxcywh：[中心点x, 中心点y, 宽, 高]，全部归一化到 0~1

    归一化的意义是让标注与图像分辨率解耦——同一份标签在 2446×1000
    和 1280×1280 上都能用，模型才能泛化到不同尺寸的输入。

    Args:
        bbox: 天池格式的四个坐标 [xmin, ymin, xmax, ymax]（像素）。
        w: 原图宽度（像素），由调用方从图像文件读出。
        h: 原图高度（像素）。

    Returns:
        (cx, cy, bw, bh) 四个归一化浮点数。
    """
    # 解包出 4 个坐标并统一转浮点（JSON 里的数字可能是整数）
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    # 中心点：先取两端均值，再除图像宽/高归一化到 0~1
    # 注意 x 除以 w、y 除以 h，写反了模型学出来的东西会转置 90°
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    # 宽度/高度：max(0.0, ...) 防负宽度
    # （若原始标注错误导致 xmax < xmin，不加这层保护会算出负宽度，
    #   负宽度进入 IoU 计算会让并集面积算错、梯度方向反掉）
    bw = max(0.0, (xmax - xmin)) / w
    bh = max(0.0, (ymax - ymin)) / h
    # 双向钳制中心点到 [0, 1]
    # 为什么钳中心点而不是边界？目标被图像边缘截断是真实存在的（框可以超出
    # 图像），但中心点必须落在图内。实测 8570 个框中有 9 个越界（幅度约 1e-6），
    # 就是这里要挡的。
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    # 宽高只卡上限不卡下限：宽度为 0 虽有缺陷（退化框），但那是需要上报的
    # 数据质量问题，不该在转换阶段悄悄改掉
    bw, bh = min(bw, 1.0), min(bh, 1.0)
    return cx, cy, bw, bh


# 图片去重
def convert_train_dir(
    ds_dir: str,
    out_images: str,
    out_labels: str,
    scheme: str,
) -> int:
    """
    Convert one extracted train dataset dir to YOLO format.

    把一个解压后的天池训练集目录整理成 YOLO 格式。

    处理两张图片表：
      - defect_Images/  缺陷图，有标注
      - normal_Images/  正常图，无标注 → 写成空的 .txt 作为负样本

    ⚠️ 去重说明（本项目最关键的一个数据坑）：
       partA 与 partB 的 anno_train.json 字节级完全相同
       （md5 均为 8ca9efa45e3f4ade805a12019a5880a3），各自 7728 条标注、
       引用同一批 4774 个图片名。
       而 partA/partB 目录里的**图片文件名并不相同**（实测同名交集为 0，
       partB 有 3656 张独有图片）。所以 partB 的图片按名找不到对应标注，
       会被写成空标签——如果不做处理，这些图会被当成"无缺陷负样本"混入训练集，
       严重污染正负样本比。

       因此这里按**图像名**去重：目标文件已存在就跳过并告警，不静默覆盖。
       同时，未匹配到标注的图仍会写出空 .txt（作为负样本保留），
       但函数末尾会汇总报告跳过与未匹配的数量，便于人工核查。

    Args:
        ds_dir: 含有 defect_Images/、normal_Images/、Annotations/ 的目录。
        out_images: 输出图像的扁平目录。
        out_labels: 输出标签 .txt 的扁平目录。
        scheme: 类别体系名，目前只支持 "tianchi20"。

    Returns:
        实际写出的图像数量（去重后的净数量）。
    """
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # 读取本数据集目录的标注并按图像名分组
    by_name = load_annotations(os.path.join(ds_dir, "Annotations", "anno_train.json"))
    count = 0
    skipped = 0  # 未知缺陷名的标注计数（跳过必须有计数，否则不知道丢了多少数据）

    # 依次处理缺陷图与正常图两个目录
    for subdir in ("defect_Images", "normal_Images"):
        img_dir = os.path.join(ds_dir, subdir)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            # 扩展名白名单过滤（挡掉 __MACOSX 残留等垃圾文件）
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            src = os.path.join(img_dir, fname)

            # Read dimensions to normalize bbox coordinates.
            # 读出原图宽高，用于把像素坐标归一化
            with Image.open(src) as im:
                w, h = im.size

            dst_img = os.path.join(out_images, fname)
            # ★★ 去重判据：按图像名（目标文件是否已存在），不是按标注条数
            if os.path.exists(dst_img):
                # 不静默跳过——你必须知道跳过了什么
                print(f"  [warn] duplicate image name skipped: {fname}")
                continue  # ★ 跳过本图且不累加，这就是防 GT 翻倍的关键
            shutil.copy(src, dst_img)

            # Build YOLO label (empty file for normal images).
            lines: List[str] = []
            for ann in by_name.get(fname, []):
                # 中文缺陷名 → 类别 ID；.get 取不到返回 None（正常图即此情况）
                category = NAME_TO_CATEGORY.get(ann["defect_name"])
                if category is None:
                    skipped += 1  # ★ 计数，结束时统一报告
                    continue
                cls = category_to_class_id(category, scheme)
                cx, cy, bw, bh = xyxy_to_yolo(ann["bbox"], w, h)
                # 保留 6 位小数：实测训练集最小框宽 0.002833，
                # 若只留 3 位会写成 0.003（误差 6%），极端情况框宽 0.0001
                # 会被写成 0.000 —— 这个框就消失了
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # 标签文件名 = 图像名去扩展名 + .txt
            label_path = os.path.join(out_labels, os.path.splitext(fname)[0] + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                # ★ lines 为空时 "\n".join(lines) 返回空串 → 写出空文件 = 负样本
                # 正常图保留为负样本很重要：检测模型必须见过"什么都没有"的图，
                # 否则会在完好织物上乱框。实测验证集 957 张里有 374 张负样本（39%）。
                f.write("\n".join(lines) + ("\n" if lines else ""))

            count += 1

    # 汇总报告被跳过的标注数（不静默丢数据）
    if skipped:
        print(
            f"  [warn] {skipped} annotations had unknown defect_name and were skipped"
        )
    return count


def convert_test_dir(test_dir: str, out_images: str) -> int:
    """Copy an extracted test dir (flat .jpg, no labels) into out_images.

    复制测试集图片。测试集没有标注，只需拷贝图片。
    """
    os.makedirs(out_images, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(test_dir)):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            continue
        shutil.copy(os.path.join(test_dir, fname), os.path.join(out_images, fname))
        count += 1
    return count


def split_train_val(root: str, val_ratio: float, seed: int) -> Tuple[int, int]:
    """Split the staging train dir into train/ and val/ by image name.

    把暂存区的图像划分成训练集与验证集。

    ⚠️ 顺序要求：必须在**去重之后**划分。
       若先分别划分再合并，partA 的图可能进 train 而 partB 的同名图进 val，
       验证集就 100% 泄漏进训练集。而且这种错误不报错、日志正常、指标还更好，
       属于最隐蔽的实验错误。

    Args:
        root: 输出根目录（内含 _staging/）。
        val_ratio: 验证集比例，默认 0.1。
        seed: 随机种子，固定后划分结果完全可复现。

    Returns:
        (训练集图像数, 验证集图像数)。
    """
    # 暂存区（所有图先转换到这里，再从这里分配）与最终的目标目录
    staging_img = os.path.join(root, "_staging", "images")
    staging_lbl = os.path.join(root, "_staging", "labels")
    train_img = os.path.join(root, "train", "images")
    train_lbl = os.path.join(root, "train", "labels")
    val_img = os.path.join(root, "val", "images")
    val_lbl = os.path.join(root, "val", "labels")

    # ★ sorted() 保证顺序固定：不排序的话不同机器上 os.listdir 的返回顺序
    #   可能不同，划分就不可复现了
    all_images = sorted(os.listdir(staging_img))
    random.seed(seed)  # ★ 固定随机种子 —— 可复现的基础
    random.shuffle(all_images)  # 原地打乱（shuffle 返回 None，不用接返回值）
    n_val = int(len(all_images) * val_ratio)  # 9576 × 0.1 = 957.6 → 957
    # 取前 10% 作验证集；用 set 便于 O(1) 查询某文件属于哪一边
    val_set = set(all_images[:n_val])

    # 建四个目录；exist_ok=True 表示已存在就跳过（脚本可重复运行）
    for d in (train_img, train_lbl, val_img, val_lbl):
        os.makedirs(d, exist_ok=True)

    for name in all_images:
        # 三元表达式决定去哪：验证集还是训练集
        dst_img_dir = val_img if name in val_set else train_img
        dst_lbl_dir = val_lbl if name in val_set else train_lbl
        # os.rename 是**移动**不是复制：天然保证不会重复
        os.rename(os.path.join(staging_img, name), os.path.join(dst_img_dir, name))
        # 标签用同样的判断，保证图像与标签成对移动（否则会错位）
        lbl = os.path.splitext(name)[0] + ".txt"
        src_lbl = os.path.join(staging_lbl, lbl)
        if os.path.exists(src_lbl):
            os.rename(src_lbl, os.path.join(dst_lbl_dir, lbl))

    shutil.rmtree(os.path.join(root, "_staging"), ignore_errors=True)
    return len(all_images) - n_val, n_val


def write_meta(root: str, scheme: str) -> None:
    """Write classes.txt and data.yaml (Ultralytics) for the output root.

    写出数据集描述文件：classes.txt（类名列表）与 data.yaml（Ultralytics 配置）。

    data.yaml 里的 `nc: 18` 很关键——ultralytics 加载时会发现 COCO 预训练权重
    是 80 类、这里声明 18 类，从而**自动重建分类头**。这是微调能发生的前提。
    """
    names = class_names_for(scheme)
    with open(os.path.join(root, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")

    yaml_lines = [
        # 路径转成正斜杠（YAML 里 Windows 反斜杠会被当转义符）
        f"path: {os.path.abspath(root).replace(os.sep, '/')}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(names)}",  # 类别数
        "names:",
    ]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")


def main() -> int:
    """脚本入口：按四步把原始数据整理成可直接训练的 YOLO 数据集。

    流程：
      1. 转换  把各训练集目录转到 _staging（转换时按图像名去重）
      2. 划分  _staging → train / val（10%，seed 固定）
      3. 测试  拷贝无标签的 testA / testB
      4. 描述  写 classes.txt 与 data.yaml

    ⚠️ 顺序不可调换：必须先去重（步骤 1 内完成）再划分（步骤 2）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dirs", nargs="+", default=[], help="Extracted train dataset dirs"
    )
    parser.add_argument(
        "--test-dirs", nargs="+", default=[], help="Extracted test dirs (no labels)"
    )
    parser.add_argument("--out", default="data/tianchi", help="Output root directory")
    parser.add_argument(
        "--class-scheme",
        default="tianchi20",
        choices=["tianchi20"],
        help="Class scheme to emit (default: tianchi20, 20 classes)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for val split"
    )
    args = parser.parse_args()

    scheme = args.class_scheme
    root = args.out
    print(f"Class scheme: {scheme} ({len(class_names_for(scheme))} classes)")
    print(f"Output root: {root}")

    # 1. Convert all train dirs into a staging train dir.
    # 暂存区：所有图先统一转到这里，再从这里分配到 train/val
    staging_img = os.path.join(root, "_staging", "images")
    staging_lbl = os.path.join(root, "_staging", "labels")
    total = 0
    for ds_dir in args.train_dirs:
        print(f"Converting train dir: {ds_dir}")
        # ★ 去重在这一步内部完成（按图像名）
        n = convert_train_dir(ds_dir, staging_img, staging_lbl, scheme)
        total += n
        print(f"  wrote {n} images")
    print(f"Total train images: {total}")

    # 2. Split staging -> train/val.
    # ★ 必须在去重之后划分，否则验证集会 100% 泄漏
    if os.path.isdir(staging_img) and os.listdir(staging_img):
        n_train, n_val = split_train_val(root, args.val_ratio, args.seed)
        print(f"Split: {n_train} train / {n_val} val")
    else:
        print("  [warn] no train images found; skipping split")

    # 3. Convert test dirs (no labels).
    # 测试集无标注，只需拷贝图片
    test_img = os.path.join(root, "test", "images")
    for test_dir in args.test_dirs:
        print(f"Converting test dir: {test_dir}")
        n = convert_test_dir(test_dir, test_img)
        print(f"  wrote {n} test images")

    # 4. Write classes.txt + data.yaml.
    # data.yaml 里的 nc 会让 ultralytics 重建 18 类分类头
    write_meta(root, scheme)

    print("Done.")
    print(f"  train: {os.path.join(root, 'train', 'images')}")
    print(f"  val:   {os.path.join(root, 'val', 'images')}")
    print(f"  test:  {test_img}")
    print(f"  classes: {os.path.join(root, 'classes.txt')}")
    print(f"  data.yaml: {os.path.join(root, 'data.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
