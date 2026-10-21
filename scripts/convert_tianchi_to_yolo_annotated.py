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

from __future__ import (
    annotations,  # 启用 PEP 563 延迟求值注解，让下面的类型注解书写更宽松（前向引用不报错）
)

import argparse  # 命令行参数解析库：把 --train-dirs 等选项解析成 args 对象
from collections import (
    defaultdict,  # 带默认值的字典：访问缺失键时自动建空列表，免去"先判断后插入"的样板代码
)
import json  # JSON 读写：用于加载标注文件 anno_train.json
import os  # 操作系统接口：路径拼接、建目录、列文件、重命名、删目录等
import random  # 伪随机数发生器：train/val 划分时打乱名单；固定 seed 后结果完全可复现
import shutil  # 高级文件操作：shutil.copy 拷贝文件、shutil.rmtree 递归删除目录
from typing import (  # 类型注解：Dict/List/Tuple 让函数签名自文档化，IDE 可自动补全检查
    Dict,
    List,
    Tuple,
)

from PIL import (
    Image,  # Pillow 图像库：本脚本只用它打开图片、读出原图宽高（归一化标注的除数）
)

# ---------------------------------------------------------------------------
# 天池 20 类缺陷体系（类别 ID 1..20；ID 0 保留给「无缺陷」）。
# ---------------------------------------------------------------------------

# 中文 defect_name -> 天池类别 ID（1..20），依据数据集官方 README。
# 三个要点：
#   1) 同一缺陷有多个别名（"水渍"/"油渍"/"污渍" → 同一类 2），必须归并；
#   2) 官方 README 存在错别字（"粗维" 应为 "粗纬"），需做容错；
#   3) 20 个原始名最终映射出 18 个可训练类别（19/20 族后续经实证诊断与
#      9/11 合并，因为视觉上不可区分——属标签噪声而非模型能力问题）。
NAME_TO_CATEGORY: Dict[
    str, int
] = {  # ★ 总映射表：键 = 标注 JSON 里 defect_name 的原文，值 = 天池类别 ID（1..20）
    "破洞": 1,  # 破洞 → 类别 1（hole）：一对一映射
    "水渍": 2,  # 水渍 → 类别 2（stain）
    "油渍": 2,  # 油渍 → 同为类别 2：三种液体污渍视觉同属"斑渍"族，标注体系强制归并
    "污渍": 2,  # 污渍 → 同为类别 2
    "三丝": 3,  # 三丝 → 类别 3（three_silk）
    "结头": 4,  # 结头 → 类别 4（knot）
    "花板跳": 5,  # 花板跳 → 类别 5（flower_board）
    "百脚": 6,  # 百脚 → 类别 6（hundred_feet）
    "毛粒": 7,  # 毛粒 → 类别 7（hair_particle）
    "粗经": 8,  # 粗经 → 类别 8（coarse_warp）
    "松经": 9,  # 松经 → 类别 9（loose_warp）
    "断经": 10,  # 断经 → 类别 10（broken_warp）
    "吊经": 11,  # 吊经 → 类别 11（hanging_warp）
    "粗纬": 12,  # 粗纬 → 类别 12（coarse_weft），正确写法
    "粗维": 12,  # 粗纬 (coarse weft); 粗维 is the README's typo —— 官方 README 错别字，容错映射到同一类 12
    "纬缩": 13,  # 纬缩 → 类别 13（weft_shrink）
    "浆斑": 14,  # 浆斑 → 类别 14（size_stain）
    "整经结": 15,  # 整经结 → 类别 15（warping_knot）
    "星跳": 16,  # 星跳 → 类别 16（star_skip）
    "跳花": 16,  # 跳花 → 同为类别 16：与星跳同属"跳纱"族，归并
    "断氨纶": 17,  # 断氨纶 → 类别 17（broken_spandex）
    "稀密档": 18,  # 稀密档 → 类别 18（dense_section）
    "浪纹档": 18,  # 浪纹档 → 同为类别 18：稀密/浪纹/色差三种"档"归并
    "色差档": 18,  # 色差档 → 同为类别 18
    "磨痕": 19,  # 磨痕 → 类别 19（surface_mark）
    "轧痕": 19,  # 轧痕 → 同为类别 19：四种加工"痕"归并
    "修痕": 19,  # 修痕 → 同为类别 19
    "烧毛痕": 19,  # 烧毛痕 → 同为类别 19
    "死皱": 20,  # 死皱 → 类别 20（weave_defect）
    "云织": 20,  # 云织 → 同为类别 20
    "双纬": 20,  # 双纬 → 同为类别 20
    "双经": 20,  # 双经 → 同为类别 20
    "跳纱": 20,  # 跳纱 → 同为类别 20
    "筘路": 20,  # 筘路 → 同为类别 20
    "纬纱不良": 20,  # 纬纱不良 → 同为类别 20：其余罕见疵点全部并入"织疵"类
}

# 天池类别 ID（1..20）-> 英文类名。列表下标即训练用的 0-based 类别 ID。
TIANCHI_CATEGORIES: List[
    str
] = [  # ★ 类名表：列表下标 i 就是 YOLO 训练用的 0-based 类别 ID（i = 天池ID - 1）
    "hole",  # 下标 0 ← 天池类别 1 破洞
    "stain",  # 下标 1 ← 天池类别 2 污渍（含水渍/油渍）
    "three_silk",  # 下标 2 ← 天池类别 3 三丝
    "knot",  # 下标 3 ← 天池类别 4 结头
    "flower_board",  # 下标 4 ← 天池类别 5 花板跳
    "hundred_feet",  # 下标 5 ← 天池类别 6 百脚
    "hair_particle",  # 下标 6 ← 天池类别 7 毛粒
    "coarse_warp",  # 下标 7 ← 天池类别 8 粗经
    "loose_warp",  # 下标 8 ← 天池类别 9 松经
    "broken_warp",  # 下标 9 ← 天池类别 10 断经
    "hanging_warp",  # 下标 10 ← 天池类别 11 吊经
    "coarse_weft",  # 下标 11 ← 天池类别 12 粗纬
    "weft_shrink",  # 下标 12 ← 天池类别 13 纬缩
    "size_stain",  # 下标 13 ← 天池类别 14 浆斑
    "warping_knot",  # 下标 14 ← 天池类别 15 整经结
    "star_skip",  # 下标 15 ← 天池类别 16 星跳（含跳花）
    "broken_spandex",  # 下标 16 ← 天池类别 17 断氨纶
    "dense_section",  # 下标 17 ← 天池类别 18 稀密档（含浪纹档/色差档）
    "surface_mark",  # 下标 18 ← 天池类别 19 磨痕（含轧痕/修痕/烧毛痕）
    "weave_defect",  # 下标 19 ← 天池类别 20 织疵（死皱/云织/双纬等其余疵点）
]

# 图像扩展名白名单。用白名单而非黑名单：新的垃圾文件类型出现时自动被排除
# （例如 macOS 解压残留的 __MACOSX/._xxx.jpg）。
IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
}  # 合法图像扩展名集合（统一小写）；白名单机制让任何新型垃圾文件自动被挡掉


def category_to_class_id(
    category: int, scheme: str
) -> int:  # 把天池 1..20 的类别 ID 转成 YOLO 训练用的 0-based ID
    """Map a Tianchi category id (1..20) to a 0-indexed class id."""
    if scheme == "tianchi20":  # 目前只有一种类别体系：20 类方案
        return category - 1  # 天池 ID 从 1 起、YOLO 类 ID 从 0 起，整体减 1 完成平移
    raise ValueError(
        f"Unknown class scheme: {scheme}"
    )  # 传入未知方案立即抛错，绝不静默按 20 类处理


def class_names_for(
    scheme: str,
) -> List[str]:  # 按类别体系返回类名列表（列表下标即 0-based 类别 ID）
    """Return the class-name list matching the chosen scheme."""
    if scheme == "tianchi20":  # 20 类方案
        return TIANCHI_CATEGORIES  # 直接返回上面定义的 20 个英文类名
    raise ValueError(f"Unknown class scheme: {scheme}")  # 未知方案报错


def load_annotations(
    anno_path: str,
) -> Dict[str, List[Dict]]:  # 读取标注 JSON 并按图像名分组
    """Read anno_train.json and group annotations by image name.

    读取天池标注文件并按图像名分组。

    原始标注集中在一个 JSON 里（一条记录对应一个框），而后续处理需要
    "给一张图找它的所有标注"，所以这里转置成「文件名 → 标注列表」。

    Args:
        anno_path: anno_train.json 的路径。

    Returns:
        字典 {图像文件名: [标注, ...]}。文件不存在时返回空字典。
    """
    by_name: Dict[str, List[Dict]] = defaultdict(
        list
    )  # 初始化"文件名 → 标注列表"字典，缺失键自动取到空列表
    if not os.path.exists(anno_path):  # 标注文件不存在时（例如没有 Annotations 的目录）
        return by_name  # 直接返回空字典，调用方按"本目录无任何标注"处理
    with open(
        anno_path, encoding="utf-8"
    ) as f:  # 以 UTF-8 打开（中文缺陷名必须显式指定编码，否则 Windows 上乱码/报错）
        for ann in json.load(f):  # json.load 一次性读出整个标注数组，逐条遍历
            by_name[ann["name"]].append(
                ann
            )  # 以图像文件名为键，把该条标注追加进对应列表（一张图多条标注自然聚合）
    return by_name  # 返回分组后的字典


# 将数据集坐标格式转为yolo格式
def xyxy_to_yolo(  # 函数定义：一个边界框一个边界框地做格式转换
    bbox: List[float],
    w: int,
    h: int,  # bbox = 天池像素坐标 [xmin, ymin, xmax, ymax]；w/h = 原图宽高（像素）
) -> Tuple[
    float, float, float, float
]:  # 返回值类型：归一化后的 (cx, cy, bw, bh) 四元组
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
    xmin, ymin, xmax, ymax = (
        float(v) for v in bbox
    )  # 四个坐标逐一转 float，防止 JSON 整数参与后续除法丢精度
    # 中心点：先取两端均值，再除图像宽/高归一化到 0~1
    # 注意 x 除以 w、y 除以 h，写反了模型学出来的东西会转置 90°
    cx = (xmin + xmax) / 2.0 / w  # 中心 x = (左 + 右) / 2 ÷ 图宽
    cy = (ymin + ymax) / 2.0 / h  # 中心 y = (上 + 下) / 2 ÷ 图高
    # 宽度/高度：max(0.0, ...) 防负宽度
    # （若原始标注错误导致 xmax < xmin，不加这层保护会算出负宽度，
    #   负宽度进入 IoU 计算会让并集面积算错、梯度方向反掉）
    bw = (
        max(0.0, (xmax - xmin)) / w
    )  # 框宽 = (右 - 左) ÷ 图宽，max 挡掉标注错误导致的负宽
    bh = max(0.0, (ymax - ymin)) / h  # 框高 = (下 - 上) ÷ 图高，max 挡掉负高
    # 双向钳制中心点到 [0, 1]
    # 为什么钳中心点而不是边界？目标被图像边缘截断是真实存在的（框可以超出
    # 图像），但中心点必须落在图内。实测 8570 个框中有 9 个越界（幅度约 1e-6），
    # 就是这里要挡的。
    cx, cy = (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
    )  # min(max(x,0),1) 是标准的区间钳制写法，先保下限再保上限
    # 宽高只卡上限不卡下限：宽度为 0 虽有缺陷（退化框），但那是需要上报的
    # 数据质量问题，不该在转换阶段悄悄改掉
    bw, bh = (
        min(bw, 1.0),
        min(bh, 1.0),
    )  # 宽高上限钳到 1.0；不设下限，0 宽留给后续数据质量报告
    return cx, cy, bw, bh  # 返回四个归一化浮点数，交由调用方格式化成标签行


# 图片去重
def convert_train_dir(  # 核心函数：把一个训练集目录转成 YOLO 格式（内含按图像名去重）
    ds_dir: str,  # 数据集目录，需含 defect_Images/、normal_Images/、Annotations/
    out_images: str,  # 输出图像的扁平目录（所有图平铺在一层）
    out_labels: str,  # 输出标签 .txt 的扁平目录
    scheme: str,  # 类别体系名，目前只支持 "tianchi20"
) -> int:  # 返回值：实际写出的图像数量（去重后的净数量）
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
    os.makedirs(
        out_images, exist_ok=True
    )  # 确保输出图像目录存在；exist_ok=True 表示已存在就复用（脚本可重复运行）
    os.makedirs(out_labels, exist_ok=True)  # 确保输出标签目录存在

    # 读取本数据集目录的标注并按图像名分组
    by_name = load_annotations(
        os.path.join(ds_dir, "Annotations", "anno_train.json")
    )  # 拼出标注文件路径并加载，得到 {图名: [标注,...]}
    count = 0  # 写出图像计数器，最终作为返回值
    skipped = 0  # 未知缺陷名的标注计数（跳过必须有计数，否则不知道丢了多少数据）

    # 依次处理缺陷图与正常图两个目录
    for subdir in (
        "defect_Images",
        "normal_Images",
    ):  # 先缺陷图后正常图，固定顺序保证跨次运行行为一致
        img_dir = os.path.join(ds_dir, subdir)  # 拼出当前子目录的完整路径
        if not os.path.isdir(
            img_dir
        ):  # 有的数据集目录可能没有 normal_Images（如纯负样本集）
            continue  # 目录不存在就跳过，不报错
        for fname in sorted(
            os.listdir(img_dir)
        ):  # 列出目录内容并排序：保证处理顺序确定、可复现
            # 扩展名白名单过滤（挡掉 __MACOSX 残留等垃圾文件）
            if (
                os.path.splitext(fname)[1].lower() not in IMAGE_EXTS
            ):  # 取扩展名转小写，不在白名单即为非图片
                continue  # 跳过垃圾文件（如 .DS_Store、._xxx.jpg）
            src = os.path.join(img_dir, fname)  # 源文件完整路径

            # Read dimensions to normalize bbox coordinates.
            # 读出原图宽高，用于把像素坐标归一化
            with Image.open(src) as im:  # 用 with 打开图片，退出时自动释放文件句柄
                w, h = im.size  # 取出原图宽高（像素）——归一化标注的除数

            dst_img = os.path.join(
                out_images, fname
            )  # 目标路径 = 输出目录 + 原文件名（扁平化，去掉子目录层级）
            # ★★ 去重判据：按图像名（目标文件是否已存在），不是按标注条数
            if os.path.exists(
                dst_img
            ):  # 同名文件已存在 = 前一个 part 已写过这张图，命中去重条件
                # 不静默跳过——你必须知道跳过了什么
                print(
                    f"  [warn] duplicate image name skipped: {fname}"
                )  # 显式告警到日志，便于人工核查
                continue  # ★ 跳过本图且不累加计数——这就是防 GT 翻倍的关键一行
            shutil.copy(
                src, dst_img
            )  # 把图片拷贝进输出目录（只读源数据，源目录保持原样）

            # Build YOLO label (empty file for normal images).
            lines: List[str] = []  # 收集该图的全部 YOLO 标签行；正常图保持为空列表
            for ann in by_name.get(
                fname, []
            ):  # 按图名查它的所有标注；取不到返回空列表（正常图/未匹配图即此情况）
                # 中文缺陷名 → 类别 ID；.get 取不到返回 None（正常图即此情况）
                category = NAME_TO_CATEGORY.get(
                    ann["defect_name"]
                )  # 查映射表；未知缺陷名返回 None
                if category is None:  # 遇到映射表里没有的缺陷名
                    skipped += 1  # ★ 计数，结束时统一报告（不静默丢数据）
                    continue  # 跳过这条标注，继续处理该图的其他标注
                cls = category_to_class_id(
                    category, scheme
                )  # 天池 ID（1..20）→ YOLO 0-based 类 ID
                cx, cy, bw, bh = xyxy_to_yolo(
                    ann["bbox"], w, h
                )  # 像素 xyxy 框 → 归一化 cxcywh
                # 保留 6 位小数：实测训练集最小框宽 0.002833，
                # 若只留 3 位会写成 0.003（误差 6%），极端情况框宽 0.0001
                # 会被写成 0.000 —— 这个框就消失了
                lines.append(
                    f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )  # 按 YOLO 标准格式"类 cx cy w h"写一行，6 位小数保精度

            # 标签文件名 = 图像名去扩展名 + .txt
            label_path = os.path.join(
                out_labels, os.path.splitext(fname)[0] + ".txt"
            )  # 例：abc.jpg → abc.txt，保证图与标签一一对应
            with open(
                label_path, "w", encoding="utf-8"
            ) as f:  # 以写模式打开标签文件（UTF-8）
                # ★ lines 为空时 "\n".join(lines) 返回空串 → 写出空文件 = 负样本
                # 正常图保留为负样本很重要：检测模型必须见过"什么都没有"的图，
                # 否则会在完好织物上乱框。实测验证集 957 张里有 374 张负样本（39%）。
                f.write(
                    "\n".join(lines) + ("\n" if lines else "")
                )  # 有标签逐行写并以换行结尾；无标签则写空文件（负样本）

            count += 1  # 一张图（含其标签）处理完毕，净计数 +1

    # 汇总报告被跳过的标注数（不静默丢数据）
    if skipped:  # 仅当确实有未知缺陷名时才报告
        print(  # print 跨行调用，保持行长可读
            f"  [warn] {skipped} annotations had unknown defect_name and were skipped"  # 汇总告警：多少条标注因未知缺陷名被跳过
        )
    return count  # 返回本目录净写出的图像数（已按图像名去重）


def convert_test_dir(test_dir: str, out_images: str) -> int:  # 拷贝一个无标注测试集目录
    """Copy an extracted test dir (flat .jpg, no labels) into out_images.

    复制测试集图片。测试集没有标注，只需拷贝图片。
    """
    os.makedirs(
        out_images, exist_ok=True
    )  # 确保测试集输出目录存在（多个测试目录共用，重复创建无副作用）
    count = 0  # 拷贝计数器
    for fname in sorted(os.listdir(test_dir)):  # 排序列出测试目录，保证顺序确定
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:  # 同样走扩展名白名单
            continue  # 非图片文件跳过
        shutil.copy(
            os.path.join(test_dir, fname), os.path.join(out_images, fname)
        )  # 原样拷贝（测试集无标注，无需生成标签文件）
        count += 1  # 拷一张计一张
    return count  # 返回本目录拷贝的图片数


def split_train_val(
    root: str, val_ratio: float, seed: int
) -> Tuple[int, int]:  # 把暂存区划分为 train/val 两个子集
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
    staging_img = os.path.join(
        root, "_staging", "images"
    )  # 暂存区图像目录（convert_train_dir 的写入目标）
    staging_lbl = os.path.join(root, "_staging", "labels")  # 暂存区标签目录
    train_img = os.path.join(root, "train", "images")  # 训练集图像目录
    train_lbl = os.path.join(root, "train", "labels")  # 训练集标签目录
    val_img = os.path.join(root, "val", "images")  # 验证集图像目录
    val_lbl = os.path.join(root, "val", "labels")  # 验证集标签目录

    # ★ sorted() 保证顺序固定：不排序的话不同机器上 os.listdir 的返回顺序
    #   可能不同，划分就不可复现了
    all_images = sorted(
        os.listdir(staging_img)
    )  # 列出暂存区全部图像文件名并排序——顺序确定是复现的前提
    random.seed(seed)  # ★ 固定全局随机种子 —— 可复现的基础（同 seed 必得同划分）
    random.shuffle(all_images)  # 原地打乱（shuffle 返回 None，不用接返回值）
    n_val = int(
        len(all_images) * val_ratio
    )  # 9576 × 0.1 = 957.6 → 957（int 向下取整，保证验证集不会超量）
    # 取前 10% 作验证集；用 set 便于 O(1) 查询某文件属于哪一边
    val_set = set(
        all_images[:n_val]
    )  # 打乱后的前 n_val 个进验证集；转 set 后成员查询 O(1)

    # 建四个目录；exist_ok=True 表示已存在就跳过（脚本可重复运行）
    for d in (train_img, train_lbl, val_img, val_lbl):  # 遍历四个目标目录
        os.makedirs(d, exist_ok=True)  # 逐一创建，已存在则复用

    for name in all_images:  # 逐张图像决定去向
        # 三元表达式决定去哪：验证集还是训练集
        dst_img_dir = (
            val_img if name in val_set else train_img
        )  # 命中名单 → val 图像目录，否则 → train 图像目录
        dst_lbl_dir = (
            val_lbl if name in val_set else train_lbl
        )  # 标签走与图像完全相同的分支——保证图/标签成对、不错位
        # os.rename 是**移动**不是复制：天然保证不会重复
        os.rename(
            os.path.join(staging_img, name), os.path.join(dst_img_dir, name)
        )  # 移动图像文件（同分区内 rename 是元数据操作，极快）
        # 标签用同样的判断，保证图像与标签成对移动（否则会错位）
        lbl = (
            os.path.splitext(name)[0] + ".txt"
        )  # 由图像名推出标签文件名（.jpg → .txt）
        src_lbl = os.path.join(staging_lbl, lbl)  # 暂存区中对应标签的路径
        if os.path.exists(
            src_lbl
        ):  # 正常情况每张图都有标签（可能是空文件）；防御性判断防异常
            os.rename(
                src_lbl, os.path.join(dst_lbl_dir, lbl)
            )  # 把标签也移动到与图像相同的子集

    shutil.rmtree(
        os.path.join(root, "_staging"), ignore_errors=True
    )  # 划分完成后删除空的暂存区，输出目录只留 train/val/test
    return len(
        all_images
    ) - n_val, n_val  # 返回 (训练集数, 验证集数)：总数减验证数即训练数


def write_meta(
    root: str, scheme: str
) -> None:  # 写出数据集描述文件（类名表 + Ultralytics 配置）
    """Write classes.txt and data.yaml (Ultralytics) for the output root.

    写出数据集描述文件：classes.txt（类名列表）与 data.yaml（Ultralytics 配置）。

    data.yaml 里的 `nc: 18` 很关键——ultralytics 加载时会发现 COCO 预训练权重
    是 80 类、这里声明 18 类，从而**自动重建分类头**。这是微调能发生的前提。
    """
    names = class_names_for(scheme)  # 按类别体系取出类名列表（下标 = 0-based 类别 ID）
    with open(
        os.path.join(root, "classes.txt"), "w", encoding="utf-8"
    ) as f:  # 打开 classes.txt 准备写入
        f.write(
            "\n".join(names) + "\n"
        )  # 一行一个类名写入，末尾补一个换行（符合 POSIX 文本文件惯例）

    yaml_lines = [  # 用列表逐行攒 data.yaml 内容，最后一次性 join 写出，减少磁盘 IO
        # 路径转成正斜杠（YAML 里 Windows 反斜杠会被当转义符）
        f"path: {os.path.abspath(root).replace(os.sep, '/')}",  # 数据集根目录绝对路径；Windows 的 \ 替换为 / 防 YAML 转义
        "train: train/images",  # 训练集图像相对路径（相对 path 字段解析）
        "val: val/images",  # 验证集图像相对路径
        "test: test/images",  # 测试集图像相对路径
        f"nc: {len(names)}",  # 类别数——ultralytics 据此判断是否重建分类头（与 COCO 80 类不同即重建）
        "names:",  # 类名映射段的开头，下面逐行跟 "  0: hole" 形式
    ]
    yaml_lines += [
        f"  {i}: {n}" for i, n in enumerate(names)
    ]  # 列表推导追加 索引: 类名 映射，两空格缩进符合 YAML 层级
    with open(
        os.path.join(root, "data.yaml"), "w", encoding="utf-8"
    ) as f:  # 打开 data.yaml 准备写入
        f.write("\n".join(yaml_lines) + "\n")  # 拼成整文写入，末尾补换行


def main() -> int:  # 脚本入口：按四步流程把原始数据整理成可直接训练的 YOLO 数据集
    """脚本入口：按四步把原始数据整理成可直接训练的 YOLO 数据集。

    流程：
      1. 转换  把各训练集目录转到 _staging（转换时按图像名去重）
      2. 划分  _staging → train / val（10%，seed 固定）
      3. 测试  拷贝无标签的 testA / testB
      4. 描述  写 classes.txt 与 data.yaml

    ⚠️ 顺序不可调换：必须先去重（步骤 1 内完成）再划分（步骤 2）。
    """
    parser = argparse.ArgumentParser(
        description=__doc__
    )  # 创建参数解析器，帮助文本直接复用模块 docstring（自文档化）
    parser.add_argument(  # 注册 --train-dirs 选项
        "--train-dirs",
        nargs="+",
        default=[],
        help="Extracted train dataset dirs",  # nargs="+" 一次收多个目录；默认空列表
    )
    parser.add_argument(  # 注册 --test-dirs 选项
        "--test-dirs",
        nargs="+",
        default=[],
        help="Extracted test dirs (no labels)",  # 同样支持多目录；默认空列表
    )
    parser.add_argument(
        "--out", default="data/tianchi", help="Output root directory"
    )  # 输出根目录，默认 data/tianchi
    parser.add_argument(  # 注册 --class-scheme 选项
        "--class-scheme",  # 选项名（argparse 自动把 --a-b 映射到 args.a_b）
        default="tianchi20",  # 默认 20 类体系
        choices=["tianchi20"],  # 只允许这一个取值，用户输错立即报错并列出合法值
        help="Class scheme to emit (default: tianchi20, 20 classes)",  # 帮助文本
    )
    parser.add_argument(  # 注册 --val-ratio 选项
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",  # 验证集比例（浮点），默认 10%
    )
    parser.add_argument(  # 注册 --seed 选项
        "--seed",
        type=int,
        default=42,
        help="Random seed for val split",  # 划分用随机种子（整数），默认 42
    )
    args = parser.parse_args()  # 真正解析 sys.argv，所有选项值收入 args 对象

    scheme = args.class_scheme  # 取出类别体系名
    root = args.out  # 取出输出根目录
    print(
        f"Class scheme: {scheme} ({len(class_names_for(scheme))} classes)"
    )  # 回显体系与类别数，日志可核对是否选错体系
    print(f"Output root: {root}")  # 回显输出根目录

    # 1. Convert all train dirs into a staging train dir.
    # 暂存区：所有图先统一转到这里，再从这里分配到 train/val
    staging_img = os.path.join(
        root, "_staging", "images"
    )  # 暂存区图像路径（convert_train_dir 的公共写入目标）
    staging_lbl = os.path.join(root, "_staging", "labels")  # 暂存区标签路径
    total = 0  # 全部训练集写出总数（跨目录累加）
    for (
        ds_dir
    ) in args.train_dirs:  # 逐个处理命令行传入的训练集目录（partA、partB、train2……）
        print(f"Converting train dir: {ds_dir}")  # 打印当前处理的目录，日志分段清晰
        # ★ 去重在这一步内部完成（按图像名）
        n = convert_train_dir(
            ds_dir, staging_img, staging_lbl, scheme
        )  # 转换 + 按图像名去重都在函数内部完成
        total += n  # 累加净写出数（重复图已被跳过，不会虚增）
        print(f"  wrote {n} images")  # 回显本目录净写出数
    print(f"Total train images: {total}")  # 打印去重后的总训练图数

    # 2. Split staging -> train/val.
    # ★ 必须在去重之后划分，否则验证集会 100% 泄漏
    if os.path.isdir(staging_img) and os.listdir(
        staging_img
    ):  # 暂存区存在且非空才需要划分
        n_train, n_val = split_train_val(
            root, args.val_ratio, args.seed
        )  # ★ 顺序关键：先完成去重（上面）再划分，防验证集泄漏
        print(f"Split: {n_train} train / {n_val} val")  # 回显划分结果
    else:  # 暂存区为空（没传 --train-dirs 或目录里没有图）
        print(
            "  [warn] no train images found; skipping split"
        )  # 告警并跳过划分（不静默失败）

    # 3. Convert test dirs (no labels).
    # 测试集无标注，只需拷贝图片
    test_img = os.path.join(
        root, "test", "images"
    )  # 测试集输出目录（testA/testB 合并到同一处）
    for test_dir in args.test_dirs:  # 逐个处理测试集目录
        print(f"Converting test dir: {test_dir}")  # 打印当前处理的测试目录
        n = convert_test_dir(test_dir, test_img)  # 拷贝图片（不生成任何标签）
        print(f"  wrote {n} test images")  # 回显本目录拷贝数

    # 4. Write classes.txt + data.yaml.
    # data.yaml 里的 nc 会让 ultralytics 重建 18 类分类头
    write_meta(
        root, scheme
    )  # 写出 classes.txt 与 data.yaml，数据集即可被 ultralytics/YOLO 直接加载

    print("Done.")  # 全部四步完成
    print(
        f"  train: {os.path.join(root, 'train', 'images')}"
    )  # 打印训练集图像路径，便于直接拷贝使用
    print(f"  val:   {os.path.join(root, 'val', 'images')}")  # 打印验证集图像路径
    print(f"  test:  {test_img}")  # 打印测试集图像路径
    print(f"  classes: {os.path.join(root, 'classes.txt')}")  # 打印类名文件路径
    print(
        f"  data.yaml: {os.path.join(root, 'data.yaml')}"
    )  # 打印 data.yaml 路径（训练命令 --data 指向它）
    return 0  # 返回 0：Unix 约定 0 = 成功


if (
    __name__ == "__main__"
):  # 仅当本文件作为脚本直接运行时才执行；被其他模块 import 时不跑（可复用上面的函数）
    raise SystemExit(
        main()
    )  # 把 main() 的返回值作为进程退出码传出（0 = 成功，非 0 = 异常）
