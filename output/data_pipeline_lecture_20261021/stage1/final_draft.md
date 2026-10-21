# 逐行代码讲解：织物瑕疵检测项目核心实现

## 这份讲义怎么读

**每一段代码后面都跟一张"逐行解读"表**，把这一行在做什么、每个参数是什么含义、为什么这么写讲清楚。

表格三列的含义：

| 列 | 含义 |
|---|---|
| **行号** | 对应代码在源文件中的真实行号（可直接跳转核对）|
| **代码** | 该行的关键部分（省略号表示中间未展示的行）|
| **讲解** | 这一行的作用、参数含义、设计意图 |

**不需要提前会编程**——所有术语第一次出现时都会解释。所有代码都取自项目真实源码，行号可核对。

## 涉及的文件与行号坐标

| 主题 | 文件路径 | 行号范围 |
|------|---------|---------|
| 坐标格式转换 | `scripts/convert_tianchi_to_yolo.py` | L134–146 |
| 去重与负样本 | `scripts/convert_tianchi_to_yolo.py` | L180–206 |
| 训练/验证划分 | `scripts/convert_tianchi_to_yolo.py` | L229–257 |
| 可微 IoU | `backend/training/dimension_rewards.py` | L61–87 |
| 分类奖励 D02 | `backend/training/dimension_rewards.py` | L259–276 |
| 奖励注入 | `backend/training/reward_rtdetr.py` | L73–103 |
| 奖励计算与漏检处理 | `backend/training/reward_rtdetr.py` | L128–199 |
| 推理预处理 | `backend/inference/rtdetr_onnx.py` | L70–83 |
| 输出解析 | `backend/inference/rtdetr_onnx.py` | L102–146 |
| 推理引擎工厂 | `backend/inference/engine.py` | L355–404 |
| 标签加载 | `backend/training/dataset.py` | L159–186 |
| 自定义 collate | `backend/training/dataset.py` | L229–259 |

---

# 第一部分 数据清洗代码

## 1.1 坐标格式转换（`convert_tianchi_to_yolo.py` L134–146）

**要解决的问题**：天池给的标注是 `xyxy` 像素坐标（四个数字：左上角 x、左上角 y、右下角 x、右下角 y），YOLO 格式要的是 `cxcywh` 归一化坐标（中心点 x、中心点 y、宽、高，全部除以图像宽高缩放到 0~1）。两者不仅格式不同，**坐标原点也不同**——必须转。

```python
def xyxy_to_yolo(
    bbox: List[float], w: int, h: int
) -> Tuple[float, float, float, float]:
    """Convert xyxy pixel bbox to normalized (cx, cy, bw, bh)."""  # 文档字符串：说明这个函数做什么
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)  # 解包出 4 个坐标并统一转浮点（JSON 里可能是整数）
    cx = (xmin + xmax) / 2.0 / w  # 中心点 x：先取两端均值，再除图像宽归一化到 0~1
    cy = (ymin + ymax) / 2.0 / h  # 中心点 y：注意除的是图像高 h（与 x 用 w 区分，写反会转置）
    bw = max(0.0, (xmax - xmin)) / w  # 宽度：max(0.0, ...) 防负宽度（标注错误时 xmax < xmin）
    bh = max(0.0, (ymax - ymin)) / h  # 高度：同样防负
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)  # 双向钳制中心点到 [0,1]：实测有框溢出 1e-6
    bw, bh = min(bw, 1.0), min(bh, 1.0)  # 宽高只卡上限不卡下限：宽度为 0 是需上报的数据质量问题
    return cx, cy, bw, bh  # 返回归一化后的四个数
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L134–135 | `def xyxy_to_yolo(bbox, w, h)` | 函数签名。`bbox` 是输入的四个坐标；`w`/`h` 是**原图的宽和高**（像素），由调用方从图像文件读出传入 |
| L136 | `) -> Tuple[float, float, float, float]:` | 返回类型标注，表示返回 4 个浮点数组成的元组。**这是类型提示**，不写也能跑，但写了 IDE 和类型检查器能帮你抓错 |
| L137 | `"""Convert xyxy pixel bbox..."""` | 文档字符串，说明这个函数做什么 |
| L138 | `xmin, ymin, xmax, ymax = (float(v) for v in bbox)` | **解包赋值**。把 `bbox` 列表里 4 个元素依次取出，同时用 `float()` 转成浮点数。写成生成器表达式是为了避免先建中间列表。`float()` 这步不能省——原始 JSON 里的数字可能是整数，除法时需要浮点语义 |
| L139 | `cx = (xmin + xmax) / 2.0 / w` | **算中心点**。`(xmin + xmax) / 2.0` 是中心点的像素坐标（`2.0` 保证是浮点除法），再 `/ w` 除以图像宽，归一化到 0~1 |
| L140 | `cy = (ymin + ymax) / 2.0 / h` | 同上，纵坐标用图像高 `h` 归一化。**注意 x 用 w、y 用 h**，写反了模型学出来的东西会转置 90° |
| L141 | `bw = max(0.0, (xmax - xmin)) / w` | 算宽度。`max(0.0, ...)` 是**防负宽度**：如果原始标注错误导致 `xmax < xmin`，宽度会算成负数。负宽度进入 IoU 计算会让并集面积算错、梯度方向反掉。**这是防御性代码，实测确有 9 个框越界** |
| L142 | `bh = max(0.0, (ymax - ymin)) / h` | 同上，高度方向防负 |
| L143 | `cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)` | **双向钳制**。`max(x, 0.0)` 先确保不小于 0，`min(..., 1.0)` 再确保不大于 1。**为什么要钳制中心点而不是边界**？因为目标被图像边缘截断是真实存在的（框可以超出图像），但**中心点必须落在图内**。实测有框的 `R=1.000001` 溢出 1e-6，就是这里要挡的 |
| L144 | `bw, bh = min(bw, 1.0), min(bh, 1.0)` | 宽高也做上限钳制。**注意这里只卡上限不卡下限**——宽度为 0 虽有缺陷（退化框），但那是需要上报的数据质量问题，不该在转换阶段悄悄改掉 |
| L145 | `return cx, cy, bw, bh` | 返回归一化后的四个数 |

### 为什么归一化

**让标注与图像分辨率解耦**。同一份标签在 2446×1000 和 1280×1280 上都能用，模型才能泛化到不同尺寸的输入。

### 配套：写入时的精度

```python
lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")  # 保留 6 位小数：实测最小框宽 0.002833，3 位会丢小目标
```

| 代码 | 讲解 |
|------|------|
| `:.6f` | 保留 6 位小数。**实测训练集最小框宽是 0.002833**——若只留 3 位会写成 0.003（误差 6%），若某框宽是 0.0001 则 3 位直接写成 0.000，**这个框就消失了**。6 位是精度与文件体积的平衡点 |

---

## 1.2 去重与负样本（`convert_tianchi_to_yolo.py` L180–206）

**要解决的问题**：partA 和 partB 的标注文件 md5 完全相同（`8ca9efa45e3f4ade805a12019a5880a3`），是同一个文件被拷了两遍。不去重的话每张图的标注会写两遍，**GT 框数量直接翻倍**。

```python
            src = os.path.join(img_dir, fname)  # 拼出源图像路径（join 会自动处理 Windows 分隔符）

            # Read dimensions to normalize bbox coordinates.
            with Image.open(src) as im:  # 上下文管理器：打开图像并保证用完自动关闭
                w, h = im.size  # 读出图像宽高，顺序是（宽, 高）

            dst_img = os.path.join(out_images, fname)  # 拼出目标路径（扁平化输出目录）
            # ★ 去重判据：目标文件已存在 → 按图像名去重
            if os.path.exists(dst_img):
                print(f"  [warn] duplicate image name skipped: {fname}")  # 打印警告：绝不静默跳过
                continue  # 负样本（正常图）跳过，不进 batch
            shutil.copy(src, dst_img)  # 复制图像到输出目录

            # Build YOLO label (empty file for normal images).
            lines: List[str] = []  # 初始化标注行列表（正常图最终保持为空）
            for ann in by_name.get(fname, []):  # 遍历本图标注；取不到返回空列表（正常图即此情况）
                category = NAME_TO_CATEGORY.get(ann["defect_name"])  # 中文缺陷名 → 类别 ID 映射，找不到返回 None
                if category is None:  # 遇到未知缺陷名
                    skipped += 1  # ★ 计数：跳过必须有计数，结束时统一报告
                    continue  # 负样本（正常图）跳过，不进 batch
                cls = category_to_class_id(category, scheme)  # 天池 1-based 类别 ID → 训练用 0-based
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L181 | `src = os.path.join(img_dir, fname)` | 拼出源图像的完整路径。`os.path.join` 比字符串 `+` 安全——它会自动处理 Windows 的反斜杠分隔符 |
| L183 | `with Image.open(src) as im:` | **上下文管理器**语法。打开图像的同时保证用完自动关闭（即使中途抛异常）。`with` 块结束时会调用 `im.close()` |
| L184 | `w, h = im.size` | 读出图像宽高。`im.size` 是 `(宽, 高)` 的元组。**注意顺序是宽在前**——传参给 `xyxy_to_yolo(bbox, w, h)` 时别弄反 |
| L186 | `dst_img = os.path.join(out_images, fname)` | 拼出目标路径（扁平化后的输出目录）|
| L187 | `if os.path.exists(dst_img):` | **去重的判据：目标文件已存在**。这意味着按**图像名**去重，而不是按标注条数 |
| L188 | `print(f"  [warn] duplicate image name skipped: {fname}")` | 打印警告。**不静默跳过**——你必须知道跳过了什么。f-string 里 `{fname}` 是变量插值 |
| L189 | `continue` | **跳过本图，不累加**。这是去重的关键：partB 里的同名图在这里被挡掉，不会产生第二份标注 |
| L190 | `shutil.copy(src, dst_img)` | 复制图像到输出目录。用 `copy` 而非 `move` 是因为 `shutil` 还要给划分阶段留余地 |
| L194 | `lines: List[str] = []` | 初始化标注行列表。`List[str]` 是类型标注：这个列表装的是字符串。**正常图片最终会保持为空** |
| L195 | `for ann in by_name.get(fname, []):` | 遍历这张图的所有标注。`by_name` 是"文件名 → 标注列表"的字典；`.get(fname, [])` 表示**取不到就返回空列表**（而不是报错）——正常图就是取不到的情况 |
| L196 | `category = NAME_TO_CATEGORY.get(ann["defect_name"])` | 查中文缺陷名到类别 ID 的映射。`.get()` 找不到返回 `None` |
| L197 | `if category is None:` | 判断是否遇到未知缺陷名 |
| L198 | `skipped += 1` | **计数**。这是数据清洗的关键习惯：跳过必须有计数，结束时统一报告，否则你永远不知道丢了多少数据 |
| L199 | `continue` | 跳过这条无法识别的标注，继续处理下一条 |
| L200 | `cls = category_to_class_id(category, scheme)` | 把 1-based 的天池类别 ID 转成 0-based 的训练类别 ID |

### 负样本的写法（同一函数 L202–206）

```python
            label_path = os.path.join(out_labels, os.path.splitext(fname)[0] + ".txt")  # 标签文件名 = 图像名去扩展名 + .txt
            with open(label_path, "w", encoding="utf-8") as f:  # 写模式打开；显式 utf-8 防止 Windows 下中文乱码
                f.write("\n".join(lines) + ("\n" if lines else ""))  # ★ lines 为空时 join 返回空串 → 写出空文件 = 负样本
```

| 行号 | 代码 | 讲解 |
|------|------|------|
| L202 | `os.path.splitext(fname)[0]` | 拆文件名去掉扩展名。`splitext` 返回 `("名字", ".jpg")`，取 `[0]` 得到不带扩展名的部分 |
| L203 | `open(label_path, "w", encoding="utf-8")` | 以**写入**模式打开。`encoding="utf-8"` 显式指定编码——不写的话 Windows 上可能用 GBK，中文标签会乱码 |
| L204 | `"\n".join(lines)` | 把标注行用换行符连接。**当 `lines` 为空时，`join` 返回空字符串 `""`**——这就是负样本能自然变成空文件的原因 |
| L204 | `+ ("\n" if lines else "")` | 条件表达式：只有 `lines` 非空时才在末尾补换行符。**三元表达式**语法是 `A if 条件 else B` |

> ⚠️ **空标注文件是数据，不是缺失数据。** 实测验证集 957 张里有 **374 张是负样本（占 39%）**。检测模型必须见过"什么都没有"的图，否则会在完好织物上乱框——真实产线里大部分布匹是好的。

---

## 1.3 训练/验证划分（`convert_tianchi_to_yolo.py` L229–257）

**要解决的问题**：怎么把 9576 张图分成训练集和验证集，以及**为什么顺序不能反**。

```python
def split_train_val(root: str, val_ratio: float, seed: int) -> Tuple[int, int]:  # 划分函数：根目录、验证集比例、随机种子
    """Split the staging train dir into train/ and val/ by image name."""  # 文档字符串
    staging_img = os.path.join(root, "_staging", "images")
    staging_lbl = os.path.join(root, "_staging", "labels")
    train_img = os.path.join(root, "train", "images")
    train_lbl = os.path.join(root, "train", "labels")
    val_img = os.path.join(root, "val", "images")
    val_lbl = os.path.join(root, "val", "labels")

    all_images = sorted(os.listdir(staging_img))  # ★ sorted() 保证顺序固定，否则不同机器划分不可复现
    random.seed(seed)  # ★ 固定随机种子 —— 可复现的基础
    random.shuffle(all_images)  # 原地打乱（shuffle 返回 None，不用接返回值）
    n_val = int(len(all_images) * val_ratio)  # 算验证集张数：9576 × 0.1 = 957.6 → 957
    val_set = set(all_images[:n_val])  # 取前 10% 作验证集；用 set 便于 O(1) 查询归属

    for d in (train_img, train_lbl, val_img, val_lbl):
        os.makedirs(d, exist_ok=True)  # 建目录；exist_ok=True 表示已存在就跳过（可重复运行）

    for name in all_images:  # 遍历所有文件名
        dst_img_dir = val_img if name in val_set else train_img  # ★ 三元表达式：决定图像去哪个目录
        dst_lbl_dir = val_lbl if name in val_set else train_lbl  # 标签用同样判断，保证图像与标签成对移动
        os.rename(os.path.join(staging_img, name), os.path.join(dst_img_dir, name))  # os.rename 是移动不是复制，天然不重复
        lbl = os.path.splitext(name)[0] + ".txt"  # 算出对应的标签文件名
        src_lbl = os.path.join(staging_lbl, lbl)
        if os.path.exists(src_lbl):  # 标签文件可能不存在
            os.rename(os.path.join(src_lbl), os.path.join(dst_lbl_dir, lbl))  # 移动标签到对应目录

    shutil.rmtree(os.path.join(root, "_staging"), ignore_errors=True)  # 删除暂存区；ignore_errors=True 删不掉也不报错
    return len(all_images) - n_val, n_val  # 返回（训练数, 验证数）
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L229 | `def split_train_val(root, val_ratio, seed)` | 三个参数：根目录、验证集比例、随机种子。**返回 `(训练数, 验证数)`** |
| L231–236 | 拼出 6 个目录路径 | `_staging` 是**暂存区**——所有图先转换到这里，再从这里分配到 train/val。`train`/`val` 各需 images 和 labels 两个子目录 |
| L238 | `all_images = sorted(os.listdir(staging_img))` | 列出暂存区所有图像文件名。**`sorted()` 保证顺序固定**——不排序的话，不同机器上 `os.listdir` 的返回顺序可能不同，划分就不可复现了 |
| L239 | `random.seed(seed)` | **设置随机种子**。Python 的 `random` 是伪随机数生成器，指定种子后序列完全确定。这是"可复现"的基础 |
| L240 | `random.shuffle(all_images)` | **原地打乱**列表顺序。`shuffle` 直接修改原列表并返回 `None`，所以不用接收返回值 |
| L241 | `n_val = int(len(all_images) * val_ratio)` | 算验证集张数。`int()` 向下取整。实测 `9576 × 0.1 = 957.6 → 957` |
| L242 | `val_set = set(all_images[:n_val])` | 取打乱后**前 10%** 作为验证集，存成 `set`（集合）以便 O(1) 查询"这个文件名在不在验证集里"。切片 `[0:n_val]` 取前 n_val 个 |
| L244–245 | `os.makedirs(d, exist_ok=True)` | 批量创建 4 个目录。`exist_ok=True` 表示"已存在就跳过"而不是报错——这样脚本可以重复运行 |
| L247 | `for name in all_images:` | 遍历所有文件名 |
| L248 | `dst_img_dir = val_img if name in val_set else train_img` | **三元表达式**决定图像去哪。在验证集里就去 `val_img`，否则去 `train_img` |
| L249 | `dst_lbl_dir = val_lbl if name in val_set else train_lbl` | 标签目录用**同样的判断**，保证图像和标签成对移动 |
| L250 | `os.rename(旧路径, 新路径)` | **`os.rename` 是移动不是复制**。天然保证不会重复，且跨盘时也能工作 |
| L251 | `lbl = os.path.splitext(name)[0] + ".txt"` | 算出对应的标签文件名（同名 + `.txt`）|
| L253 | `if os.path.exists(src_lbl):` | 标签文件可能不存在（比如负样本虽然也写了空文件，但保险起见判断一下）|
| L254 | `os.rename(...)` | 移动标签到对应目录 |
| L256 | `shutil.rmtree(..., ignore_errors=True)` | 删除暂存区。`ignore_errors=True` 表示删不掉也不报错 |
| L257 | `return len(all_images) - n_val, n_val` | 返回训练集数量和验证集数量 |

### ⚠️ 最关键的一点：顺序

整个流程是：

```
原始目录 → 逐个转换到 _staging（转换时按图像名去重）→ 划分 train/val → 删除 _staging
```

**先去重，再划分。** 如果顺序反了（先分别划分再合并），partA 的图可能进 train，而 partB 的同名图进 val——**验证集 100% 泄漏进训练集**（因为 partA/partB 标注完全相同，不是部分泄漏）。

而且这种错误**不报错、日志正常、指标还更好**，是典型的隐蔽实验错误。

---

# 第二部分 奖励机制代码

## 2.1 可微 IoU（`dimension_rewards.py` L61–87）

**要解决的问题**：IoU（交并比）是检测的核心指标，但标准写法用 Python 内置 `max()`/`min()`，**会切断计算图，梯度直接归零**。

```python
def differentiable_iou(
    pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, eps: float = 1e-6  # 预测框/真值框形状 (N,4)；eps 防除零
) -> torch.Tensor:
    px1, py1, pw, ph = pred_boxes.unbind(1)  # ★ unbind(1) 拆成 4 个 (N,)：是视图不复制，且梯度正常传递
    gx1, gy1, gw, gh = gt_boxes.unbind(1)  # 真值框同理（g 前缀 = ground truth）
    px2, py2 = px1 + pw, py1 + ph  # xywh → xyxy：算出预测框的右下角
    gx2, gy2 = gx1 + gw, gy1 + gh  # 真值框的右下角

    ix1 = torch.maximum(px1, gx1)  # ★ 必须用 torch.maximum！Python 内置 max 会切断计算图
    iy1 = torch.maximum(py1, gy1)  # 交集框上边界，同理
    ix2 = torch.minimum(px2, gx2)  # ★ 同理，不能用内置 min()
    iy2 = torch.minimum(py2, gy2)  # 交集框下边界

    inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)  # ★ 交集面积：clamp 把负面积截断到 0（两框不相交时）
    union = pw * ph + gw * gh - inter  # 并集面积：容斥原理（两面积之和减交集）
    return inter / (union + eps)  # IoU = 交集 ÷ 并集；+eps 防并集为 0
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L61–63 | `def differentiable_iou(pred_boxes, gt_boxes, eps=1e-6)` | `pred_boxes` 形状 `(N, 4)`，每行是 `[x, y, w, h]`；`gt_boxes` 同形状。`eps=1e-6` 是**数值稳定性项**——防止除以 0 |
| L75 | `px1, py1, pw, ph = pred_boxes.unbind(1)` | **`unbind(1)` 在第 1 维解绑**，把 `(N,4)` 拆成 4 个 `(N,)` 的张量。**为什么用 unbind 而不是切片 `[:, 0]`？** 因为 unbind 返回的是**视图（view）**，不复制数据、更高效，而且**梯度会正常传递**。切片也能传梯度，但 unbind 更简洁 |
| L76 | `gx1, gy1, gw, gh = gt_boxes.unbind(1)` | 同样拆开真实框。变量名 `g` 前缀表示 ground truth（真值）|
| L77 | `px2, py2 = px1 + pw, py1 + ph` | **把 xywh 转成 xyxy**。`px1 + pw` 就是右边界的 x 坐标。加法保持梯度 |
| L78 | `gx2, gy2 = gx1 + gw, gy1 + gh` | 真值框同理 |
| L80 | `ix1 = torch.maximum(px1, gx1)` | **交集框的左边界** = 两框左边界的较大者。⚠️ **必须用 `torch.maximum` 而不是 Python 内置 `max()`**——内置 max 返回的是 Python 数值，**计算图在这里就断了**，`loss.backward()` 会直接抛异常 |
| L81 | `iy1 = torch.maximum(py1, gy1)` | 交集框上边界，同理 |
| L82 | `ix2 = torch.minimum(px2, gx2)` | 交集框右边界 = 两框右边界的较小者。`torch.minimum` 同理不可用内置 `min()` |
| L83 | `iy2 = torch.minimum(py2, gy2)` | 交集框下边界 |
| L85 | `inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)` | **算交集面积**。`ix2 - ix1` 可能是负数（两框不相交），`clamp(min=0.0)` 把它截断到 0。**如果不 clamp，负面积会让 IoU 变成负数，梯度方向完全错误** |
| L86 | `union = pw * ph + gw * gh - inter` | **算并集面积**。用容斥原理：两框面积之和减去重复计算的交集部分 |
| L87 | `return inter / (union + eps)` | IoU = 交集 ÷ 并集。`+ eps` 防止并集为 0 时除零 |

### 实测验证（课堂现场可跑）

```python
import torch
from backend.training.dimension_rewards import differentiable_iou

g = torch.tensor([[0.5, 0.5, 0.2, 0.2]])

# 情况A：输入不带梯度
p1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
print('A →', differentiable_iou(p1, g).requires_grad)   # False

# 情况B：预测框带梯度（训练时的真实情况）
p2 = torch.tensor([[0.5, 0.5, 0.2, 0.2]], requires_grad=True)
out = differentiable_iou(p2, g)
print('B →', out.requires_grad)      # True
out.backward()
print(p2.grad)                       # [[0.0, 0.0, 6.294e-05, 6.294e-05]]
```

**本机实测输出**：

```
A 输入无requires_grad → False
B 预测框带梯度 → True
  反传后 p.grad = [[0.0, 0.0, 6.29425048828125e-05, 6.29425048828125e-05]]
```

**教学价值**：情况 A 的 `False` 解释了很多同学"写了可微函数却发现没梯度"的困惑——**不是函数不可微，是输入张量没带 `requires_grad=True`**。情况 B 的梯度出现在 w/h 方向，正是"框的位置还能被调整"的直接证据。

---

## 2.2 分类奖励 D02（`dimension_rewards.py` L259–276）

**要解决的问题**：怎么用一个可微的量代替"预测对不对"这个硬判断。

```python
    def compute_cls_reward(  # 方法：D02 分类维度奖励
        self,
        predictions: Dict[str, torch.Tensor],  # 预测字典：含 logits (N,C) 与 boxes (N,4)
        targets: List[Dict],  # 真值列表：每项含 boxes 与 labels
    ) -> torch.Tensor:
        aligned = self._aligned_logits_labels(predictions, targets)  # 对齐预测与真值，返回 (logits, labels)；失败返回 None
        if aligned is None:  # 对齐失败的兜底
            return torch.tensor(-1.0)  # 返回最差奖励 -1（D02 值域 [-1,1]）；常量张量不参与梯度但也不报错
        logits, labels = aligned  # 元组解包：logits (N,C)、labels (N,)
        probs = self._class_probs(logits)  # ★ 关键：logits→概率。softmax 或 sigmoid 按 head_type 分支
        p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # ★ 取真实类别上的概率：unsqueeze 加维→gather 按索引取→squeeze 压维
        return (2.0 * p_true - 1.0).mean()  # 映射到 [-1,1] 后对 batch 求均值：p_true=1→+1，=0→-1
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L259 | `def compute_cls_reward(self, predictions, targets)` | 方法定义在 `DimensionRewardComputer` 类里，`self` 是实例。`predictions` 是 `{"logits": (N,C), "boxes": (N,4)}` 字典；`targets` 是 `[{"boxes": [[...]], "labels": [int]}, ...]` 列表 |
| L260–262 | 类型标注 | `Dict[str, torch.Tensor]` 表示字典的值都是张量；`List[Dict]` 表示列表里每个元素是字典 |
| L267 | `aligned = self._aligned_logits_labels(predictions, targets)` | 调用内部方法做**对齐**：把预测和真值按索引一一配对，返回 `(logits, labels)`。对齐失败返回 `None` |
| L268 | `if aligned is None:` | 对齐失败的兜底分支 |
| L269 | `return torch.tensor(-1.0)` | **返回最差奖励 −1**。D02 的取值范围是 [−1, 1]，−1 表示"完全没分类信息"。注意这里返回的是**常量张量**，不参与梯度——但也不会报错 |
| L270 | `logits, labels = aligned` | **元组解包**，拿到 logits（形状 `(N, C)`，C=18）和 labels（形状 `(N,)`）|
| L271 | `probs = self._class_probs(logits)` | **关键一步**：把 logits 转成概率。内部按 `head_type` 分支——`"softmax"` 走 `torch.softmax(logits, dim=-1)`（单标签头）；`"sigmoid"` 走 `torch.sigmoid(logits)`（RT-DETR 的逐类 focal/vfl 头，每类独立二分类）|
| L272 | `p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)` | **取真实类别上的概率**。三个操作拆开看：<br>• `labels.unsqueeze(1)`：形状 `(N,)` → `(N,1)`，多一维<br>• `.gather(1, ...)`：沿第 1 维（类别维）按索引取值，每个样本取出自己真实类别那一列<br>• `.squeeze(1)`：`(N,1)` → `(N,)`，把长度 1 的维度压掉 |
| L273 | `return (2.0 * p_true - 1.0).mean()` | **映射到 [−1, 1] 并取均值**。`p_true=1`（完全确定）→ 奖励 +1；`p_true=0`（完全错）→ 奖励 −1。`.mean()` 对 batch 内所有样本取平均 |

### 为什么不用 argmax

| 硬决策 | 替换为 | 梯度 |
|------|--------|------|
| `argmax(logits)` | `softmax(logits)` 概率质量 | ✅ 有 |

**softmax + 交叉熵的梯度公式是 `∂loss/∂z = p − onehot(y)`**。当模型把真实类概率压到 0 时，梯度不是 0 而是接近 −1，模型仍能学习。

而 `argmax` 是硬决策：梯度为 0，**模型学不动**。这就是整个七维奖励设计的基础。

---

## 2.3 奖励注入损失（`reward_rtdetr.py` L73–103）

**要解决的问题**：怎么把奖励加进 ultralytics 的训练循环，**且不改一行框架代码**。

```python
    def forward(  # 重写父类 forward：签名必须与 RTDETRDetectionLoss 一致
        self,
        preds: tuple,  # 元组 (dec_bboxes, dec_scores)：形状 (L,B,N,4) 与 (L,B,N,C)
        batch: Dict[str, Any],  # 标注字典：bboxes / cls / gt_groups
        dn_bboxes: Optional[torch.Tensor] = None,  # 去噪查询框（DENOISING 是 DETR 的训练加速机制）
        dn_scores: Optional[torch.Tensor] = None,  # 去噪查询分数
        dn_meta: Optional[Dict[str, Any]] = None,  # 去噪元信息
    ) -> Dict[str, torch.Tensor]:
        loss = super().forward(  # ★ 调用父类：拿到原生三项损失（VFL分类 / L1回归 / GIoU）
            preds,
            batch,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        dec_bboxes, dec_scores = preds  # 从 preds 元组解出框和分数
        loss["loss_reward"] = self._reward_loss(dec_bboxes[-1], dec_scores[-1], batch)  # ★★ 核心一行：往字典加一个键，ultralytics 会自动求和
        return loss  # 返回带奖励项的损失字典
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L73–78 | `def forward(self, preds, batch, dn_bboxes=None, ...)` | **重写父类方法**。父类 `RTDETRDetectionLoss.forward` 的签名必须一致，否则框架调用时会报参数不匹配 |
| L74 | `preds: tuple` | 元组 `(dec_bboxes, dec_scores)`，形状分别是 `(L, B, N, 4)` 和 `(L, B, N, C)`。**L 是解码器层数（6）、B 是批大小、N 是 query 数** |
| L75 | `batch: Dict[str, Any]` | 标注字典，含 `bboxes`（真实框）、`cls`（真实类别）、`gt_groups`（每图有几个框）|
| L76–78 | `dn_bboxes / dn_scores / dn_meta` | **去噪查询**相关张量。DENOISING 是 DETR 的训练加速机制：给 query 加噪声构造难样本。这些参数原样传给父类，不参与奖励计算 |
| L79 | `) -> Dict[str, torch.Tensor]:` | 返回**损失字典**——这是关键，返回的是字典不是单个张量 |
| L94–100 | `loss = super().forward(preds, batch, dn_bboxes=..., ...)` | **调用父类实现**。`super()` 在 Python 里表示"父类"，这是标准的继承覆写写法。父类算出的三项损失（分类 VFL、框回归 L1、GIoU）原封不动拿到手 |
| L101 | `dec_bboxes, dec_scores = preds` | 从 `preds` 元组里解出框和分数 |
| L102 | `loss["loss_reward"] = self._reward_loss(dec_bboxes[-1], dec_scores[-1], batch)` | **本项目的核心一行**。`[-1]` 取最后一层解码器的输出（最精细的预测）。然后**往损失字典里加一个键** |
| L103 | `return loss` | 把带奖励项的字典返回 |

### 为什么这样就够了

ultralytics 的 `RTDETRDetectionModel.loss()` 内部会做类似：

```python
loss, loss_items = self.criterion(preds, batch)  # ultralytics 内部：criterion 返回字典
loss.sum().backward()  # ★ 字典每一项都被加进来 → 奖励自动参与反传，无需改框架
```

**criterion 返回的字典里每一项都会被加起来**。所以只要塞一个 `loss_reward` 键进去，奖励就自动被折进监督目标，**完全不用改动 ultralytics 的训练循环**。

最终损失：

```
L_total = L_giou + L_class(VFL) + L_bbox(L1) + β × (−R_scalar)
          └────── ultralytics 原生三项 ──────┘└──── 本项目注入 ────┘
```

`β` 是超参数：`β=0` 即退化为纯监督训练，作为 A/B 对照基线。

---

## 2.4 奖励计算与漏检处理（`reward_rtdetr.py` L128–199）

**要解决的问题**：奖励该算在哪些预测上？以及——**DETR 系模型里怎么定义"漏检"**。

```python
        match_indices = self.matcher(  # ★ 复用监督损失已算过的那次匈牙利匹配（同一 matcher 同一输入）
            pred_bboxes, pred_scores, batch["bboxes"], batch["cls"], batch["gt_groups"]  # 匹配器需要的 5 个参数
        )

        pred_logits: List[torch.Tensor] = []  # 收集参与奖励的预测分数
        pred_boxes: List[torch.Tensor] = []  # 收集参与奖励的预测框
        targets: List[Dict] = []  # 收集标注
        n_missed = 0  # 漏检计数，用于日志
        gt_groups = batch["gt_groups"]  # 每图各有几个 GT；GT 索引是全 batch 扁平的
        gt_offset = 0  # 当前图在扁平 GT 数组中的起始偏移
        for i, (src, dst) in enumerate(match_indices):  # 遍历每张图：i=图下标，src/dst=该图的匹配对
            for k in range(len(src)):
                pred_logits.append(pred_scores[i][src[k]])  # 取出被匹配 query 的分类分数 (C,)
                pred_boxes.append(pred_bboxes[i][src[k]])  # 取出该 query 的框 (4,)
                targets.append(
                    {
                        "boxes": [batch["bboxes"][dst[k]].tolist()],  # 对应真值框；.tolist() 切断梯度是对的（真值本就不该有梯度）
                        "labels": [int(batch["cls"][dst[k]])],  # 对应真值类别
                    }
                )
```

### 逐行解读（前半段）

| 行号 | 代码 | 讲解 |
|------|------|------|
| L132–134 | `match_indices = self.matcher(pred_bboxes, pred_scores, batch["bboxes"], batch["cls"], batch["gt_groups"])` | **复用监督损失已经算过的那次匈牙利匹配**。`self.matcher` 是继承来的匈牙利匹配器，`match_indices` 是 `[(src_indices, dst_indices), ...]` 列表——每图一对：`src` 是被选中的 query 编号，`dst` 是对应的 GT 编号。**这是本段最重要的设计**：奖励只评价监督信号已选中的检测对，不会另起一套标准 |
| L136 | `pred_logits: List[torch.Tensor] = []` | 收集参与奖励计算的预测分数 |
| L137 | `pred_boxes: List[torch.Tensor] = []` | 收集参与奖励计算的预测框 |
| L138 | `targets: List[Dict] = []` | 收集对应的真值。**三者是平行数组**：第 k 个元素互相对应 |
| L139 | `n_missed = 0` | 漏检计数，用于日志 |
| L140 | `gt_groups = batch["gt_groups"]` | 每张图各有几个 GT 框。**GT 索引是全 batch 扁平的**，所以下面要用 `gt_offset` 累加定位 |
| L141 | `gt_offset = 0` | 当前图在扁平 GT 数组中的起始偏移 |
| L142 | `for i, (src, dst) in enumerate(match_indices):` | 遍历每张图。`i` 是图在 batch 内的下标；`enumerate` 同时给出下标和值 |
| L144 | `for k in range(len(src)):` | 遍历这张图的每个匹配对 |
| L145 | `pred_logits.append(pred_scores[i][src[k]])` | **取出被匹配到的 query 的分类分数**。`pred_scores[i]` 取第 i 张图的 `(N, C)`，`[src[k]]` 再取第 src[k] 个 query → 得到 `(C,)` |
| L146 | `pred_boxes.append(pred_bboxes[i][src[k]])` | 同样取出该 query 的框 `(4,)` |
| L147–152 | `targets.append({...})` | 组装对应的真值。`batch["bboxes"][dst[k]]` 是第 dst[k] 个 GT 框；`.tolist()` 把张量转成 Python 列表（**这里切断梯度是对的**——真值本来就不该有梯度）|

### 漏检处理的认知纠错

```python
            if self.include_misses:  # 开关：是否把漏检也送进奖励
                gt_to_query = {  # 建立"GT编号→query编号"反查表
                    int(g): int(q)  # 字典推导式：g 是 GT 号，q 是 query 号
                    for g, q in zip(dst.tolist(), src.tolist(), strict=True)  # zip 配对；strict=True 长度不一致就报错
                }
                matched = {int(q) for q in src.tolist()}  # 已被占用的 query 集合
                for g in range(gt_offset, gt_offset + int(gt_groups[i])):  # 遍历本图的所有 GT
                    label = int(batch["cls"][g])  # 该 GT 的真实类别
                    class_ids = self._critical_ids_for(label)  # 查该类别属于哪个关键类集合（断经{9,15}/跳花{14,17}）
                    if class_ids is None:  # 非关键类
                        continue  # 负样本（正常图）跳过，不进 batch
                    if (
                        self._p_crit(pred_scores[i][gt_to_query[g]], class_ids)  # 该 GT 匹配到的 query 在关键类上的置信度
                        >= self.miss_threshold  # ★ 达到阈值说明模型其实认出来了，不算漏
                    ):
                        continue  # 负样本（正常图）跳过，不进 batch
                    spare = self._best_spare_query(i, class_ids, matched, pred_scores)  # ★ 找空闲 query：未被占用中关键类置信度最高的
                    if spare is None:  # 没有空闲 query
                        continue  # 负样本（正常图）跳过，不进 batch
```

| 行号 | 代码 | 讲解 |
|------|------|------|
| L154 | `if self.include_misses:` | 开关。是否启用"把漏检也送进奖励"的逻辑 |
| L155–158 | `gt_to_query = {g: q for g, q in zip(...)}` | **字典推导式**，建立"GT 编号 → query 编号"的反查表。`zip(dst.tolist(), src.tolist(), strict=True)` 把两个列表配对；`strict=True` 表示长度必须一致，不一致就报错（比默认的静默截断更安全）|
| L159 | `matched = {int(q) for q in src.tolist()}` | **已被占用的 query 集合**。集合推导式 `{... for ...}` |
| L160 | `for g in range(gt_offset, gt_offset + int(gt_groups[i])):` | 遍历这张图的所有 GT。范围用 `gt_offset` 定位到本图的 GT 段 |
| L161 | `label = int(batch["cls"][g])` | 该 GT 的真实类别 |
| L162 | `class_ids = self._critical_ids_for(label)` | 查出这个类别属于哪个"关键类集合"（断经类 `{9,15}` 或跳花类 `{14,17}`），不属于则返回 `None` |
| L163–164 | `if class_ids is None: continue` | **只关心关键类**，非关键类的漏检不额外处理 |
| L165–169 | `if self._p_crit(...) >= self.miss_threshold: continue` | **判断是否真的漏检**：该 GT 匹配到的 query 在关键类集合上的置信度若已达阈值，说明模型其实认出来了，不算漏 |
| L170 | `spare = self._best_spare_query(i, class_ids, matched, pred_scores)` | **找一个空闲 query**——在没被任何 GT 占用的 query 里，选关键类置信度最高的那个 |
| L171 | `if spare is None: continue` | 没有空闲 query 就跳过（query 数 ≥ GT 数时通常不会发生）|

### 为什么要这样处理

**这是全项目最有价值的一处认知纠错**：

| 模型类型 | "漏检"的含义 |
|---------|-------------|
| YOLO 系（one-to-many）| 没有匹配的框 |
| **DETR 系（one-to-one）** | **匹配上的那个 query 对关键类置信度太低** |

因为 DETR 的匈牙利匹配保证**每个 GT 都有 query 被分配**（query 数 N ≥ GT 数），所以"漏检"绝不是"没匹配上"。

代码的做法是：**拿一个空闲的 spare query 去指向这个漏检的 GT**，让奖励项去惩罚它。这等于强行给一个空闲 query 施加召回压力——**而不是让漏检变成零梯度**。

---

# 第三部分 推理与加载代码

## 3.1 推理预处理（`rtdetr_onnx.py` L70–83）

**要解决的问题**：怎么把一张 BGR 的 uint8 原图，变成能喂进网络的张量。

```python
    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:  # 方法：原始 BGR 图 → 网络输入张量（本引擎自己完成预处理）
        import cv2  # ★ 故意放在函数内：顶层不依赖 cv2，导 ONNX 等轻量操作不必装 opencv

        resized = cv2.resize(
            image_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR  # ★ 不等比拉伸成正方形（不是 letterbox 填充）
        )
        rgb = resized[:, :, ::-1]  # ★ BGR→RGB：负步长切片是视图，不复制内存
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC→CHW；转 float32；只除 255（不做 ImageNet 均值方差）
        return np.ascontiguousarray(tensor[None, ...])  # ★ 加批次维 + 保证内存连续（transpose 后非连续，ONNX Runtime 要求连续）
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L70 | `def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:` | 输入是**原始 BGR 格式**的图像数组，输出是网络输入张量。**注意：这个引擎自己完成预处理，不要求调用方先转好** |
| L76 | `import cv2` | **把 import 放在函数内部**是刻意的——这个文件顶层不依赖 cv2，只有真正做推理时才需要它。这样导ONNX 之类的轻量操作不必装 opencv |
| L78–80 | `cv2.resize(image_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)` | **拉伸成正方形**。`cv2.resize` 的第二个参数是 `(宽, 高)` 元组。`INTER_LINEAR` 是双线性插值——缩小时比最近邻效果好得多。注意这是**不等比拉伸**（2446×1000 → 1280×1280），不是等比缩放加填充 |
| L81 | `rgb = resized[:, :, ::-1]` | **BGR 转 RGB**。`::-1` 是负步长切片，表示"倒着取"，把第 0 通道和第 2 通道交换。**这是视图操作，不复制内存**，性能开销极小 |
| L82 | `tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0` | 三步：<br>• `transpose(2,0,1)`：**HWC → CHW**。OpenCV/PIL 给的是高-宽-通道，PyTorch 要的是通道-高-宽<br>• `.astype(np.float32)`：uint8（0~255）转 float32<br>• `/ 255.0`：缩放到 0~1。**注意：这里没有做 ImageNet 均值方差归一化**——RT-DETR 只除 255 |
| L83 | `return np.ascontiguousarray(tensor[None, ...])` | 两个操作：<br>• `tensor[None, ...]`：**加一个批次维度**，`(3,1280,1280)` → `(1,3,1280,1280)`。`None` 等价于 `np.newaxis`<br>• `np.ascontiguousarray`：**确保内存连续**。`transpose` 产生的是非连续视图，而 ONNX Runtime 要求连续内存，不转会在运行时出错 |

### 与通用预处理的差异（答辩高频考点）

| | 通用预处理 | RT-DETR |
|---|---|---|
| 缩放 | letterbox（等比 + 黑边）| **scale_fill（不等比拉伸）** |
| 归一化 | `/255` + ImageNet mean/std | **只 `/255`** |
| 坐标反变换 | 需还原 padding 偏移 | **直接乘原图宽高** |

**为什么必须拉伸**：letterbox 会在上下加黑边（2446×1000 填进 1280×1280，约 78% 的行是纯黑），模型训练时（`rect=True`）没见过黑边，推理时突然出现会造成**分布偏移**。而且拉伸后没有 padding 偏移量要还原，坐标回原图只需乘以宽高。

---

## 3.2 输出解析（`rtdetr_onnx.py` L102–146）

**要解决的问题**：网络输出的是归一化坐标，怎么变回原图像素位置。

```python
        preds = output[0]  # (300, 6)
        if preds.size == 0:  # 空输出兜底
            return _empty_detection()  # 返回统一格式的空结果，调用方无需特殊处理

        boxes_n = preds[:, :4]  # 前 4 列 = [cx,cy,w,h]，归一化到 0~1（_n = normalized）
        scores = preds[:, 4]  # 第 5 列 = 置信度；★ RT-DETR 没有 objectness 通道
        labels = preds[:, 5].astype(np.int64)  # 第 6 列 = 类别 ID，转整数

        cx, cy, w, h = boxes_n[:, 0], boxes_n[:, 1], boxes_n[:, 2], boxes_n[:, 3]  # 分别取出 4 列，每个 (300,)
        x1 = (cx - w / 2) * orig_w  # ★ cxcywh→xyxy 并还原像素；能这么简单是因为输入用拉伸无 padding
        y1 = (cy - h / 2) * orig_h  # 同上（y 方向）
        x2 = (cx + w / 2) * orig_w  # 同上（右下角 x）
        y2 = (cy + h / 2) * orig_h  # 同上（右下角 y）

        keep = scores >= self.conf_threshold  # 置信度过滤，返回布尔掩码
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L118 | `preds = output[0]` | 网络输出形状是 `(1, 300, 6)`。`output[0]` 去掉批次维 → `(300, 6)`。**300 是预设的最大检测数**，decoder 内部已做 top-k 选择 |
| L119–120 | `if preds.size == 0: return _empty_detection()` | **空输出兜底**。返回统一格式的空结果，调用方无需特殊处理 |
| L122 | `boxes_n = preds[:, :4]` | 取前 4 列 = 框坐标 `[cx, cy, w, h]`，**归一化到 0~1**。变量名后缀 `_n` 表示 normalized |
| L123 | `scores = preds[:, 4]` | 第 5 列 = 置信度。**注意这里没有 objectness 通道**——YOLO 是 `4+1+nc`（第 5 列是 objectness），RT-DETR 是 `4+1`（第 5 列直接就是最终分数）|
| L124 | `labels = preds[:, 5].astype(np.int64)` | 第 6 列 = 类别 ID。`astype(np.int64)` 转成整数类型（网络输出的是浮点）|
| L126 | `cx, cy, w, h = boxes_n[:, 0], ...` | 把 4 列分别取出，每个是 `(300,)` 的向量 |
| L127 | `x1 = (cx - w / 2) * orig_w` | **cxcywh → xyxy**：中心点减半宽得到左边界，再乘原图宽变回像素。**这里能这么简单，正是因为输入用的是拉伸而非 letterbox** |
| L128–130 | 同上 | 其余三个边界同理 |
| L132 | `keep = scores >= self.conf_threshold` | **置信度过滤**。返回布尔掩码数组 |
| L133 | `np.stack([x1, y1, x2, y2], axis=1)[keep]` | `np.stack` 把 4 个向量拼成 `(300, 4)` 矩阵；`[keep]` 用布尔掩码筛掉低分项 → `(N, 4)` |

### 剩下的部分

```python
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)[keep].astype(np.float32)  # 统一成 float32，与其他后端一致
        scores = scores[keep].astype(np.float32)  # 同步过滤分数
        classes = labels[keep]  # 同步过滤类别
        class_names = [
            self.class_names[c] if c < len(self.class_names) else str(c)  # ★ 越界保护：非法类别 ID 回退成字符串而不抛异常
            for c in classes  # 列表推导式：为每个类别取名字
        ]
```

| 代码 | 讲解 |
|------|------|
| `.astype(np.float32)` | 统一成 float32，与其他后端保持一致 |
| `self.class_names[c] if c < len(self.class_names) else str(c)` | **越界保护**。网络若输出非法类别 ID，回退成字符串而不是抛异常。**列表推导式**：`[表达式 for 变量 in 序列]` |

**三个要点**：① 300 是预设最大检测数，已做 top-k，**所以不需要 NMS**；② 没有 objectness 通道；③ 有越界保护。

---

## 3.3 推理引擎工厂（`engine.py` L355–404）

**要解决的问题**：怎么让上层代码不关心底层用哪个推理后端。

```python
def create_engine(
    backend: str = "dummy",  # 后端名：dummy / onnx / pytorch / rtdetr
    model_path: str = "",  # 模型路径（dummy 不需要）
    **kwargs,  # 收集任意额外关键字参数（input_size / conf_threshold / device 等）
) -> InferenceEngine:  # ★ 返回抽象基类类型，调用方只依赖接口
    if backend == "rtdetr":  # RT-DETR 引擎参数结构不同，走单独构造路径
        return _create_rtdetr_engine(model_path, **kwargs)  # 延迟导入在函数内部，不用 RT-DETR 时不加载它

    engines = {  # ★ 字典做注册表：键=名字，值=类对象（不是实例）
        "dummy": DummyInferenceEngine,  # 随机数造框，用于无模型跑通全流程
        "onnx": ONNXInferenceEngine,  # ONNX Runtime 引擎（生产用，可配 TensorRT）
        "pytorch": PyTorchInferenceEngine,  # 原生 PyTorch 引擎（开发调试用）
    }
    engine_cls = engines.get(backend)  # 按名字取类；找不到返回 None
    if engine_cls is None:  # 未知后端
        raise ValueError(f"Unknown backend: {backend}. Options: {list(engines.keys())}")  # ★ 立刻报错并列出可用选项，而不是让用户面对 NoneType 错误

    engine = engine_cls(model_path=model_path, **kwargs)  # 实例化；四个后端走同一行代码
    if backend != "dummy":  # ★ Dummy 引擎不需要加载模型
        engine.load()  # 加载模型到内存（ONNX session / torch.load）
        engine.warmup()  # ★ 预热：让 cuDNN 选好算法、显存分配完成，否则前几次推理特别慢
    return engine  # 返回配置好的实例
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L355–359 | `def create_engine(backend="dummy", model_path="", **kwargs)` | **工厂函数**。`**kwargs` 是"收集任意额外关键字参数"——调用方可以传 `input_size`、`confidence_threshold`、`device` 等，引擎构造器按需接收 |
| L360 | `) -> InferenceEngine:` | 返回类型是**抽象基类**，不是具体类。这是关键——调用方只依赖接口 |
| L361–362 | `if backend == "rtdetr": return _create_rtdetr_engine(...)` | RT-DETR 引擎**参数结构不同**（要 `imgsz`、`conf_threshold`、`class_names`），所以走单独的构造路径。**延迟导入**在函数内部，避免不用 RT-DETR 时也加载它 |
| L365–369 | `engines = {...}` | **字典做注册表**。键是名字字符串，值是类对象（不是实例）。这样新增后端只要往字典里加一行 |
| L370 | `engine_cls = engines.get(backend)` | 按名字取类。`.get()` 找不到返回 `None` |
| L371–372 | `if engine_cls is None: raise ValueError(...)` | **未知后端立刻报错并列出可用选项**，而不是让用户面对一个莫名其妙的 `NoneType` 错误。f-string 里的 `list(engines.keys())` 列出所有可选后端 |
| L374 | `engine = engine_cls(model_path=model_path, **kwargs)` | **实例化**。用变量 `engine_cls` 调用，这样四个后端走同一行代码 |
| L375 | `if backend != "dummy":` | **Dummy 引擎不需要加载模型**——它用随机数造假框 |
| L376 | `engine.load()` | 加载模型到内存（ONNX Runtime session / torch.load）|
| L377 | `engine.warmup()` | **预热**。用随机数据跑几次，让 cuDNN 选好算法、显存分配完成。**跳过预热的话，前几次推理会特别慢**——生产环境这是必须做的 |
| L378 | `return engine` | 返回配置好的实例 |

### 架构价值

```python
class InferenceEngine(ABC):  # ★ 抽象基类：所有推理后端的统一接口
    @abstractmethod
    def infer(self, images: np.ndarray) -> List[Dict]: ...  # ★ @abstractmethod 强制子类实现；实例化抽象类会直接报错
    @abstractmethod
    def load(self): ...  # 同上，强制实现模型加载

    def warmup(self, iterations: int = 3):  # 非抽象方法：所有子类免费获得预热能力（模板方法模式）
        if self._warmed_up:  # 已预热过就跳过
            return
        dummy = np.random.randn(1, 3, *self.input_size).astype(np.float32)  # 造一批随机输入（形状须与真实输入一致）
        for _ in range(iterations):  # 跑若干次
            self.infer(dummy)  # 用推理来预热
        self._warmed_up = True  # 标记已预热

    def benchmark(self, batch_size: int = 1, iterations: int = 100) -> Dict[str, float]:  # 同上：子类免费获得测速能力
        # ★ 模板方法：子类免费获得测速能力（实现见下方逐行解读表）
        ...
```

| 设计 | 讲解 |
|------|------|
| `@abstractmethod` | 强制子类实现。**实例化抽象类会直接报错**，避免"写了一半才发现漏方法" |
| `warmup` / `benchmark` 放在基类 | **模板方法模式**——推理逻辑各子类不同，但"预热""测速"是共性行为，继承下来即可。所以四个引擎都能直接 `.benchmark()` |
| `DummyInferenceEngine` | 用随机数造假框。**价值：没有 GPU、没有权重也能跑通全流程和 205 个测试**。测试桩不是 hack，是架构的一部分 |

**课堂演示**：

```python
from backend.inference.engine import create_engine  # 导入工厂函数
e = create_engine('dummy')  # 用 dummy 后端造一个引擎（无需模型）
print(type(e).__name__)  # 打印引擎类名
print(e.benchmark(batch_size=1, iterations=5))  # 测速：基类免费提供的方法
```

---

## 3.4 标签加载（`dataset.py` L159–186）

**要解决的问题**：磁盘上存的是归一化坐标，但 IoU 计算要像素坐标。

```python
    def _load_yolo(self) -> List[Dict]:  # 方法：读 YOLO 格式标签并反归一化回像素坐标
        annotations = []  # 累积每张图的标注
        for img_path in self._image_files:  # 遍历图像；★ 顺序必须与 _image_files 一致（后面按索引对齐）
            base = os.path.splitext(os.path.basename(img_path))[0]  # 三步拿无扩展名文件名：basename→splitext→取[0]
            label_path = os.path.join(self.label_dir, base + ".txt")  # 拼出标签文件路径
            boxes, labels = [], []  # ★ 初始为空 → 负样本自然保持空，无需特殊分支
            if os.path.exists(label_path):  # 标签文件可能不存在
                with open(label_path, "r") as f:  # 逐行读；空文件→循环不执行→boxes 保持空
                    for line in f:  # 遍历每行标注
                        parts = line.strip().split()  # strip 去首尾空白，split 按空白切成字符串列表
                        if len(parts) >= 5:  # ★ 防御性检查：每行应有 5 个字段（类别 + 4 坐标）
                            cls = int(parts[0])  # 类别 ID 转整数
                            cx, cy, w, h = map(float, parts[1:5])  # map 转浮点并解包；切片取索引 1~4
                            iw, ih = self.image_size  # 图像宽高；注意与框的 w/h 区分（iw/ih = image width/height）
                            boxes.append(
                                [
                                    int((cx - w / 2) * iw),
                                    int((cy - h / 2) * ih),
                                    int(w * iw),
                                    int(h * ih),
                                ]
                            )
                            labels.append(cls)  # 收集类别
            annotations.append({"boxes": boxes, "labels": labels})  # 追加这张图的标注
        return annotations  # 返回全部标注
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L160 | `annotations = []` | 累积每张图的标注 |
| L161 | `for img_path in self._image_files:` | 遍历图像列表。**注意顺序必须与 `self._image_files` 一致**——后面用索引对齐 |
| L162 | `os.path.splitext(os.path.basename(img_path))[0]` | 三步拿无扩展名的文件名：取 basename（去掉目录）→ 拆扩展名 → 取前半部分 |
| L163 | `label_path = os.path.join(self.label_dir, base + ".txt")` | 拼出标签文件路径 |
| L164 | `boxes, labels = [], []` | 这张图的框和类别，**初始为空**——负样本保持空列表 |
| L165 | `if os.path.exists(label_path):` | 标签文件可能不存在（虽然转换脚本总会写空文件）|
| L166 | `with open(label_path, "r") as f:` | 逐行读。空文件 → 循环体不执行 → `boxes` 保持空 → **负样本被自然处理，不需要特殊分支** |
| L167 | `for line in f:` | 遍历每行标注 |
| L168 | `parts = line.strip().split()` | `strip()` 去掉首尾空白（含换行符），`split()` 按空白切分成字符串列表 |
| L169 | `if len(parts) >= 5:` | **防御性检查**。每行应有 5 个字段：类别 + 4 个坐标 |
| L170 | `cls = int(parts[0])` | 类别 ID 转整数 |
| L171 | `cx, cy, w, h = map(float, parts[1:5])` | `map(float, ...)` 把字符串列表转成浮点数并解包。`parts[1:5]` 取索引 1~4 |
| L172 | `iw, ih = self.image_size` | 取图像宽高。**注意变量名是 `iw/ih`（image width/height）**，和前面 `w/h` 区分开——`w/h` 是框的宽高，`iw/ih` 是图像的宽高 |
| L173–178 | `int((cx - w/2) * iw)` | **反归一化**：归一化中心点减半宽 → 乘图像宽 → 像素坐标。`int()` 取整（YOLO 标注本身就是整数像素）|
| L179 | `labels.append(cls)` | 收集类别 |
| L185 | `annotations.append({...})` | 追加这张图的标注 |
| L186 | `return annotations` | 返回全部 |

### 数据流的双向性

```
磁盘上的 YOLO 格式（归一化 cxcywh）
    ↓ 读入时反归一化
内存中的像素坐标 xywh
```

**为什么不同模块用不同坐标系**：

| 层 | 坐标系 | 理由 |
|---|--------|------|
| 存储层 | 归一化 | 与分辨率解耦，通用 |
| IoU 计算层 | 像素 | 避免反复乘除，且与 ultralytics 内部一致 |
| 奖励计算层 | 像素 `xyxy`/`xywh` | `dimension_rewards.py` 里 `unbind(1)` 后直接算 |

---

## 3.5 自定义 collate（`dataset.py` L229–259）

**要解决的问题**：PyTorch 默认的 collate 要求每个样本结构一致，但检测任务每张图的框数不同。

```python
def defect_collate_fn(batch) -> Tuple[torch.Tensor, List[Dict]]:
    images: List[torch.Tensor] = []  # 收集图像张量
    targets: List[Dict] = []  # 收集标注
    for img, target in batch:  # 遍历 batch 里每个（图像, 标注）
        boxes = target.get("boxes", [])  # .get(key, default) 安全取值
        labels = target.get("labels", [])  # 同上
        if not boxes or not labels:  # ★ 空列表是 falsy → not boxes 等价"没有框"；这里丢弃负样本
            continue  # 负样本（正常图）跳过，不进 batch
        images.append(img)  # 收集图像
        targets.append({"boxes": [list(boxes[0])], "labels": [int(labels[0])]})  # ★ 单框设计：只取每图第一个（主缺陷）框

    if not images:  # 整个 batch 全是负样本的极端情况
        return torch.empty(0, 3, 640, 640), []  # ★ 返回空张量+空列表让训练器跳过，而不是崩溃

    return torch.stack(images), targets  # ★ stack 把 N 个 (3,H,W) 拼成 (N,3,H,W)，要求形状完全一致
```

### 逐行解读

| 行号 | 代码 | 讲解 |
|------|------|------|
| L229 | `def defect_collate_fn(batch) -> Tuple[torch.Tensor, List[Dict]]:` | **collate_fn** 是"把多个样本拼成一个 batch"的回调函数。PyTorch DataLoader 接受 `collate_fn=` 参数覆盖默认行为 |
| L230–231 | `images`, `targets` | 两个平行列表，最终要 `torch.stack` 和直接返回 |
| L232 | `for img, target in batch:` | 遍历 batch 里的每个 `(图像张量, 标注字典)` |
| L233 | `boxes = target.get("boxes", [])` | **`.get(key, default)`** 安全的字典取值，取不到用默认值 |
| L234 | `labels = target.get("labels", [])` | 同上 |
| L235 | `if not boxes or not labels:` | **空列表是 falsy**，所以 `not boxes` 等价于"没有框"。**这就是丢弃负样本的地方** |
| L236 | `continue` | 负样本（正常图）被跳过，不进 batch |
| L237 | `images.append(img)` | 收集图像张量 |
| L238 | `targets.append({"boxes": [list(boxes[0])], "labels": [int(labels[0])]})` | **单框设计**：只取每张图的第一个（主缺陷）框。外层 `[...]` 是为了保持"每图一个框列表"的结构一致性 |
| L242 | `if not images:` | 整个 batch 全是负样本的极端情况 |
| L243 | `return torch.empty(0, 3, 640, 640), []` | 返回**空张量 + 空列表**，让训练器能识别并跳过这个 batch。**而不是崩溃** |
| L245 | `return torch.stack(images), targets` | **`torch.stack` 把 N 个 `(3,H,W)` 拼成 `(N,3,H,W)`**。注意 stack 要求所有张量形状完全一致——这正是要丢掉负样本的原因（图像尺寸都相同，但目标结构不同）|

### 为什么只取第一个框

因为 `dimension_rewards.py` 的对齐逻辑是"预测 i 对应 GT i"：

```python
    def _aligned_boxes(self, predictions, targets):  # 内部方法：把预测框与真值框按索引对齐
        gt_list = [list(t["boxes"][0]) for t in targets if t.get("boxes")]  # 取每图第一个真值框（与单框设计一致）
        ...
        n = min(pred.shape[0], gt.shape[0])  # 取两者的较小长度，防止维度不匹配
        return pred[:n], gt[:n]  # ★ 返回对齐后的 (N,4) 预测框与真值框
```

**明确的取舍**：牺牲多框场景的精度，换取奖励信号与监督信号的精确对齐。源码注释写得很清楚——"the single-box detector aligns prediction i with ground-truth i"。

> ⚠️ **注意两条路径的行为差异**：自研单框训练器**丢弃负样本**，而主训练路径（ultralytics）**保留全部 374 张负样本**。这是有意为之——自研训练器没有"无缺陷"这个类别可用。

---

# 第四部分 逐行讲解的常见误区汇总

| 误区 | 正确做法 | 本项目位置 |
|------|---------|-----------|
| 用 Python `max()`/`min()` 算 IoU | 用 `torch.maximum`/`torch.minimum` | `dimension_rewards.py` L80–83 |
| 用 `argmax` 判断分类对错 | 用 `softmax` 概率质量 | `dimension_rewards.py` L271–272 |
| 以为可微函数就一定有梯度 | 输入张量要带 `requires_grad=True` | 见 2.1 实测 |
| 坐标转换不钳制 | 中心点双向钳制 + 宽高防负 | `convert_tianchi_to_yolo.py` L141–144 |
| 标注精度随手取 3 位 | 至少 6 位（最小框宽 0.002833）| `convert_tianchi_to_yolo.py` L202 |
| 先划分再去重 | 必须先去重再划分 | `convert_tianchi_to_yolo.py` L238–257 |
| 丢掉负样本 | 空 .txt 文件保留 | `convert_tianchi_to_yolo.py` L202–206 |
| 跳过未知类别不计数 | `skipped += 1` 并汇总报告 | `convert_tianchi_to_yolo.py` L198 |
| 混淆 xyxy 与 xywh | 天池是 xyxy，YOLO 是 cxcywh | `convert_tianchi_to_yolo.py` L134 |
| 以为 DETR 漏检是"没匹配上" | 是"匹配上但置信度低" | `reward_rtdetr.py` L128–131 |
| 推理用 letterbox 但训练用 rect | RT-DETR 必须 scale_fill | `rtdetr_onnx.py` L78–80 |
| 只除 255 却做了 ImageNet 归一化 | RT-DETR 只除 255 | `rtdetr_onnx.py` L82 |
| 忘记 transpose 后内存不连续 | `np.ascontiguousarray` | `rtdetr_onnx.py` L83 |

---

# 附录 一分钟讲稿

> 这份讲义的核心是**逐行读懂项目最关键的六段代码**。
>
> **第一段是坐标转换**（`convert_tianchi_to_yolo.py` L134–146）。天池给的是 xyxy 像素坐标，YOLO 要 cxcywh 归一化。逐行看：解包出四个坐标、算中心点除以图像宽高、宽度用 `max(0.0, ...)` 防负、中心点用 `min(max(x,0),1)` 双向钳制。最后一行 `bw, bh = min(bw, 1.0), min(bh, 1.0)` 只卡上限不卡下限——因为宽度为 0 是需要上报的数据质量问题，不该在转换阶段悄悄改掉。我实测确认 8570 个框里有 9 个越界，幅度 1e-6，所以这道钳制不是过度设计。
>
> **第二段是去重**（L180–206）。partA 和 partB 的标注文件 md5 完全相同。判据是 `os.path.exists(dst_img)`——目标文件已存在就 `continue` 跳过，**按图像名去重而不是按标注条数**。紧接着 `lines` 列表保持为空，`"\n".join(lines)` 就写出空文件，这就是负样本的生成方式——验证集里 374 张负样本占 39%。
>
> **第三段是划分**（L229–257）。`random.seed(seed)` 固定种子、`sorted()` 保证顺序、`shuffle` 原地打乱。**最关键的是顺序**：先去重再划分，反了会让验证集 100% 泄漏且不报错。
>
> **第四段是可微 IoU**（`dimension_rewards.py` L61–87）。逐行看 `torch.maximum` 不能写成 Python 内置 `max()`——内置 max 切断计算图。`unbind(1)` 拆张量且保持梯度。`clamp(min=0.0)` 把负的交集面积截断到 0。我在现场跑过验证：输入不带 `requires_grad` 时 `requires_grad` 是 False，带上之后 `backward()` 得到的梯度是 `[[0, 0, 6.29e-05, 6.29e-05]]`，出现在宽高方向。
>
> **第五段是分类奖励**（L259–276）。`probs.gather(1, labels.unsqueeze(1)).squeeze(1)` 三步取出真实类别上的概率。关键是这里用 `softmax` 而不是 `argmax`——因为 softmax 交叉熵的梯度是 `p − onehot(y)`，真实类概率压到 0 时梯度仍接近 −1，模型还能学；argmax 则梯度直接归零。
>
> **第六段是奖励注入**（`reward_rtdetr.py` L73–103）。继承 `RTDETRDetectionLoss`，先 `super().forward(...)` 拿到原生三项损失，然后 `loss["loss_reward"] = ...` 往字典里加一个键。**ultralytics 会把字典里每一项加起来，所以完全不用改框架代码**。最后那段漏检处理是最有价值的认知纠错——DETR 的匈牙利匹配保证每个 GT 都有 query，所以"漏检"不是"没匹配上"，而是"匹配上了但关键类置信度低"。代码拿一个空闲 query 去指向漏检的 GT，强行施加召回梯度。
>
> 这六段代码串起来就是这个项目的方法论内核：**数据决定上限，损失函数决定方向，而可微性是让奖励真正起作用的技术前提。**
