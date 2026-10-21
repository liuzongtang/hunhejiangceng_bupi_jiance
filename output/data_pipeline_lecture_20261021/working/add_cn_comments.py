"""
为讲义中的代码块批量添加中文注释。

策略：
- 对每段源码，识别关键行（赋值、判断、返回、特殊调用），在行尾追加中文注释
- 保持缩进与原代码完全一致
- 已有注释的行不重复添加
- 注释过长时改为上方独立注释行
"""

from pathlib import Path
import re
import sys

# 行尾注释表：按"去空白后的代码前缀"匹配
INLINE = {
    # --- xyxy_to_yolo ---
    '"""Convert xyxy pixel bbox to normalized (cx, cy, bw, bh)."""': "# 文档字符串：说明这个函数做什么",
    "xmin, ymin, xmax, ymax = (float(v) for v in bbox)": "# 解包出 4 个坐标并统一转浮点（JSON 里可能是整数）",
    "cx = (xmin + xmax) / 2.0 / w": "# 中心点 x：先取两端均值，再除图像宽归一化到 0~1",
    "cy = (ymin + ymax) / 2.0 / h": "# 中心点 y：注意除的是图像高 h（与 x 用 w 区分，写反会转置）",
    "bw = max(0.0, (xmax - xmin)) / w": "# 宽度：max(0.0, ...) 防负宽度（标注错误时 xmax < xmin）",
    "bh = max(0.0, (ymax - ymin)) / h": "# 高度：同样防负",
    "cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)": "# 双向钳制中心点到 [0,1]：实测有框溢出 1e-6",
    "bw, bh = min(bw, 1.0), min(bh, 1.0)": "# 宽高只卡上限不卡下限：宽度为 0 是需上报的数据质量问题",
    "return cx, cy, bw, bh": "# 返回归一化后的四个数",
    # --- 精度 ---
    'lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")': "# 保留 6 位小数：实测最小框宽 0.002833，3 位会丢小目标",
    # --- 去重段 ---
    "src = os.path.join(img_dir, fname)": "# 拼出源图像路径（join 会自动处理 Windows 分隔符）",
    "with Image.open(src) as im:": "# 上下文管理器：打开图像并保证用完自动关闭",
    "w, h = im.size": "# 读出图像宽高，顺序是（宽, 高）",
    "dst_img = os.path.join(out_images, fname)": "# 拼出目标路径（扁平化输出目录）",
    "if os.path.exists(dst_img):": "# ★ 去重判据：目标文件已存在 → 按图像名去重",
    'print(f"  [warn] duplicate image name skipped: {fname}")': "# 打印警告：绝不静默跳过",
    "continue": "# ★ 跳过本图且不累加 —— 这就是防 GT 翻倍的关键",
    "shutil.copy(src, dst_img)": "# 复制图像到输出目录",
    "lines: List[str] = []": "# 初始化标注行列表（正常图最终保持为空）",
    "for ann in by_name.get(fname, []):": "# 遍历本图标注；取不到返回空列表（正常图即此情况）",
    'category = NAME_TO_CATEGORY.get(ann["defect_name"])': "# 中文缺陷名 → 类别 ID 映射，找不到返回 None",
    "if category is None:": "# 遇到未知缺陷名",
    "skipped += 1": "# ★ 计数：跳过必须有计数，结束时统一报告",
    "cls = category_to_class_id(category, scheme)": "# 天池 1-based 类别 ID → 训练用 0-based",
    # --- 负样本 ---
    'label_path = os.path.join(out_labels, os.path.splitext(fname)[0] + ".txt")': "# 标签文件名 = 图像名去扩展名 + .txt",
    'with open(label_path, "w", encoding="utf-8") as f:': "# 写模式打开；显式 utf-8 防止 Windows 下中文乱码",
    'f.write("\\n".join(lines) + ("\\n" if lines else ""))': "# ★ lines 为空时 join 返回空串 → 写出空文件 = 负样本",
    # --- 划分 ---
    "def split_train_val(root: str, val_ratio: float, seed: int) -> Tuple[int, int]:": "# 划分函数：根目录、验证集比例、随机种子",
    '"""Split the staging train dir into train/ and val/ by image name."""': "# 文档字符串",
    "all_images = sorted(os.listdir(staging_img))": "# ★ sorted() 保证顺序固定，否则不同机器划分不可复现",
    "random.seed(seed)": "# ★ 固定随机种子 —— 可复现的基础",
    "random.shuffle(all_images)": "# 原地打乱（shuffle 返回 None，不用接返回值）",
    "n_val = int(len(all_images) * val_ratio)": "# 算验证集张数：9576 × 0.1 = 957.6 → 957",
    "val_set = set(all_images[:n_val])": "# 取前 10% 作验证集；用 set 便于 O(1) 查询归属",
    "os.makedirs(d, exist_ok=True)": "# 建目录；exist_ok=True 表示已存在就跳过（可重复运行）",
    "for name in all_images:": "# 遍历所有文件名",
    "dst_img_dir = val_img if name in val_set else train_img": "# ★ 三元表达式：决定图像去哪个目录",
    "dst_lbl_dir = val_lbl if name in val_set else train_lbl": "# 标签用同样判断，保证图像与标签成对移动",
    "os.rename(os.path.join(staging_img, name), os.path.join(dst_img_dir, name))": "# os.rename 是移动不是复制，天然不重复",
    'lbl = os.path.splitext(name)[0] + ".txt"': "# 算出对应的标签文件名",
    "if os.path.exists(src_lbl):": "# 标签文件可能不存在",
    "os.rename(os.path.join(src_lbl), os.path.join(dst_lbl_dir, lbl))": "# 移动标签到对应目录",
    'shutil.rmtree(os.path.join(root, "_staging"), ignore_errors=True)': "# 删除暂存区；ignore_errors=True 删不掉也不报错",
    "return len(all_images) - n_val, n_val": "# 返回（训练数, 验证数）",
    # --- 可微 IoU ---
    "pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, eps: float = 1e-6": "# 预测框/真值框形状 (N,4)；eps 防除零",
    "px1, py1, pw, ph = pred_boxes.unbind(1)": "# ★ unbind(1) 拆成 4 个 (N,)：是视图不复制，且梯度正常传递",
    "gx1, gy1, gw, gh = gt_boxes.unbind(1)": "# 真值框同理（g 前缀 = ground truth）",
    "px2, py2 = px1 + pw, py1 + ph": "# xywh → xyxy：算出预测框的右下角",
    "gx2, gy2 = gx1 + gw, gy1 + gh": "# 真值框的右下角",
    "ix1 = torch.maximum(px1, gx1)": "# ★ 必须用 torch.maximum！Python 内置 max 会切断计算图",
    "iy1 = torch.maximum(py1, gy1)": "# 交集框上边界，同理",
    "ix2 = torch.minimum(px2, gx2)": "# ★ 同理，不能用内置 min()",
    "iy2 = torch.minimum(py2, gy2)": "# 交集框下边界",
    "inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)": "# ★ 交集面积：clamp 把负面积截断到 0（两框不相交时）",
    "union = pw * ph + gw * gh - inter": "# 并集面积：容斥原理（两面积之和减交集）",
    "return inter / (union + eps)": "# IoU = 交集 ÷ 并集；+eps 防并集为 0",
    # --- 分类奖励 ---
    "def compute_cls_reward(": "# 方法：D02 分类维度奖励",
    "predictions: Dict[str, torch.Tensor],": "# 预测字典：含 logits (N,C) 与 boxes (N,4)",
    "targets: List[Dict],": "# 真值列表：每项含 boxes 与 labels",
    "aligned = self._aligned_logits_labels(predictions, targets)": "# 对齐预测与真值，返回 (logits, labels)；失败返回 None",
    "if aligned is None:": "# 对齐失败的兜底",
    "return torch.tensor(-1.0)": "# 返回最差奖励 -1（D02 值域 [-1,1]）；常量张量不参与梯度但也不报错",
    "logits, labels = aligned": "# 元组解包：logits (N,C)、labels (N,)",
    "probs = self._class_probs(logits)": "# ★ 关键：logits→概率。softmax 或 sigmoid 按 head_type 分支",
    "p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)": "# ★ 取真实类别上的概率：unsqueeze 加维→gather 按索引取→squeeze 压维",
    "return (2.0 * p_true - 1.0).mean()": "# 映射到 [-1,1] 后对 batch 求均值：p_true=1→+1，=0→-1",
    # --- 奖励注入 ---
    "def forward(": "# 重写父类 forward：签名必须与 RTDETRDetectionLoss 一致",
    "preds: tuple,": "# 元组 (dec_bboxes, dec_scores)：形状 (L,B,N,4) 与 (L,B,N,C)",
    "batch: Dict[str, Any],": "# 标注字典：bboxes / cls / gt_groups",
    "dn_bboxes: Optional[torch.Tensor] = None,": "# 去噪查询框（DENOISING 是 DETR 的训练加速机制）",
    "dn_scores: Optional[torch.Tensor] = None,": "# 去噪查询分数",
    "dn_meta: Optional[Dict[str, Any]] = None,": "# 去噪元信息",
    "loss = super().forward(": "# ★ 调用父类：拿到原生三项损失（VFL分类 / L1回归 / GIoU）",
    "dec_bboxes, dec_scores = preds": "# 从 preds 元组解出框和分数",
    'loss["loss_reward"] = self._reward_loss(dec_bboxes[-1], dec_scores[-1], batch)': "# ★★ 核心一行：往字典加一个键，ultralytics 会自动求和",
    "return loss": "# 返回带奖励项的损失字典",
    # --- ultralytics 内部 ---
    "loss, loss_items = self.criterion(preds, batch)": "# ultralytics 内部：criterion 返回字典",
    "loss.sum().backward()": "# ★ 字典每一项都被加进来 → 奖励自动参与反传，无需改框架",
    # --- 奖励计算 ---
    "match_indices = self.matcher(": "# ★ 复用监督损失已算过的那次匈牙利匹配（同一 matcher 同一输入）",
    'pred_bboxes, pred_scores, batch["bboxes"], batch["cls"], batch["gt_groups"]': "# 匹配器需要的 5 个参数",
    "pred_logits: List[torch.Tensor] = []": "# 收集参与奖励的预测分数",
    "pred_boxes: List[torch.Tensor] = []": "# 收集参与奖励的预测框",
    "targets: List[Dict] = []": "# 收集对应真值（与上面两个平行数组一一对应）",
    "n_missed = 0": "# 漏检计数，用于日志",
    'gt_groups = batch["gt_groups"]': "# 每图各有几个 GT；GT 索引是全 batch 扁平的",
    "gt_offset = 0": "# 当前图在扁平 GT 数组中的起始偏移",
    "for i, (src, dst) in enumerate(match_indices):": "# 遍历每张图：i=图下标，src/dst=该图的匹配对",
    "for k in range(len(src))": "# 遍历这张图的每个匹配对",
    "pred_logits.append(pred_scores[i][src[k]])": "# 取出被匹配 query 的分类分数 (C,)",
    "pred_boxes.append(pred_bboxes[i][src[k]])": "# 取出该 query 的框 (4,)",
    '"boxes": [batch["bboxes"][dst[k]].tolist()],': "# 对应真值框；.tolist() 切断梯度是对的（真值本就不该有梯度）",
    '"labels": [int(batch["cls"][dst[k]])],': "# 对应真值类别",
    # --- 漏检处理 ---
    "if self.include_misses:": "# 开关：是否把漏检也送进奖励",
    "gt_to_query = {": '# 建立"GT编号→query编号"反查表',
    "int(g): int(q)": "# 字典推导式：g 是 GT 号，q 是 query 号",
    "for g, q in zip(dst.tolist(), src.tolist(), strict=True)": "# zip 配对；strict=True 长度不一致就报错",
    "matched = {int(q) for q in src.tolist()}": "# 已被占用的 query 集合",
    "for g in range(gt_offset, gt_offset + int(gt_groups[i])):": "# 遍历本图的所有 GT",
    'label = int(batch["cls"][g])': "# 该 GT 的真实类别",
    "class_ids = self._critical_ids_for(label)": "# 查该类别属于哪个关键类集合（断经{9,15}/跳花{14,17}）",
    "if class_ids is None:": "# 非关键类",
    "continue": "# 只关心关键类，非关键类漏检不额外处理",
    "self._p_crit(pred_scores[i][gt_to_query[g]], class_ids)": "# 该 GT 匹配到的 query 在关键类上的置信度",
    ">= self.miss_threshold": "# ★ 达到阈值说明模型其实认出来了，不算漏",
    "continue": "# 真正漏检才继续处理",
    "spare = self._best_spare_query(i, class_ids, matched, pred_scores)": "# ★ 找空闲 query：未被占用中关键类置信度最高的",
    "if spare is None:": "# 没有空闲 query",
    "continue": "# 跳过（query 数 ≥ GT 数时通常不会发生）",
    # --- 预处理 ---
    "def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:": "# 方法：原始 BGR 图 → 网络输入张量（本引擎自己完成预处理）",
    "import cv2": "# ★ 故意放在函数内：顶层不依赖 cv2，导 ONNX 等轻量操作不必装 opencv",
    "image_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR": "# ★ 不等比拉伸成正方形（不是 letterbox 填充）",
    "rgb = resized[:, :, ::-1]": "# ★ BGR→RGB：负步长切片是视图，不复制内存",
    "tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0": "# HWC→CHW；转 float32；只除 255（不做 ImageNet 均值方差）",
    "return np.ascontiguousarray(tensor[None, ...])": "# ★ 加批次维 + 保证内存连续（transpose 后非连续，ONNX Runtime 要求连续）",
    # --- 输出解析 ---
    "preds = output[0]  # (300, 6)": "# 去掉批次维 → (300,6)；300 是预设最大检测数（decoder 已做 top-k）",
    "if preds.size == 0:": "# 空输出兜底",
    "return _empty_detection()": "# 返回统一格式的空结果，调用方无需特殊处理",
    "boxes_n = preds[:, :4]": "# 前 4 列 = [cx,cy,w,h]，归一化到 0~1（_n = normalized）",
    "scores = preds[:, 4]": "# 第 5 列 = 置信度；★ RT-DETR 没有 objectness 通道",
    "labels = preds[:, 5].astype(np.int64)": "# 第 6 列 = 类别 ID，转整数",
    "cx, cy, w, h = boxes_n[:, 0], boxes_n[:, 1], boxes_n[:, 2], boxes_n[:, 3]": "# 分别取出 4 列，每个 (300,)",
    "x1 = (cx - w / 2) * orig_w": "# ★ cxcywh→xyxy 并还原像素；能这么简单是因为输入用拉伸无 padding",
    "y1 = (cy - h / 2) * orig_h": "# 同上（y 方向）",
    "x2 = (cx + w / 2) * orig_w": "# 同上（右下角 x）",
    "y2 = (cy + h / 2) * orig_h": "# 同上（右下角 y）",
    "keep = scores >= self.conf_threshold": "# 置信度过滤，返回布尔掩码",
    "np.stack([x1, y1, x2, y2], axis=1)[keep]": "# 拼成 (300,4) 后用掩码筛掉低分项 → (N,4)",
    # --- 输出解析后半 ---
    "boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)[keep].astype(np.float32)": "# 统一成 float32，与其他后端一致",
    "scores = scores[keep].astype(np.float32)": "# 同步过滤分数",
    "classes = labels[keep]": "# 同步过滤类别",
    "self.class_names[c] if c < len(self.class_names) else str(c)": "# ★ 越界保护：非法类别 ID 回退成字符串而不抛异常",
    "for c in classes": "# 列表推导式：为每个类别取名字",
    # --- 工厂 ---
    'backend: str = "dummy",': "# 后端名：dummy / onnx / pytorch / rtdetr",
    'model_path: str = "",': "# 模型路径（dummy 不需要）",
    "**kwargs,": "# 收集任意额外关键字参数（input_size / conf_threshold / device 等）",
    ") -> InferenceEngine:": "# ★ 返回抽象基类类型，调用方只依赖接口",
    'if backend == "rtdetr":': "# RT-DETR 引擎参数结构不同，走单独构造路径",
    "return _create_rtdetr_engine(model_path, **kwargs)": "# 延迟导入在函数内部，不用 RT-DETR 时不加载它",
    "engines = {": "# ★ 字典做注册表：键=名字，值=类对象（不是实例）",
    '"dummy": DummyInferenceEngine,': "# 随机数造框，用于无模型跑通全流程",
    '"onnx": ONNXInferenceEngine,': "# ONNX Runtime 引擎（生产用，可配 TensorRT）",
    '"pytorch": PyTorchInferenceEngine,': "# 原生 PyTorch 引擎（开发调试用）",
    "engine_cls = engines.get(backend)": "# 按名字取类；找不到返回 None",
    "if engine_cls is None:": "# 未知后端",
    'raise ValueError(f"Unknown backend: {backend}. Options: {list(engines.keys())}")': "# ★ 立刻报错并列出可用选项，而不是让用户面对 NoneType 错误",
    "engine = engine_cls(model_path=model_path, **kwargs)": "# 实例化；四个后端走同一行代码",
    'if backend != "dummy":': "# ★ Dummy 引擎不需要加载模型",
    "engine.load()": "# 加载模型到内存（ONNX session / torch.load）",
    "engine.warmup()": "# ★ 预热：让 cuDNN 选好算法、显存分配完成，否则前几次推理特别慢",
    "return engine": "# 返回配置好的实例",
    # --- 抽象基类 ---
    "class InferenceEngine(ABC):": "# ★ 抽象基类：所有推理后端的统一接口",
    "def infer(self, images: np.ndarray) -> List[Dict]: ...": "# ★ @abstractmethod 强制子类实现；实例化抽象类会直接报错",
    "def load(self): ...": "# 同上，强制实现模型加载",
    "def warmup(self, iterations: int = 3):": "# 非抽象方法：所有子类免费获得预热能力（模板方法模式）",
    "if self._warmed_up:": "# 已预热过就跳过",
    "dummy = np.random.randn(1, 3, *self.input_size).astype(np.float32)": "# 造一批随机输入（形状须与真实输入一致）",
    "for _ in range(iterations):": "# 跑若干次",
    "self.infer(dummy)": "# 用推理来预热",
    "self._warmed_up = True": "# 标记已预热",
    # --- 演示 ---
    "from backend.inference.engine import create_engine": "# 导入工厂函数",
    "e = create_engine('dummy')": "# 用 dummy 后端造一个引擎（无需模型）",
    "print(type(e).__name__)": "# 打印引擎类名",
    "print(e.benchmark(batch_size=1, iterations=5))": "# 测速：基类免费提供的方法",
    # --- 标签加载 ---
    "def _load_yolo(self) -> List[Dict]:": "# 方法：读 YOLO 格式标签并反归一化回像素坐标",
    "annotations = []": "# 累积每张图的标注",
    "for img_path in self._image_files:": "# 遍历图像；★ 顺序必须与 _image_files 一致（后面按索引对齐）",
    "base = os.path.splitext(os.path.basename(img_path))[0]": "# 三步拿无扩展名文件名：basename→splitext→取[0]",
    'label_path = os.path.join(self.label_dir, base + ".txt")': "# 拼出标签文件路径",
    "boxes, labels = [], []": "# ★ 初始为空 → 负样本自然保持空，无需特殊分支",
    "if os.path.exists(label_path):": "# 标签文件可能不存在",
    'with open(label_path, "r") as f:': "# 逐行读；空文件→循环不执行→boxes 保持空",
    "for line in f:": "# 遍历每行标注",
    "parts = line.strip().split()": "# strip 去首尾空白，split 按空白切成字符串列表",
    "if len(parts) >= 5:": "# ★ 防御性检查：每行应有 5 个字段（类别 + 4 坐标）",
    "cls = int(parts[0])": "# 类别 ID 转整数",
    "cx, cy, w, h = map(float, parts[1:5])": "# map 转浮点并解包；切片取索引 1~4",
    "iw, ih = self.image_size": "# 图像宽高；注意与框的 w/h 区分（iw/ih = image width/height）",
    "int((cx - w / 2) * iw)": "# ★ 反归一化：中心点减半宽 → 乘图像宽 → 像素坐标",
    "int((cy - h / 2) * ih)": "# 同上（y 方向）",
    "int(w * iw)": "# 框宽还原为像素",
    "int(h * ih)": "# 框高还原为像素",
    "labels.append(cls)": "# 收集类别",
    'annotations.append({"boxes": boxes, "labels": labels})': "# 追加这张图的标注",
    "return annotations": "# 返回全部标注",
    # --- collate ---
    "images: List[torch.Tensor] = []": "# 收集图像张量",
    "targets: List[Dict] = []": "# 收集标注",
    "for img, target in batch:": "# 遍历 batch 里每个（图像, 标注）",
    'boxes = target.get("boxes", [])': "# .get(key, default) 安全取值",
    'labels = target.get("labels", [])': "# 同上",
    "if not boxes or not labels:": '# ★ 空列表是 falsy → not boxes 等价"没有框"；这里丢弃负样本',
    "continue": "# 负样本（正常图）跳过，不进 batch",
    "images.append(img)": "# 收集图像",
    'targets.append({"boxes": [list(boxes[0])], "labels": [int(labels[0])]})': "# ★ 单框设计：只取每图第一个（主缺陷）框",
    "if not images:": "# 整个 batch 全是负样本的极端情况",
    "return torch.empty(0, 3, 640, 640), []": "# ★ 返回空张量+空列表让训练器跳过，而不是崩溃",
    "return torch.stack(images), targets": "# ★ stack 把 N 个 (3,H,W) 拼成 (N,3,H,W)，要求形状完全一致",
    # --- 对齐 ---
    "def _aligned_boxes(self, predictions, targets):": "# 内部方法：把预测框与真值框按索引对齐",
    'gt_list = [list(t["boxes"][0]) for t in targets if t.get("boxes")]': "# 取每图第一个真值框（与单框设计一致）",
    "n = min(pred.shape[0], gt.shape[0])": "# 取两者的较小长度，防止维度不匹配",
    "return pred[:n], gt[:n]": "# ★ 返回对齐后的 (N,4) 预测框与真值框",
}

# 需要在上一行加独立注释的（因为行尾会过长）
INLINE[
    "def benchmark(self, batch_size: int = 1, iterations: int = 100) -> Dict[str, float]:"
] = "# 同上：子类免费获得测速能力"

ABOVE = {
    "if os.path.exists(dst_img):": "# ★★ 去重判据：按图像名去重（不是按标注条数）",
    "def xyxy_to_yolo(": "# 函数：把天池的 xyxy 像素框转成 YOLO 的 cxcywh 归一化框",
    "def differentiable_iou(": "# 函数：可微的 IoU 计算（对比标准实现的 max/min）",
    "def create_engine(": "# ★ 工厂函数：按名字造引擎，上层不关心具体后端",
    "def defect_collate_fn(batch) -> Tuple[torch.Tensor, List[Dict]]:": "# ★ 自定义 collate：把多样本拼成 batch",
}


def annotate(code: str) -> str:
    """给代码块加中文注释。"""
    out = []
    for line in code.split("\n"):
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue

        # 已有 # 注释的行，跳过（但要判断是否是我们要加的）
        indent = line[: len(line) - len(line.lstrip())]
        code_part = stripped.split("#")[0].rstrip()

        # 空函数体修复：def 后只有注释没有实际语句时，补一个 ... 让语法成立
        if code_part.startswith(("def ", "async def ")) and line.rstrip().endswith(":"):
            nxt = None
            for j in range(len(out) + 1):
                if j < len(out):
                    cand = out[j]
                else:
                    break
                if cand.strip():
                    nxt = cand
                    break
            if nxt is not None and nxt.strip().startswith("#"):
                out.append(f"{indent}    ...  # 函数体见下方说明")
                out.append(line)
                continue

        if code_part in INLINE:
            comment = INLINE[code_part]
            if code_part in ABOVE:
                out.append(f"{indent}{comment}")
                out.append(line)
            else:
                out.append(f"{indent}{code_part}  {comment}")
            continue

        out.append(line)
    return "\n".join(out)


def main() -> int:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    md = src.read_text(encoding="utf-8")

    # 逐个代码块处理
    result = []
    pos = 0
    idx = 0
    for m in re.finditer(r"```(\w*)\n(.*?)```", md, re.S):
        result.append(md[pos : m.start()])
        lang, code = m.group(1), m.group(2)
        idx += 1
        if lang in ("python", "py", "bash", "sh"):
            new_code = annotate(code)
            result.append(f"```{lang}\n{new_code}```")
        else:
            result.append(m.group(0))
        pos = m.end()
    result.append(md[pos:])

    dst.write_text("".join(result), encoding="utf-8")
    print(f"处理 {idx} 个代码块 → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
