# 附录三 源码地址索引与演示脚本

本附录用于课堂现场演示：给出全部关键源文件的绝对路径、精确行号坐标，以及经本机实测通过的演示命令。

## 附录三说明

项目根目录：

```
C:\Users\master\Desktop\维度模型(1)
```

下表所有路径均为该根目录下的相对路径，配合根目录即可拼出完整绝对路径。

---

## 一、源码总览：代码分布

| 目录 | 行数 | 职责 | 讲解相关度 |
|------|------|------|-----------|
| `scripts/` | 14048 | 训练、评估、诊断、可视化脚本 | ★★★★★ |
| `backend/training/` | 4679 | 七维奖励、损失、训练器、数据集 | ★★★★★ |
| `backend/inference/` | 1509 | 推理引擎、预处理、后处理 | ★★★★ |
| `mixed_reward/` | 786 | RLHF 混合奖惩独立包 | ★★ |
| `backend/schemas/` | 656 | Pydantic 模型、18 类枚举 | ★★★★ |
| `backend/services/` | 661 | 检测落库、告警升级、模型版本 | ★★★ |
| `backend/routers/` | 646 | REST 端点 | ★★★ |
| `backend/models/` | 448 | SQLAlchemy ORM | ★★ |
| **合计** | **约 27000** | | |

> `scripts/` 占比过半——因为它承载了全部实验与诊断工作。这本身就是一个值得讲的点：**科研项目的代码量大头在"验证"而非"实现"。**

---

## 二、关键源码地址与行号坐标

### 2.1 数据工程（第二章讲义对应）

| 功能 | 文件路径 | 关键行号 | 讲解要点 |
|------|---------|---------|---------|
| 中文缺陷名映射表 | `scripts/convert_tianchi_to_yolo.py` | **L44–80** | 20 名 → 18 类，别名归并、错别字容错 |
| 坐标格式转换 | `scripts/convert_tianchi_to_yolo.py` | **L134–146** | `xyxy_to_yolo`，三处防御 |
| 单目录转换（含去重） | `scripts/convert_tianchi_to_yolo.py` | **L148–214** | L189 是去重的 warn 打印 |
| 训练/验证划分 | `scripts/convert_tianchi_to_yolo.py` | **L229–257** | seed=42，os.rename 成对移动 |
| 数据集描述文件生成 | `scripts/convert_tianchi_to_yolo.py` | **L260–276** | 写 `data.yaml`，nc=18 |
| 主流程入口 | `scripts/convert_tianchi_to_yolo.py` | **L279–341** | 四步：转换→划分→测试集→写 meta |

### 2.2 七维奖励与训练（第三、四、五章讲义对应）

| 功能 | 文件路径 | 关键行号 | 讲解要点 |
|------|---------|---------|---------|
| 可微 IoU | `backend/training/dimension_rewards.py` | **L61–87** | `torch.maximum` vs Python `max` |
| 奖励计算器类 | `backend/training/dimension_rewards.py` | **L90–127** | 构造参数、head_type 分支 |
| D01 定位 | `backend/training/dimension_rewards.py` | **L238–253** | `2·IoU−1` |
| D02 分类 | `backend/training/dimension_rewards.py` | **L259–276** | `p_true` 替代 argmax |
| D03 校准 | `backend/training/dimension_rewards.py` | **L282–300** | `−|conf−p_true|` |
| D04 漏检 | `backend/training/dimension_rewards.py` | **L306–323** | 软阈值 sigmoid |
| D05 误检 | `backend/training/dimension_rewards.py` | **L329–350** | 置信度加权 |
| D06/D07 敏感度 | `backend/training/dimension_rewards.py` | **L372–398** | 集合概率质量 + 外积判定 |
| 奖励张量堆叠 | `backend/training/dimension_rewards.py` | **L439–443** | `compute_reward_tensor` |
| 奖励注入损失 | `backend/training/reward_rtdetr.py` | **L37–103** | 子类化 `RTDETRDetectionLoss` |
| 奖励项计算 | `backend/training/reward_rtdetr.py` | **L105–199** | 复用匈牙利匹配 |
| 空闲 query 选择 | `backend/training/reward_rtdetr.py` | **L215–243** | 处理"漏检"的关键逻辑 |
| 注入安装函数 | `backend/training/reward_rtdetr.py` | **L259–292** | 同时 patch `criterion` 与 `init_criterion` |
| 损失函数 | `backend/training/loss_functions.py` | **L20–67** | `detection_loss`：CE + SmoothL1 |
| 标量奖励损失 | `backend/training/loss_functions.py` | **L70–95** | `L_scalar = −R_scalar` |
| 训练器（可学习权重） | `backend/training/hybrid_reward_trainer.py` | **L86–98** | `dim_weights` 与模型同优化器 |
| 数据集加载器 | `backend/training/dataset.py` | **L70–128** | `FabricDataset` |
| YOLO 标签读取 | `backend/training/dataset.py` | **L159–186** | 归一化 → 像素反变换 |
| COCO 标签读取 | `backend/training/dataset.py` | **L192–226** | `image_id` 映射 |
| 自定义 collate | `backend/training/dataset.py` | **L229–259** | 单框对齐、丢弃负样本 |

### 2.3 推理与服务（第五章讲义对应）

| 功能 | 文件路径 | 关键行号 | 讲解要点 |
|------|---------|---------|---------|
| 抽象基类 | `backend/inference/engine.py` | **L25–93** | `ABC` + `@abstractmethod` |
| 预热与测速 | `backend/inference/engine.py` | **L65–93** | 模板方法，子类免费获得 |
| Dummy 引擎 | `backend/inference/engine.py` | **L96–152** | 无模型跑通全链路 |
| ONNX 引擎 | `backend/inference/engine.py` | **L155–238** | YOLO 风格输出解析 |
| PyTorch 引擎 | `backend/inference/engine.py` | **L241–352** | 原生 torch |
| 工厂函数 | `backend/inference/engine.py` | **L355–404** | `create_engine("dummy"\|"onnx"\|"pytorch"\|"rtdetr")` |
| RT-DETR ONNX 引擎 | `backend/inference/rtdetr_onnx.py` | **L26–48** | 构造参数，conf 默认 0.25 |
| 五步预处理 | `backend/inference/rtdetr_onnx.py` | **L70–83** | scale_fill + 只除 255 |
| 输出解析 | `backend/inference/rtdetr_onnx.py` | **L102–146** | `(1,300,6)` → 像素 xyxy |
| 通用预处理 | `backend/inference/preprocessing.py` | **L21–83** | letterbox + ImageNet |
| FastAPI 入口 | `backend/main.py` | **L39–148** | `create_app()`、CORS、WebSocket |
| WebSocket 端点 | `backend/main.py` | **L116–129** | `/ws/alerts` |
| 18 类枚举（事实来源） | `backend/schemas/defect.py` | 全文件 211 行 | `TIANCHI_CLASS_NAMES` |

### 2.4 训练入口与超参

| 功能 | 文件路径 | 关键行号 | 讲解要点 |
|------|---------|---------|---------|
| 主训练脚本 | `scripts/train_rtdetr.py` | 全文件 164 行 | 梯度裁剪 monkey patch 在 L87–94 |
| 最佳配方脚本 | `scripts/train_mosaic1280.py` | 全文件 71 行 | bf16 补丁 L30–34，BN 哨兵 L45–57 |
| 参数定义 | `scripts/train_rtdetr.py` | **L35–78** | `--amp`、`--clip-grad`、`--mosaic` |

### 2.5 评估与诊断工具

| 功能 | 文件路径 | 行数 | 用途 |
|------|---------|------|------|
| 官方三档 IoU 评估 | `scripts/eval_multIou.py` | 192 | 产出官方口径 mean mAP |
| 实验臂对比 | `scripts/compare_runs.py` | 198 | 读 `results.csv` 出对比表 |
| 混淆诊断 | `scripts/diagnose_confusion.py` | — | 逐类 GT→预测错分表 |
| 类别族聚类 | `scripts/cluster_defect_families.py` | 159 | 层次聚类 + bootstrap 稳定性 |
| 奖励消融 | `scripts/ablate_reward.py` | — | 七维奖励逐项消融 |
| 阈值校准 | `scripts/eval_threshold_calibration.py` | — | 置信度校准曲线 |
| 预测可视化 | `scripts/visualize_predictions.py` | 189 | 画框到图上 |

### 2.6 前端与测试

| 资源 | 路径 | 说明 |
|------|------|------|
| 中文监控看板 | `backend/dashboard.html` | Chart.js + WebSocket |
| 产线模拟器 | `backend/simulator.html` | Canvas 织物纹理渲染 |
| 一键启动（Win） | `scripts/start.bat` | `test` / `backend` / `e2e` / `demo` |
| 一键启动（Linux） | `scripts/start.sh` | 同上 |
| 奖励公式单测 | `backend/training/tests/test_dimension_rewards.py` | **21 个测试，实测 3.04s 通过** |
| 奖励注入单测 | `backend/training/tests/test_reward_rtdetr.py` | 注入逻辑验证 |
| 推理引擎单测 | `backend/inference/tests/test_engine.py` | 工厂与四种后端 |
| 告警升级单测 | `backend/tests/test_alert_escalation.py` | 严重度升级逻辑 |

---

## 三、演示环境准备

### 3.1 Python 解释器（重要）

**必须用 anaconda3 base**，它带 torch、ultralytics、CUDA：

```
C:\Users\master\anaconda3\python.exe
```

本项目下的 `.workbuddy/binaries` 托管 Python **没有 torch**，误用会直接 ImportError。

### 3.2 依赖与权重

```bash
cd "C:/Users/master/Desktop/维度模型(1)"

# 安装（可编辑模式）
pip install -e .

# COCO 预训练权重（仓库内已有，66 MB）
ls -la rtdetr-l.pt
```

### 3.3 训练好的产物

| 路径 | 说明 |
|------|------|
| `runs/rtdetr/tianchi18/weights/best.pt` | 18 类基线最优权重 |
| `runs/rtdetr/tianchi18_full/weights/best.pt` | 18 类全量训练权重 |
| `runs/rtdetr/tianchi18_mosaic1280/weights/best.pt` | **最佳配方权重（推荐演示用）** |
| `runs/rtdetr/tianchi18_nomosaic1280/` | 因子解耦对照臂 |
| `runs/multi_iou/*.json` | 官方口径评估结果 |
| `data/tianchi/` | 清洗后的数据集（8619+957） |

---

## 四、演示脚本（全部经本机实测）

### 4.1 演示一：标注重复的哈希验证（30 秒，冲击力最强）

```bash
cd "C:/Users/master/Desktop/维度模型(1)"

md5sum shujuji/*partA/partA/Annotations/anno_train.json \
       shujuji/*partB/partB/Annotations/anno_train.json
```

**预期输出**（两行 md5 完全相同）：

```
8ca9efa45e3f4ade805a12019a5880a3  .../partA/Annotations/anno_train.json
8ca9efa45e3f4ade805a12019a5880a3  .../partB/Annotations/anno_train.json
```

**讲解话术**：这就是"不去重会让 GT 翻倍"的铁证。

### 4.2 演示二：数据规模与负样本比例（30 秒）

```bash
# 训练/验证规模
echo -n "train images: "; ls data/tianchi/train/images | wc -l
echo -n "train labels: "; ls data/tianchi/train/labels | wc -l
echo -n "val images:   "; ls data/tianchi/val/images | wc -l
echo -n "val 负样本:   "; find data/tianchi/val/labels -name "*.txt" -empty | wc -l
```

**预期输出**：`8619` / `8619` / `957` / `374`

### 4.3 演示三：类别分布长尾（1 分钟）

```bash
cat data/tianchi/train/labels/*.txt | awk '{print $1}' | sort -n | uniq -c | sort -k2 -n
```

**预期输出**（节选）：

```
    273 0      # hole       破洞
   1779 3      # knot       结头
    117 4      # flower_board 花板跳
   1229 11     # weft_shrink  纬缩
```

**讲解话术**：最大 1779、最小 117，差 15.2 倍。小类召回低是数据问题不是模型问题。

### 4.4 演示四：越界标注检测（1 分钟，证明钳制的必要性）

```bash
python -c "
import glob
n=0; tot=0
for f in glob.glob('data/tianchi/train/labels/*.txt'):
    for line in open(f):
        p=line.split()
        if len(p)<5: continue
        tot+=1
        c,cx,cy,w,h=map(float,p[:5])
        if cx-w/2<-1e-9 or cx+w/2>1+1e-9 or cy-h/2<-1e-9 or cy+h/2>1+1e-9:
            n+=1
print(f'越界框 {n} 个 / 共 {tot} 个')
"
```

**预期输出**：`越界框 9 个 / 共 8570 个`

### 4.5 演示五：图像尺寸一致性（30 秒）

```bash
python -c "
from PIL import Image
import os
d='data/tianchi/train/images'
fs=[f for f in os.listdir(d) if f.endswith('.jpg')][:100]
sizes={Image.open(os.path.join(d,f)).size for f in fs}
print('尺寸集合:', sizes)
print('宽高比:', [round(w/h,2) for w,h in sizes])
"
```

**预期输出**：

```
尺寸集合: {(2446, 1000)}
宽高比: [2.45]
```

**讲解话术**：全部一致 → letterbox 会浪费 78% 行 → 解释 rect=True 的收益机制。

### 4.6 演示六：可微性的直接证明（2 分钟，算法核心）

这是讲七维奖励时最有说服力的一段——**当场证明梯度能回传**。

```bash
python -c "
import sys; sys.path.insert(0,'.')
import torch
from backend.training.dimension_rewards import differentiable_iou

g = torch.tensor([[0.5, 0.5, 0.2, 0.2]])

# 情况A：输入不带梯度（学生常见误解）
p1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
print('A 输入无requires_grad →', differentiable_iou(p1, g).requires_grad)

# 情况B：预测框带梯度（训练时的真实情况）
p2 = torch.tensor([[0.5, 0.5, 0.2, 0.2]], requires_grad=True)
out = differentiable_iou(p2, g)
print('B 预测框带梯度 →', out.requires_grad)
out.backward()
print('  反传后 p.grad =', p2.grad.tolist())
"
```

**本机实测输出**：

```
A 输入无requires_grad → False
B 预测框带梯度 → True
  反传后 p.grad = [[0.0, 0.0, 6.29425048828125e-05, 6.29425048828125e-05]]
```

**讲解话术**：情况 A 的 `False` 解释了为什么很多同学写了可微函数却发现"没有梯度"——**不是函数不可微，是输入张量没带 `requires_grad`**。情况 B 的梯度值出现在 w/h 方向，正是"框的位置还能被调整"的直接证据。

### 4.7 演示七：Dummy 引擎无模型跑通（1 分钟）

```bash
python -c "
import sys; sys.path.insert(0,'.')
from backend.inference.engine import create_engine
from backend.schemas.defect import TIANCHI_CLASS_NAMES

print('类别数:', len(TIANCHI_CLASS_NAMES))
e = create_engine('dummy')
print('引擎:', type(e).__name__)
print('benchmark:', e.benchmark(batch_size=1, iterations=5))
"
```

**讲解话术**：没有 GPU、没有权重也能跑通全链路——这是抽象基类 + 工厂模式的直接价值。

### 4.8 演示八：奖励公式单测（30 秒，证明公式可验证）

```bash
python -m pytest backend/training/tests/test_dimension_rewards.py -v --tb=short
```

**本机实测输出**：`21 passed in 3.04s`

### 4.9 演示九：启动服务看实时看板（2 分钟）

```bash
# 方式一：一键脚本
scripts\start.bat backend

# 方式二：直接启动
python -m backend.main
```

然后浏览器打开：

| 地址 | 页面 |
|------|------|
| http://localhost:8000/simulator | 产线模拟器（Canvas 织物纹理 + 缺陷标注动画）|
| http://localhost:8000/dashboard | 监控看板（4 标签 + Chart.js + WebSocket 实时告警）|
| http://localhost:8000/docs | Swagger API 文档 |

**讲解话术**：看板上的告警是 WebSocket 实时推送的，后端 `/ws/alerts` 端点；`ping`/`pong` 是心跳机制。

### 4.10 演示十：因子解耦对照表（1 分钟）

```bash
for d in tianchi18_full tianchi18_nomosaic1280 tianchi18_mosaic1280 tianchi18_hi1536_v4; do
  f="runs/rtdetr/$d/results.csv"
  [ -f "$f" ] && python -c "
import pandas as pd
df=pd.read_csv(r'$f'); df.columns=[c.strip() for c in df.columns]
c=[x for x in df.columns if 'mAP50(B)' in x][0]
print(f'$d: best mAP50={df[c].max():.4f} @ep{df[c].idxmax()}')
"
done
```

**本机实测输出**：

```
tianchi18_full:           best mAP50=0.5414 @ep55
tianchi18_nomosaic1280:   best mAP50=0.5707 @ep49
tianchi18_mosaic1280:     best mAP50=0.5887 @ep52
tianchi18_hi1536_v4:      best mAP50=0.5693 @ep59
```

**讲解话术**：这四条构成完整解耦链——基线 0.5414 →加 rect 0.5707（+0.029）→ 加 mosaic 0.5887（+0.018）→ 换 1536 降到 0.5693（−0.019，无净增益）。

> **注意口径**：以上是 `mAP50(B)` 单口径。官方口径 mean mAP（IoU0.1/0.3/0.5 三档平均）下 rect +0.036、mosaic +0.027。**两口径下 rect 贡献都大于 mosaic，但引用时必须标明是哪个口径。**

### 4.11 演示十一：查看各类 AP 差异（1 分钟）

```bash
python scripts/compare_runs.py \
    --run baseline=runs/rtdetr/tianchi18_full \
    --run mosaic=runs/rtdetr/tianchi18_mosaic1280 \
    --out runs/compare/demo.md
```

会输出逐类 AP@0.5 差值表与关键类对比。

---

## 五、现场演示的时间分配建议

| 演示 | 时长 | 讲什么 | 优先级 |
|------|------|--------|--------|
| 一 哈希验证 | 30s | 数据坑的铁证 | ★★★★★ |
| 三 类别分布 | 1min | 长尾问题 | ★★★★ |
| 六 可微性 | 2min | 七维奖励的立身之本 | ★★★★★ |
| 十 因子解耦 | 1min | 方法论 | ★★★★ |
| 九 启动看板 | 2min | 全链路闭环 | ★★★ |
| 四 越界检测 | 1min | 防御性编码 | ★★ |
| 五 尺寸一致性 | 30s | rect 收益机制 | ★★ |
| 八 单测 | 30s | 公式可验证 | ★★ |
| 七 Dummy 引擎 | 1min | 架构价值 | ★★ |

**如果只有 5 分钟**：演示一 + 演示六 + 演示十。

---

## 六、演示时的常见问题预案

**Q：能不能现场重新训练？**

不建议。60 epoch 在本机约需 1.5–2 小时，且**训练期间严禁并发任何 GPU 进程**——并行的推理脚本被终止时，飞行中的 CUDA kernel 会造成 GPU 级故障，杀死同卡训练进程（CUDA illegal memory access，与训练代码无关）。演示时只跑评估与推理。

**Q：为什么用 18 类而不是 20 类？**

原始 20 类中有两对视觉上几乎无法区分：断经↔磨痕、纬缩↔粗纬。合并后关键类召回显著回升（纬缩+粗纬 0.40→0.88，断经+磨痕 0.52→0.79）。这是标签噪声天花板，不是模型能力问题。详见 `backend/schemas/defect.py` 与讲义第六章。

**Q：18 类是怎么合并的？会不会是我主观决定的？**

不是。项目做了数据驱动的聚类（`scripts/cluster_defect_families.py`），层次聚类 + bootstrap 稳定性检验证明 **k=3 族稳定（support=1.00）**，k=6 族不被数据支持（稳定性 0.286）。合并的 18 类本身来自对混淆矩阵加裁剪图的人工复核。

**Q：模型指标和公开方法比如何？**

Tianchi 20 类口径下 mAP50 = 59.4%，与 Fab-ME（YOLOv8s）持平，低于 SPFFNet（YOLOv11）的 65.8%。**注意 SPFFNet 报告的是 8 类子集口径，不可直接对比。**

**Q：为什么 fp16 会让 mAP 变成 0？**

fp16 最大值 65504，neck 中 39 个 BN 层激活值平方后逼近上限，cuDNN 的 BN 方差统计溢出导致 `running_var` 归零；推理时 eval 模式用 `sqrt(running_var + eps)`，被 eps 放大 100–400 倍。修复用 bf16（指数位与 fp32 同为 8 位）。**这个坑要 9 个 epoch 以上才显现，1–2 epoch 冒烟测不出来。**

---

## 七、一页速查：所有演示命令

```bash
cd "C:/Users/master/Desktop/维度模型(1)"
PY="C:/Users/master/anaconda3/python.exe"

# 1 哈希验证
md5sum shujuji/*partA/partA/Annotations/anno_train.json shujuji/*partB/partB/Annotations/anno_train.json

# 2 数据规模
ls data/tianchi/train/images | wc -l; ls data/tianchi/val/images | wc -l
find data/tianchi/val/labels -name "*.txt" -empty | wc -l

# 3 类别分布
cat data/tianchi/train/labels/*.txt | awk '{print $1}' | sort -n | uniq -c

# 4 越界检测
$PY -c "import glob;n=0;t=0
for f in glob.glob('data/tianchi/train/labels/*.txt'):
    for l in open(f):
        p=l.split()
        if len(p)<5: continue
        t+=1;c,cx,cy,w,h=map(float,p[:5])
        if cx-w/2<-1e-9 or cx+w/2>1+1e-9 or cy-h/2<-1e-9 or cy+h/2>1+1e-9: n+=1
print(f'越界 {n}/{t}')"

# 5 图像尺寸
$PY -c "from PIL import Image;import os
d='data/tianchi/train/images'
fs=[f for f in os.listdir(d) if f.endswith('.jpg')][:100]
print({Image.open(os.path.join(d,f)).size for f in fs})"

# 6 可微性
$PY -c "import sys;sys.path.insert(0,'.');import torch
from backend.training.dimension_rewards import differentiable_iou
g=torch.tensor([[0.5,0.5,0.2,0.2]])
p=torch.tensor([[0.5,0.5,0.2,0.2]],requires_grad=True)
o=differentiable_iou(p,g);o.backward()
print('可微:',o.requires_grad,'grad:',p.grad.tolist())"

# 7 Dummy 引擎
$PY -c "import sys;sys.path.insert(0,'.')
from backend.inference.engine import create_engine
e=create_engine('dummy');print(type(e).__name__,e.benchmark(1,5))"

# 8 奖励单测
$PY -m pytest backend/training/tests/test_dimension_rewards.py -q

# 9 启动服务
$PY -m backend.main
# 浏览器 http://localhost:8000/dashboard

# 10 因子解耦
for d in tianchi18_full tianchi18_nomosaic1280 tianchi18_mosaic1280 tianchi18_hi1536_v4; do
  f="runs/rtdetr/$d/results.csv"
  [ -f "$f" ] && $PY -c "
import pandas as pd
df=pd.read_csv(r'$f');df.columns=[c.strip() for c in df.columns]
c=[x for x in df.columns if 'mAP50(B)' in x][0]
print('$d', round(df[c].max(),4))"
done
```
