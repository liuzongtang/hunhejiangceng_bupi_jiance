## 附录二 专有名词解释

本节汇总讲义中出现的核心术语，按字母与拼音顺序排列，供答辩与复习时查阅。

### A–C 字母区间

**AMP（Automatic Mixed Precision，自动混合精度）**

用低精度浮点数（fp16/bf16）执行部分运算以加速训练的技术。本项目踩过两个相关坑：ultralytics 硬编码的梯度裁剪阈值 10.0 过松导致 NaN 发散；fp16 因动态范围上限只有 65504，使 neck 中 39 个 BN 层的 running_var 被清零，推理时被 sqrt(eps) 放大百倍，mAP 直接崩为 0。最终改用 bf16 修复。

**BN（Batch Normalization，批归一化）**

对每层输出做标准化以稳定训练的网络层。分训练态（用当前批次的均值方差）与推理态（用累积的 running_mean / running_var）两种模式。本项目的 BN 污染问题正是出在推理态读取了被污染的 running_var。

**Backbone（骨干网络）**

主干特征提取网络。本项目为 ResNet + 混合注意力结构，从 COCO 预训练权重初始化，在织物数据上全参数微调。

**bf16（bfloat16，脑浮点数）**

一种 16 位浮点格式，指数位与 fp32 同为 8 位，因而动态范围与 fp32 一致，不会像 fp16 那样溢出。是本项目最终采用的混合精度方案。

**criterion（损失准则对象）**

ultralytics 中负责计算损失并以字典形式返回各项损失的类。本项目通过子类化 RTDETRDetectionLoss 并往返回字典追加一个 loss_reward 键，实现了奖励项对训练循环的零侵入注入。

**DETR（DEtection TRansformer）**

一种基于 Transformer 的目标检测器，核心机制是用二分图匹配（匈牙利算法）做预测框与真实框的一对一匹配，因此不需要 NMS。RT-DETR 是其改进版本。

### D–H 字母区间

**Decoder（解码器）**

RT-DETR 中逐层 refining 查询集合的模块，共 6 层。本项目使用可变形注意力（deformable attention），其内部 grid_sample 在 fp16 下数值不稳定，是 18 类训练发散的关键位置。

**D01–D07（七维奖励维度）**

项目提出的七类可微奖励：定位（2·IoU−1）、分类（2·p_true−1）、校准（−|conf−p_true|）、漏检（−λ·sigmoid((θ−IoU)/τ)）、误检（−λ·conf·(1−matched)）、断经敏感（3·p_crit−2）、跳花敏感（3·p_crit−2）。每一维都把硬决策替换为软指示，以保留梯度。

**denoising（去噪查询）**

DETR 系模型训练时的辅助机制，通过给查询加入噪声构造难样本以加速收敛。

**epoch（轮次）**

模型完整遍历一次训练集称为一个 epoch。本项目主训练 60 epoch，patience=20 表示验证指标 20 轮不提升则提前停止。

**FPN（Feature Pyramid Network，特征金字塔网络）**

本项目 Neck 层的结构，通过自顶向下路径融合多尺度特征，让细线状缺陷在高分辨率特征图上获得表达。这也是 39 个 BN 层所在的位置。

**focal loss / VFL（变分焦点损失）**

一种针对类别不平衡的分类损失，降低易分样本的权重。本项目 RT-DETR 的分类头采用 VFL，且为逐类 sigmoid 输出而非 softmax——这是奖励代码里 head_type 分支存在的原因。

**factor decoupling（因子解耦）**

本项目的方法论核心，指通过单变量对照实验分离各配置因素的独立贡献。例如补齐 nomosaic1280 对照臂后，才能确认主贡献来自 rect 而非 mosaic。

### J–M 字母区间

**jpeg / jpg**

图像压缩格式。天池数据集的缺陷图均为该格式，文件名为哈希串如 002775affdee8d951419014376.jpg。

**label noise（标签噪声）**

标注本身错误或不可判别的训练数据缺陷。本项目最重要的发现：断经与磨痕、纬缩与粗纬在图像上视觉几乎相同、双向混淆，构成标签噪声天花板，导致断经召回仅约 0.45，且任何训练干预都无法突破。

**letterbox vs scale_fill（缩放方式）**

两种把非方形图像变成方形输入的方法。letterbox 等比缩放后填充黑边；scale_fill 不等比拉伸、无填充。本项目 RT-DETR 采用 scale_fill，因为 letterbox 的黑边会造成训练与推理的分布偏移，且拉伸后坐标反变换只需乘以原始宽高。

**long-tail distribution（长尾分布）**

少数类样本量极大、多数类极小的类别分布。本项目训练集最大类（三丝）1779 个框、最小类（花板跳）117 个框，相差 15.2 倍。

**mAP（mean Average Precision，平均精度均值）**

检测任务的综合指标。先由混淆矩阵算出 Precision 与 Recall，绘制 PR 曲线求面积得 AP，再对所有类别与不同 IoU 阈值取平均。天池官方口径为 mean(mAP@IoU0.1, mAP@IoU0.3, mAP@IoU0.5)。

**mosaic（四图拼接增强）**

数据增强手段，把四张图拼成一张训练图。本项目实测贡献 +0.018 mAP50。

### N–Z 字母区间

**negative sample（负样本）**

不含任何缺陷的图像。检测任务必须见过负样本，否则模型会在完好织物上乱框。本项目验证集 957 张里有 374 张是负样本，占 39%。

**NMS（Non-Maximum Suppression，非极大值抑制）**

传统检测器的后处理，用于去除同一物体的重复框。RT-DETR 因采用一对一集合预测，天然不需要 NMS——这是它相比 YOLO 的核心结构优势。

**nn.Parameter（可学习参数）**

PyTorch 中注册为模型参数的张量，会自动获得梯度。本项目把七维奖励权重定义为 nn.Parameter，与模型参数一起被 AdamW 更新，而非写死在超参中。

**objectness（目标置信度）**

YOLO 系输出中的第 5 个通道，表示该框包含物体的概率。RT-DETR 输出无此通道，其 (1,300,6) 张量格式为 [cx, cy, w, h, score, class]。

**precision / recall（精确率 / 召回率）**

precision = TP/(TP+FP)，recall = TP/(TP+FN)。精确率衡量"报出来的对不对"，召回率衡量"该报的有没有漏"。本项目奖励注入的实测效果是召回 +0.030、精确率 −0.022——典型的零和权衡。

**running_var（BN 层的运行方差）**

BN 层在推理时使用的方差累计量。本项目 fp16 实验中该值被清零，是 AMP 污染的判据，也是 BN 哨兵监控的检查对象。

**rect（矩形批次）**

按图像真实长宽比组批，避免传统做法把宽图拉伸成正方形再填充大量黑边。本项目实测贡献 +0.029 mAP50，是三个因子中最大的一项，同时节省 59% 算力。

**RT-DETR-L**

本项目使用的检测器，L 为 Large 规格。输入 1280×1280 拉伸方图，输出 (1,300,6) 张量，COCO 预训练后在天池 18 类数据上全参数微调。

**scale_fill**

见 letterbox 条目。

**soft IoU（软 IoU）**

用 torch.maximum / torch.minimum 等可微算子实现的 IoU，相对于 Python 内置 max/min 能保留梯度。本项目不同iable_iou 函数的核心设计。

**std / mean（均值与标准差）**

归一化参数。通用预处理用 ImageNet 的 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]；RT-DETR 预处理则只除以 255.0，不做均值方差归一化——这是两套预处理的关键差异之一。

**spare query（空闲查询）**

DETR 系模型中未匹配到任何真实框的查询槽位。本项目奖励代码用空闲 query 指向被漏检的关键类目标，强行施加召回梯度，避免漏检变成零梯度。

**τ / lambda（奖励超参）**

τ 为 sigmoid 软指示的温度系数，控制软阈值的平滑程度；lambda 为漏检与误检惩罚的权重，本项目默认 lambda_miss=1.5、lambda_fp=1.0。

**Ultralytics**

本项目使用的目标检测训练框架，提供 YOLO / RT-DETR 的统一 API。项目通过 monkey patch 与子类化两种方式对其做零侵入扩展。

**YOLO 格式**

一种标注格式约定：images 子目录放图、labels 子目录放同名 .txt，每行为 class cx cy w h（均归一化到 0~1）。空 .txt 文件表示负样本。

**grid_sample**

PyTorch 中按坐标采样图像像素的可微插值操作。RT-DETR 的可变形注意力依赖它，该操作在 fp16 下数值溢出，是 18 类训练 NaN 的根因。

**focal / vfl head（检测头类型）**

逐类输出 sigmoid 概率的检测头，与 softmax 单标签头相对。本项目奖励代码通过 head_type 参数兼容两者——sigmoid 头取关键类集合的最大值，softmax 头取概率求和。

**monkey patch（猴子补丁）**

运行时替换库函数而不修改库源码的做法。本项目用它把 ultralytics 的 autocast 换成 bf16 版本、把梯度裁剪阈值从 10.0 降到 1.0。
