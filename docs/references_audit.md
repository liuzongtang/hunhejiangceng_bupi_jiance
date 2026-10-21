# 外部来源核实记录（references_audit）

**日期**：2026-10-21（阶段 0，计划文档 §四-阶段 0）
**核实人**：研究助理（arXiv 精确检索 + 官方 proceedings + DOI 交叉验证）
**铁律**：未核实的外部论文不进参考文献；预印本必须标注"arXiv preprint"，不得冒充正式发表；只保留概念不保留署名仅适用于"未找到"条目。

---

## 一、逐条核实结果

### 1. Dr. Seg —— ✅ 已核实（arXiv 预印本）

| 项 | 内容 |
|---|---|
| 标题 | Dr. Seg: Revisiting GRPO Training for Visual Large Language Models through Perception-Oriented Design |
| arXiv | **2603.00152**（2026-03 首次提交） |
| 作者 | Haoxiang Sun, Tao Wang, Chenwei Tang, Li Yuan, Jiancheng Lv（四川大学 + 北京大学） |
| 发表状态 | **arXiv 预印本，未见正式发表记录**（引用须标 preprint） |
| 代码 | github.com/eVI-group-SCU/Dr-Seg；模型 HuggingFace hao05/Dr_Seg |

**核心内容**：GRPO 从语言推理迁移到视觉感知任务不最优；提出 Look-to-Confirm 机制 + **Distribution-Ranked Reward 模块**（用滚动历史队列中的经验分位数/排名替换原始奖励值，获得尺度不变奖励，**防止高方差目标主导优化**）。

**与本项目的对应**：B2「分布排序七维评分」的概念源头。**注意边界**：Dr. Seg 面向 VLLM 推理分割/检测（VLLM+SAM2 管线），非封闭集检测器；我们的 B2 是把"分布排序"思想迁移到产线七维评分的回放分析，属概念迁移而非方法复现。

### 2. Mixed-R1 —— ✅ 已核实（arXiv 预印本）

| 项 | 内容 |
|---|---|
| 标题 | Mixed-R1: Unified Reward Perspective For Reasoning Capability in Multimodal Large Language Models |
| arXiv | **2505.24164**（2025-05-30） |
| 作者 | Shilin Xu, Yanwei Li, Rui Yang, Tao Zhang, … Xiangtai Li, Hao Fei（字节跳动 + 北大 + NUS） |
| 发表状态 | **arXiv 预印本**（Google Scholar 记录 16 次引用） |
| 代码 | github.com/xushilin1/mixed-r1 |

**核心内容**：混合奖励函数设计（匹配奖励/图表奖励/**IoU 奖励**/开放式 BMAS 奖励）+ Mixed-45K 混合后训练数据集，解决多源 MLLM 任务稳定 RL 问题。

**与本项目的对应**：多维度奖励设计的先例（七维奖惩的相关工作引用）。IoU reward for grounding 与我们检测任务奖励设计最相关。

### 3. DINO-R1 —— ✅ 已核实（arXiv 预印本）★与 B3 最相关

| 项 | 内容 |
|---|---|
| 标题 | DINO-R1: Incentivizing Reasoning Capability in Vision Foundation Models |
| arXiv | **2505.24025**（2025-05） |
| 作者 | Chenbin Pan, Liu Ren, Wenbin He, Zhengzhong Tu |
| 发表状态 | **arXiv 预印本**，未见正式发表记录 |
| 基线 | Grounding DINO（query-based 检测器，与 RT-DETR 同族） |

**核心内容**：提出 **GRQO（Group Relative Query Optimization）**：① **query 级相对奖励模块**——每个 query 在组内评估质量，相对组均值计算归一化奖励（鼓励每个 query 超过动态组基线，提供比 one-to-one 匹配更密的训练信号）；② **objectness 分布上的 KL 正则化**——缓解训练漂移与灾难遗忘。

**与本项目的对应**：B3「查询级组内归一化辅助奖励 + KL」的概念源头，且架构最近（query-based 检测器）。论文自报超参：奖励权重 10³、KL 权重 0.04 时最优——B3 设计时可作起点参考（仍须按我们协议重新扫描）。

### 4. MMS-PRM —— ✅ 已核实（IJCAI 2026 正式发表）★与 C6 最相关

| 项 | 内容 |
|---|---|
| 标题 | Improving Multimodal Reasoning via Worst Dimension Optimization |
| 会议 | **IJCAI 2026 正式 proceedings**（ijcai.org/proceedings/2026/166） |
| arXiv | 2606.07801 |
| 作者 | Haocheng Lv, Huaping Zhang, Qiuchi Li, Lei Li, Chunxiao Gao（北京理工大学） |
| 代码 | github.com/leibniz-Man/MMS-PRM |

**核心内容**：MMS-PRM（Multimodal Multi-Dimensional Scalarization PRM）——**非补偿式多目标过程奖励**：平均分会掩盖单维度失败；层次化细粒度奖励空间 + Chebyshev 引导 MCTS（聚焦最差维度）+ 课程式 DPO。自报数据：SFT 67.4 → full 79.7（M3CoT val）；加权和 MCTS 75.3 vs Chebyshev MCTS 79.7。

**与本项目的对应**：C6「最差维度定向优化（broken_spandex）」的概念源头；也是论文 5.5「平均分掩盖局部失败」论述的最佳外部支撑——**这是我们唯一一条正式发表（非预印本）的新引用**。

### 5. PDCR —— ✅ 已核实（arXiv 预印本，疑 CVPR 2026）

| 项 | 内容 |
|---|---|
| 标题 | PDCR: Perception-Decomposed Confidence Reward for Vision-Language Reasoning |
| arXiv | **2605.13467**（2026-05） |
| 作者 | Hee Suk Yoon, Chang D. Yoo（KAIST） |
| 发表状态 | **CVPR 2026 正式发表**（CVF Open Access: openaccess.thecvf.com/content/CVPR2026/html/Yoon_PDCR_...，pp. 18881–18891；dblp: conf/cvpr/YoonYHEKHDLY26） |

**核心内容**：**簇内归一化**（intra-cluster normalization）——全局奖励在异构任务上造成 "mixture-induced signal degradation"（视觉稀疏步骤的信号被文本密集步骤统计扭曲）；用 Visual Dependence Score（真实图 vs 白图对数概率差）+ Otsu 无监督聚类分解感知/推理步骤，各簇内独立归一化置信增益。

**与本项目的对应**：C5「训练侧簇内归一化」的概念源头。**审稿意见警示（pith.science 收录的 referee report）**：摘要无定量结果、硬二值聚类、增益幅度存疑——论文引用时应仅引"簇内归一化"思想，不引其效果数字。

### 6. "方差霸权" —— ❌ 未找到出处（概念性表述）

arXiv 中英文检索均无该术语的技术文献；返回结果均为无关的中文政论/金融博文（"梯度、方差、屏障"文明论、均值-方差模型等）。

**处置（按计划 §二 硬规则）**：
- 论文中仅作**概念性表述**，不署名、不进参考文献；
- 该机理（多目标中高方差项主导梯度）在 Dr. Seg（2603.00152）与 PDCR（2605.13467）中有明确的技术诊断与命名（"high-variance objectives dominate gradients" / "mixture-induced signal degradation"）——**机理段可引这两篇已核实文献支撑，原创表述归我们**。

---

## 二、可进参考文献列表的条目（均须标注发表状态）

| # | 文献 | 状态标注 |
|---|---|---|
| N1 | Sun et al., Dr. Seg, arXiv:2603.00152, 2026 | arXiv preprint |
| N2 | Xu et al., Mixed-R1, arXiv:2505.24164, 2025 | arXiv preprint |
| N3 | Pan et al., DINO-R1, arXiv:2505.24025, 2025 | arXiv preprint |
| N4 | Lv et al., Improving Multimodal Reasoning via Worst Dimension Optimization, IJCAI 2026 | **正式发表**（ proceedings 页码 166） |
| N5 | Yoon et al., PDCR, CVPR 2026, pp. 18881–18891 | **正式发表**（2026-10-21 CVF Open Access 二次确认） |

**不进列表**："方差霸权"（未找到出处）；MMS-PRM 与 MMS-PRM 概念相关的其他 MLLM PRM（Perception-R1 ICLR 2026 等）未逐一核实，本轮不引。

## 三、对论文的影响

1. 原 19 条文献全部为此前已核实条目，**不受本次新增影响**（0.3 项通过）。
2. 若论文 4.16（B2/B3/C5/C6 相关工作）需要外部支撑，按上表 N1–N5 标注状态引用；机理段（阶段 4.1）引 N1/N5 的诊断、原创表述归我们。
3. B3/C5/C6 实验设计前，先读对应原文方法段（DINO-R1 GRQO / PDCR 簇内归一化 / MMS-PRM 最差维度），确保概念迁移边界在论文中写清（面向 VLLM/开放集，我们是封闭集 RT-DETR）。

## 四、遗留待办

- [x] ~~PDCR 的 CVPR 2026 接收状态~~ → **已确认：CVPR 2026 正式论文**（CVF Open Access + dblp 双源）
- [ ] Dr. Seg / DINO-R1 是否被后续会议接收：检索一次官方 proceedings（每轮定稿前复查）
