# 混合奖惩机制重构再出发 —— 交接执行清单

| 项 | 内容 |
|---|---|
| 文档版本 | v1.0（2026-10-21） |
| 交接对象 | 项目第二位成员（下称"执行人"） |
| 上游依据 | `docs/plan_next_steps.md` v2.0、`docs/plan_stage1.3_rerun.md`、论文 `docs/paper/paper_draft.md` v2.1、`scripts/RUNBOOK.md` |
| 目标 | 在"部分推翻重来 + 允许换模型/换架构"的前提下，系统性冲击"让七维混合奖惩在该领域产生正收益"；所有结论三种子、预注册判据、失败有明确转向 |
| 当前最强基线 | mosaic1280，官方口径 mean mAP **0.6349**，权重 `runs/rtdetr/tianchi18_mosaic1280/weights/best.pt` |
| GPU 现状 | Copy-Paste 训练链占用中（ETA 约 10-22 清晨，见 `logs/stage3_cp1280.log`）；**启动任何 GPU 任务前必须先做 R0 检查** |

> **给执行人的第一句话**：你不需要读懂整个项目。按本文档顺序执行即可；每个实验都给了"命令 + 成功判据 + 失败后怎么办"。遇到本文档未覆盖的情况，记录到 §13 结果表并提问，**不要自行发明判据**。

---

## 0. 三条铁律与工作方式

### 铁律 1：GPU 全程不并发

训练链运行期间，**严禁启动任何其他 GPU 进程**（包括推理、评测、可视化）。2026-10-20 事故：并行的推理脚本被工具超时 SIGTERM 时，飞行中的 CUDA kernel 造成 GPU 级故障，连带杀死了同卡训练进程（CUDA illegal memory access，与训练代码无关）。CPU 任务（回放分析、探针、数据构建）可随时并行。

每次启动 GPU 任务前执行：

```bash
nvidia-smi --query-gpu=memory.used --format=csv   # 应 < 2000 MiB
```

### 铁律 2：三种子 + 预注册判据

- 单种子显著性线 ±0.005 对种子间离散度（实测 0.0093）**过紧**。任何"转正/阴性"结论必须有 3 个种子（42 / 123 / 2024）。
- 每个实验臂的判据在开跑前写进 `runs/<阶段>/<臂>/PREREG.md`（一句话即可），**事后不移动**。
- 判据模板：**转正 = 官方 mean > 同协议对照 +0.005 且 crit_recall 不降**；|Δ| ≤ 0.0093 = 噪声，补种子；双降 = 见 §11 失败手册。

### 铁律 3：先探针，后实验

头量探针（§3，CPU 为主，约 1 天）决定后面哪些臂值得上 GPU。**禁止跳过探针直接开训**——本项目已经为"没探针就开训"付过 12+  GPU 天的学费。

### 工作方式

- 每个臂一个 run 名：`<阶段><臂>_<seed>`，结果落 `runs/<阶段>/<臂名>/`。
- **官方口径评测只认 `scripts/eval_multIou.py` 管道**，输出 JSON 存档到 `runs/multi_iou/<臂名>_seed<N>_official.json`。
- 每阶段结束写一页 `runs/<阶段>/REPORT.md`：假设 / 命令 / 结果 / 判定 / 转向。
- 所有判定填入 §13 总表。

### 第 1 天操作卡（按顺序打勾）

```
[ ] R0  检查 GPU 空闲（Copy-Paste 链是否结束）
[ ] R1  环境自检（python/torch/ultralytics/GPU）
[ ] R2  数据完整性校验（957/583/374/953/泄漏 0）
[ ] R3  基线复现（mean 必须 = 0.6349 ± 0.002），记下 B*
[ ] R4  抄写统一对照协议（微调协议 / 从头训协议）
[ ] R5  冒烟协议自验（train_adaptive_loss.py --fraction 0.05 跑通）
[ ] P0.1 类别侧 oracle（eval_confusion_decomp.py --tau 0.001）
[ ] P0.2 定位侧 oracle 快速版（读 handoff_baseline.json 的 AP@0.1−AP@0.5 差）
[ ] S0  测试时选择实验（CPU，见 §3.4）
[ ] A2  门控奖励回放（CPU，可与后续 GPU 臂并行，见 §5.2）
```

---

## 1. 五分钟背景（不需要读代码）

### 1.1 任务、数据与指标

- **任务**：天池织物瑕疵检测。给定布匹图像，检出 18 类缺陷的框；产线上还需要逐图判断"有/无瑕疵"。
- **数据**：`data/tianchi/`（YOLO 格式，勿改动 `data.yaml`，UTF-8）。train 8619 张 / val 957 张；val = 374 张无瑕疵 + 583 张有瑕疵（共 953 个 GT 框）。测试集 A/B 为比赛提交集。
- **官方指标**：mean(mAP@IoU0.1, mAP@IoU0.3, mAP@IoU0.5)；初赛总分 = 0.2·ACC + 0.8·mAP。**ACC 是图像级判定，与检测 mAP 是两个轴**（这一点是 §5 阶段 A 的根据）。
- **关键类**（召回最差的三类，本项目全程跟踪）：hole 破洞 / broken_warp 断经 / broken_spandex 断氨纶。

### 1.2 资产速查表

| 资产 | 路径 |
|---|---|
| 当前最强权重 | `runs/rtdetr/tianchi18_mosaic1280/weights/best.pt`（mean 0.6349 / mAP50 0.5917） |
| 旧基线权重 | `runs/rtdetr/tianchi18_full/weights/best.pt`（mAP50 0.541，mean 0.572） |
| 从头训配方 | `scripts/train_mosaic1280.py`（含 bf16 补丁 + BN 哨兵，直接抄） |
| 官方口径评测 | `scripts/eval_multIou.py`（IoU 0.1/0.3/0.5，方形 letterbox，conf 0.001） |
| 微调对照协议样板 | `scripts/train_adaptive_loss.py`（文件头注释即协议定义） |
| 七维奖惩（检测版） | `scripts/train_rtdetr_reward.py` + `backend/training/dimension_rewards.py`、`reward_rtdetr.py` |
| 其他干预臂 | `scripts/train_rtdetr_{boxweighted,costsensitive,nwd,objectness,grpo}.py` |
| 混淆分解（权威） | `scripts/eval_confusion_decomp.py` → `runs/confusion/decomp_*.json` |
| 图像级二分类回放 | `scripts/eval_binary_replay.py` → `runs/binary_replay/REPORT.md` |
| 缺陷族结构与分类器 | `runs/family_clustering/families.yaml`、`family_cls_best.pt`（3 族 top-1 0.9119） |
| 阈值标定 | `runs/threshold_calibration/threshold_sweep.csv` |
| 论文（结论回填处） | `docs/paper/paper_draft.md` |

### 1.3 七维混合奖惩是什么

七个可微奖励维度，以 `loss_reward = −β·mean(R)` 并入检测损失（β=0.5，A/B）：

| 维度 | 名称 | 工业语义 |
|---|---|---|
| D01 | 定位 soft IoU | 框回归质量 |
| D02 | 分类 smooth accuracy | 类别判别正确性 |
| D03 | 校准 gap | 置信度可信度 |
| D04 | 漏检 soft miss | 带病出厂代价最高 |
| D05 | 误检 soft FP | 良品误判报废 |
| D06 | 断经·断氨纶敏感度 | 高危关键类 |
| D07 | 星跳·死皱敏感度 | 高影响关键类 |

三个接口层：Layer-1 可微奖励（注入损失）；Layer-3 GRPO（查询噪声 REINFORCE，已证伪）；决策层回放（对缓存预测事后打分，CPU）。

> 注意：`mixed_reward/` 包和 `configs/mixed_reward_*.yaml` 是 LLM/verl 路线的残留（Qwen + accuracy/safety/completeness/format），**本任务不使用**，勿被其误导。

### 1.4 失败史与机理——本清单的出发点

**已失败的全部尝试**（均有存档，细节见论文 4.4–4.16）：

| 实验 | 结果签名 | 存档 |
|---|---|---|
| 七种干预（弱基线） | 全部不敌均匀微调 0.606 | 论文 4.7 |
| R2 box 严重度加权 | mean **−0.023** 换 crit_recall +0.104 | `runs/multi_iou/strong_r2_*.json` |
| Direction B 分类加权 | mean **−0.029** 换 crit_recall +0.184 | 同上 |
| R3 NWD 度量替换 | mean **−0.015** 换 crit_recall +0.124 | 同上 |
| Layer-1 可微奖励 | 单 seed 表观 +0.0007，**三种子复测 Δmean −0.0096±0.0093（1/3 正）→ 确认阴性** | `runs/multi_iou/l1_seed*`、`strong_l1_*` |
| GRPO | mean −0.306，模型崩溃 | `strong_l3_grpo.json` |
| UW 自适应加权 | 四种子仅 1/4 转正（噪声） | `runs/multi_iou/uw_seed*` |
| 置信度重校准 | isotonic 修好 ECE（0.106→0.037）但 **AUC 0.763/0.823 严格不变** | 论文 4.13 |
| 分布排序七维评分 | 0.9248 < 门 0.9373，**NEG** | `runs/alarm_replay/stage1/` |
| 集成不确定性 | 阴性 | 论文 4.15 |

**机理三段论**（论文 5.1/5.2/5.5，本清单的全部策略由此推出）：

1. **PR 曲线滑动**：已收敛检测器的精度-召回曲线由表征决定。在收敛权重上做任何"损失重加权/度量替换"，只能沿曲线滑动——召回↑必然精度↓，mean mAP 净降。R2/B/NWD 三臂同一签名（mean −0.015~−0.029 换 crit_recall +0.10~+0.18）证明这是系统性结构特征，不是调参问题。
2. **类别侧几乎没有头量**：GT 实例级混淆分解显示错分仅 **7/166 ≈ 4.2%**，且错分全部落在同族簇内。任何针对分类侧重加权的干预，最多能动这 4%，还得付出精度代价——零和是注定。
3. **真正的瓶颈**：① 漏检——但 conf 地板处召回上限 89%~100%，说明"检得出但排得靠后"（置信度排序问题，TP/FP 可分性 AUC 仅 0.76~0.82）；② FP 质量。这两者都不是"重加权"能解决的，需要表征/架构变化，或换一个不触发零和的决策轴。

**推论：奖惩要发光，必须离开"收敛模型损失重加权"这个接口。**

### 1.5 总策略——奖惩的四个新位置

| 新位置 | 逻辑 | 对应阶段 |
|---|---|---|
| ① 决策层（图像级/产线轴） | 图像级判定的正确率、停机档误报与检测 mAP **不构成 PR 零和**（两个指标轴）；已有正证据：二分类回放 93.73%/AUC 0.9808 有头量，objectness 探针是唯一正结果 | 阶段 A（+S0） |
| ② 子问题层（族级/定位） | 族级 3 分类与低 IoU 定位是"更容易的子问题"，其 PR 曲线离最优更远，头量更大；定位头量上界 mean(AP@0.1−AP@0.5)=0.074 | 阶段 B / D |
| ③ 表征层（训练早期） | 收敛模型上奖惩只能滑动曲线；从头训**前 10 epoch** 施加奖励，可能在曲线定型前外移它 | 阶段 C |
| ④ 推理层（测试时计算） | 不改权重、只改选择——零和在构造上不可能发生；奖励当候选检测集的打分器 | S0 → 阶段 S2 |

**总路线**：先用 1 天 CPU 探针（§3）测定头量分布 → 按路由表启动 GPU 队列（§4）→ 每臂三种子判定 → 全负则按 §12 停止条件转入价值整理（负结果链 + 产线决策层交付本身就是完整成果）。

---

## 2. 阶段 R：地基重建与基线锁定（第 1 天上午）

"推翻重来"指：不继承任何来路不明的中间状态，从环境/数据/基线/协议四样重建，后面所有臂与同一把尺子比。

### R0 前置检查（GPU）

```bash
tail -5 logs/stage3_cp1280.log          # Copy-Paste 链日志，仍在更新=在跑
nvidia-smi --query-gpu=memory.used --format=csv
```

- 训练还在跑 → **等待**（ETA 约 10-22 清晨），期间只做 CPU 任务（§3 的 P0.*、S0、A2、附录脚本）。
- 已结束 → 读 `runs/stage3_cp/` 判定报告：若 Copy-Paste 转正（mean > 0.6399），把本清单所有门的基线值 **B\*** 更新为新值，并在 §13 表记录。

### R1 环境自检

```bash
cd "C:/Users/master/Desktop/维度模型(1)"
C:/Users/master/anaconda3/python.exe -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

期望输出 `2.8.0+cu126 True 8.4.91`。**不要使用 `.workbuddy/binaries` 下的 Python（无 torch）**。

### R2 数据完整性校验

```bash
C:/Users/master/anaconda3/python.exe - <<'EOF'
from pathlib import Path
root = Path('data/tianchi')
for split in ['train','val','test']:
    imgs = list((root/split/'images').glob('*'))
    labs = list((root/split/'labels').glob('*.txt'))
    print(split, 'images', len(imgs), 'labels', len(labs))
defect = empty = boxes = 0
for lp in (root/'val'/'labels').glob('*.txt'):
    lines = [l for l in lp.read_text(encoding='utf-8').splitlines() if l.strip()]
    if lines: defect += 1; boxes += len(lines)
    else: empty += 1
print('val defect/empty/boxes =', defect, empty, boxes)
tr = {p.stem for p in (root/'train'/'images').glob('*')}
va = {p.stem for p in (root/'val'/'images').glob('*')}
print('train/val 泄漏:', len(tr & va))
EOF
```

期望：`val images 957`、`defect 583`、`empty 374`、`boxes 953`、`泄漏 0`。任何不符 → **停止并上报**（历史坑：partA/partB 的 anno_train.json 字节相同、各 7728 条标注引用同一批 4774 个图片名，而 partB 的图片名与 partA 无交集——后果是 partB 图片匹配不上标注被写成空 txt、**被误标为无缺陷负样本，污染正负样本比**；当前 data/tianchi 已是修好版，此检查是防回归）。

### R3 基线锁定

```bash
C:/Users/master/anaconda3/python.exe scripts/eval_multIou.py \
  --weights runs/rtdetr/tianchi18_mosaic1280/weights/best.pt \
  --data data/tianchi/data.yaml \
  --out runs/multi_iou/handoff_baseline.json
C:/Users/master/anaconda3/python.exe -c "import json; d=json.load(open('runs/multi_iou/handoff_baseline.json')); m=d['mAP']; print({k: round(v,4) for k,v in m.items()}); print('mean =', round((m['mAP@0.1']+m['mAP@0.3']+m['mAP@0.5'])/3,4))"
```

期望 mean = **0.6349**（mAP50 0.5917，18 类在场，953 实例）。偏差 >0.004 → 按 §11 F6 处理。同时把 per_class 表抄入 `runs/multi_iou/handoff_baseline.json` 供 P0.2 使用。**记下此值为 B\***。

### R4 统一对照协议（所有臂的同一把尺子）

**微调协议**（A/B/D/S4 臂，从 best.pt 出发；与历史 strong_r2_uniform 对照完全同口径）：

```
mosaic1280 best.pt 起点 | 5 epoch | batch 4 | imgsz 1280 方形 letterbox
lr0 1e-4 | amp OFF（fp16 会毒化 BN，见 §14 坑 #4）| mosaic 0 | workers 0 | seed 42
```

历史对照值：mean 0.6158 / crit_recall 0.6260（存档 `runs/multi_iou/strong_r2_uniform.json`）。

**从头训协议**（阶段 C/S1，抄 `scripts/train_mosaic1280.py`）：

```
rtdetr-l.pt COCO 初始化 | rect=True | mosaic=1.0（e50 关）| bf16（补丁 autocast+check_amp）
MuSGD(auto) | lr0 5e-4 | lrf 0.01 | 60 epoch | patience 20 | batch 4 | workers 0 | seed 42
单 epoch 约 15~20 分钟，全程约 15~17 小时
```

### R5 冒烟协议（每个新臂开跑前必做）

```bash
C:/Users/master/anaconda3/python.exe scripts/train_adaptive_loss.py --method uw --epochs 1 --fraction 0.05 --name smoke_check
```

通过标准：管道跑通；`logs/BN_POISONED.marker` 未生成；`results.csv` 的 mAP50 不为 0。冒烟不过 → 不进正式队列。

### R6 归档规范

```
runs/<阶段>/<臂名>_seed<N>/          # ultralytics 自带产物（results.csv 等）
runs/multi_iou/<臂名>_seed<N>_official.json   # 官方口径评测（唯一可信数字来源）
runs/<阶段>/<臂名>/PREREG.md         # 开跑前一句话判据
runs/<阶段>/REPORT.md                # 阶段小结
```

---

## 3. 阶段 0：头量探针与 S0（第 1 天下午–第 2 天，CPU 为主）

目的：用便宜实验测定"奖惩在哪个轴上还有头量"，避免把 GPU 天数花在没有头量的轴上。

### P0.1 类别侧 oracle（约 2 小时）

```bash
C:/Users/master/anaconda3/python.exe scripts/eval_confusion_decomp.py \
  --weights runs/rtdetr/tianchi18_mosaic1280/weights/best.pt --tau 0.001
```

读 `runs/confusion/decomp_tianchi18_mosaic1280_tau0.001.json` 的 `misclassified` 列。类别侧理论上限 = (correct+misclassified)/total。历史参照（旧基线）：hole 0/39、broken_warp 2/73、broken_spandex 1/54 错分。

**判据**：mosaic1280 上三关键类错分合计 >5 个 → 分类侧有残余头量 → GPU 队列中 B1 提到最前；≤5 个 → 分类侧无头量，分类侧重加权预期零和（与 R2/B/NWD 历史一致），跳过所有分类侧重加权臂。

### P0.2 定位侧 oracle（快速版即刻，精确版可选）

快速版（无代码）：读 R3 产出的 `handoff_baseline.json`，逐类算 AP@0.1 − AP@0.5。当前值：mean 0.6659 − 0.5917 = **0.074**；差距最大的类：size_stain 0.264、loose_warp 0.187、broken_warp 0.127、hanging_warp 0.119、knot 0.103——全是薄框/线性族。即"框再准一点"的头上界就是 0.074。

精确版（可选，规格见附录 A）：oracle box swap 后重算 mean。

**判据**：oracle 增益 ≥ +0.01 → 启动阶段 D（低 IoU 定位奖励）；< +0.005 → 定位头量在该特征下不可达，跳过 D。

### P0.3 召回侧 oracle（读 P0.1 同一 JSON 的 missed 列）

历史值：conf 地板处 hole 0/39、broken_warp 6/73、broken_spandex 3/54 漏检 → 检测能力上限 89%~100%。**结论：检测能力不是瓶颈，置信度排序才是**（与校准审计 AUC 0.763/0.823 互证）。本探针不单独启动任何臂，但它是"为什么弃权/FP 轴有头量"的依据。

### P0.4 FP 构成（约 2 小时，规格见附录 A 的扩展模式）

在工作 conf=0.25 下把 FP 分为：(a) 干净图上的背景 FP；(b) 缺陷图上的重复/邻近 FP；(c) 错类 FP。

**判据**：(a) 占比 >40% → 弃权轴（阶段 A）头量确认；(b) 占比高 → 考虑推理层去重（S0 的候选生成已覆盖）；(c) 占比高 → 与 P0.1 矛盾时以 P0.1 为准（P0.1 是 GT 级权威口径）。

### S0 测试时选择实验（约半天，CPU/轻推理，规格见附录 C 思路）

候选检测集生成：max_det 前缀 {100, 200, 300} × TTA {imgsz 1280, 1536} × {原图, 水平翻转} 的加权融合。选择规则三组：① 固定基线（现状）；② 七维奖励手工打分；③ 交叉拟合学习组合器（val 上 5-fold，防过拟合）。

**判据（预注册）**：任一规则 mean > B*+0.005 → 推理层奖励成立，进入阶段 S2 深化；全负 → "决策层/选择层已最优"的强证据，奖励只剩训练早期（阶段 C）一条路。

### 头量路由表（探针结束后按此启动 §4 队列）

| 探针结果 | 路由 |
|---|---|
| P0.1 分类头量 >5 个错分 | B1（族检测器探针）提前到队列第 1 位 |
| P0.2 定位 oracle ≥+0.01 | D1 进入队列（第 3 位） |
| P0.4 FP 中背景 FP >40% | A1 保持队列第 1 位（默认即此） |
| 三项全无头量 | 跳过 B/D，主线 = A1 → C1 screen → M1；奖励价值主张转向决策层（A2） |
| S0 阳性 | 插入 S2（推理层奖励产品化），优先级仅次于 A1 |

---

## 4. GPU 实验队列总览（严格串行）

```
                    ┌─ A1 objectness 联合微调（1-2 天）──── 默认第 1 位
R 完成 ── 探针路由 ──┼─ B1 族检测器探针（约 6 小时）
                    ├─ D1 低 IoU 定位奖励（约 1 天，P0.2 过线才启动）
                    └─ S4 单类检测器（1-2 天，B1 判定后）
                              │
                    A1/B1/D 判定完毕 ──→ C1 screen（15 小时，大赌注）
                              │
              screen 阳性 ──→ C1 confirm（3 种子，60 小时）
              screen 阴性 ──→ M1 AP 代理损失（1 天）── 阳性则与最强奖励臂组合拳
                              │
                    全部阴性 ──→ B2 全级联（若 B1 阳性，2-3 天）/ S1 换 YOLO11（15 小时+）
                              │
                    仍全部阴性 ──→ 按 §12 停止条件收尾，转入价值整理
```

CPU 并行轨（不占 GPU）：P0.*、S0、A2、附录 A/B 脚本、族数据集构建、§13 记录。

预算：顺利路径约 5~6 GPU 天；最坏路径约 13~14 GPU 天。

---

## 5. 阶段 A：弃权 / 图像级轴（最有证据的轴，默认第 1 位）

**假设**：图像级"有/无瑕疵"判定、停机档误报与检测 mAP 是两个指标轴；在图像级轴上优化奖惩**不触发检测 PR 零和**。

**证据**：二分类回放 93.73%（tau=0.35，AUC 0.9808；对照旧基线 92.06%）；objectness 线性探针是历史上唯一正结果（域内不如 max-conf，跨域 FPR 低约 12 点）；TP/FP 可分性 AUC 0.763/0.823 有头量但单调重校准无效（论文 4.13）。

### A1 objectness 联合微调（GPU，1-2 天）

```bash
# 先复现安全探针（应有正结果，作为管道自检）
C:/Users/master/anaconda3/python.exe scripts/train_rtdetr_objectness.py --probe-only --epochs 3

# 正式臂：联合微调（代价敏感 BCE，关键类 query 升权、背景 query 降权）
C:/Users/master/anaconda3/python.exe scripts/train_rtdetr_objectness.py --epochs 5 --lr0 1e-4 --name a1_obj_joint_s42
C:/Users/master/anaconda3/python.exe scripts/eval_multIou.py \
  --weights runs/rtdetr-objectness/a1_obj_joint_s42/weights/best.pt \
  --data data/tianchi/data.yaml --out runs/multi_iou/a1_obj_joint_s42_official.json
# 种子 123/2024 同样两遍
# 图像级指标：
C:/Users/master/anaconda3/python.exe scripts/eval_binary_replay.py \
  --weights runs/rtdetr-objectness/a1_obj_joint_s42/weights/best.pt --tag a1_s42
```

**预注册判据**：域内图像级最佳正确率 ≥ 94.2%（93.73%+0.5）**且**检测 mean 降幅 ≤0.005；次级指标 AITEX 跨域 FPR ≤ 66.6%（71.6%−5 点）。

**失败转向**：mean 掉 >0.005 → objectness 梯度干扰检测 → lr0 降为 5e-5 重试一次；仍掉 → 记阴性转 A2（CPU）。图像级正确率不动 → 说明该轴分数已饱和，同样转 A2。

### A2 门控分数奖励回放（CPU，1 天，与 GPU 臂并行）

用 `scripts/eval_binary_replay.py` 对多个候选门控分数跑回放（`runs/binary_replay/mosaic1280/` 已有 max_conf 缓存）：

| 候选分数 | 来源 |
|---|---|
| max_conf | 现状基线（93.73%） |
| objectness | A1 产物 |
| 七维图像级聚合分 | D03（校准）+ D05（误检）两维重定义为图像级量后聚合——**这是七维机制在决策层的正式接口** |
| sev×conf | 阶段 1.2 已证优势来自严重度先验，作为对照基准（注意：它不是七维，是对照） |

**预注册判据**：最佳正确率 > 93.73%+0.5 且工作 tau 必须经 5-fold 交叉拟合选择（禁止在 val 上直接搜 tau 后报数）。

**意义**：A2 若转正 → 七维中的 D03/D05 在图像级门控上"发光"，成为大创/论文的决策层贡献；若全负（含不如 max-conf）→ 与 B2/C8 历史一致，决策层分数无头量，奖励彻底转向训练侧（阶段 C）。

---

## 6. 阶段 B：族级子问题轴（换架构的便宜版）

**假设**：3 族（A 线性 / B 斑渍 / C 点状，bootstrap support 1.00）是比 18 类"更容易的子问题"，其 PR 曲线离最优更远——族级子问题上的奖惩有头量。已有资产：族映射 `runs/family_clustering/families.yaml`、crop 级 3 族分类器 top-1 0.9119（CI [0.894, 0.929]）。

### B1 族检测器探针（GPU，约 6 小时）

```bash
# 1) 构建族标注数据集（规格见附录 B；核查 val 框数仍为 953）
C:/Users/master/anaconda3/python.exe scripts/build_family_dataset.py \
  --src data/tianchi --dst data/tianchi_family3 --map runs/family_clustering/families.yaml

# 2) 从 mosaic1280 best.pt 换头（18→3）微调 20 epoch（抄 train_mosaic1280.py 配方，epochs=20）
C:/Users/master/anaconda3/python.exe scripts/train_family_detector.py \
  --init runs/rtdetr/tianchi18_mosaic1280/weights/best.pt \
  --data data/tianchi_family3/data.yaml --epochs 20 --name b1_family3_s42

# 3) 两个口径评测：3 族 mAP（官方管道，nc=3）+ 18 类预测按 families.yaml 聚合成族后的 3 族 mAP
```

**预注册判据**：专用族检测器的族级 AP@0.5 比"18 类聚合"高 ≥0.02 → 族级子问题有头量 → 启动 B2；否则阶段 B 关闭，把算力让给 C/M。

### B2 族级联 + 族级奖励（GPU，2-3 天，条件启动）

- **stage-1**：3 族检测器从头训 60ep（train_mosaic1280.py 配方 + 族数据），**前 10 epoch 施加族级 D01/D04 奖励**（复用 `backend/training/dimension_rewards.py`，作用对象换成族标签）——这是七维机制在族级子问题上的正式接口。
- **stage-2**：族内细分类器，每族一个 crop 级 ResNet（A 族 7 类 / B 族 5 类 / C 族 6 类，复用 `scripts/train_family_classifier.py` 模式）。
- **融合评测**：stage-1 框 + stage-2 类 → `eval_multIou.py` 口径（融合脚本规格见附录 A 的 match 协议）。

**预注册判据**：融合 mean > B*+0.005，且逐类 AP 无任何一类掉 >0.05（级联不应伤害稀有类）；三种子。

**诚实风险提示**：级联的类别正确率上限 = 族准确率 × 族内准确率，而当前 18 类检测器的类正确率已经很高（错分 7/166）。B2 是"用可能的类精度换召回/框质量"的赌注，**只有在 B1 证实族级有头量时才值得下注**。

---

## 7. 阶段 C：早期表征奖励（大赌注，15h screen / 60h confirm）

**假设**：收敛模型上奖惩只能沿 PR 曲线滑动；若在从头训的**前 10 epoch**（曲线定型前）施加可微奖励，可能外移曲线本身。这是"接口没错、时机错了"假说的直接检验。

### C1 screen（GPU，15 小时，单 seed）

```bash
# 复制 scripts/train_mosaic1280.py → scripts/train_mosaic1280_earlyreward.py
# 唯一差异：前 10 epoch 以 β=0.5 注入 Layer-1 可微奖励
#（注入机制抄 scripts/train_rtdetr_reward.py 的 on_train_start 挂载 + reward 注入），
# 后 50 epoch β=0 纯监督。配方其余部分逐行一致。
C:/Users/master/anaconda3/python.exe scripts/train_mosaic1280_earlyreward.py
C:/Users/master/anaconda3/python.exe scripts/eval_multIou.py \
  --weights runs/rtdetr/tianchi18_earlyreward/weights/best.pt \
  --data data/tianchi/data.yaml --out runs/multi_iou/c1_earlyreward_s42_official.json
```

**关键设计**：既有 mosaic1280 就是同 seed、同配方、β=0 全程的严格对照——**screen 阶段零额外对照成本**。

**预注册判据**：screen Δmean ≥ +0.005 → confirm；≤ +0.005 → 记阴性，**不进 confirm**（省 60 小时）。

### C1 confirm（GPU，60 小时，仅 screen 阳性）

seed 123 / 2024 各跑臂 + 对应 β=0 对照（共 4 个 60ep 跑）。判据同铁律 2。

### C2 变体（screen 阳性但 confirm 阴性时）

β 线性 ramp（0→0.5 覆盖前 20 epoch）一臂 seed 42。仍阴性 → 阶段 C 关闭，写入负结果链。

---

## 8. 阶段 D：低 IoU 定位奖励（约 1 天，前置 P0.2 ≥ +0.01）

**假设**：比赛指标权重在 IoU 0.1/0.3 上，薄框类的 AP@0.1→AP@0.5 落差（mean 0.074，最大 0.26）主要是定位问题；只作用于框回归的奖励可能在不触发分类零和的前提下收回部分落差。

**与历史 R3/NWD 臂的区别**（务必在 PREREG 里写明）：NWD 是**度量替换**（整体损失形式换掉，已零和 −0.015）；D1 是**只加不换**的定位奖励，且 IoU 口径对齐比赛指标（0.1/0.3 而非 0.5）。

```bash
# 新臂脚本规格：抄 train_rtdetr_boxweighted.py 的微调协议，奖励项只作用于匹配框的
# soft-IoU（IoU 0.1/0.3 口径），不动分类项、不动 giou/L1 默认增益；β 从 0.1 起
C:/Users/master/anaconda3/python.exe scripts/train_rtdetr_locreward.py --beta 0.1 --epochs 5 --name d1_loc_s42
# 官方评测 + 种子 123/2024
```

**预注册判据**：mean 增益 ≥+0.005，且增益主要来自 AP@0.3/@0.5 的薄框类（loose_warp / size_stain / broken_warp / hanging_warp）；crit_recall 不降。

**失败转向**：若签名同 R3（mean 降、crit_recall 升）→ 定位侧在该特征下也不可外移，记阴性；若双降 → 按 §11 F2 处理。

---

## 9. 阶段 S：换模型 / 换架构（前述臂全负后启动）

### S1 YOLO11 重建（GPU，15 小时 + 臂）

启动条件：A1/B1/D 全负且 C1 screen 阴性。

1. 确认能下载 `yolo11l.pt`（ultralytics 自动下载；被墙则用本地 `yolo11s.pt` 并在 PREREG 记录口径变化）。
2. 同数据同配方从头训 60ep（imgsz 1280 / rect / mosaic / batch 4 / MuSGD）。
3. 复制 `eval_multIou.py` 为 `eval_multIou_yolo.py`：把 `RTDETRValidator` 换成 `DetectionValidator`，其余不动。
4. sanity：YOLO 的 mAP@0.5 必须与其 `results.csv` 一致（±0.004）。
5. 判据：YOLO 基线 mean ≥ 0.62（sanity 线，预期 0.62~0.66 区间）；把当时最强的奖励臂移植到 YOLO 训练循环（ultralytics 原生 loss，挂钩比 RT-DETR 更容易），Δ > +0.005 且三种子。

**理由**：RT-DETR 免 NMS top-300 与 YOLO 的 NMS 密集预测 PR 几何不同；零和可能是 RT-DETR 接口特有的。另：YOLO11 是产线部署的现实选择，重建基线本身有独立价值。

### S2 推理期奖励产品化（轻 GPU / CPU）

S0 阳性后的延伸：把"候选生成 + 奖励选择/融合"写成 `backend/inference` 里的正式模块（多尺度 TTA + 奖励融合 + 工作点选择）。判据同 S0，加一条：推理开销 ≤ 3× 基线（产线可接受）。

### S3 DiffusionDet（条件启动，不建议首轮）

仅在 S0/S2 证明"推理期奖励有效"后考虑：扩散检测器的去噪过程天然支持奖励引导采样（classifier-guidance 式），是"训练期不动、推理期发光"的最佳载体。但需从零训练 300+ epoch（约 80+ GPU 小时），**排在所有其他方向之后**，且需先确认代码可复现（小数据冒烟）。

### S4 单类检测器（GPU，1-2 天，旧计划 C1 合并）

18 类 → 1 类（defect），去掉多类竞争。从 mosaic1280 best.pt 换头微调，协议同 R4。判预注册：召回不降且误报更低（图像级 FPR ≤17.4%）。与 B1 同属"降类别数"家族，信息互补——B1 显示族级有头量则 S4 更有希望。

---

## 10. 阶段 M：直接优化指标（备胎与组合拳）

### M1 AP 代理损失（GPU，约 1 天）

```bash
# 新臂：抄 train_rtdetr_reward.py 微调协议，在 GIoU/L1 旁加可微 AP 代理
#（Smooth-AP 风格 pairwise ranking loss，作用于匹配 query 的分数序），lr0 5e-5 起
C:/Users/master/anaconda3/python.exe scripts/train_rtdetr_aploss.py --epochs 5 --name m1_aploss_s42
# 官方评测 + 种子 123/2024
```

**预注册判据**：mean > B*+0.005。

**组合拳**：M1 阳性 → 在 M1 收敛模型上重试当时最强的奖励臂（奖惩的头量可能在更优的 PR 曲线上重新出现）。M1 也阴性 → "检测 mAP 在该数据/架构上已近天花板"成为强结论，项目价值主张转向决策层（A2）+ 诊断层（论文负结果链）+ 产线工作点（sev×conf + 三档，已有交付）。

---

## 11. 失败判读与转向手册

任何一臂判定前，先对签名，再查下表。**签名不明时禁止下结论**。

| 编号 | 失败签名 | 根因 | 下一步 |
|---|---|---|---|
| F1 | mean ↓ >0.005 且 crit_recall ↑ >0.02、精度 ↓ | PR 曲线滑动（历史 R2/B/NWD 同一签名） | 记入负结果链（机理证据）；该接口放弃，转阶段 C（早期奖励）或 S（换架构） |
| F2 | mean ↓ 且 crit_recall ↓（双降） | 优化不稳定：干预梯度与检测梯度冲突 | 干预权重/lr 减半重试一次；仍双降 → 用 `train_adaptive_loss.py`（UW，已修 bug 版）做梯度平衡再试；再失败 → 接口不可救，换轴 |
| F3 | 单 seed \|Δ\| ≤ 0.0093 | 种子噪声（实测离散度 0.0093） | **必须补到 3 种子再判定**；禁止单 seed 宣科 |
| F4 | mAP≈0，或 epoch ≥9 后 BN running_var 归零 | fp16-AMP 毒化 BN（neck 激活平方后超 fp16 上限，39 层 running_var 清零，eval 放大 100~400×） | 改 amp=False 或 bf16 补丁（抄 `train_mosaic1280.py` 补丁三行）重跑；查 `logs/BN_POISONED.marker` |
| F5 | 自定义评测召回系统性崩坏 | RT-DETR val **强制方形 letterbox**（ultralytics `models/rtdetr/val.py` 硬编码 rect=False，外部传 rect=True 被忽略） | 自写评测管道一律 rect=False；用 `eval_multIou.py` 复评确认 |
| F6 | mAP@0.5 与训练 results.csv 偏差 >0.004 | 评测管道分叉 | 停。先复现 sanity contract（eval_multIou.py 文件头），修好再继续 |
| F7 | 奖励曲线持续上涨不收敛 | 奖励黑客（reward hacking） | 降 β；加 KL 锚或组基线（GRPO 臂的地板修复经验：eps 1e-2→1e-4 反而放大伤害，说明单纯降噪无效，应降奖励强度） |

---

## 12. 预算、排期与停止条件

### 预算

| 阶段 | 内容 | 资源 | 机时 | 启动条件 |
|---|---|---|---|---|
| R | 地基重建 | CPU + 1 次评测 | 0.5 天 | 立即 |
| 0 | 头量探针 | CPU | 0.5~1 天 | R 后 |
| S0 | 测试时选择 | CPU/轻推理 | 0.5 天 | R 后 |
| A2 | 门控奖励回放 | CPU | 1 天 | R 后（与 GPU 并行） |
| A1 | objectness 联合 | GPU | 1~2 天 | R 后 |
| B1 | 族检测器探针 | GPU | ~6 小时 | R 后 |
| D1 | 定位奖励 | GPU | ~1 天 | P0.2 ≥+0.01 |
| S4 | 单类检测器 | GPU | 1~2 天 | B1 判定后 |
| C1 screen | 早期表征奖励 | GPU | 15 小时 | A1/B1/D 判定后 |
| M1 | AP 代理损失 | GPU | ~1 天 | 任意时点可插队 |
| C1 confirm | 早期奖励 ×3 种子 | GPU | 60 小时 | screen 阳性 |
| B2 | 族级联 | GPU | 2~3 天 | B1 阳性 |
| S1 | YOLO11 重建 | GPU | 15 小时 + 臂 | 前述全负 |

顺利路径 5~6 GPU 天；最坏 13~14 GPU 天。全部串行，遵守铁律 1。

### 停止条件（三条全满足即收尾，不再开新臂）

1. A1 / B1 / D1 / M1 全部三种子阴性（或按路由表跳过且 S0 阴性）；
2. S0 阴性 + A2 阴性（决策层/选择层无头量）；
3. C1 screen 阴性（表征层无头量）。

此时输出（本身就是完整科研成果，勿理解为"失败"）：
- 论文负结果链扩展至 6+ 类干预接口，机理统一为"PR 曲线滑动 + 双轴解耦"（§1.4 三段论的实验版）；
- 产线决策层交付：sev×conf 排序 + 三档工作点（复核档 tau=0.25 告警率 17.4%、停机档高阈值 FPR 4.3%）+ 图像级门控 93.73%；
- 大创结题报告：检测性能线（0.572→0.6349）+ 诊断/决策层贡献线。

---

## 13. 结果记录总表（执行中持续填写）

| 臂 | 协议 | seeds | mean | mAP50 | crit_recall | Δmean vs 对照 | 判定 | 存档 |
|---|---|---|---|---|---|---|---|---|
| B* 基线 | mosaic1280 | 42 | 0.6349 | 0.5917 | — | — | 参照 | `runs/multi_iou/handoff_baseline.json` |
| A1 s42 | 微调 5ep | 42 | | | | | | |
| A1 s123 | | 123 | | | | | | |
| B1 | 换头 20ep | 42 | | | | | | |
| D1 | 微调 5ep | 42/123/2024 | | | | | | |
| C1 screen | 从头 60ep | 42 | | | | | | |
| M1 | 微调 5ep | 42/123/2024 | | | | | | |
| S0 | CPU 回放 | — | | | | | | |
| A2 | CPU 回放 | — | 图像级正确率 | | | | | |

规则：官方数字只来自 `runs/multi_iou/*_official.json`；crit_recall = broken_warp+broken_spandex 组召回 @max-F1（与历史对照同口径）；每臂判定后立即更新"判定"列，不留空过夜。

---

## 14. 常见坑速查（全部为实测坑，按出现频率排序）

1. **GPU 不并发**：训练链运行期间开推理 → 推理被 SIGTERM 时可能 GPU 级故障连带杀训练（2026-10-20 事故）。
2. **单种子噪声**：种子间离散度 0.0093 > 单 seed 效应量常模；三种子是底线。
3. **fp16-AMP 毒化 BN**：≥9 epoch 长跑清零 39 层 running_var → mAP=0；1-2 epoch 冒烟测不出。用 amp=False 或 bf16 补丁（`train_mosaic1280.py` 的补丁三行：TU/TR.autocast 打补丁 + TR.check_amp=True）。
4. **ultralytics RT-DETR val 强制方形 letterbox**：`models/rtdetr/val.py` 硬编码 rect=False，外部 rect=True 被忽略；自写评测必须方形，否则召回系统性崩坏。
5. **workers=0**：Windows 下 workers>0 报 WinError 1455。
6. **MuSGD 勿换 AdamW**：从头训换优化器会塌陷。
7. **桌面 3D 软件抢 GPU**：曾把 20min/epoch 拖到 160min/epoch；开训前关。
8. **data.yaml 勿动**：UTF-8 编码，中文路径下改动会破坏数据集加载。
9. **partA/partB anno_train.json 字节相同（md5 一致）**：两份标注引用同一批 4774 个图片名，但 partB 图片与 partA 无交集 → partB 图片匹配不上标注被写成空 txt 误标为负样本，污染正负样本比。重建数据必须按图像名核对标注匹配率。
10. **原生 copy_paste 对检测静默无效**（segments 为空直接 return）：必须 `install_detection_copypaste()`（`backend/training/copypaste_det.py`）。
11. **混淆矩阵仅在 val plots=True 时更新**；矩阵索引为 [pred, gt]，需转置为 [gt, pred]。
12. **优化器 auto 会覆盖 lr0/momentum**：微调脚本里设 lr0 前先确认 optimizer 参数。
13. **1280 batch 8 约占 21GB 显存，勿再加大**；从头训用 batch 4。
14. **`optimizer=auto` + MuSGD 下 lr0=5e-4/lrf=0.01 是验证过的组合**：换 lr 前先跑冒烟。

---

## 15. 与既有计划的关系 & 回填

- 本清单**吸收并取代** `docs/plan_next_steps.md` v2.0 的阶段 3+：C1（单类检测器）→ S4；C5（簇内归一化）→ 若 B1 阳性再议；B3（查询级组内归一化辅助奖励）→ 可作为 C1 的奖励变体（在 PREREG 注明）；C6（最差维度定向）→ 分布排序实验已部分证伪（B2 NEG），不单列。
- **Copy-Paste 链结果（约 10-22 清晨出）是本清单的输入**：若转正（mean > 0.6399），B* 更新，所有门同步上移；若阴性，负结果链 +1，本清单不受影响。
- 论文回填：每臂判定（无论正负）写入 `docs/paper/paper_draft.md` 对应新小节（接 4.16 之后）与 §5.5 负结果链；大创材料同步。**负结果同样必须回填**——本项目的机理结论正是由负结果链支撑的。
- 升级路径：执行人遇到判据争议、GPU 故障、或发现本文档事实错误（路径/数值）时，找 刘宗棠 决议，不要自行改判据。

---

## 附录 A：Oracle 头量探针脚本规格（P0.2 / P0.4 用）

新脚本 `scripts/oracle_probe.py`，约 150 行。复用两个现有模式：官方 AP 计算抄 `eval_multIou.py`；匹配协议抄 `eval_confusion_decomp.py`（GT 级、类感知、IoU 0.5 贪婪匹配）。

```
用法: python scripts/oracle_probe.py --weights <best.pt> --mode {box,cls,fp} --out runs/headroom/<tag>_<mode>.json

流程:
1. 官方 val 管道推理一遍（RTDETRDataset, rect=False 方形, imgsz=1280, conf=0.001, max_det=300），
   缓存每图 preds (N,6: xyxy+conf+cls) 与 GT（同类感知匹配，抄 eval_confusion_decomp.py）。
2. mode=box : 每个与 GT 匹配（IoU>=0.5, 任意类）的 pred，框替换为 GT 框（cls/conf 不变）。
   mode=cls : 每个匹配的 pred，cls 替换为所匹配 GT 的 cls（框/conf 不变）。
   mode=fp  : 在 conf=0.25 下统计 FP 按 (a) 干净图背景 / (b) 缺陷图重复邻近 / (c) 错类 三类计数。
3. 用 ultralytics 的 match_predictions + ap_per_class 按 IoU=[0.1,0.3,0.5] 重算 mAP
   （与 eval_multIou.py 同一算法；函数签名以 ultralytics 8.4.91 源码为准，
    ultralytics/utils/metrics.py）。

验收（不满足则脚本有 bug，不得采信数字）:
- mode=none（原样重算）在 mosaic1280 上必须复现 mean 0.6349 / mAP50 0.5917；
  在 tianchi18_full 上必须复现 mAP50 0.541。
- box/cls oracle 的 mAP 必须 ≥ 原样（单调性 sanity）。
```

## 附录 B：族标注数据集构建脚本规格（B1/B2 用）

新脚本 `scripts/build_family_dataset.py`，约 80 行。

```
用法: python scripts/build_family_dataset.py --src data/tianchi --dst data/tianchi_family3 --map runs/family_clustering/families.yaml

流程:
1. 读 families.yaml 的 class_to_family3（18 类 → 3 族）。
2. 在 --dst 下建 train/val(/test)/images 与 labels：images 用符号链接或复制；
   labels 逐行重映射 class id（0-17 → 0-2），保持归一化 cxcywh 不变。
3. 写 data.yaml（nc=3, names: A_linear/B_blotch/C_point, train/val/test 指向上级目录）。
4. 自检打印：val 标签文件数 = 957；GT 框总数 = 953（与 18 类版一致，重映射不增删框）；
   抽 10 个标签文件人工核对族映射正确。

注意: 勿改 data/tianchi/data.yaml（UTF-8）；新数据集是独立目录，互不影响。
```

## 附录 C：S0 测试时选择实验规格

新脚本 `scripts/eval_ttc_selection.py`，CPU 为主（957 张图，RT-DETR-L@1280 CPU 推理约 10~30 分钟；GPU 空闲时约 10 分钟）。

```
候选检测集（每图）:
  C1 = 官方管道输出（基线）
  C2 = C1 的 max_det=100 / 200 前缀
  C3 = 多尺度 TTA: imgsz {1280, 1536} x {原图, hflip} 四路输出的加权框融合
      （权重在 val 上网格搜索 {0.1..0.9}，与选择规则分离，防泄漏）
选择规则:
  R1 = 固定用 C1（基线）
  R2 = 七维奖励对候选打分选最优（D01-D07 聚合，权重用论文表 1 默认）
  R3 = 学习型组合器：以 GT 计算每候选的 mean-AP 贡献作标签，5-fold 交叉拟合
      （防过拟合；报告 crossfit 均值）
判据: 任一规则 mean > B*+0.005（R3 用 crossfit 均值判定）。
输出: runs/ttc/REPORT.md + 各规则 mean/mAP50。
```

## 附录 D：七维奖惩检测版代码地图（改奖励时用）

| 组件 | 位置 |
|---|---|
| 七维奖励计算 | `backend/training/dimension_rewards.py` |
| RT-DETR 奖励注入（criterion 替换） | `backend/training/reward_rtdetr.py` |
| Layer-1 训练入口 | `scripts/train_rtdetr_reward.py`（--beta / --include-misses / --lambda-miss） |
| GRPO（已证伪，存档参考） | `backend/training/grpo.py`、`grpo_trainer.py` |
| UW/GradNorm（已修 bug） | `backend/training/adaptive_loss.py` + `scripts/train_adaptive_loss.py` |
| objectness 头 | `backend/training/objectness.py` + `scripts/train_rtdetr_objectness.py` |
| 挂载模式 | on_train_start 替换 trainer.model.criterion（ultralytics 优化器先于 criterion 重建，新参数不会被优化——UW 类需手动解析更新，见 `adaptive_loss.py` 实现） |

---

*文档结束。执行中任何与本文档冲突的事实（路径不存在、数值不复现），以复现结果为准并记录到 §13，同时通知 刘宗棠 更新本文档。*
