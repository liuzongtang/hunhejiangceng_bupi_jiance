# 奖励注入 RT-DETR 训练 — 终端跑法

Layer 1 训练（7 维规则奖励以可微 `loss_reward` 注入 ultralytics RT-DETR 训练循环）的日常运行手册。

## 环境（已就绪，无需重装）

| 项 | 值 |
|----|----|
| Python | `C:/Users/master/anaconda3/python.exe`（若 `python` 已指向 anaconda 则直接用 `python`） |
| 依赖 | torch 2.8.0+cu126、ultralytics 8.4.91 |
| GPU | RTX 4090 D（24GB） |
| 数据 | `data/tianchi/`（8619 train / 957 val，YOLO 格式，20 类） |
| 预训练权重 | `runs/rtdetr/tianchi20/weights/best.pt`（mAP50=0.594 基线） |

## 从项目根目录运行

```bash
cd "C:/Users/master/Desktop/维度模型(1)"
```

## 命令

```bash
# 1. A/B 基线（无奖励，纯监督精调）
python scripts/train_rtdetr_reward.py --beta 0   --epochs 5 --name beta0_e5

# 2. 奖励开
python scripts/train_rtdetr_reward.py --beta 0.5 --epochs 5 --name beta0.5_e5

# 3. 更多 epoch / 更强奖励 / 更大漏检惩罚
python scripts/train_rtdetr_reward.py --beta 1.0 --epochs 20 --lambda-miss 2.0 --name beta1.0_e20

# 4. 漏检纳入奖励（验证根因）：匹配查询置信度低于阈值的关键类 GT，
#    用一条「空闲 query」额外喂进奖励，让 D04/D06/D07 见到漏检并给出召回梯度
python scripts/train_rtdetr_reward.py --beta 0.5 --epochs 5 --include-misses --name beta0.5_miss
```

> **关键机制澄清**：RT-DETR 的 Hungarian 匹配器会把**每个 GT 都分给某个 query**（N=300 ≥ GT 数），所以不存在「未匹配的 GT」。所谓「漏检」= 匹配上的 query 对关键类集合的 sigmoid 置信度 `p_crit < --miss-threshold`（默认 0.5）。`--include-misses` 对这类 GT 再取一条**未被占用的空闲 query**（`p_crit` 最高者）喂入奖励，D06/D07 得到 `3*p_crit-2` 的负信号、D04 得到漏检惩罚，梯度流向那条空闲 query。

## 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--weights` | `runs/rtdetr/tianchi20/weights/best.pt` | 起跑权重 |
| `--beta` | 0.5 | 奖励权重；`0` = 关闭（纯监督基线） |
| `--include-misses` | off | 漏检关键类 GT 用空闲 query 喂进奖励（验证根因） |
| `--miss-threshold` | 0.5 | 关键类集合 sigmoid 置信度低于此值即算「漏检」 |
| `--lambda-miss` | 1.5 | D04 漏检惩罚系数 |
| `--epochs` | 5 | |
| `--imgsz` | 1280 | 与原始训练一致 |
| `--batch` | 8 | 1280 下约占 21GB 显存，**勿再加大** |
| `--lr0` | 0.0001 | 注意 `optimizer=auto` 会自动覆盖 lr0/momentum |
| `--log-every` | 50 | 每 N step 记一条 `batch_reward` 日志 |
| `--device` | 0 | |
| `--name` | — | run 子目录名（结果落在 `runs/rtdetr-reward/<name>/`） |

## 结果位置

| 内容 | 路径 |
|------|------|
| 实时进度条 | 终端（giou/cls/l1 loss + 每 step 奖励日志） |
| 结构化 JSON 日志 | `logs/training.log`（`batch_reward` / `epoch_metrics` / `training_complete`） |
| 每 step 奖励明细 | `runs/rtdetr-reward/<name>/reward_history.csv` |
| 四子图曲线 | `runs/rtdetr-reward/<name>/reward_training.png` |
| ultralytics 自带 | `runs/rtdetr-reward/<name>/results.csv` + `results.png` |

## 重绘曲线（不重训）

```bash
python scripts/plot_reward_training.py --dir runs/rtdetr-reward/<name> [--window 50]
```

## 判断收敛/稳定

- **收敛**：终端 `mAP50` 连续 2~3 epoch 平台化（当前 ~0.573）。
- **稳定**：`epoch_metrics` 里的 `reward_mean_step` 全程窄幅波动（当前 0.01~0.03，std ~0.08），无发散、无持续上涨（持续上涨 = 奖励黑客）。
- **结论归因**：必须 `--beta 0` 与 `--beta 0.5` 各跑一遍（同 epoch/超参），对比 4 个关键类 recall（断经 9 / 断氨纶 16 / 星跳 15 / 死皱 19）vs 整体 mAP。
- **漏检根因验证**：`--beta 0.5`（matched-only）与 `--beta 0.5 --include-misses` 各跑一遍。`reward_history.csv` 的 `n_missed` 列 > 0 说明漏检关键类确实被喂进了奖励；若 recall 仍不动，则根因不在「奖励只看匹配对」，而在 Layer-1 注入本身对召回无效，应转向 Layer-3 GRPO。

## 注意事项

- 在 Git Bash 下运行；项目路径含中文，直接 `cd` 根目录即可，**勿改动 `data/tianchi/data.yaml`**（UTF-8 编码）。
- 单 epoch ~18min（imgsz 1280），5 epoch ~90min。
- `--beta 0` 时无 `reward_history.csv` / `reward_training.png`（没有奖励信号可画），只有 `results.csv`。

## Layer 3 GRPO（自研最小循环）

用 query 噪声做随机策略，`G` 组候选检测集 + 7 维奖励 → 组内相对优势 → REINFORCE（+ KL 锚定到冻结参考策略）。核心在 `backend/training/grpo.py`（纯 torch 原语）+ `backend/training/grpo_trainer.py`（循环 + RT-DETR 适配器 `RTDETRGRPOPolicy`）。单测：`backend/training/tests/test_grpo.py`（14 个，CPU 可跑）。

```bash
# 纯 GRPO：只训练探索温度 log_std（模型冻结），小而安全
python scripts/train_rtdetr_grpo.py --epochs 1 --num-groups 4 --name grpo_skel

# 混合：GRPO 组基线 + 可微奖励梯度进 decoder（较重，注意显存）
python scripts/train_rtdetr_grpo.py --beta-reward 0.5 --train-decoder --epochs 3
```

> **关键机制**：RT-DETR 的 content query 在训练时被 detach（`_get_decoder_input` 里 `embeddings.detach()`），所以 `RTDETRGRPOPolicy` 给 decoder 的 `_get_decoder_input` 打补丁，往 content query（去噪前缀之后的 `num_queries` 行）加 `std*eps` 噪声，并记录 log-prob。log-prob 用**似然比得分**（对固定样本 `z.detach()` 求导，而非重参数化路径）计算：`d log_prob/d log_std = mean(eps²)-1`、`d log_prob/d mean = eps/std`，两者都非零。但因为 content query 被 detach，均值的梯度到不了模型，所以**纯 GRPO 只调探索温度，detector 的均值要靠 `--beta-reward` 的可微奖励梯度来推**（这正是 Layer-1 失败后、Layer-3 要补的那块：组基线防奖励黑客 + 可微梯度推均值）。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num-groups` | 4 | 每图采样 G 组；`G=4` × batch 8 的纯 GRPO 模式每次 rollout 图被 detach，显存≈单前向 |
| `--log-std-init` | -2.3 | 初始噪声 ~0.1 |
| `--ref-log-std` | -4.6 | 参考策略噪声 ~0.01（KL 锚） |
| `--beta-kl` | 0.01 | KL 惩罚权重，防探索温度坍缩/发散 |
| `--beta-reward` | 0.0 | 可微奖励梯度权重；>0 需 `--train-decoder`，且显存翻倍 |
| `--lr` | 1e-3 | decoder/模型权重学习率（仅 `--train-decoder` 时生效） |
| `--lr-log-std` | 0.1 | `log_std` 单标量的独立学习率（无 weight decay）。REINFORCE 信号在 ~B·300·256 维上很弱，需比模型 LR 大得多 |
| `--advantage-eps` | 1e-2 | 组内优势 `(R-mean)/(std+eps)` 的 std 地板。奖励对 query 噪声「局部平坦」→ `std_reward`~1e-4，不加地板优势会爆到 ~100 |
| `--advantage-clip` | 5.0 | 优势硬裁剪到 `[-clip, clip]`（兜底） |
| `--train-decoder` | off | 解冻 decoder head（仅配合 `--beta-reward > 0` 有意义） |
| `--freeze-backbone` | on | 冻结 backbone+neck |

结果：`runs/rtdetr-grpo/<name>/grpo_history.csv`（每步 loss/loss_policy/mean_reward/std_reward）+ `logs/training.log`（`grpo_batch`/`grpo_epoch`/`grpo_complete`）。

> **注意**：这是最小自研循环，不是 ultralytics 全量 `model.train`——无 EMA/自动验证/自动保存 best.pt。骨架阶段只验证「采样→奖励→优势→REINFORCE」链路与 `log_std` 能被调起来；全量接入（EMA、val 曲线、best.pt 保存、backbone 解冻策略）是后续迭代。
