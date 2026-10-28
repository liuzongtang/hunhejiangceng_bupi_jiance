# 阶段 2 再出发 —— 本机执行计划 v1.1

> 修订原因：`runs/rtdetr/tianchi18_mosaic1280/weights/best.pt` 与全部历史 runs 存档在本机不存在，仓主也无法提供。
> 因此 handoff v1.0 的 **R3「复现 0.6349」不可执行**，必须改为 **R3′「重训自建基线」**；所有对照值重新测量。
> 判据结构（铁律 2：同协议对照 + 三种子 + 预注册）**不变**。

## 0. 本机环境（已就绪）

```
PY      = C:\Users\B6\Documents\deepseek-harness\default-workspace\envs\tianchi\Scripts\python.exe
ROOT    = C:\Users\B6\Documents\deepseek-harness\default-workspace\repo
GPU     = RTX 4090D（铁律 1：任何 GPU 任务前 nvidia-smi 应 < 2000 MiB）
```

每一个 handoff 命令里的 `C:/Users/master/anaconda3/python.exe` 都替换成上面的 `PY`；
每个命令前 `cd` 到 `ROOT`（脚本用相对路径 `data/tianchi`、`runs/...`）。

## 1. 阻塞清单（只有 2 个）

| # | 资产 | 谁解决 | 代价 |
|---|---|---|---|
| B1 | `data/tianchi/`（8619 train / 957 val，YOLO 格式） | **你**：天池登录后下载 `smartdiagnosisofclothflaw` | 一次性 |
| B2 | `runs/rtdetr/tianchi18_mosaic1280/weights/best.pt` | **我**：从数据重训 60 ep | 15–17 h GPU |

B2 只能由 B1 推导出来。**B1 一到，立刻开跑 R3′，其余全部并行等待。**

## 2. 新的阶段顺序

```
B1 数据到位
   │
   ├─ R2  数据完整性校验（5 分钟，必须过：957 / 583 / 374 / 953 / 泄漏 0）
   │
   ├─ R3′ 重训 mosaic1280（60 ep，seed 42，15–17 h）  ← 关键路径
   │      └─ 产出：新 B* = eval_multIou 的 mean（不再假设 =0.6349）
   │
   ├─ R3″ 5 ep 均匀微调对照（1–2 h）→ 新对照值（替代 0.6158 / 0.6260）
   │
   ├─ R5  冒烟：train_adaptive_loss.py --method uw --epochs 1 --fraction 0.05
   │
   ├─ P0.1 类别侧 oracle（eval_confusion_decomp.py --tau 0.001，~30 min）
   ├─ P0.2 定位侧 oracle（读 R3′ 的 per_class，AP@0.1−AP@0.5）
   ├─ P0.3 召回侧（读同一 JSON 的 missed 列）
   ├─ P0.4 FP 构成（oracle_probe.py --mode fp）
   ├─ S0   测试时选择（eval_ttc_selection.py）
   └─ A2   图像级回放（eval_binary_replay.py --weights <新B*>*）
   │
   └─ 按 P0.* 结果走 handoff §3 路由表 → §4 GPU 队列（A1 / B1 / D1 / C1 / M1）
```

**并行规则**（铁律 1 的落地版）：GPU 上**同一时刻只有一个进程**；`P0.*`、`S0`、A2 是 CPU 任务，可与 GPU 训练并行，但**不要**在训练期间跑 GPU 推理。

## 3. Day-1 操作卡（按顺序打勾）

```
[ ] B1  你提供 data/tianchi
[ ] R0  nvidia-smi --query-gpu=memory.used --format=csv   # 期望 < 2000 MiB
[ ] R1  PY -c "import torch,ultralytics;print(torch.__version__,torch.cuda.is_available(),ultralytics.__version__)"
        # 期望 2.8.0+cu126 True 8.4.91
[ ] R2  PY scripts/check_data_integrity.py            # 见 §5，或按 handoff R2 的内联脚本
[ ] R3′ PY scripts/train_mosaic1280.py                # 后台 15–17 h，日志 logs/stage_r3prime.log
[ ] R3″ PY scripts/train_adaptive_loss.py --epochs 5 --name r3pp_uniform_s42
[ ] R5  PY scripts/train_adaptive_loss.py --method uw --epochs 1 --fraction 0.05 --name smoke_check
[ ] P0.1 ...（R3′ 完成后）
```

## 4. 判定规则（与 handoff 一致，只有基线值换新）

- 转正 = 官方 mean > **同协议对照** + 0.005 且 crit_recall 不降
- |Δ| ≤ 0.0093 = 噪声，补种子到 3 个（42 / 123 / 2024）
- 双降 = 按 handoff §11 F2 处理
- 官方数字只来自 `runs/multi_iou/*_official.json`

> `run/…/PREREG.md` 必须在开跑前写、事后不移动。模板见 §6。

## 5. R2 数据完整性校验（本机版，替代 handoff 的 heredoc）

PowerShell 下 heredoc 语法不通用，写成独立脚本更稳：

```python
# scripts/check_data_integrity.py （本轮新增）
from pathlib import Path
root = Path("data/tianchi")
for split in ["train", "val", "test"]:
    imgs = list((root / split / "images").glob("*"))
    labs = list((root / split / "labels").glob("*.txt"))
    print(split, "images", len(imgs), "labels", len(labs))
defect = empty = boxes = 0
for lp in (root / "val" / "labels").glob("*.txt"):
    lines = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
    if lines:
        defect += 1; boxes += len(lines)
    else:
        empty += 1
print("val defect/empty/boxes =", defect, empty, boxes)
tr = {p.stem for p in (root / "train" / "images").glob("*")}
va = {p.stem for p in (root / "val" / "images").glob("*")}
print("train/val 泄漏:", len(tr & va))
# 期望：val images 957 / defect 583 / empty 374 / boxes 953 / 泄漏 0
```

## 6. PREREG 模板（每个臂开跑前复制到 `runs/<阶段>/<臂>/PREREG.md`）

```markdown
# <臂名> 预注册

- 臂：<A1 objectness 联合 / D1 低 IoU 定位奖励 / ...>
- 协议：<微调 5ep from <权重> / 从头 60ep>
- 唯一变量：<一句话>
- 对照：<run 名>（同 seed、同配方、同评测管道）
- 判据（开跑前写死，事后不移动）：官方 mean > 对照 + 0.005 且 crit_recall 不降
- 种子计划：42 → 若 |Δ| ≤ 0.0093 则补 123 / 2024
- 失败转向：<按 handoff §11 的哪一条>
- 开跑时间 / 预期时长：
```

## 7. 与 handoff v1.0 的差异表

| handoff v1.0 | v1.1 | 原因 |
|---|---|---|
| R3 复现 mean=0.6349±0.002 | **R3′ 重训并重新定义 B\*** | best.pt 不存在 |
| 对照 0.6158 / crit_recall 0.6260 直接引用 | R3″ 重新测量 | `strong_r2_uniform.json` 不存在 |
| P0.2 快速版读 `handoff_baseline.json` | 读 R3′ 的 `*_official.json` | 同上 |
| A2 复用 `runs/binary_replay/mosaic1280/` 缓存 | `eval_binary_replay.py` 现场推理 | 缓存不存在（脚本本就现场跑） |
| `families.yaml` 为 B1 必需 | 用脚本内置 3 族表 | 3 族构成是代码常量，见 AUDIT §4.1 |
| C1「复制 train_mosaic1280.py 加一行」 | 已重写为 argparse 版本，配方落盘 design.json | 原脚本无 main()，import 即训练 |
| 所有 Python 路径 `C:/Users/master/...` | `envs\tianchi\Scripts\python.exe` | 本机用户名 B6 |
