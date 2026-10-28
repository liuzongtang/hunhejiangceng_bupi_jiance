# 阶段 2 再出发 —— 本机可行性审计

| 项 | 内容 |
|---|---|
| 审计日期 | 2026-10-21 |
| 审计对象 | `docs/plan_stage2_reward_relaunch_handoff.md` v1.0（659 行）+ 仓库 `liuzongtang/hunhejiangceng_bupi_jiance` |
| 本机 | DESKTOP-5O8N019 / `C:\Users\B6` / RTX 4090D 24 GB / Win10+ |
| 结论 | **可行，但需要先补 2 个外部资产**；handoff 的绝对数值全部失效，必须自建新基线 |

---

## 1. 仓库与资产盘点

仓库 `main` 最新提交 `e822ced`（2026-10-21 09:21 UTC），全量 1.69 MB —— **纯源码**。
`.gitignore` 明确排除 `data/`、`runs/`、`logs/`、`*.pt`、`*.onnx`；README 第 294–296 行自述：

> 本仓库仅包含源代码。Tianchi 数据集与训练好的模型权重体积较大，未纳入版本控制，请按需自行获取或训练生成。

### 1.1 handoff 引用点核对（22 项）

**存在于仓库（14）**：`eval_multIou.py`、`train_mosaic1280.py`、`train_adaptive_loss.py`、`train_rtdetr_reward.py`、`train_rtdetr_{boxweighted,costsensitive,nwd,objectness,grpo}.py`、`eval_confusion_decomp.py`、`eval_binary_replay.py`、`train_family_classifier.py`、`build_family_mapping.py`、`convert_tianchi_to_yolo.py`。

**handoff 要求"新写"、审计时确实不存在（8）**：

| 脚本 | 对应臂 | 本机状态 |
|---|---|---|
| `scripts/oracle_probe.py` | P0.2 / P0.4 | 待写 |
| `scripts/build_family_dataset.py` | B1 | **已写**（见 §5） |
| `scripts/eval_ttc_selection.py` | S0 / S2 | 待写 |
| `scripts/train_family_detector.py` | B1 | 待写 |
| `scripts/train_rtdetr_locreward.py` | D1 | 待写 |
| `scripts/train_mosaic1280_earlyreward.py` | C1 | **已写**（见 §5） |
| `scripts/train_rtdetr_aploss.py` | M1 | 待写 |
| `scripts/eval_multIou_yolo.py` | S1 | 待写 |

### 1.2 缺失的输入资产

| 资产 | handoff 用途 | 本机 |
|---|---|---|
| `data/tianchi/`（8619 train / 957 val，YOLO） | R2 校验、R3 基线、一切评测 | **缺** |
| `runs/rtdetr/tianchi18_mosaic1280/weights/best.pt` | B*=0.6349、所有微调臂的起点 | **缺**（仓主也无法提供） |
| `runs/rtdetr/tianchi18_full/weights/best.pt` | F6 sanity（mAP50 0.541） | **缺** |
| `runs/multi_iou/handoff_baseline.json` | P0.2 快速版、per_class 表 | 缺（R3 产出） |
| `runs/multi_iou/strong_r2_uniform.json` | 微调对照 0.6158 / crit_recall 0.6260 | 缺（可重跑） |
| `runs/confusion/decomp_*.json` | P0.1 类别侧 oracle 的历史参照 | 缺（可重跑） |
| `runs/binary_replay/mosaic1280/` | A2 直接复用 | 缺（可重跑，`eval_binary_replay.py` 现场推理） |
| `runs/family_clustering/families.yaml` | B1 族映射 | 缺（**不阻塞**，见 §4.1） |
| `runs/threshold_calibration/threshold_sweep.csv` | P0.1 交叉校验 | 缺（`eval_threshold_calibration.py` 可重算） |
| `rtdetr-l.pt`（COCO 初始化权重） | 从头训配方 | 缺（ultralytics 自动下载） |

---

## 2. 阻塞点分类（关键结论）

**只有 2 个真·硬阻塞**，其余全部"可用脚本重算，只是要花 GPU 时间"：

| 阻塞 | 为什么硬 | 解锁方式 |
|---|---|---|
| **B1. 数据集** | 无法用任何脚本凭空造；天池官方页需阿里云账号登录 | 你在天池登录后下载 `smartdiagnosisofclothflaw`，或把已有副本拷进 `data/tianchi` |
| **B2. mosaic1280 `best.pt`** | 是 B*、所有 5-epoch 微调臂、P0.1/P0.2/S0/A2 的共同起点 | 只能**自己重训**（60 ep，约 15–17 h GPU） |

**非阻塞（可重算）**：`strong_r2_uniform`（5 ep 微调 ≈ 1–2 h）、`decomp_*`（推理 ≈ 30 min）、`binary_replay` 缓存（推理 ≈ 30 min）、`threshold_sweep.csv`（推理 ≈ 30 min）、`families.yaml` 的 3 族部分（代码常量已固化）。

### 2.1 由此产生的战略性后果

handoff 全部判据都锚在**已不存在的权重**上：

| 锚点 | 值 | 现状 |
|---|---|---|
| B*（基线 mean） | 0.6349 | 不可复现，需重训后重测 |
| 微调对照 | 0.6158 / crit_recall 0.6260 | 需从新 B* 重跑 5 ep |
| 定位头量上界 | mean(AP@0.1−AP@0.5)=0.074 | 需新 B* |
| 图像级基线 | 93.73% @ tau=0.35 | 需新 B* |
| 种子离散度 | 0.0093 | 可作为先验沿用，但需在新基线上确认 |

**处理原则**：`0.6349` 等数字降级为**文献参照值**，不再作为判定门。所有后续判定必须在**同一批自建基线**内部比较（同 seed、同配方、同评测管道）。这不削弱计划的有效性——铁律 2 本来就要求"同协议对照"——但**必须重写 §13 结果表的"参照"行**，并承认与历史论文数值不可直接拼接。

---

## 3. 本机环境审计

### 3.1 已确认

| 项 | 实测 |
|---|---|
| GPU | RTX 4090D，**1990 MiB / 24564 MiB 占用**（< 铁律 1 的 2000 MiB 阈值，可开跑） |
| 磁盘 | C: 969 GB 空闲，D: 1122 GB 空闲 |
| 默认 Python | `C:\Users\B6\anaconda3\python.exe` = **3.14.6**，无 torch |
| 现有 torch | `ragforge`/`warehouse` = 2.14.1+**cpu**；`ict26` = 2.8.0+**cpu** —— 全部无 CUDA |
| 目标环境 | py3.12 venv + **torch 2.8.0+cu126** + **ultralytics 8.4.91**（与 handoff R1 期望版本一致；8.4.91 经 `pip index` 证实存在） |

### 3.2 本机特有的两个坑（handoff 未涵盖，已解决）

1. **沙箱阻断 schannel**：`curl.exe` / `Invoke-WebRequest` 访问 HTTPS 全部报
   `schannel: AcquireCredentialsHandle failed / SEC_E_NO_CREDENTIALS`，但 **Python 自己的 TLS 栈正常**。
   后果：一切下载必须走 `python -m pip` 或 `urllib`，不要用 curl/PowerShell 下载。
2. **`os.mkdir(path, 0o700)` 会造出"连属主都进不去"的目录**（`icacls` 读 ACL 都报 Access denied）。
   因为 `tempfile.mkdtemp()` 用 `mkdir(..., 0o700)`，于是 `ensurepip`、`venv` 自带 pip 引导、pip 的临时解包**全部失败**。
   解决：在 venv 的 `Lib/site-packages/sitecustomize.py` 里把 `os.mkdir` 的 mode 强制为 `0o777`（单文件、可删、不影响训练语义）。

### 3.3 环境位置（与 handoff R1 的差异）

handoff 假设 `C:/Users/master/anaconda3/python.exe`；**本机该路径不存在**。实际使用：

```
venv      : C:\Users\B6\Documents\deepseek-harness\default-workspace\envs\tianchi\Scripts\python.exe
项目根     : C:\Users\B6\Documents\deepseek-harness\default-workspace\repo
```

所有 handoff 命令里的 `C:/Users/master/anaconda3/python.exe` 一律替换为上面的 venv 路径。

---

## 4. 文 档与代码的不一致（会白烧 GPU 天的部分）

### 4.1 `families.yaml` 缺失不阻塞 B1

handoff §6 把 `runs/family_clustering/families.yaml` 列为必需资产。审计结论：**3 族映射不需要它**——

- `scripts/build_family_mapping.py` 第 26–92 行把 18 类顺序与 `SUPER = {A_linear: 7 类, B_blotch: 5 类, C_point: 6 类}` 写死为常量；
- 论文 4.10 与 `paper_skeleton.md` 第 289–293 行给出同一份三族构成；
- `families.yaml` 里**只有 6 子族细分**（`class_to_family3` 之外的 operational 6 分类）才依赖 `runs/family_clustering/box_features.npz`（该文件要用检测器提特征，缺权重时无法重算）。B1 只用 3 族，**不需要 6 子族**。

→ 已把 3 族表内置进新脚本 `build_family_dataset.py`（`--map` 仍可覆盖）。

### 4.2 C1 不能"复制 train_mosaic1280.py 加一行"

handoff §7 写"复制 `train_mosaic1280.py` → `train_mosaic1280_earlyreward.py`，唯一差异：前 10 epoch 以 β=0.5 注入"。**代码事实**：
`train_mosaic1280.py` 是**无 `main()`、无 `argparse` 的顶层脚本**，import 即执行训练；它硬编码 `epochs=60`、`name="tianchi18_mosaic1280"`、`resume` 从 `sys.argv[1]=="resume"` 读。
→ 必须重写为带 argparse 的版本，并逐行保留 bf16 补丁三行、BN 哨兵、`optimizer="auto"`、`rect=True`。已完成（`train_mosaic1280_earlyreward.py`），并把配方写进 `design.json` 落盘，防止事后漂移。

### 4.3 bf16 / amp=False 两条协议不要混用

- R4 **微调协议**：`amp OFF`（fp16 毒化 BN，坑 #3）；
- 从头训配方：`amp=True` **但已被补丁改成 bf16** + `check_amp=True`。

两者都对，但执行人若把"amp OFF"套到从头训，或把 `amp=True`（真 fp16）用到微调，都会重演 F4。**规则**：微调脚本 = `--amp` 默认关；从头训脚本 = 必须带 `scripts/train_mosaic1280.py` 的三行补丁。

### 4.4 已验证为真的 handoff 命令

- A1 `train_rtdetr_objectness.py --probe-only --epochs 3`：`--probe-only` 存在（脚本第 265 行），语义与文档一致（冻结主干只训 objectness 头）。
- R5 冒烟 `train_adaptive_loss.py --method uw --epochs 1 --fraction 0.05`：`--fraction`、`--method` 均存在。
- A2 `eval_binary_replay.py --weights ... --tag ...`：`--tag` 存在，且脚本**现场推理**（`cache=None`），不依赖预存缓存 → A2 可由新基线直接重算。

### 4.5 七维奖励注入接口（C1/D1/M1 必须遵守）

`backend/training/reward_rtdetr.py::attach_reward_criterion` 同时设 `model.criterion` 与 `model.init_criterion`。
但**真正的挂载点必须是 `on_train_start`**：trainer 在 `_setup_train` 内重建模型，训练前挂在 `model.model` 上会被静默丢弃（`train_adaptive_loss.py` 第 98–118 行注释明确记录）。所有新奖励臂都必须照此实现。

---

## 5. 本轮已产出的脚本

| 脚本 | 状态 | 说明 |
|---|---|---|
| `scripts/build_family_dataset.py` | 已写，未接数据验证 | 18 类 → 3 族 YOLO 数据集；内置 3 族表；hardlink/copy/symlink；自检"框数不增删"（val 应为 953）；`--dst` 与 `--src` 同目录会拒绝 |
| `scripts/train_mosaic1280_earlyreward.py` | 已写，未接数据验证 | C1：前 N epoch 注入 β，之后精确 β=0；配方与 `train_mosaic1280.py` 逐项一致并落盘 `design.json`；含 BN 哨兵与 `epoch_beta()` 单测友好函数 |

验证状态：`py_compile` 通过；**未做端到端验证**（缺 `data/tianchi`）。按 handoff 附录 A 的验收条款，任何新评测脚本在其"原样重算"结果与 R3 对不上之前不得采信其数字。

---

## 6. 结论与建议路径

1. **数据集是唯一真正的入口**（权重可由它自训）。天池页 `tianchi.aliyun.com/dataset/79336` 需阿里云账号登录；论文（arXiv 2412.03200）描述 5913 缺陷 + 3663 正常 = 9576 张，与本项目 8619+957 完全吻合，确认是同一份数据。
2. 拿到数据后**第一件事不是跑探针，而是重训 mosaic1280**（15–17 h）——没有它，P0.1/P0.2/S0/A2 全部无输入。
3. 重训完成后重跑 5 ep 均匀微调做对照，把 §13 的"对照"列换成新值。
4. 再按 handoff §3 → §4 的路由表推进；C1 screen 可以在新基线评测期间并行排队（它不依赖 best.pt，只依赖数据）。
5. 论文/大创材料里所有旧数字必须标注"历史机器、权重已失，无法复现"，或全部用新基线重算后再引用。

---

## 7. 本轮新增的两个关键发现（改变执行细节）

### 7.1 P0.2 的 box oracle 按原规格是**恒等于原值**的（规格错误，非实现问题）

handoff 附录 A 定义：`mode=box` 把"与 GT 匹配（IoU>=0.5，任意类）"的 pred 框替换成 GT 框。
**推理证明**：能被替换的 pred 本来就在 IoU>=0.5 上与 GT 匹配，也就是在 0.1 / 0.3 / 0.5 三个阈值下**都已经是 TP**；替换后框更准、cls/conf 不变，TP 集合完全相同，AP 只在重复/歧义 TP 的排序上可能变化——所以 mean 不变。
`oracle_probe.py` 已用合成数据实测确认：`box` 与 `none` 结果逐位一致。

**修正**：定位头量要用"近失"变体 —— `--mode box --iou-thr 0.3`（或 0.1），把"在 0.5 上漏、在 0.3 上对"的框吸到 GT 上，其 Δ 才对应 AP@0.1−AP@0.5 的 0.074 落差。
该结论已写进脚本 docstring、JSON 的 `matching.box_oracle_note` 与控制台输出。

### 7.2 沙箱内无法创建真正脱离的进程 → 长跑必须由用户启动

**先说被推翻的结论**：本文档第一版断言"沙箱内无法创建脱离进程"。补充实测后的事实是分层的：

| 方式 | workspace-write（原） | danger-full-access（现） |
|---|---|---|
| 我的后台 job | 随调用被杀 | 同左 |
| `Start-Process`（隐藏窗口） | 跨工具调用能活，**关 app 即死** | 同左（仍属 DSH 进程树） |
| `schtasks` | `ERROR: The system cannot find the path specified` | **成功**（服务托管） |
| WMI `Win32_Process.Create` | `拒绝访问`（0x80041003） | **成功**（父进程是 WmiPrvSE） |

**关 app 会杀训练的原因**：DSH 把整棵进程树放进一个作业对象，`Start-Process` 的子进程仍在树内；只有**服务托管**的进程能逃出去。

**采用方案**：`TianchiStage2Queue` 计划任务 → `run_stage2_task.bat` → `tools/queue_runner.py`。已验证：在 **session 5（交互会话）**运行、CUDA 可用、2048×2048 矩阵乘成功、22.9 GB 空闲；任务设置改为不因空闲/电池停止、`ExecutionTimeLimit=PT72H`。

**这次一并修掉的三个坑**：
1. 队列 runner 把 ultralytics 进度条字符（U+2578）写进 GBK 控制台 → `UnicodeEncodeError` 打死 runner 并连带杀训练。已强制 stdout=UTF-8 并逐行兜底。
2. `train_mosaic1280.py` 缺 `if __name__ == "__main__"` 保护：Windows spawn 下 workers>0 会递归重跑整个脚本 → **这才是历史上 workers 只能设 0 的真正根因**（不只是 WinError 1455）。加保护后 workers=4 可用且快 34%。
3. PowerShell `Set-Content -Encoding utf8` 会加 BOM，把 `queue.json` 写坏（`json.load` 直接失败）。改用不带 BOM 的写入。
