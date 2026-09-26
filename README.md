# 织物瑕疵检测系统

> 纺织物布匹瑕疵检测系统 — 计算机视觉（RT-DETR）+ 7 维度混合奖惩训练 + 中文实时监控

## 项目概述

面向纺织生产线的 AI 视觉布匹瑕疵检测系统，基于 **RT-DETR-L** 检测器、**18 类 Tianchi 布匹瑕疵**分类体系（由 20 类合并两对视觉易混淆类而来），支持 4 种推理后端、7 维度混合奖惩训练（含 GRPO），以及中文 Web 实时监控看板。

### 核心指标

| 指标 | 目标 / 现状 | 说明 |
|------|------------|------|
| 缺陷类型 | **18 类**（20 类合并后） | 断经↔磨痕、纬缩↔粗纬合并；Tianchi `smartdiagnosisofclothflaw` |
| 检测精度 | **mAP50 = 0.594（20 类）/ 0.541（18 类）** | RT-DETR-L，端到端免 NMS |
| 推理后端 | **4 种** | Dummy / ONNX / PyTorch / RT-DETR |
| 训练机制 | **7 维度混合奖惩 + GRPO** | 定位 + 分类 + 校准 + 漏检 + 误检 + 断经 + 星跳/死皱 |
| 实时监控 | WebSocket + 健康检查 | 中文看板 + 产线模拟器 |

## 快速启动

```bash
# 安装
pip install -e .

# 启动后端服务
python -m backend.main

# 浏览器访问
http://localhost:8000/simulator    # 中文产线模拟器
http://localhost:8000/dashboard    # 中文监控看板
http://localhost:8000/docs         # API 文档 (Swagger)
```

### 一键启动

```cmd
scripts\start.bat              # Windows: 安装 → 测试 → 训练
scripts\start.bat backend      # 启动 API 服务器
scripts\start.bat e2e          # 产线仿真
scripts\start.bat demo         # 混合奖惩演示
scripts\start.bat test         # 仅运行测试
```

```bash
bash scripts/start.sh          # Linux/macOS 同上
```

## 前端页面

| 地址 | 页面 | 语言 | 功能 |
|------|------|------|------|
| `/simulator` | 产线模拟器 | 中文 | Canvas 织物纹理生成、20 类缺陷标注动画、流水线可视化 |
| `/dashboard` | 监控看板 | 中文 | 4 标签、多种图表、WebSocket 实时告警、20 类分布 |
| `/docs` | API 文档 | 英文 | Swagger UI 交互式调试 |

## 系统架构

```
                          ┌──────────────────────┐
                          │   Web 前端             │
                          │   /simulator (模拟器)  │
                          │   /dashboard (看板)    │
                          └──────────┬───────────┘
                                     │ WebSocket
┌──────────┐    ┌──────────┐    ┌────▼────┐    ┌──────────┐
│ 工业相机 │───▶│ 预处理   │───▶│ 推理检测 │───▶│ 图像存储 │
│ (图像)   │    │ (1280)   │    │ (RT-DETR)│    │ (MinIO)  │
└──────────┘    └──────────┘    └────┬────┘    └──────────┘
                                     │
                              ┌──────▼──────┐
                              │ 告警服务     │
                              │ (MQTT/WS)   │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │ PostgreSQL  │
                              └─────────────┘
```

## 项目结构

```
├── backend/                   # 后端服务
│   ├── main.py                # FastAPI 入口 (REST + WebSocket)
│   ├── config.py / config.yaml
│   ├── database.py            # SQLAlchemy 2.0 async 引擎
│   ├── storage.py             # MinIO/S3 图像存储 (本地回退)
│   ├── websocket.py           # WebSocket 实时告警推送
│   ├── models/                # ORM 模型 (defect / report / ...)
│   ├── schemas/               # Pydantic Schema + 20 类枚举 (唯一事实来源)
│   ├── services/              # 检测 / 模型 / 告警
│   ├── routers/               # detection / model / system / storage
│   ├── training/              # 可微 7 维奖励 + 奖励注入 (reward_rtdetr) + GRPO (grpo)
│   ├── inference/             # Dummy / ONNX / PyTorch / RT-DETR 推理引擎
│   ├── deploy/                # 导出 / 量化 / 边缘部署
│   ├── dashboard.html         # 中文监控看板
│   └── simulator.html         # 中文产线模拟器
├── mixed_reward/              # 混合奖惩机制 + 数字识别 CNN
├── scripts/                   # 训练 / 奖励 / GRPO / 混淆诊断 / 可视化 / 导出脚本
├── tests/                     # 单元测试
├── configs/                   # YAML 配置
└── requirements*.txt          # 依赖
```

## API 参考

### 检测

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/detection/report` | 提交检测报告 |
| POST | `/api/v1/detection/infer` | 运行推理检测 |
| GET | `/api/v1/detection/reports` | 查询历史报告 |

### 模型管理

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/model/update` | 注册模型更新 |
| GET | `/api/v1/model/current` | 当前活跃模型 |
| GET | `/api/v1/model/history` | 版本历史 |

### 系统监控

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/system/status` | 设备 + 系统资源 |
| GET | `/api/v1/system/stats` | 聚合统计 (1h/24h/7d/30d) |
| GET | `/api/v1/system/health` | 健康检查 |

### 图像存储

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/storage/upload` | 上传图像 |
| GET | `/api/v1/storage/download/{name}` | 下载图像 |
| GET | `/api/v1/storage/list` | 列出图像 |
| DELETE | `/api/v1/storage/{name}` | 删除图像 |

### 实时推送

| 协议 | 端点 | 说明 |
|------|------|------|
| WebSocket | `/ws/alerts` | 实时告警推送 |

## 缺陷分类（18 类 Tianchi，20→18 合并）

> 由 20 类合并两对视觉易混淆类而来：**断经(broken_warp)↔磨痕(surface_mark)**、**纬缩(weft_shrink)↔粗纬(coarse_weft)**——两者图像上视觉几乎相同、双向混淆，属标签噪声而非模型可训练的差距（详见下方「关键研究结论」）。合并后关键类召回显著回升。

模型输出的 class id 0..17 与下表顺序一致。分类体系的唯一事实来源为 `backend/schemas/defect.py`。

| id | 代码 | 名称 | 英文 (枚举值) | 严重度 |
|----|------|------|--------------|--------|
| 0  | HO-01 | 破洞 | `hole` | 严重 critical |
| 1  | ST-01 | 污渍 | `stain` | 中等 medium |
| 2  | SL-01 | 三丝 | `three_silk` | 轻微 minor |
| 3  | KN-01 | 结头 | `knot` | 轻微 minor |
| 4  | FL-01 | 花板跳 | `flower_board` | 主要 major |
| 5  | HF-01 | 百脚 | `hundred_feet` | 主要 major |
| 6  | HP-01 | 毛粒 | `hair_particle` | 轻微 minor |
| 7  | CW-01 | 粗经 | `coarse_warp` | 轻微 minor |
| 8  | LW-01 | 松经 | `loose_warp` | 轻微 minor |
| 9  | BW-01 | 断经 | `broken_warp` | 严重 critical |
| 10 | HW-01 | 吊经 | `hanging_warp` | 主要 major |
| 11 | WS-01 | 纬缩 | `weft_shrink` | 轻微 minor |
| 12 | ST-02 | 浆斑 | `size_stain` | 中等 medium |
| 13 | KN-02 | 整经结 | `warping_knot` | 轻微 minor |
| 14 | SK-01 | 星跳 | `star_skip` | 主要 major |
| 15 | BS-01 | 断氨纶 | `broken_spandex` | 严重 critical |
| 16 | DS-01 | 稀密档 | `dense_section` | 主要 major |
| 17 | WD-01 | 死皱 | `weave_defect` | 主要 major |

> 另有 `other`（代码 `OT-01`，严重度 `info`）作为未知类型的防御性兜底，模型正常只输出上表 18 类。

## 7 维度混合奖惩（可微）

奖励对 logits 与 boxes **可微**：硬决策（argmax、集合成员、硬 IoU 阈值）已替换为 softmax/sigmoid 软指示，因此奖励信号能直接反传进模型本身，而不是只调维度权重。

| 维度 | 公式 | 范围 | 说明 |
|------|------|------|------|
| D01 定位 | `2*IoU - 1`（软 IoU） | [-1, 1] | 框定位精度 |
| D02 分类 | `2*p_true - 1` | [-1, 1] | softmax 在 GT 类上的质量（平滑准确率） |
| D03 校准 | `-|conf - p_true|` | [-1, 0] | 置信度 vs 软准确率 |
| D04 漏检 | `-λ·sigmoid((θ-IoU)/τ)` | [-λ, 0] | 软漏检惩罚 |
| D05 误检 | `-λ·sigmoid((θ-IoU)/τ)` | [-λ, 0] | 软误检惩罚 |
| D06 断经 | `3*p_broken - 2` | [-2, 1] | 断经 / 断氨纶敏感度 (class {9, 15}) |
| D07 跳花 | `3*p_skip - 2` | [-2, 1] | 星跳 / 死皱敏感度 (class {14, 17}) |

`L_total = L_det + β·L_scalar + γ·L_consistency`

基于该奖励的两条训练路线：
- **Layer-1 奖励注入**：`backend/training/reward_rtdetr.py` 把可微奖励 `loss_reward` 注入 ultralytics RT-DETR 训练循环。
- **Layer-3 GRPO**：`backend/training/grpo.py` + `grpo_trainer.py` 自研 query 噪声 REINFORCE 循环（组内相对优势 + KL 锚定冻结参考策略）。

## 模型训练（RT-DETR）

```bash
# 训练 18 类 Tianchi 检测器 (RT-DETR-L, 60 epochs)
python scripts/train_rtdetr.py --amp False --batch 4

# 导出 ONNX
python scripts/export_rtdetr_onnx.py

# 推理预测（自动识别 .pt / .onnx）
python scripts/predict_defect.py --weights runs/rtdetr/tianchi20/weights/best.onnx --limit 10

# 可视化预测结果
python scripts/visualize_predictions.py --limit 6
```

> ⚠️ **18 类训练必须 `--amp False`**。RT-DETR 的 deformable attention 走 `grid_sample`，是 fp16 AMP 的不稳定点，`amp=True` 下 18 类训练必然 NaN 发散（20 类同配方却不炸）。fp32 代价是显存翻倍、batch 8→4，最终 mAP50=0.541 全程 0 NaN。

RT-DETR 输入为 1280 拉伸方图（BGR→RGB、/255，无 ImageNet 均值/方差），ONNX 输出 `(1, 300, 6)` = `[cx, cy, w, h, score, class]`（归一化坐标），一对一匹配、无需 NMS。

### 奖励注入 / GRPO 训练

```bash
# Layer-1 可微奖励注入（--beta 0 为纯监督 A/B 基线）
python scripts/train_rtdetr_reward.py --beta 0.5 --epochs 5 --name beta0.5_e5

# Layer-3 查询噪声 GRPO
python scripts/train_rtdetr_grpo.py --epochs 1 --num-groups 4 --name grpo_skel

# 重绘奖励训练曲线 / A/B 召回对比
python scripts/plot_reward_training.py --dir runs/rtdetr-reward/<name>
python scripts/plot_ab_recall.py
```

完整跑法、关键参数与判断标准见 [`scripts/RUNBOOK.md`](scripts/RUNBOOK.md)。

## 关键研究结论与告警兜底

针对「关键类召回」（断经 / 星跳 / 断氨纶 / 死皱）做了一整轮排查（奖励注入、GRPO、置信度校准、过采样精调、混淆矩阵 + 裁剪图人工复核），结论是：

> **断经（broken_warp）召回 ~0.45 是标签噪声天花板，不是模型可训练的差距。** 断经↔磨痕、纬缩↔粗纬在图像上视觉几乎相同、混淆双向（label-noise 特征），任何训练干预（奖励/GRPO/校准/过采样）都无法实质提升。合并视觉相同类后召回显著回升（纬缩+粗纬→0.88，断经+磨痕→0.79）。

据此在**告警层**做了兜底：磨痕（minor）视觉上可能就是断经（critical），告警时按 `effective_alert_severity()` 升级到 critical 触发停机，避免关键缺陷被静默降级。存储的 `severity` 保持规范值，升级级别以 `escalated_severity` 独立落库（`alarm_type="suspected_critical"`），可审计。唯一事实来源在 `backend/schemas/defect.py`。

## 论文与同类对比

已产出**一篇学术论文骨架**（双语，诚实口径 A+B），位于 [`docs/paper/paper_skeleton.md`](docs/paper/paper_skeleton.md)（Markdown + Word），配 3 张图（技术路线 / 训练曲线 / 奖励 A/B）于 `docs/paper/figures/`。

核心创新点与贡献：

1. **RT-DETR-L Tianchi 基线**：20 类 mAP50 = 0.594、18 类 mAP50 = 0.541，处公开一流水平。
2. **七维混合奖惩**：D01–D07 可微奖励注入训练循环（`loss_reward = -β·mean(R)`）+ Layer-3 GRPO（query 噪声 REINFORCE + 组内相对优势 + KL 锚定）。奖励注入 A/B：漏检注入使全局召回 **+0.030**（0.523→0.553），精确 −0.022，mAP50 持平（+0.004）。
3. **标签噪声天花板诊断**：混淆矩阵 + 双向判定 + 裁剪复核，定位断经↔磨痕、纬缩↔粗纬为标签噪声；合并后召回 纬缩+粗纬 0.40→0.88、断经+磨痕 0.52→0.79。
4. **fp16 AMP 溢出修复**：18 类 NaN 发散根因为 deformable-attention `grid_sample` fp16 溢出，`amp=False` 修复。

**已核验同类对比（Tianchi 20 类，arXiv/Crossref）**：

| 方法 | mAP50 | 说明 |
|------|-------|------|
| SPFFNet（YOLOv11） | 65.8% | 公开 SOTA |
| 改进 RT-DETR（R18，浙大学报） | 60.0% | — |
| Fab-ME（YOLOv8s） | 59.4% | — |
| **本项目 RT-DETR-L（20 类基线）** | **59.4%** | 与 Fab-ME 完全一致 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python + FastAPI |
| 数据库 | PostgreSQL / SQLite + SQLAlchemy 2.0 async |
| 推理 | RT-DETR / ONNX Runtime / PyTorch |
| 存储 | MinIO / S3 / 本地回退 |
| 前端 | Chart.js + WebSocket (纯 HTML，中文界面) |
| 训练 | PyTorch + 7 维度混合奖惩 |

## 测试

```bash
# 全部测试（205 个，含可微奖励 / GRPO / 告警升级等）
pytest tests/ backend/tests/ backend/training/tests/ backend/inference/tests/ backend/deploy/tests/ -v

# 功能验证
python scripts/func_test.py
```

## 部署

```bash
# 生成部署产物
python -c "from backend.deploy import generate_all; generate_all('./deploy')"

# Docker
docker-compose -f deploy/docker-compose.yml up -d

# 边缘节点 (systemd)
sudo cp deploy/fabric-defect.service /etc/systemd/system/
sudo systemctl enable --now fabric-defect
```

## 数据与模型权重

本仓库仅包含源代码。Tianchi 数据集（`smartdiagnosisofclothflaw`）与训练好的模型权重（`runs/rtdetr/tianchi20/weights/`）体积较大，未纳入版本控制，请按需自行获取或训练生成。
