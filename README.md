# 织物瑕疵检测系统

> 纺织物布匹瑕疵检测系统 — 计算机视觉（RT-DETR）+ 7 维度混合奖惩训练 + 中文实时监控

## 项目概述

面向纺织生产线的 AI 视觉布匹瑕疵检测系统，基于 **RT-DETR-L** 检测器、**20 类 Tianchi 布匹瑕疵**分类体系，支持 4 种推理后端、7 维度混合奖惩训练，以及中文 Web 实时监控看板。

### 核心指标

| 指标 | 目标 / 现状 | 说明 |
|------|------------|------|
| 缺陷类型 | **20 类** | Tianchi `smartdiagnosisofclothflaw` 分类体系 |
| 检测精度 | **mAP50 = 0.594 / mAP50-95 = 0.334** | RT-DETR-L，50 epochs |
| 推理后端 | **4 种** | Dummy / ONNX / PyTorch / RT-DETR |
| 训练机制 | **7 维度混合奖惩** | 定位 + 分类 + 校准 + 漏检 + 误检 + 断经 + 跳花 |
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
│   ├── training/              # 7 维度混合奖惩训练器
│   ├── inference/             # Dummy / ONNX / PyTorch / RT-DETR 推理引擎
│   ├── deploy/                # 导出 / 量化 / 边缘部署
│   ├── dashboard.html         # 中文监控看板
│   └── simulator.html         # 中文产线模拟器
├── mixed_reward/              # 混合奖惩机制 + 数字识别 CNN
├── scripts/                   # 训练 / 预测 / 可视化 / 导出 / 仿真脚本
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

## 缺陷分类（20 类 Tianchi）

模型输出的 class id 0..19 与下表顺序一致。分类体系的唯一事实来源为 `backend/schemas/defect.py`。

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
| 11 | CF-01 | 粗纬 | `coarse_weft` | 轻微 minor |
| 12 | WS-01 | 纬缩 | `weft_shrink` | 轻微 minor |
| 13 | ST-02 | 浆斑 | `size_stain` | 中等 medium |
| 14 | KN-02 | 整经结 | `warping_knot` | 轻微 minor |
| 15 | SK-01 | 星跳 | `star_skip` | 主要 major |
| 16 | BS-01 | 断氨纶 | `broken_spandex` | 严重 critical |
| 17 | DS-01 | 稀密档 | `dense_section` | 主要 major |
| 18 | SM-01 | 磨痕 | `surface_mark` | 轻微 minor |
| 19 | WD-01 | 死皱 | `weave_defect` | 主要 major |

> 另有 `other`（代码 `OT-01`，严重度 `info`）作为未知类型的防御性兜底，模型正常只输出上表 20 类。

## 7 维度混合奖惩

| 维度 | 公式 | 范围 | 说明 |
|------|------|------|------|
| D01 定位 | `2*IoU - 1` | [-1, 1] | 框定位精度 |
| D02 分类 | `±1` | {-1, 1} | 类别判定 |
| D03 校准 | `-ECE` | (-∞, 0] | 置信度校准 |
| D04 漏检 | `-λ·miss` | (-∞, 0] | 漏检惩罚 |
| D05 误检 | `-λ·fp` | (-∞, 0] | 误检惩罚 |
| D06 断经 | `+1 / -2` | {-2, 1} | 断经 / 断氨纶敏感度 (class {9, 16}) |
| D07 跳花 | `+1 / -2` | {-2, 1} | 星跳 / 死皱敏感度 (class {15, 19}) |

`L_total = L_det + β·L_scalar + γ·L_consistency`

## 模型训练（RT-DETR）

```bash
# 训练 20 类 Tianchi 检测器 (RT-DETR-L, 50 epochs)
python scripts/train_rtdetr.py

# 导出 ONNX
python scripts/export_rtdetr_onnx.py

# 推理预测（自动识别 .pt / .onnx）
python scripts/predict_defect.py --weights runs/rtdetr/tianchi20/weights/best.onnx --limit 10

# 可视化预测结果
python scripts/visualize_predictions.py --limit 6
```

RT-DETR 输入为 1280 拉伸方图（BGR→RGB、/255，无 ImageNet 均值/方差），ONNX 输出 `(1, 300, 6)` = `[cx, cy, w, h, score, class]`（归一化坐标），一对一匹配、无需 NMS。

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
# 全部测试
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
