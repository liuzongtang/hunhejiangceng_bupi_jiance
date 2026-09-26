# -*- coding: utf-8 -*-
"""Fill 附件2 校级学生科研项目立项申请书 with project content (research-only fields).

Leaves personal fields (学号/姓名/学院/专业/指导教师/成员/论文/签名/公章) untouched.
Writes a NEW file so the blank template is preserved.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

SRC = "附件2：2026年度校级学生科研项目立项申请书.docx"
DST = "附件2：2026年度校级学生科研项目立项申请书（已填写）.docx"

PROJECT_NAME = "织瑕智鉴——基于RT-DETR与七维混合奖惩的织物瑕疵智能检测系统"


def set_font(run, bold=False, size=12):
    run.font.name = "宋体"
    run.font.size = Pt(size)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "宋体")


def write_cell(cell, blocks):
    """blocks: each is a list of (text, bold) runs, or ("IMG", path, w_cm), or ("CAP", text)."""
    tc = cell._tc
    for p in tc.findall(qn("w:p")):
        tc.remove(p)
    for block in blocks:
        para = cell.add_paragraph()
        if isinstance(block, tuple) and block[0] == "IMG":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.add_run().add_picture(block[1], width=Cm(block[2]))
        elif isinstance(block, tuple) and block[0] == "CAP":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.space_after = Pt(4)
            r = para.add_run(block[1])
            set_font(r, bold=False, size=9)
            r.font.color.rgb = RGBColor(0x89, 0x87, 0x81)
        else:
            for text, bold in block:
                set_font(para.add_run(text), bold=bold)
            para.paragraph_format.space_after = Pt(0)
            para.paragraph_format.line_spacing = 1.25
            # first-line indent 2 chars for body paragraphs (non-heading)
            if not (len(block) == 1 and block[0][1]):
                para.paragraph_format.first_line_indent = Pt(24)


def set_text(cell, text, bold=False, size=12):
    """Set a cell to a single-run paragraph (for short fields like 项目名称)."""
    tc = cell._tc
    for p in tc.findall(qn("w:p")):
        tc.remove(p)
    para = cell.add_paragraph()
    set_font(para.add_run(text), bold=bold, size=size)


# ---------------------------------------------------------------- content
def B(t):
    """Bold heading paragraph."""
    return [(t, True)]


def N(t):
    """Normal paragraph."""
    return [(t, False)]


def IMG(name, w=13.5):
    """Figure paragraph."""
    return ("IMG", str(FIG / name), w)


def CAP(t):
    """Caption paragraph."""
    return ("CAP", t)


SEC1 = [
    B("1.研究状况与选题价值（1000字）"),
    B("（一）国内外研究现状"),
    N(
        "织物瑕疵检测经历了从传统图像处理到深度学习的范式演进。早期方法依赖 Gabor 滤波、局部二值模式（LBP）、灰度共生矩阵（GLCM）等手工特征，对光照与纹理变化敏感、泛化能力弱，难以应对多类别、弱对比的复杂瑕疵。卷积神经网络兴起后，Faster R-CNN、SSD 与 YOLO 系列等目标检测器被引入纺织质检，将检测精度与实时性提升到工业可用水平；国内天池“布匹疵点智能识别”等公开竞赛进一步推动了数据集与评测基准的规范化。近年来，以 DETR、RT-DETR 为代表的 Transformer 检测器凭借全局注意力与一对一匹配、免 NMS 的端到端优势在通用检测上表现突出，但在织物瑕疵这一细粒度、长尾、类别高度易混淆的场景中仍处于探索初期。本项目基于 RT-DETR-L 构建 20 类 Tianchi 检测器，50 轮训练即收敛至 mAP50=0.594、mAP50-95=0.334，验证了 Transformer 检测器在该场景的可行性。"
    ),
    IMG("fig1_training_curves.png"),
    CAP("图1  RT-DETR-L 训练收敛曲线（50 epochs，mAP50 收敛至 0.594）"),
    B("（二）现有研究不足"),
    N(
        "其一，训练目标单一。现有方法普遍以检测损失（分类+回归）为唯一优化目标，未将“漏检/误检的工业成本差异”纳入训练，导致断经、断氨纶等关键缺陷的召回难以针对性提升。其二，Transformer 检测器在织物场景应用不足，RT-DETR 的端到端优势未被系统挖掘。其三，对“视觉相似类混淆”这一本质问题缺乏系统诊断——本项目逐类分析发现检测精度呈明显长尾分布：毛粒、纬缩、稀密档、断经等类 mAP50 低于 0.45，其中断经（严重缺陷）仅 0.446，远低于 0.592 的平均水平，而同为严重缺陷的破洞（0.813）、断氨纶（0.694）表现良好，说明瓶颈并非类别稀有度，而是特定类间的视觉混淆。其四，检测到告警之间缺少闭环，关键缺陷易被静默降级。"
    ),
    IMG("fig2_per_class_map.png", 12),
    CAP("图2  逐类 mAP50 长尾分布（断经 0.446 为关键类瓶颈）"),
    B("（三）选题理论价值"),
    N(
        "针对上述不足，本项目提出“七维混合奖惩”机制，将定位（D01）、分类（D02）、校准（D03）、漏检（D04）、误检（D05）及关键类敏感度（D06 断经/断氨纶、D07 星跳/死皱）七个维度构造成对 logits 与边界框可微的奖励信号，并进一步以 GRPO 组内相对优势与 KL 锚定探索“奖励引导的检测器训练”，为“以工业代价为目标的端到端检测训练”提供新框架。实验上，将漏检纳入奖励后召回率提升 +0.030（0.523→0.553），但精确率同步下降、mAP50 持平，表明仅靠训练端干预难以突破瓶颈。进一步的混淆诊断揭示问题本质：断经↔磨痕、纬缩↔粗纬的混淆呈双向、裁剪图视觉几乎相同，属标签噪声特征而非模型可训练差距；合并视觉同源类后召回率显著回升——纬缩+粗纬 0.40→0.88、断经+磨痕 0.52→0.79。这一“标签噪声天花板”的发现及其诊断方法（混淆矩阵+双向性判定+裁剪图复核），对同类细粒度长尾识别任务具有可迁移的方法论价值。"
    ),
    IMG("fig3_reward_ab.png"),
    CAP("图3  奖励注入 A/B：召回 +0.030 但精确率下降、mAP 持平"),
    IMG("montage_broken_warp_surface_mark.png", 10),
    CAP("图4a  断经↔磨痕 裁剪图（视觉几乎同源）"),
    IMG("montage_weft_shrink_coarse_weft.png", 10),
    CAP("图4b  纬缩↔粗纬 裁剪图（视觉几乎同源）"),
    IMG("fig4_merge_recovery.png"),
    CAP("图5  混淆类合并后召回恢复（纬缩+粗纬 0.40→0.88）"),
    B("（四）实践应用意义与成果预期"),
    N(
        "系统基于 RT-DETR-L 构建 20 类 Tianchi 检测器，支持 4 种推理后端（Dummy/ONNX/PyTorch/RT-DETR）与 ONNX 边缘部署，以 WebSocket 实时告警结合中文看板实现“检测—告警—停机”闭环。针对易混淆关键类，系统在告警层实施 effective_alert_severity() 升级策略，使视觉上疑似断经的磨痕不再被降级为轻微告警，切实保障产线安全。全链条从问题诊断到成果落地逻辑自洽，兼顾学术创新与工程应用双重价值。"
    ),
]

SEC2 = [
    B("2.研究内容与目标（2000字）"),
    B("一、拟解决的关键问题"),
    B("问题一：如何将工业代价差异纳入检测器的训练目标"),
    N(
        "传统检测器以分类与回归损失为唯一优化目标，隐含假设“所有错误同等重要”。但在纺织质检中，漏检一个断经（严重缺陷，导致整匹布报废）与漏检一个毛粒（轻微缺陷）的工业代价相差悬殊。如何在训练目标中显式编码这种代价差异，使模型优先学会识别高代价缺陷，是第一个关键问题。"
    ),
    B("问题二：如何在不牺牲总体精度的前提下提升关键缺陷召回"),
    N(
        "直接对关键类加权或注入奖励，往往以精确率下降、总体 mAP 持平为代价——本项目实测将漏检注入奖励后，召回率 +0.030 但精确率 −0.022、mAP50 基本持平。如何在“召回—精度”权衡中取得更优解，而非简单的指标搬移，是第二个关键问题。"
    ),
    B("问题三：如何区分“模型可训练差距”与“标签噪声天花板”"),
    N(
        "本项目前期发现，断经召回约 0.45 的瓶颈并非类别稀有度（同为稀有类的断氨纶表现良好），而是断经↔磨痕、纬缩↔粗纬等视觉同源类的双向混淆，属标签噪声特征。如何系统化诊断并界定这一“数据/标注层面的天花板”，避免在错误方向上投入无效训练干预，是第三个关键问题。"
    ),
    B("问题四：如何在“检测—告警”闭环中提供安全兜底"),
    N(
        "即使模型无法完全区分视觉同源类，产线也不能因此停机错误或漏停。如何在告警层对易混淆类进行保守升级，避免严重缺陷被静默降级为轻微告警，是第四个关键问题。"
    ),
    B("二、主要内容"),
    [
        ("（一）20 类 Tianchi 织物瑕疵检测器。", True),
        (
            "基于 RT-DETR-L 构建端到端检测模型，覆盖破洞、污渍、断经、断氨纶等 20 类瑕疵；输入 1280 拉伸方图、一对一匈牙利匹配、免 NMS，输出归一化的 [cx,cy,w,h,score,class]；支持 PyTorch 与 ONNX 双推理后端，便于边缘部署。",
            False,
        ),
    ],
    [
        ("（二）七维混合奖惩机制。", True),
        (
            "将定位、分类、校准、漏检、误检及关键类敏感度七个维度构造为对 logits 与边界框可微的奖励信号，形成 L_total=L_det+β·L_scalar+γ·L_consistency，实现“工业代价导向”的训练目标。",
            False,
        ),
    ],
    [
        ("（三）奖励注入与 GRPO 训练路线。", True),
        (
            "两条互补路线：Layer-1 将可微奖励直接注入 ultralytics RT-DETR 训练循环，实现零侵入的监督增强；Layer-3 以查询噪声 REINFORCE + 组内相对优势 + KL 锚定，探索奖励引导的强化学习训练，并解决 RT-DETR 内容查询被 detach 导致均值梯度无法回传的问题。",
            False,
        ),
    ],
    [
        ("（四）混淆诊断与标签噪声分析。", True),
        (
            "通过逐类 mAP 画像、混淆矩阵、双向性判定与裁剪图人工复核，定位视觉同源类，量化“合并后召回恢复”收益，界定标签噪声天花板，输出合并/重标注建议。",
            False,
        ),
    ],
    [
        ("（五）告警兜底与系统集成。", True),
        (
            "设计 effective_alert_severity() 升级策略，构建 FastAPI 后端 + WebSocket 实时告警 + 中文看板 + 产线模拟器的完整系统，支持 ONNX 边缘部署。",
            False,
        ),
    ],
    B("三、核心目标"),
    [
        ("总目标：", True),
        (
            "构建一套基于 RT-DETR 与七维混合奖惩、面向 20 类织物瑕疵的端到端智能检测系统，在保证总体检测精度的同时，显著提升关键缺陷的召回与告警可靠性。",
            False,
        ),
    ],
    [
        ("具体目标（可量化）：", True),
        (
            "（1）检测精度：20 类 mAP50≥0.59，并给出逐类精度画像；（2）奖励机制：七维奖励对模型参数可微，奖励注入与 GRPO 训练给出严格 A/B 对照结论；（3）诊断方法：形成“混淆矩阵+双向性判定+裁剪复核”的标签噪声诊断流程，输出可执行的合并/重标注建议；（4）系统落地：4 种推理后端 + ONNX 边缘部署 + 实时告警 + 中文看板，通过全部功能与单元测试。",
            False,
        ),
    ],
    B("四、重要观点"),
    [
        ("观点一：检测器的优化目标应反映工业代价，而非仅最小化平均误差。", True),
        (
            "七维混合奖惩将“漏检/误检/关键类敏感度”等代价项落为可微损失分量，使训练目标与生产目标对齐，是对“平均精度最大化”范式的重要补充。",
            False,
        ),
    ],
    [
        (
            "观点二：关键缺陷的召回瓶颈可能是“标签噪声天花板”，而非模型可训练差距。",
            True,
        ),
        (
            "视觉同源、双向混淆的类，任何训练干预（奖励注入、GRPO、置信度校准、过采样）都无法实质提升，本项目多项 A/B 实验均证实；应转向数据层（合并/重标注）或告警层（保守升级）解决。",
            False,
        ),
    ],
    [
        ("观点三：“检测—告警”必须闭环设计安全兜底。", True),
        (
            "在模型无法完全区分易混淆类的前提下，告警层应“宁可信其有”——将疑似高影响类的低影响类告警升级为疑似严重缺陷，从系统层面杜绝关键缺陷被静默降级，这是工业质检系统可靠性的底线要求。",
            False,
        ),
    ],
]

SEC3 = [
    B("3.研究思路与方法（1000字）"),
    B("一、研究思路"),
    N(
        "本项目遵循“构建基线→诊断瓶颈→定向干预→闭环落地”的递进式研究思路，以实验数据驱动研究走向，避免盲目调参。（一）构建基线：以 RT-DETR-L 训练 20 类检测器，建立 mAP50=0.594 的精度基准与逐类性能画像。（二）诊断瓶颈：通过逐类 mAP、混淆矩阵与裁剪图复核，定位关键缺陷召回不足的根因——是类别稀有度、训练不足，还是视觉同源/标签噪声。（三）定向干预：分层实施并逐一 A/B 验证——Layer-1 奖励注入、Layer-3 GRPO、过采样与置信度校准。（四）闭环落地：将安全兜底落到告警层，并通过系统集成与测试保障可行性。"
    ),
    B("二、研究框架"),
    N(
        "整体框架自上而下分六层：数据层（Tianchi 20 类数据集，1280 拉伸方图预处理）→ 模型层（RT-DETR-L 检测器，PyTorch/ONNX 双后端）→ 奖励层（七维混合奖惩 D01~D07，可微）→ 诊断层（逐类 mAP + 混淆矩阵 + 双向性判定 + 裁剪复核）→ 告警层（effective_alert_severity() 易混淆类保守升级）→ 应用层（FastAPI + WebSocket + 中文看板 + 边缘部署）。"
    ),
    IMG("fig5_framework.png"),
    CAP("图6  系统研究框架（六层）"),
    B("三、具体方法"),
    [
        ("（一）检测器训练。", True),
        (
            "采用 ultralytics RT-DETR-L，输入 1280 拉伸方图（BGR→RGB、/255，无 ImageNet 均值方差），一对一匈牙利匹配、免 NMS。",
            False,
        ),
    ],
    [
        ("（二）七维奖励可微化。", True),
        (
            "将奖励中的硬决策（argmax、集合成员、硬 IoU 阈值）替换为 softmax/sigmoid 软指示，使奖励对模型参数可微、能直接反传，而非仅调节维度权重。",
            False,
        ),
    ],
    [
        ("（三）奖励注入（Layer-1）。", True),
        (
            "RewardRTDETRDetectionLoss 在损失中追加 loss_reward=−β·mean(R)，与检测损失线性叠加；include-misses 参数将漏检关键类的最佳备用查询纳入惩罚。",
            False,
        ),
    ],
    [
        ("（四）GRPO（Layer-3）。", True),
        (
            "向内容查询注入高斯噪声 z=mean+std·eps，以似然比得分估计策略梯度，避免梯度恒为零的陷阱；组内相对优势采用 eps 下限 + clip 防优势爆炸，KL 锚定冻结参考策略防策略坍塌；beta_reward 通道回传可微奖励以驱动被 detach 的查询均值。",
            False,
        ),
    ],
    [
        ("（五）标签噪声诊断。", True),
        (
            "以混淆矩阵判断混淆是否双向（双向即标签噪声特征，单向为系统偏差）；合并视觉同源类重新评测召回，量化“合并收益”。",
            False,
        ),
    ],
    [
        ("（六）告警兜底。", True),
        (
            "CONFUSABLE_HIGHER_IMPACT 映射 + effective_alert_severity()，以 max(原级别, 易混淆高影响类级别) 作为告警级别，规范级别不变、升级级别独立落库可审计。",
            False,
        ),
    ],
    B("四、研究逻辑与方法应用流程"),
    N(
        "问题提出（关键类召回不足）→ 基线画像（逐类 mAP 长尾）→ 根因假设（稀有度/训练不足/标签噪声）→ 分层干预与 A/B 验证（奖励/GRPO/校准/过采样）→ 结论判定（标签噪声天花板）→ 转向数据层（合并/重标注）+ 告警层（保守升级）→ 系统集成与测试落地。关键在每一步都以可量化证据（mAP/召回/混淆矩阵）作为下一步决策依据，使研究走向可追溯、可复现。"
    ),
]

SEC4 = [
    B("4.研究基础与保障（1000字）"),
    B("一、前期研究积累"),
    [
        ("（一）检测器与工具链。", True),
        (
            "已完成 RT-DETR-L 在 20 类 Tianchi 数据上的训练（mAP50 0.594、mAP50-95 0.334），并具备 ONNX 导出（1280 输入、输出 [cx,cy,w,h,score,class]）、推理预测、训练曲线与逐类 mAP 可视化等完整脚本。",
            False,
        ),
    ],
    [
        ("（二）七维奖励与训练路线。", True),
        (
            "七维混合奖惩已完成可微化改造，Layer-1 奖励注入与 Layer-3 GRPO（查询噪声 REINFORCE + 组内优势 + KL 锚定）均已实现并通过单元测试。",
            False,
        ),
    ],
    [
        ("（三）混淆诊断结论。", True),
        (
            "已完成逐类分析、混淆矩阵与裁剪图人工复核，形成“断经召回约 0.45 为标签噪声天花板”的确定性结论，并据此实现告警层易混淆类升级兜底。",
            False,
        ),
    ],
    [
        ("（四）系统框架。", True),
        (
            "FastAPI 后端、SQLAlchemy 2.0 异步、WebSocket 实时告警、中文监控看板与产线模拟器均已就绪，累计 205 项测试通过，工程规范已确立。",
            False,
        ),
    ],
    B("二、人员与平台条件保障"),
    [
        ("（一）算力平台。", True),
        (
            "训练基于 NVIDIA RTX 4090 D（24GB 显存），满足 RT-DETR-L 1280 分辨率训练与 GRPO 显存需求；推理支持 CPU/GPU，可经 ONNX 边缘部署。",
            False,
        ),
    ],
    [
        ("（二）软件环境。", True),
        (
            "Python 3.13 + PyTorch 2.8（CUDA）+ ultralytics 8.4 + ONNX Runtime + FastAPI，环境稳定、版本明确、可复现。",
            False,
        ),
    ],
    [
        ("（三）数据保障。", True),
        (
            "采用公开天池“布匹疵点智能识别”数据集，具备 20 类标准化标注；标签噪声问题已诊断清楚，可在数据层通过合并/重标注解决。",
            False,
        ),
    ],
    [
        ("（四）工程规范。", True),
        (
            "项目遵循 Router→Service→Model 分层架构、Pydantic 双层校验、结构化 JSON 日志、模块级单例等规范，测试覆盖充分，保障可维护性。",
            False,
        ),
    ],
    B("三、实施可行性论证"),
    [
        ("（一）技术可行性。", True),
        (
            "RT-DETR 端到端检测 + 七维可微奖励 + GRPO 训练路线均有成熟实现与验证，无原理性障碍；前期实验已给出明确技术结论与改进方向。",
            False,
        ),
    ],
    [
        ("（二）数据可行性。", True),
        (
            "公开数据集可获取、标注规范；标签噪声天花板已通过混淆矩阵双向性与裁剪图复核定位，合并/重标注路径明确、收益可量化。",
            False,
        ),
    ],
    [
        ("（三）资源可行性。", True),
        (
            "单卡 4090 D 即可完成训练，GRPO 采用冻结主干轻量策略、显存可控；推理端支持 ONNX 边缘部署，落地成本低。",
            False,
        ),
    ],
    [
        ("（四）时间与风险可控。", True),
        (
            "研究路线以 A/B 对照实验驱动，每步有量化判断标准；潜在风险（如 GRPO 收敛不稳定）已通过优势 eps 下限 + clip、KL 锚定等机制缓解，并保留纯监督与奖励注入两条稳健路线兜底。",
            False,
        ),
    ],
    N(
        "综上，本项目在前期积累、算力、数据、工程规范与风险控制上均已具备实施条件，技术路线可行、目标可达、风险可控。"
    ),
]

# ---------------------------------------------------------------- fill
doc = Document(SRC)

# Table 0 项目名称
doc.tables[0].rows[0].cells[1].text = PROJECT_NAME

# Table 1 项目名称 (merged name cell) + 起止年月
t1 = doc.tables[1]
set_text(t1.rows[0].cells[2], PROJECT_NAME, bold=True)
set_text(t1.rows[1].cells[2], "2026年10月 至 2027年9月")

# Table 4 项目设计论证 (4 sections)
t4 = doc.tables[4]
write_cell(t4.rows[0].cells[0], SEC1)
write_cell(t4.rows[1].cells[0], SEC2)
write_cell(t4.rows[2].cells[0], SEC3)
write_cell(t4.rows[3].cells[0], SEC4)

# Table 5 预期阶段计划与预期成果
t5 = doc.tables[5]
stages = [
    (
        "1",
        "2026年10月—2026年11月",
        "国内外研究现状综述与技术路线确定",
        "调研报告",
        "项目组",
    ),
    (
        "2",
        "2026年11月—2026年12月",
        "RT-DETR-L 20类检测器训练与ONNX导出（mAP50≥0.59）",
        "模型权重、工具链",
        "项目组",
    ),
    (
        "3",
        "2026年12月—2027年03月",
        "七维可微奖励、奖励注入与GRPO训练及A/B结论",
        "源代码、实验报告",
        "项目组",
    ),
    (
        "4",
        "2027年03月—2027年05月",
        "逐类mAP画像、混淆诊断与标签噪声分析",
        "分析报告",
        "项目组",
    ),
    (
        "5",
        "2027年05月—2027年07月",
        "告警升级策略、系统集成与边缘部署",
        "可运行系统、测试报告",
        "项目组",
    ),
    ("6", "2027年07月—2027年09月", "结题报告与成果凝练", "结题报告", "项目组"),
]
for i, (no, period, name, form, who) in enumerate(stages):
    row = t5.rows[1 + i]
    set_text(row.cells[1], no)
    set_text(row.cells[2], period)
    set_text(row.cells[3], name)
    set_text(row.cells[4], form)
    set_text(row.cells[5], who)

final = [
    ("1", "织物瑕疵智能检测系统（原型）", "软件系统"),
    ("2", "学术论文（或软件著作权）", "论文/软著"),
    ("3", "结题研究报告", "报告"),
]
for i, (no, name, form) in enumerate(final):
    row = t5.rows[10 + i]
    set_text(row.cells[1], no)
    set_text(row.cells[2], name)
    set_text(row.cells[4], form)

BENEFITS_AND_VISION = [
    N(
        "学术上，提出七维混合奖惩机制与标签噪声诊断方法，为细粒度长尾缺陷检测提供新范式；"
        "应用上，系统可迁移至纺织质检产线，提升关键缺陷召回与告警可靠性、减少漏检停机损失；"
        "育人上，培养学生在深度学习、强化学习与系统工程方面的综合实践能力。"
    ),
    B("应用愿景（系统建成后的五级硬件落地规划）"),
    [
        ("第一级·实时标记与报警：", True),
        (
            "驱动喷码/喷雾装置（PLC 控制 + 编码器同步追踪布料位置）在瑕疵处精准打标并声光报警；"
            "对易混淆关键类采用 UV 荧光标记，支持下游二次识别。",
            False,
        ),
    ],
    [
        ("第二级·产线控制与联动：", True),
        (
            "对破洞、断经等关键性严重瑕疵，经 PLC 自动停机/减速织机或验布机，"
            "或联动执行机构转运返修、分拣剔除，实现闭环停机。",
            False,
        ),
    ],
    [
        ("第三级·质量分级与分拣：", True),
        (
            "综合瑕疵类型、数量、面积与分布自动评定 A/B/C 质量等级并生成瑕疵热力图，"
            "由智能叉车/立库分级分拣入库。",
            False,
        ),
    ],
    [
        ("第四级·数据追溯与系统集成：", True),
        (
            "生成含坐标、分类、面积、色差的结构化瑕疵数据并导出 CSV/PDF 报告，"
            "与 MES/ERP 打通实现全链路追溯；裁剪环节投影瑕疵分布图以避开瑕疵排版，最大化面料利用率。",
            False,
        ),
    ],
    [
        ("第五级·人工复核与干预：", True),
        (
            "保留人机协同通道，操作员经人机界面确认/修正检测结果，"
            "形成“检测—复核—重标注”数据飞轮，持续提升模型精度。",
            False,
        ),
    ],
    N(
        "五级愿景与项目告警升级语义（确诊关键缺陷 vs 疑似关键类）深度融合，"
        "确保硬件投入与当前检测能力精准匹配、逐步落地。"
    ),
]

write_cell(t5.rows[13].cells[1], BENEFITS_AND_VISION)

doc.save(DST)
print("saved:", DST)
