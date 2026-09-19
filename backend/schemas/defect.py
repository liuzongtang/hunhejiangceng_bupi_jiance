"""
Defect classification enums, codes, severity and descriptions.

This is the single source of truth for the 20-class Tianchi
"smartdiagnosisofclothflaw" taxonomy used by the trained RT-DETR model.

The ``DefectType`` member order matches the model's class-id order (0..19), so
``list(DefectType)`` minus the defensive ``OTHER`` fallback is exactly the model
head's class name list (see ``TIANCHI_CLASS_NAMES``).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Set


class DefectType(str, Enum):
    """Tianchi cloth-defect categories (model class id 0..19) plus OTHER."""

    HOLE = "hole"  # HO-01 破洞
    STAIN = "stain"  # ST-01 污渍
    THREE_SILK = "three_silk"  # SL-01 三丝
    KNOT = "knot"  # KN-01 结头
    FLOWER_BOARD = "flower_board"  # FL-01 花板跳
    HUNDRED_FEET = "hundred_feet"  # HF-01 百脚
    HAIR_PARTICLE = "hair_particle"  # HP-01 毛粒
    COARSE_WARP = "coarse_warp"  # CW-01 粗经
    LOOSE_WARP = "loose_warp"  # LW-01 松经
    BROKEN_WARP = "broken_warp"  # BW-01 断经
    HANGING_WARP = "hanging_warp"  # HW-01 吊经
    COARSE_WEFT = "coarse_weft"  # CF-01 粗纬
    WEFT_SHRINK = "weft_shrink"  # WS-01 纬缩
    SIZE_STAIN = "size_stain"  # ST-02 浆斑
    WARPING_KNOT = "warping_knot"  # KN-02 整经结
    STAR_SKIP = "star_skip"  # SK-01 星跳
    BROKEN_SPANDEX = "broken_spandex"  # BS-01 断氨纶
    DENSE_SECTION = "dense_section"  # DS-01 稀密档
    SURFACE_MARK = "surface_mark"  # SM-01 磨痕
    WEAVE_DEFECT = "weave_defect"  # WD-01 死皱
    OTHER = "other"  # OT-01 其他（防御性兜底）


class Severity(str, Enum):
    """Defect severity levels (production impact)."""

    CRITICAL = "critical"
    MAJOR = "major"
    MEDIUM = "medium"
    MINOR = "minor"
    INFO = "info"


class DefectCode(str, Enum):
    """Standard defect codes."""

    HO_01 = "HO-01"  # 破洞
    ST_01 = "ST-01"  # 污渍
    SL_01 = "SL-01"  # 三丝
    KN_01 = "KN-01"  # 结头
    FL_01 = "FL-01"  # 花板跳
    HF_01 = "HF-01"  # 百脚
    HP_01 = "HP-01"  # 毛粒
    CW_01 = "CW-01"  # 粗经
    LW_01 = "LW-01"  # 松经
    BW_01 = "BW-01"  # 断经
    HW_01 = "HW-01"  # 吊经
    CF_01 = "CF-01"  # 粗纬
    WS_01 = "WS-01"  # 纬缩
    ST_02 = "ST-02"  # 浆斑
    KN_02 = "KN-02"  # 整经结
    SK_01 = "SK-01"  # 星跳
    BS_01 = "BS-01"  # 断氨纶
    DS_01 = "DS-01"  # 稀密档
    SM_01 = "SM-01"  # 磨痕
    WD_01 = "WD-01"  # 死皱
    OT_01 = "OT-01"  # 其他


# The 20 model classes (id 0..19). Excludes the OTHER fallback so the model
# head width stays 20.
TIANCHI_CLASS_NAMES: List[str] = [
    dt.value for dt in DefectType if dt is not DefectType.OTHER
]

# Primary Chinese label per class (parallel to TIANCHI_CLASS_NAMES).
TIANCHI_CLASS_NAMES_ZH: List[str] = [
    "破洞", "污渍", "三丝", "结头", "花板跳",
    "百脚", "毛粒", "粗经", "松经", "断经",
    "吊经", "粗纬", "纬缩", "浆斑", "整经结",
    "星跳", "断氨纶", "稀密档", "磨痕", "死皱",
]

# DefectType -> DefectCode
DEFECT_CODES: Dict[DefectType, DefectCode] = {
    DefectType.HOLE: DefectCode.HO_01,
    DefectType.STAIN: DefectCode.ST_01,
    DefectType.THREE_SILK: DefectCode.SL_01,
    DefectType.KNOT: DefectCode.KN_01,
    DefectType.FLOWER_BOARD: DefectCode.FL_01,
    DefectType.HUNDRED_FEET: DefectCode.HF_01,
    DefectType.HAIR_PARTICLE: DefectCode.HP_01,
    DefectType.COARSE_WARP: DefectCode.CW_01,
    DefectType.LOOSE_WARP: DefectCode.LW_01,
    DefectType.BROKEN_WARP: DefectCode.BW_01,
    DefectType.HANGING_WARP: DefectCode.HW_01,
    DefectType.COARSE_WEFT: DefectCode.CF_01,
    DefectType.WEFT_SHRINK: DefectCode.WS_01,
    DefectType.SIZE_STAIN: DefectCode.ST_02,
    DefectType.WARPING_KNOT: DefectCode.KN_02,
    DefectType.STAR_SKIP: DefectCode.SK_01,
    DefectType.BROKEN_SPANDEX: DefectCode.BS_01,
    DefectType.DENSE_SECTION: DefectCode.DS_01,
    DefectType.SURFACE_MARK: DefectCode.SM_01,
    DefectType.WEAVE_DEFECT: DefectCode.WD_01,
    DefectType.OTHER: DefectCode.OT_01,
}

# DefectType -> Severity (production impact).
DEFECT_SEVERITY: Dict[DefectType, Severity] = {
    DefectType.HOLE: Severity.CRITICAL,
    DefectType.STAIN: Severity.MEDIUM,
    DefectType.THREE_SILK: Severity.MINOR,
    DefectType.KNOT: Severity.MINOR,
    DefectType.FLOWER_BOARD: Severity.MAJOR,
    DefectType.HUNDRED_FEET: Severity.MAJOR,
    DefectType.HAIR_PARTICLE: Severity.MINOR,
    DefectType.COARSE_WARP: Severity.MINOR,
    DefectType.LOOSE_WARP: Severity.MINOR,
    DefectType.BROKEN_WARP: Severity.CRITICAL,
    DefectType.HANGING_WARP: Severity.MAJOR,
    DefectType.COARSE_WEFT: Severity.MINOR,
    DefectType.WEFT_SHRINK: Severity.MINOR,
    DefectType.SIZE_STAIN: Severity.MEDIUM,
    DefectType.WARPING_KNOT: Severity.MINOR,
    DefectType.STAR_SKIP: Severity.MAJOR,
    DefectType.BROKEN_SPANDEX: Severity.CRITICAL,
    DefectType.DENSE_SECTION: Severity.MAJOR,
    DefectType.SURFACE_MARK: Severity.MINOR,
    DefectType.WEAVE_DEFECT: Severity.MAJOR,
    DefectType.OTHER: Severity.INFO,
}

# Defects that must halt the production line.
CRITICAL_DEFECTS: Set[DefectType] = {
    DefectType.HOLE,
    DefectType.BROKEN_WARP,
    DefectType.BROKEN_SPANDEX,
}

# Defect descriptions (Chinese).
DEFECT_DESCRIPTIONS: Dict[DefectCode, str] = {
    DefectCode.HO_01: "破洞 — 织物破损穿孔",
    DefectCode.ST_01: "污渍 — 油污、色渍",
    DefectCode.SL_01: "三丝 — 毛发、羽毛等异纤",
    DefectCode.KN_01: "结头 — 纱线接头",
    DefectCode.FL_01: "花板跳 — 提花错花、跳花",
    DefectCode.HF_01: "百脚 — 缺纬造成的纬向疵点",
    DefectCode.HP_01: "毛粒 — 纱线表面毛粒",
    DefectCode.CW_01: "粗经 — 经纱局部粗大",
    DefectCode.LW_01: "松经 — 经纱松弛",
    DefectCode.BW_01: "断经 — 经纱断裂",
    DefectCode.HW_01: "吊经 — 经纱张力过大吊起",
    DefectCode.CF_01: "粗纬 — 纬纱局部粗大",
    DefectCode.WS_01: "纬缩 — 纬纱收缩",
    DefectCode.ST_02: "浆斑 — 上浆残留斑渍",
    DefectCode.KN_02: "整经结 — 整经接头",
    DefectCode.SK_01: "星跳 — 跳纱星点",
    DefectCode.BS_01: "断氨纶 — 氨纶断裂",
    DefectCode.DS_01: "稀密档 — 经纬密度不匀",
    DefectCode.SM_01: "磨痕 — 表面磨损痕迹",
    DefectCode.WD_01: "死皱 — 不可回复皱痕",
    DefectCode.OT_01: "其他 — 未分类缺陷",
}
