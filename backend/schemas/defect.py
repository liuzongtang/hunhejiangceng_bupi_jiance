"""
Defect classification enums and code mappings.

Derived from Section 4.1 and 4.2 of the project document.
"""

from enum import Enum


class DefectType(str, Enum):
    """Defect category codes."""

    BROKEN_YARN = "broken_yarn"  # BY-01
    MISSING_STITCH = "missing_stitch"  # MS-01
    SKIP_STITCH = "skip_stitch"  # MS-02
    HOLE = "hole"  # HO-01
    STAIN = "stain"  # ST-01
    COLOR_DIFF = "color_diff"  # CD-01
    THICK_YARN = "thick_yarn"  # TH-01
    THIN_YARN = "thin_yarn"  # TH-02
    CREASE = "crease"  # WR-01
    OTHER = "other"  # OT-01


class Severity(str, Enum):
    """Defect severity levels."""

    CRITICAL = "critical"
    MAJOR = "major"
    MEDIUM = "medium"
    MINOR = "minor"
    INFO = "info"


class DefectCode(str, Enum):
    """Standard defect codes per Section 4.1."""

    BY_01 = "BY-01"  # 断纱
    MS_01 = "MS-01"  # 漏针
    MS_02 = "MS-02"  # 跳针
    HO_01 = "HO-01"  # 破洞
    ST_01 = "ST-01"  # 污渍
    CD_01 = "CD-01"  # 色差
    TH_01 = "TH-01"  # 粗节
    TH_02 = "TH-02"  # 细节
    WR_01 = "WR-01"  # 折痕
    OT_01 = "OT-01"  # 其他


# Mapping from DefectType to DefectCode
DEFECT_CODES = {
    DefectType.BROKEN_YARN: DefectCode.BY_01,
    DefectType.MISSING_STITCH: DefectCode.MS_01,
    DefectType.SKIP_STITCH: DefectCode.MS_02,
    DefectType.HOLE: DefectCode.HO_01,
    DefectType.STAIN: DefectCode.ST_01,
    DefectType.COLOR_DIFF: DefectCode.CD_01,
    DefectType.THICK_YARN: DefectCode.TH_01,
    DefectType.THIN_YARN: DefectCode.TH_02,
    DefectType.CREASE: DefectCode.WR_01,
    DefectType.OTHER: DefectCode.OT_01,
}

# Critical defects that trigger immediate stop
CRITICAL_DEFECTS = {
    DefectType.BROKEN_YARN,
    DefectType.MISSING_STITCH,
    DefectType.HOLE,
}

# Defect descriptions (Chinese)
DEFECT_DESCRIPTIONS = {
    DefectCode.BY_01: "断纱 — 经纱或纬纱断裂",
    DefectCode.MS_01: "漏针 — 针织组织缺失",
    DefectCode.MS_02: "跳针 — 线圈错位",
    DefectCode.HO_01: "破洞 — 织物破损穿孔",
    DefectCode.ST_01: "污渍 — 油污、色渍",
    DefectCode.CD_01: "色差 — 颜色不一致",
    DefectCode.TH_01: "粗节 — 纱线局部粗大",
    DefectCode.TH_02: "细节 — 纱线局部细弱",
    DefectCode.WR_01: "折痕 — 折叠产生的痕迹",
    DefectCode.OT_01: "其他 — 未分类缺陷",
}
