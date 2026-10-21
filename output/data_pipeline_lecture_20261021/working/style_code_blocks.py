"""给代码块表格批量添加底纹与边框。"""

import json
import subprocess
import sys

PY = r"C:\Users\master\anaconda3\python.exe"
EDSDK = r"D:\workbuddy\resources\app.asar.unpacked\resources\plugins\workbuddy-builtin\skills\workbuddy-local-office-edit\edsdk.py"
FID = r"C:/Users/master/Desktop/维度模型(1)/output/data_pipeline_lecture_20261021/stage3/逐行代码讲解讲义_含中文注释.docx"

BORDERS = {
    "top": {"style": "single", "size": 4, "color": "D1D5DB"},
    "bottom": {"style": "single", "size": 4, "color": "D1D5DB"},
    "left": {"style": "single", "size": 18, "color": "1D4ED8"},
    "right": {"style": "single", "size": 4, "color": "D1D5DB"},
    "insideH": {"style": "none", "size": 0, "color": "auto"},
    "insideV": {"style": "none", "size": 0, "color": "auto"},
}


def call(tool: str, payload: dict) -> dict:
    args = [
        PY,
        EDSDK,
        "call",
        tool,
        "--json",
        json.dumps(payload, ensure_ascii=False),
    ]
    r = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    out = (r.stdout or "").strip()
    # edsdk 可能先打印一行人类可读文本再打印 JSON
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"ok": False, "raw": out[:200], "stderr": (r.stderr or "")[:200]}


def main() -> int:
    res = call("doc_list_tables", {"file_id": FID})
    tables = res.get("tables") or res.get("items") or []
    if not tables and isinstance(res.get("result"), list):
        tables = res["result"]
    print(f"表格总数: {len(tables)}")

    # 代码块 = 1 行 1 列的表格
    code_tables = [
        t for t in tables if t.get("row_count") == 1 and t.get("col_count") == 1
    ]
    print(f"代码块表格(1x1): {len(code_tables)}")

    ok = 0
    fail = 0
    for i, t in enumerate(code_tables, 1):
        tid = t.get("table_id")
        if not tid:
            continue
        r = call(
            "doc_set_table_properties",
            {
                "file_id": FID,
                "table_id": tid,
                "borders": BORDERS,
                "cell_fills": [{"condition": "whole_table", "color": "F7F8FA"}],
                "cell_margin": {"top": 90, "bottom": 90, "left": 120, "right": 120},
            },
        )
        if r.get("ok") is not False and not r.get("error"):
            ok += 1
        else:
            fail += 1
            if fail <= 3:
                print(f"  失败 {tid}: {r.get('error') or r.get('raw', '')[:120]}")
        if i % 15 == 0:
            print(f"  进度 {i}/{len(code_tables)}")

    print(f"完成: 成功 {ok}  失败 {fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
