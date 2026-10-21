"""
把附录 Markdown 转为 doc-typeset 规范的 HTML 片段。

与 doc-typeset 的约束对齐：
- 正文首字体 SimSun/宋体（由 body 继承，此处只关心块级元素）
- 章节标题用 h1~h3
- 表格用 <table>，禁用 grid/dl
- 代码块用 <pre><code>，保留换行与缩进
- 禁止 <span> 模拟 badge、禁止装饰性 hr
"""

from html import escape
import re
import sys


def inline(text: str) -> str:
    """行内 Markdown 转 HTML：行内代码 → 加粗 → 斜体。顺序不可颠倒。"""
    # 先抽出行内代码，避免其内容被后续规则改写
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00CODE{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = escape(text)
    # 加粗
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    # 斜体（避免与加粗混淆：__x__ 形式不用，此处只处理单星号包裹的非粗体）
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    # 还原行内代码
    for i, c in enumerate(codes):
        text = text.replace(
            f"\x00CODE{i}\x00",
            f'<code style="font-family: var(--ff-mono); font-size: var(--fs-small); background: var(--color-code-bg); padding: var(--pad-inline-code);">{escape(c)}</code>',
        )
    return text


def table_to_html(rows: list[list[str]], align: list[str]) -> str:
    out = [
        '<table style="border-collapse: collapse; width: 100%; margin-bottom: var(--spacing-paragraph);">'
    ]
    header, body = rows[0], rows[1:]
    out.append("<thead><tr>")
    for cell in header:
        out.append(
            f'<th style="padding: var(--pad-cell); border: 1px solid var(--color-border); '
            f"text-align: left; text-indent: 0; font-weight: var(--fw-bold); "
            f'background: var(--color-highlight); font-family: var(--ff-heading);">{inline(cell)}</th>'
        )
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>")
        for cell in r:
            out.append(
                f'<td style="padding: var(--pad-cell); border: 1px solid var(--color-border); '
                f'text-align: left; text-indent: 0;">{inline(cell)}</td>'
            )
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 代码块
        if line.strip().startswith("```"):
            lang = line.strip()[3:].strip()
            i += 1
            buf: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            # 关键：把换行转成显式 <br/>，否则转换引擎会把 \n 折叠成空格
            code = "<br/>".join(escape(line) for line in buf)
            out.append(
                '<pre style="font-family: var(--ff-mono); font-size: var(--fs-code); '
                "line-height: 1.55; background: var(--color-code-bg); "
                "border: 1px solid var(--color-border); "
                "border-left: 3px solid var(--color-primary); "
                "padding: var(--pad-code); margin-top: var(--spacing-paragraph); "
                "margin-bottom: var(--spacing-paragraph); white-space: pre-wrap; "
                'word-break: break-all; text-indent: 0;"><code style="background: none; '
                "padding: 0; font-size: var(--fs-code); font-family: var(--ff-mono); "
                f'line-height: 1.55;">{code}</code></pre>'
            )
            continue

        # 表格
        if (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[i + 1])
        ):
            rows: list[list[str]] = []

            def cells(r: str) -> list[str]:
                r = r.strip()
                if r.startswith("|"):
                    r = r[1:]
                if r.endswith("|"):
                    r = r[:-1]
                return [c.strip() for c in r.split("|")]

            rows.append(cells(lines[i]))
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            out.append(table_to_html(rows, []))
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            text = inline(m.group(2))
            if level == 1:
                out.append(
                    f'<h1 style="font-family: var(--ff-heading); font-size: var(--fs-h1); '
                    f"line-height: var(--lh-heading); font-weight: var(--fw-bold); "
                    f"text-indent: 0; margin-top: var(--spacing-section); "
                    f'margin-bottom: var(--spacing-paragraph); color: var(--color-heading);">{text}</h1>'
                )
            elif level == 2:
                out.append(
                    f'<h2 style="font-family: var(--ff-heading); font-size: var(--fs-h2); '
                    f"line-height: var(--lh-heading); font-weight: var(--fw-bold); "
                    f"text-indent: 0; margin-top: var(--spacing-section); "
                    f'margin-bottom: var(--spacing-paragraph); color: var(--color-primary);">{text}</h2>'
                )
            else:
                out.append(
                    f'<h3 style="font-family: var(--ff-heading); font-size: var(--fs-h3); '
                    f"line-height: var(--lh-heading); font-weight: var(--fw-bold); "
                    f"text-indent: 0; margin-top: var(--spacing-sub); "
                    f'margin-bottom: var(--spacing-tight); color: var(--color-subheading);">{text}</h3>'
                )
            i += 1
            continue

        # 无序列表
        if re.match(r"^\s*[-*]\s+", line):
            items: list[str] = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(inline(re.sub(r"^\s*[-*]\s+", "", lines[i])))
                i += 1
            out.append(
                '<ul style="padding-left: var(--spacing-list); margin-bottom: var(--spacing-paragraph);">'
                + "".join(
                    f'<li style="margin-bottom: var(--spacing-tight); text-indent: 0;">{it}</li>'
                    for it in items
                )
                + "</ul>"
            )
            continue

        # 有序列表
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(inline(re.sub(r"^\s*\d+\.\s+", "", lines[i])))
                i += 1
            out.append(
                '<ol style="padding-left: var(--spacing-list); margin-bottom: var(--spacing-paragraph);">'
                + "".join(
                    f'<li style="margin-bottom: var(--spacing-tight); text-indent: 0;">{it}</li>'
                    for it in items
                )
                + "</ol>"
            )
            continue

        # 引用块
        if line.strip().startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(inline(lines[i].strip().lstrip(">").strip()))
                i += 1
            inner = "".join(
                f'<p style="text-indent: 0; margin-bottom: var(--spacing-tight);">{b}</p>'
                for b in buf
            )
            out.append(
                '<blockquote style="border-left: 3px solid var(--color-primary); '
                "background: var(--color-callout-bg); padding: var(--pad-callout); "
                'margin-bottom: var(--spacing-paragraph); font-size: var(--fs-small);">'
                f"{inner}</blockquote>"
            )
            continue

        # 空行
        if not line.strip():
            i += 1
            continue

        # 普通段落（连续非空行合并）
        buf = []
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^\s*([-*]\s|\d+\.\s|#{1,4}\s|>|```|\|)", lines[i])
        ):
            buf.append(inline(lines[i].strip()))
            i += 1
        if buf:
            out.append(
                '<p style="margin-bottom: var(--spacing-paragraph); text-indent: 2em;">'
                + "<br/>".join(buf)
                + "</p>"
            )
        else:
            i += 1
    return "\n".join(out)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    md = open(src, encoding="utf-8").read()
    html = convert(md)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"转换完成: {dst}")
    print(f"  长度 {len(html)} 字符")
    print(
        f"  pre 块 {html.count('<pre')}  表格 {html.count('<table')}  h1 {html.count('<h1')}  h2 {html.count('<h2')}  h3 {html.count('<h3')}"
    )
