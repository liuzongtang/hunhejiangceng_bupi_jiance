"""
组装完整讲义 HTML：封面 + 目录 + 正文 + 附录三。
从 stage1/final_draft.md 与 working/appendix_source.md 重新生成，
保证所有代码块都经由 <pre> 渲染（底纹/边框/保留换行缩进）。
"""

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from md2html import convert

ROOT = Path(__file__).resolve().parent.parent
BODY_MD = ROOT / "stage1" / "final_draft.md"
APX_MD = ROOT / "working" / "appendix_source.md"
OUT = ROOT / "stage2" / "formatted-数据集收集清洗与模型输入专题讲义.html"

STYLE = """  :root {
    --page-content-width: 15.6cm;
    --fs-title: 26pt;
    --fs-subtitle: 13pt;
    --fs-h1: 18pt;
    --fs-h2: 15pt;
    --fs-h3: 13pt;
    --fs-body: 12pt;
    --fs-small: 10.5pt;
    --fs-code: 9.5pt;
    --ff-heading: "SimHei", "黑体", sans-serif;
    --ff-body: "SimSun", "宋体", serif;
    --ff-mono: "Consolas", "JetBrains Mono", monospace;
    --lh-body: 1.65;
    --lh-heading: 1.35;
    --fw-bold: 700;
    --color-primary: #1D4ED8;
    --color-heading: #0F2E6B;
    --color-subheading: #1E3A8A;
    --color-callout-bg: #EFF6FF;
    --color-text: #111827;
    --color-muted: #6B7280;
    --color-border: #D1D5DB;
    --color-bg: #FFFFFF;
    --color-highlight: #F3F4F6;
    --color-code-bg: #F7F8FA;
    --spacing-paragraph: 0.6em;
    --spacing-section: 1.8em;
    --spacing-sub: 1.2em;
    --spacing-list: 2.4em;
    --spacing-tight: 0.3em;
    --spacing-toc-indent: 1.6em;
    --pad-code: 9px 12px;
    --pad-cell: 5px 8px;
    --pad-callout: 7px 10px;
    --pad-inline-code: 1px 3px;
  }
  @page { @bottom-center { content: counter(page); } }
  @page cover { @bottom-center { content: none; } }
  section[role="cover"] { page: cover; }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--ff-body);
    font-size: var(--fs-body);
    line-height: var(--lh-body);
    color: var(--color-text);
    background: var(--color-bg);
    max-width: var(--page-content-width);
    margin-left: auto;
    margin-right: auto;
  }
  h1, h2, h3 {
    font-family: var(--ff-heading);
    line-height: var(--lh-heading);
    font-weight: var(--fw-bold);
    text-indent: 0;
  }
  p { margin-bottom: var(--spacing-paragraph); text-indent: 2em; }
  ul, ol { padding-left: var(--spacing-list); margin-bottom: var(--spacing-paragraph); }
  li { margin-bottom: var(--spacing-tight); text-indent: 0; }
  strong { font-weight: var(--fw-bold); }
  code {
    font-family: var(--ff-mono);
    font-size: var(--fs-small);
    background: var(--color-code-bg);
    padding: var(--pad-inline-code);
  }
  pre {
    font-family: var(--ff-mono);
    font-size: var(--fs-code);
    line-height: 1.55;
    background: var(--color-code-bg);
    border: 1px solid var(--color-border);
    border-left: 3px solid var(--color-primary);
    padding: var(--pad-code);
    margin-top: var(--spacing-paragraph);
    margin-bottom: var(--spacing-paragraph);
    white-space: pre-wrap;
    word-break: break-all;
    text-indent: 0;
  }
  pre code {
    background: none;
    padding: 0;
    font-size: var(--fs-code);
    font-family: var(--ff-mono);
    line-height: 1.55;
    white-space: pre-wrap;
  }
  table { border-collapse: collapse; width: 100%; margin-bottom: var(--spacing-paragraph); }
  th, td { padding: var(--pad-cell); border: 1px solid var(--color-border); text-align: left; text-indent: 0; }
  th { font-weight: var(--fw-bold); background: var(--color-highlight); font-family: var(--ff-heading); }
  blockquote {
    border-left: 3px solid var(--color-primary);
    background: var(--color-callout-bg);
    padding: var(--pad-callout);
    margin-bottom: var(--spacing-paragraph);
    font-size: var(--fs-small);
  }
  blockquote p { text-indent: 0; margin-bottom: var(--spacing-tight); }
  blockquote p:last-child { margin-bottom: 0; }
  .toc-list, .toc-list ol, .toc-list ul { list-style: none; list-style-type: none; padding-left: var(--spacing-toc-indent); }
  .toc-list li { margin-bottom: var(--spacing-tight); text-indent: 0; }
  .toc-list a { color: var(--color-primary); text-decoration: none; }
  .toc-title {
    font-family: var(--ff-heading);
    font-size: var(--fs-h2);
    font-weight: var(--fw-bold);
    text-indent: 0;
    margin-bottom: var(--spacing-sub);
  }
  .cover-eyebrow {
    text-align: center; font-size: var(--fs-small); color: var(--color-muted);
    font-family: var(--ff-heading); margin-bottom: var(--spacing-sub); text-indent: 0;
  }
  .cover-title {
    text-align: center; font-size: var(--fs-title); font-family: var(--ff-heading);
    font-weight: var(--fw-bold); color: var(--color-heading);
    margin-bottom: var(--spacing-paragraph); text-indent: 0; line-height: 1.4;
  }
  .cover-subtitle {
    text-align: center; font-size: var(--fs-subtitle); color: var(--color-muted);
    margin-bottom: var(--spacing-section); text-indent: 0;
  }
  .cover-meta {
    text-align: center; font-size: var(--fs-small); color: var(--color-muted);
    text-indent: 0; line-height: 2;
  }
"""


def build_toc(body_html: str) -> str:
    """从正文 h1/h2 生成目录条目。"""
    items = []
    sec = 0
    for m in re.finditer(r"<h2[^>]*>(.*?)</h2>", body_html, re.S):
        sec += 1
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        items.append(f'<li><a href="#section-{sec}">{title}</a></li>')
    # 附录三
    sec += 1
    items.append(f'<li><a href="#section-{sec}">附录三 源码地址索引与演示脚本</a></li>')
    return '<ol class="toc-list">\n    ' + "\n    ".join(items) + "\n  </ol>"


def main() -> int:
    body_md = BODY_MD.read_text(encoding="utf-8")
    apx_md = APX_MD.read_text(encoding="utf-8")

    body_html = convert(body_md)
    apx_html = convert(apx_md)

    # 正文里 h1 是文档大标题（封面已承载），实际章节是 h2 —— 把 h2 提升为 section 锚点
    counter = {"n": 0}

    def add_id(m: re.Match) -> str:
        counter["n"] += 1
        return m.group(0)[:-1] + f' id="section-{counter["n"]}">'

    body_html = re.sub(r"<h2 [^>]*>", add_id, body_html)
    n_sections = counter["n"]

    # 保持章节为 h2、小节为 h3（原文档大标题 h1 保留在正文首，封面另有一份）

    # 附录 h1 改为 section-N+1
    apx_html = apx_html.replace("<h1 ", f'<h1 id="section-{n_sections + 1}" ', 1)

    toc = build_toc(body_html)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="docx-page-size" content="A4">
  <title>数据集收集、清洗与模型输入：一条链路的完整拆解</title>
  <style>
{STYLE}  </style>
</head>
<body>

<section role="cover">
  <p class="cover-eyebrow">人工智能框架课 · 数据链路专题讲义</p>
  <p class="cover-title">数据集收集、清洗<br>与模型输入</p>
  <p class="cover-subtitle">一条链路的完整拆解<br>从原始压缩包到网络输入张量</p>
  <p class="cover-meta">
    数据集：天池 smartdiagnosisofclothflaw<br>
    训练集 8619 张 / 8570 框 · 验证集 957 张 / 374 负样本<br>
    全部图像尺寸 2446 × 1000，宽高比恒为 2.45:1<br>
    讲义所有数字均为项目实测 · 附源码地址与演示脚本
  </p>
</section>

<section role="cover">
<nav class="doc-toc" aria-label="文档目录">
  <p class="toc-title">目录</p>
  {toc}
</nav>
</section>

<section role="body" data-page-restart="1">

{body_html}

{apx_html}

</section>
</body>
</html>
"""

    OUT.write_text(html, encoding="utf-8")
    print(f"生成完成: {OUT}")
    print(f"  总长度 {len(html)} 字符")
    print(f"  章节数 {n_sections} + 附录 1")
    print(
        f"  pre 块 {html.count('<pre')}  表格 {html.count('<table')}  h1 {html.count('<h1')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
