#!/usr/bin/env python3
"""render_shot.py — 将真实命令输出渲染为终端风格 PNG 截图（Edge headless）。

用法:
    python tools/render_shot.py "窗口标题" input.txt output.png
    some_command | python tools/render_shot.py "窗口标题" - output.png
"""

from __future__ import annotations

import html
import subprocess
import sys
import tempfile
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
WIDTH = 1280


def render(title: str, text: str, out_png: str, width: int = WIDTH) -> None:
    """把 text 渲染成暗色终端风格图片。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.count("\n") + 2
    height = min(200 + lines * 19, 4000)
    dots = "&#9679; &#9679; &#9679;"
    bar = f'<div class="bar"><span class="dot">{dots}</span>&nbsp;&nbsp;{html.escape(title)}</div>'
    body = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    body{{background:#0c0c0c;margin:0;font-family:Consolas,'Cascadia Mono',monospace}}
    .bar{{background:#2d2d2d;padding:9px 14px;color:#c9c9c9;font-size:13px}}
    .dot{{color:#5a5a5a}}
    .wrap{{padding:16px 20px}}
    .t{{color:#4ec9b0;font-size:13px;font-weight:bold;margin:0 0 8px}}
    pre{{color:#d4d4d4;font-size:13px;line-height:1.4;white-space:pre-wrap;word-break:break-word;margin:0}}
    </style></head><body>
    {bar}
    <div class="wrap"><pre>{html.escape(text)}</pre></div>
    </body></html>"""
    tmp = Path(tempfile.mkdtemp(prefix="shot_"))
    html_file = tmp / "shot.html"
    html_file.write_text(body, encoding="utf-8")
    out = Path(out_png).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    url = "file:///" + str(html_file).replace("\\", "/")
    subprocess.run(
        [
            EDGE,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--user-data-dir={tmp / 'prof'}",
            f"--screenshot={out}",
            f"--window-size={width},{height}",
            url,
        ],
        check=True,
        capture_output=True,
    )
    print(f"[shot] {out}")


if __name__ == "__main__":
    title = sys.argv[1]
    in_file = sys.argv[2] if len(sys.argv) > 2 else "-"
    out_png = sys.argv[3] if len(sys.argv) > 3 else "shot.png"
    if in_file and in_file != "-":
        text = Path(in_file).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()
    render(title, text, out_png)
