#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""render_pipeline.py — scripts/fig_pipeline.html을 figs/fig_pipeline.png로 찍는다 (2배 해상도, 페이지 전체)

  python3 scripts/render_pipeline.py
필요: playwright (python) + chromium, 글꼴 Noto Sans CJK KR
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'scripts' / 'fig_pipeline.html'
OUT = ROOT / 'figs' / 'fig_pipeline.png'

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 960, 'height': 1000}, device_scale_factor=2)
    pg.goto(SRC.resolve().as_uri())
    pg.wait_for_timeout(300)
    h = pg.evaluate('document.body.scrollHeight')
    pg.set_viewport_size({'width': 960, 'height': h})
    pg.screenshot(path=str(OUT), full_page=True)
    b.close()
print(f'✔ {OUT.relative_to(ROOT)}  ({960 * 2} × {h * 2} px)')
