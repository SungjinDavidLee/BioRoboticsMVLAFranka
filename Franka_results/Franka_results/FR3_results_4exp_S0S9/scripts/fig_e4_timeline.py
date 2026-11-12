#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_e4_timeline.py — E4 판 하나의 손끝 높이 · 손가락 폭 시간축 (배경 = 그 순간 움직인 쪽: VLA / 수식)

  python3 scripts/fig_e4_timeline.py <media_1112 폴더> [판 메모]      (기본 T06-hyb-2)
  → figs/fig_e4_timeline.png
입력: <폴더>/timeline/<판>.csv (scripts/media_export_1112.py), <폴더>/picked.json의 hybrid 기록(넘겨받은 시각·수식 동작 시간)
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
M = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('media_1112')
TAG = sys.argv[2] if len(sys.argv) > 2 else 'T06-hyb-2'
for p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'):
    if Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'font.size': 10, 'axes.unicode_minus': False})
SURF, INK, INK2, MUTE, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
VLA_BG, FOR_BG, VLA_T, FOR_T = '#e3eefb', '#fbe3d8', '#1c5cab', '#c4521f'

R = list(csv.DictReader(open(M / 'timeline' / f'{TAG}.csv')))
t = [float(r['t_s']) for r in R]
z = [float(r['z_m']) * 1000 for r in R]
w = [float(r['grip_w_m']) * 1000 for r in R]
P = json.loads((M / 'picked.json').read_text())[TAG]
H = P['hybrid']
ev = {e['kind']: float(e['t']) for e in H['events'] if e.get('t') is not None and not e.get('why')}
g0, g1 = ev['grasp'], ev['grasp'] + float(H['grasp']['move_s'])
p0, p1 = ev['place'], min(t[-1], ev['place'] + float(H['place']['move_s']))
spans = [(0, g0, 'VLA 접근', 'v'), (g0, g1, '수식 잡기 · 들기', 'f'), (g1, p0, 'VLA 옮기기', 'v'), (p0, p1, '수식 꽂기', 'f')]

fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), dpi=200, sharex=True, gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.12})
fig.patch.set_facecolor(SURF)
for ax, y, lab in ((axs[0], z, '손끝 높이 (mm, 로봇 받침 기준)'), (axs[1], w, '손가락 폭 (mm)')):
    ax.set_facecolor(SURF)
    for a, b, _, c in spans:
        ax.axvspan(a, b, color=VLA_BG if c == 'v' else FOR_BG, lw=0, zorder=0)
    ax.plot(t, y, color=INK, lw=1.6, zorder=3)
    ax.set_ylabel(lab, color=INK2, fontsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
top = max(z) + 25
axs[0].set_ylim(min(z) - 15, top + 18)
for a, b, lab, c in spans:
    axs[0].text((a + b) / 2, top + 4, lab, ha='center', va='bottom', fontsize=9, fontweight='bold', color=VLA_T if c == 'v' else FOR_T)
gp = H['grasp']
pl = H['place']
axs[1].annotate(f"쥔 폭 {gp['grip_mm']:.1f} mm", xy=(g1 - 0.5, gp['grip_mm']), xytext=(g1 - 0.5, gp['grip_mm'] + 28),
                ha='center', fontsize=8.6, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTE, lw=0.8))
ins = (pl.get('insert') or {}).get('above_mm')
if ins is not None:
    iz = min(range(len(t)), key=lambda k: z[k] if p0 <= t[k] <= p1 else 1e9)
    axs[0].annotate(f'꽂기 끝: 학습 놓는 높이 + {ins:.1f} mm', xy=(t[iz], z[iz]), xytext=(t[iz] - 7, max(z) - 8), fontsize=8.6,
                    color=INK2, arrowprops=dict(arrowstyle='-', color=MUTE, lw=0.8))
axs[1].set_xlabel('시작부터 시간 (s)', color=INK2)
axs[1].set_xlim(0, t[-1])
fig.text(0.012, 0.975, f'E4 {TAG} — 누가 언제 움직였나', ha='left', va='top', fontsize=13.5, fontweight='bold', color=INK)
fig.text(0.012, 0.925, f"배경 파랑 = VLA · 주황 = 수식 · 수식이 넘겨받은 시각: 잡기 {g0:.1f} s, 꽂기 {p0:.1f} s · 높이·폭은 관절·손가락 실측(10 Hz)",
         ha='left', va='top', fontsize=9.3, color=INK2)
fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.1)
fig.savefig(ROOT / 'figs' / 'fig_e4_timeline.png', facecolor=SURF)
print('✔ figs/fig_e4_timeline.png', f'잡기 {g0:.2f}–{g1:.2f} s · 꽂기 {p0:.2f}–{p1:.2f} s · 끝 {t[-1]:.1f} s')
