#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_layouts_overlay.py — 수집 데이터 200판의 위치 분포 위에 평가 배치 T01–T10을 겹친 그림 (위에서 본 로봇 기준)

  python3 scripts/fig_layouts_overlay.py
  → figs/fig_layouts_train_vs_T.png, data/layouts_T_nearest.csv
입력:
  data/train_positions_200.csv  수집 200판의 실제 손끝 위치 (관절 실측 FK). 잡은 자리 = 그리퍼를 닫은 순간, 꽂은 자리 = 연 순간.
                                episode_000000–000099 = 첫 100판(11-01까지, E1 학습), 나머지 = 추가 100판(11-03~05, E2·E4는 200판 전체)
  data/layouts_T.csv            평가 배치 계획 위치 (빨간 튜브·빨간 링 칸·랙 중심과 방향)
랙 방향(deg)은 칸 번호가 커지는 방향을 로봇 +x에서 +y 쪽으로 잰 각도. 랙 크기 149 × 39 mm(장면 검출 실측).
layouts_T.csv의 red_*·ring_* 값은 스테이션 받침 모델(구멍 간격 29.5 mm)로 계산돼 있다. 실제 받침 구멍 간격은 37.2 mm라
여기서는 랙 중심 + (칸 번호 − 2.5) × 37.2 mm로 다시 계산한다(바깥 칸에서 최대 약 12 mm 차이).
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent
for p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'):
    if Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'font.size': 11, 'axes.unicode_minus': False})
SURF, INK, INK2, MUTE, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
C1, C2 = '#2a78d6', '#1baf7a'          # 첫 100판 · 추가 100판 (참고 팔레트 1·3번)
RACK_L, RACK_W = 0.149, 0.039
PITCH = 0.0372                        # 실제 받침 구멍 간격 (m)
HOME = (0.4375, 0.0163)                # 홈 자세 손끝 (x, y)

tr = list(csv.DictReader(open(ROOT / 'data' / 'train_positions_200.csv')))
T = list(csv.DictReader(open(ROOT / 'data' / 'layouts_T.csv')))
f = lambda r, k: float(r[k])


def slot_xy(r, which):
    pre, k = ('tube_rack', 'red_slot') if which == 'red' else ('empty_rack', 'ring_slot')
    cx, cy, deg = f(r, f'{pre}_x_m'), f(r, f'{pre}_y_m'), math.radians(f(r, f'{pre}_deg'))
    s = (int(r[k]) - 2.5) * PITCH
    return cx + s * math.cos(deg), cy + s * math.sin(deg)


for r in T:
    r['red_x_m'], r['red_y_m'] = (str(v) for v in slot_xy(r, 'red'))
    r['ring_x_m'], r['ring_y_m'] = (str(v) for v in slot_xy(r, 'ring'))


def nearest(rows, kx, ky, x, y):
    return min(math.hypot(f(r, kx) - x, f(r, ky) - y) for r in rows) * 1000


# 표: 평가 배치마다 가장 가까운 수집 판까지 거리 (mm)
first = [r for r in tr if r['set'] == 'first100']
out = []
for t in T:
    tx, ty, hx, hy = f(t, 'red_x_m'), f(t, 'red_y_m'), f(t, 'ring_x_m'), f(t, 'ring_y_m')
    row = {'layout': t['layout']}
    for tag, rows in (('100', first), ('200', tr)):
        row[f'tube_nearest_mm_{tag}'] = round(nearest(rows, 'grasp_x_m', 'grasp_y_m', tx, ty), 1)
        row[f'slot_nearest_mm_{tag}'] = round(nearest(rows, 'release_x_m', 'release_y_m', hx, hy), 1)
    out.append(row)
with open(ROOT / 'data' / 'layouts_T_nearest.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)


def rack(ax, cx, cy, deg):
    u = (math.cos(math.radians(deg)), math.sin(math.radians(deg)))
    v = (-u[1], u[0])
    pts = []
    for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        x = cx + a * u[0] * RACK_L / 2 + b * v[0] * RACK_W / 2
        y = cy + a * u[1] * RACK_L / 2 + b * v[1] * RACK_W / 2
        pts.append((y, x))                       # 그림 가로 = y, 세로 = x
    ax.add_patch(Polygon(pts, closed=True, fill=False, ec=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2, alpha=0.75))


fig, axs = plt.subplots(2, 1, figsize=(11.5, 11.2), dpi=180)
fig.patch.set_facecolor(SURF)
panels = [
    ('(가) 잡은 자리 — 빨간 튜브', 'grasp', 'red', 'tube_rack', '그리퍼를 닫은 순간 손끝'),
    ('(나) 꽂은 자리 — 빨간 링 칸', 'release', 'ring', 'empty_rack', '그리퍼를 연 순간 손끝'),
]
for ax, (ttl, key, tkey, rk, what) in zip(axs, panels):
    ax.set_facecolor(SURF)
    for r in T:
        rack(ax, f(r, f'{rk}_x_m'), f(r, f'{rk}_y_m'), f(r, f'{rk}_deg'))
    for sub, col, mk, lab in (('first100', C1, 'o', '수집 첫 100판 (11-01까지 · E1 학습)'),
                              ('added100', C2, '^', '수집 추가 100판 (11-03~05 · E2·E4는 200판 전체)')):
        rows = [r for r in tr if r['set'] == sub]
        ax.scatter([f(r, f'{key}_y_m') for r in rows], [f(r, f'{key}_x_m') for r in rows], s=26, marker=mk,
                   c=col, alpha=0.62, lw=0.6, ec=SURF, zorder=3, label=lab)
    for r in T:
        y, x = f(r, f'{tkey}_y_m'), f(r, f'{tkey}_x_m')
        ax.scatter([y], [x], s=270, c=INK, ec=SURF, lw=1.6, zorder=5)
        ax.text(y, x, r['layout'][1:].lstrip('0'), ha='center', va='center', color='white', fontsize=10,
                fontweight='bold', zorder=6)
    ax.scatter([HOME[1]], [HOME[0]], marker='*', s=180, c=INK2, zorder=4)
    ax.annotate('홈 (손끝)', (HOME[1], HOME[0]), xytext=(8, -4), textcoords='offset points', fontsize=9.5, color=INK2)
    ax.set_xlim(0.31, -0.31)
    ax.set_ylim(0.415, 0.685)
    ax.set_aspect('equal')
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9.5)
    ax.set_ylabel('앞 x (m)', color=INK2, fontsize=10)
    ax.set_title(f'{ttl}   ·   수집: {what} (관절 실측)', loc='left', fontsize=12.5, fontweight='bold', color=INK, pad=8)
axs[1].set_xlabel('← 로봇 왼쪽 +y (m)', color=INK2, fontsize=10)
h, l = axs[0].get_legend_handles_labels()
h += [plt.Line2D([], [], marker='o', ls='', ms=8, mfc=INK, mec=SURF),
      plt.Line2D([], [], ls=(0, (4, 3)), color=INK2, lw=1.0)]
l += ['평가 배치 T01–T10 계획 위치 (숫자 = 배치 번호)', '평가 배치의 랙 (위: 튜브 랙 · 아래: 빈 랙)']
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005), columnspacing=2.2, markerscale=1.6)
fig.text(0.04, 0.985, '수집 데이터 200판 위에 평가 배치 T01–T10 겹쳐 보기 (위에서 본 로봇 기준)', ha='left', va='top',
         fontsize=15, fontweight='bold', color=INK)
fig.text(0.04, 0.958, '점 = 자동 시연 판마다 실제 손끝 위치 (200판)  ·  검은 원 = 평가 배치 계획 위치 (수집에 쓰지 않음 · 사람이 그림을 보고 놓아 실제 자리는 날마다 10–30 mm 다를 수 있음)',
         ha='left', va='top', fontsize=10, color=INK2)
fig.subplots_adjust(left=0.07, right=0.98, top=0.905, bottom=0.085, hspace=0.22)
fig.savefig(ROOT / 'figs' / 'fig_layouts_train_vs_T.png', facecolor=SURF)
print('✔ figs/fig_layouts_train_vs_T.png  ✔ data/layouts_T_nearest.csv')
for r in out:
    print(r)
