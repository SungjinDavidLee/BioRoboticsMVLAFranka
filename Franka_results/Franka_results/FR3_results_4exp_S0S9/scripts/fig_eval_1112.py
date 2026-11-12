#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_eval_1112.py — README 도표 (실험 4개 S0–S9 단계 평가)

  python3 scripts/fig_eval_1112.py          # 저장소 폴더에서 실행 → figs/*.png
입력: data/stage_runs.csv (판마다 단계 채점), data/layouts_T.csv (평가 배치 계획 위치)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
D, F = ROOT / 'data', ROOT / 'figs'
F.mkdir(exist_ok=True)
for p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'):
    if Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'font.size': 10, 'axes.unicode_minus': False})

SURF, INK, INK2, MUTE, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
SER = {'E1': '#2a78d6', 'E2': '#eb6834', 'E3': '#1baf7a', 'E4': '#eda100'}      # 범주 색 순서 고정 (검증함)
NAME = {'E1': 'E1 실물 100판', 'E2': 'E2 실물 200판', 'E3': 'E3 실물 + 시뮬', 'E4': 'E4 VLA + 수식'}
BLUE = {250: '#86b6ef', 350: '#5598e7', 450: '#2a78d6', 550: '#1c5cab', 650: '#104281',
        100: '#cde2fb', 200: '#9ec5f4', 300: '#6da7ec', 400: '#3987e5', 500: '#256abf', 600: '#184f95', 700: '#0d366b'}
GOOD, WARN, CRIT = '#0ca30c', '#fab219', '#d03b3b'
ST = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
STL = ['S1\n목표 선택', 'S2\n접근', 'S3\n정렬', 'S4\n접촉', 'S5\n파지', 'S6\n들기', 'S7\n이탈', 'S8\n이송', 'S9\n꽂힘']
E = ['E1', 'E2', 'E3', 'E4']

R = list(csv.DictReader(open(D / 'stage_runs.csv', encoding='utf-8')))
by = {e: [r for r in R if r['exp'] == e] for e in E}


def axstyle(ax, grid='y'):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def title(fig, t, sub=None):
    fig.text(0.012, 0.975, t, ha='left', va='top', fontsize=13.5, fontweight='bold', color=INK)
    if sub:
        fig.text(0.012, 0.915, sub, ha='left', va='top', fontsize=9.5, color=INK2)


def cum(e):
    X = by[e]
    return [100 * sum(1 for r in X if all(r[j] == '1' for j in ST[:k + 1])) / len(X) for k in range(9)]


# 1) 단계별 누적 통과율
def fig_cumulative():
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(SURF)
    axstyle(ax)
    x = list(range(9))
    for e in E:
        y = cum(e)
        ax.plot(x, y, color=SER[e], lw=2, marker='o', ms=6.5, mec=SURF, mew=1.5, label=f'{NAME[e]} (n={len(by[e])})', zorder=3)
    ends = sorted(((cum(e)[-1], e) for e in E))
    ly, last = {}, -99
    for v, e in ends:                       # 끝 글자 겹치지 않게
        yy = max(v, last + 7)
        ly[e], last = yy, yy
    for e in E:
        ax.annotate(f'{NAME[e]}  {cum(e)[-1]:.0f}%', (8, cum(e)[-1]), xytext=(8.18, ly[e]), va='center', fontsize=9,
                    color=INK, arrowprops=dict(arrowstyle='-', color=GRID, lw=0.8))
    ax.set_xticks(x, STL, fontsize=8.6, color=INK2)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.3, 10.6)
    ax.set_yticks(range(0, 101, 20), [f'{v}%' for v in range(0, 101, 20)])
    ax.set_ylabel('그 단계까지 모두 통과한 판 비율', color=INK2)
    for x0, x1, lab in ((-0.4, 0.5, 'G'), (0.5, 5.5, 'P'), (5.5, 7.5, 'T'), (7.5, 8.5, 'L')):
        ax.text((x0 + x1) / 2, 102.5, lab, ha='center', va='bottom', fontsize=8.5, color=MUTE, fontweight='bold')
        ax.axvline(x1, color=GRID, lw=0.8, ls=':', zorder=1) if x1 < 8.5 else None
    ax.legend(loc='lower left', frameon=False, fontsize=8.8, labelcolor=INK)
    title(fig, '단계별 누적 통과율 (S1–S9)', '처음 막힌 단계에서 멈춘 것으로 셈 · S9 = 빨간 링 구멍에 꽂힘 · G 목표 선택 · P 잡고 들기 · T 옮기기 · L 꽂기')
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.14)
    fig.savefig(F / 'fig_stage_cumulative.png', facecolor=SURF)
    plt.close(fig)


# 2) 처음 막힌 단계 (분류별, 100% 막대)
def fig_first_fail():
    grp = [('G 실패 (S1)', ['S1'], BLUE[250], 'G'), ('P 실패 (S2–S6)', ['S2', 'S3', 'S4', 'S5', 'S6'], BLUE[400], 'P'),
           ('T 실패 (S7–S8)', ['S7', 'S8'], BLUE[550], 'T'), ('L 실패 (S9)', ['S9'], BLUE[700], 'L'), ('꽂힘 (성공)', [''], GOOD, '꽂힘')]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    axstyle(ax, grid=None)
    for i, e in enumerate(E[::-1]):
        X = by[e]
        n = len(X)
        left = 0
        for k, (lab, ss, col, short) in enumerate(grp):
            c = sum(1 for r in X if r['first_fail'] in ss)
            w = 100 * c / n
            if c:
                ax.barh(i, w - 0.35, left=left + 0.175, height=0.62, color=col, edgecolor=SURF, lw=0)
                txt_c = INK if k == 0 else '#ffffff'
                if w >= 6:
                    ax.text(left + w / 2, i, f'{short} {c}' if w >= 9 else f'{c}', ha='center', va='center', fontsize=9.5,
                            color=txt_c, fontweight='bold')
            left += w
        ax.text(101.5, i, f'n = {n}', va='center', fontsize=9, color=INK2)
    ax.set_yticks(range(4), [NAME[e] for e in E[::-1]], fontsize=9.5, color=INK)
    ax.set_xlim(0, 109)
    ax.set_xticks(range(0, 101, 25), [f'{v}%' for v in range(0, 101, 25)], color=INK2)
    hs = [Rectangle((0, 0), 1, 1, color=g[2]) for g in grp]
    ax.legend(hs, [g[0] for g in grp], loc='upper center', bbox_to_anchor=(0.47, -0.1), ncol=5,
              frameon=False, fontsize=8.8, handlelength=1.1, labelcolor=INK)
    title(fig, '판마다 처음 막힌 단계', '막대 안 숫자 = 판 수 · 파랑이 진할수록 더 뒤 단계에서 막힘 · 초록 = 빨간 링 구멍에 꽂힘')
    fig.subplots_adjust(left=0.17, right=0.98, top=0.8, bottom=0.2)
    fig.savefig(F / 'fig_first_fail.png', facecolor=SURF)
    plt.close(fig)


# 3) 배치별 성공 (격자)
def fig_layout_success():
    L = [f'T{i:02d}' for i in range(1, 11)]
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=200)
    fig.patch.set_facecolor(SURF)
    axstyle(ax, grid=None)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    steps = [100, 250, 400, 550, 700]
    for i, e in enumerate(E):
        for j, l in enumerate(L):
            X = [r for r in by[e] if r['layout'] == l]
            y = 3 - i
            if not X:
                ax.add_patch(Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, color='#f0efec', lw=0))
                ax.text(j + 0.5, y + 0.5, '—', ha='center', va='center', color=MUTE, fontsize=10)
                continue
            k = sum(1 for r in X if r['result_human'] == 'success')
            p = k / len(X)
            col = BLUE[steps[min(4, int(round(p * 4)))]]
            ax.add_patch(Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, color=col, lw=0))
            ax.text(j + 0.5, y + 0.5, f'{k}/{len(X)}', ha='center', va='center', fontsize=10,
                    color='#ffffff' if p >= 0.5 else INK, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_xticks([j + 0.5 for j in range(10)], L, color=INK2)
    ax.set_yticks([3.5 - i for i in range(4)], [NAME[e] for e in E], color=INK)
    ax.xaxis.tick_top()
    title(fig, '배치별 성공 (사람 판정)', '칸 = 성공 판 / 돌린 판 · 진할수록 성공 비율이 높음 · — = 돌리지 않음 (E3는 6개 배치에서 멈춤)')
    fig.subplots_adjust(left=0.17, right=0.985, top=0.72, bottom=0.04)
    fig.savefig(F / 'fig_layout_success.png', facecolor=SURF)
    plt.close(fig)


# 4) 평가 배치 위치 (계획)
def fig_layouts():
    T = list(csv.DictReader(open(D / 'layouts_T.csv', encoding='utf-8')))
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    axstyle(ax, grid='both')
    for r in T:
        ry, rx = float(r['red_y_m']), float(r['red_x_m'])
        yy, yx = float(r['yellow_y_m']), float(r['yellow_x_m'])
        hy, hx = float(r['ring_y_m']), float(r['ring_x_m'])
        ax.annotate('', xy=(hy, hx), xytext=(ry, rx),
                    arrowprops=dict(arrowstyle='-|>', color=BLUE[300], lw=1.2, shrinkA=9, shrinkB=7, mutation_scale=9))
        ax.plot(yy, yx, 'o', ms=6, mfc=SURF, mec=WARN, mew=1.6, zorder=3)
        ax.plot(hy, hx, 's', ms=9, mfc=CRIT, mec=SURF, mew=1.5, zorder=4)
        ax.plot(ry, rx, 'o', ms=13, mfc=BLUE[550], mec=SURF, mew=1.5, zorder=5)
        ax.text(ry, rx, r['layout'][1:].lstrip('0'), ha='center', va='center', fontsize=7.5, color='#ffffff', fontweight='bold', zorder=6)
    ax.set_xlim(0.3, -0.3)
    ax.set_ylim(0.42, 0.66)
    ax.set_aspect('equal')
    ax.set_xlabel('← 로봇 왼쪽 +y (m)', color=INK2)
    ax.set_ylabel('앞 x (m)', color=INK2)
    hs = [plt.Line2D([], [], ls='', marker='o', ms=10, mfc=BLUE[550], mec=SURF),
          plt.Line2D([], [], ls='', marker='o', ms=6, mfc=SURF, mec=WARN, mew=1.6),
          plt.Line2D([], [], ls='', marker='s', ms=8, mfc=CRIT, mec=SURF)]
    ax.legend(hs, ['빨간 튜브 (숫자 = 배치 번호)', '노란 튜브', '빨간 링 구멍 (목표 칸)'], loc='upper center',
              bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    title(fig, '평가 배치 T01–T10 (위에서 본 로봇 기준 계획 위치)',
          '수집에 쓰지 않은 배치 · 사람이 배치 그림을 보고 손으로 놓음 → 실제 놓인 자리는 날마다 10–30 mm 다를 수 있음')
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.2)
    fig.savefig(F / 'fig_layouts_T01_T10.png', facecolor=SURF)
    plt.close(fig)


# 5) E4 조명 — 시작 화면 밝기
def fig_e4_lighting():
    X = sorted(by['E4'], key=lambda r: r['rollout_folder'])
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    axstyle(ax)
    out = [i for i, r in enumerate(X) if float(r['start_brightness']) > 147 or float(r['start_saturation_pct']) >= 15]
    ax.axvspan(min(out) - 0.5, max(out) + 0.5, color='#f0efec', lw=0, zorder=0)
    ax.text((min(out) + max(out)) / 2, 182, '조명 범위 밖 14판 (10:50–11:11)', ha='center', fontsize=9, color=INK2)
    ax.axhline(147, color=MUTE, lw=1, ls='--', zorder=1)
    ax.text(-0.4, 148.2, '기준 132 + 15', fontsize=8.5, color=INK2, va='bottom')
    sty = {'success': ('o', GOOD, '성공 (꽂힘)'), 'partial': ('s', WARN, '부분'), 'fail': ('X', CRIT, '실패')}
    for k, (mk, col, lab) in sty.items():
        I = [i for i, r in enumerate(X) if r['result_human'] == k]
        ax.scatter(I, [float(X[i]['start_brightness']) for i in I], marker=mk, s=60, c=col, edgecolors=SURF, linewidths=1.2,
                   zorder=4, label=lab)
    ns = [i for i, r in enumerate(X) if r['tube_found'] == '0']
    ax.scatter(ns, [float(X[i]['start_brightness']) for i in ns], marker='o', s=210, facecolors='none', edgecolors=INK,
               linewidths=1, zorder=3, label='시작 장면에서 뚜껑을 못 찾음')
    br = next(i for i in range(1, len(X)) if X[i]['rollout_folder'][-6:] > '130000' and X[i - 1]['rollout_folder'][-6:] < '130000')
    ax.axvline(br - 0.5, color=INK2, lw=1, zorder=2)
    ax.text(br - 0.3, 183.5, '쉬었다가\n14:19 재개', fontsize=8.3, color=INK2, va='top')
    labs, pos = [], []
    for i, r in enumerate(X):
        if i == 0 or r['layout'] != X[i - 1]['layout'] or i == br:
            j = i
            while j + 1 < len(X) and X[j + 1]['layout'] == r['layout'] and j + 1 != br:
                j += 1
            pos.append((i + j) / 2)
            labs.append(r['layout'])
    ax.set_xticks(pos, labs, color=INK2)
    ax.set_xlim(-0.6, len(X) - 0.4)
    ax.set_ylim(132, 186)
    ax.set_ylabel('시작 3인칭 화면 평균 밝기', color=INK2)
    ax.legend(loc='upper left', frameon=False, fontsize=8.6, labelcolor=INK, ncol=2, bbox_to_anchor=(0.0, 0.9))
    title(fig, 'E4 판마다 시작 화면 밝기 (실행 순서)', '성공 범위 밖 5/14 vs 범위 안 11/16 (p = 0.14, 배치와 겹쳐 따로 떼어 낼 수 없음)')
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.09)
    fig.savefig(F / 'fig_e4_lighting.png', facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    fig_cumulative()
    fig_first_fail()
    fig_layout_success()
    fig_layouts()
    fig_e4_lighting()
    print('✔', sorted(p.name for p in F.glob('fig_*.png')))
