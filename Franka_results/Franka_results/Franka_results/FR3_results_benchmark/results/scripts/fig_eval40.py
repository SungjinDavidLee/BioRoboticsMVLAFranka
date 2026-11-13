"""평가 결과 그림 (eval_report.py 출력 기반)
fig5 학습 데이터 배치와 평가 배치 (위에서 본 그림) · fig6 학습 데이터와의 거리 vs 성공 수 · fig7 꽂기 단계 손 위치 (랙 기준)"""
import json, sys
import numpy as np
from svgkit import SVG, INK, MUTE, LINE, render

D = sys.argv[1]           # real_report 폴더
A = json.load(open(f'{D}/x/eval_report/data/analysis.json'))
LF = {o['L']: o for o in json.load(open(f'{D}/layout_final.json'))}
T = A['trials']
for r in T:
    if r['folder'].endswith('163440'):
        r['layout'], r['k'] = 3, 1
    if r['folder'].endswith('190717'):
        r['layout'], r['k'] = 14, 1
V = [r for r in T if r['result'] in ('success', 'partial', 'fail')]
TR = A['train']
RAMP = ['#eba25a', '#c8721a', '#9a4d08', '#5f2c00']          # 성공 0·1·2·3 (validate_palette --ordinal 통과)
GRAY = '#b4bac3'
TX = {
 'ko': {'t5': '학습 데이터 배치(회색 100판)와 평가 배치 14개 — 위에서 본 그림',
        'base': '↓ 로봇 쪽', 'xl': '로봇 y (왼쪽 +) [m]', 'yl': '로봇 x (앞 +) [m]',
        'leg': ['학습: 튜브 잡은 자리', '학습: 빨간 링 구멍', '평가: 빨간 튜브 (●) → 빨간 링 (□)', '배치 4·8에서 실제로 닫은 자리 (×)'],
        'legs': '평가 배치 색 = 성공 비율 (배치 14는 1회)', 'nums': ['0', '1/3', '2/3', '3/3'],
        't6': '학습 데이터와의 거리와 성공 비율 (배치 14개)', 'x6': '가장 가까운 학습 시행까지 거리 — 튜브·구멍 위치 합 (mm)', 'y6': '성공 비율',
        'a6': ['35 mm 안 (배치 1·2·6·10·11·14): 13/16 성공', '35 mm 밖 (배치 3·4·5·7·8·9·12·13): 4/24 성공', '순위상관 ρ = −0.84 (배치 14개)'],
        't7': '꽂기 단계: 쥔 채 구멍에 가장 가까이 간 손 위치 (빈 랙 기준)', 'x7': '랙 방향 (mm) — 옆 구멍은 ±32 mm', 'y7': '랙 수직 방향 (mm)',
        'l7': ['성공 (놓은 자리)', '부분 (들어 올렸지만 못 꽂음)'], 'h0': '구멍 중심 (가운데 = 목표 구멍)', 'h1': '옆 구멍'},
 'en': {'t5': 'Training layouts (grey, 100 episodes) vs the 14 evaluation layouts — top view',
        'base': '↓ robot', 'xl': 'robot y (left +) [m]', 'yl': 'robot x (forward +) [m]',
        'leg': ['training: tube grasp point', 'training: red-ring hole', 'evaluation: red tube (●) → red ring (□)', 'layouts 4 & 8: where the gripper closed (×)'],
        'legs': 'evaluation colour = success rate (layout 14: 1 trial)', 'nums': ['0', '1/3', '2/3', '3/3'],
        't6': 'Distance to the training data vs success rate (14 layouts)', 'x6': 'distance to the nearest training episode — tube + hole position (mm)', 'y6': 'success rate',
        'a6': ['within 35 mm (layouts 1, 2, 6, 10, 11, 14): 13/16', 'beyond 35 mm (layouts 3, 4, 5, 7, 8, 9, 12, 13): 4/24', 'Spearman ρ = −0.84 (14 layouts)'],
        't7': 'Insertion: hand position closest to the hole while holding (empty-rack frame)', 'x7': 'along the rack (mm) — neighbouring holes at ±32 mm', 'y7': 'across the rack (mm)',
        'l7': ['success (release point)', 'partial (lifted, not inserted)'], 'h0': 'hole centre (middle = target)', 'h1': 'neighbour'},
}


def fig5(lang):
    L = TX[lang]
    W, H = 1140, 462
    s = SVG(W, H, lang)
    s.text(14, 28, L['t5'], size=16.5, weight=700)
    y0, y1, x0, x1 = 0.32, -0.27, 0.43, 0.70          # 가로 = 로봇 y (왼쪽이 +), 세로 = 로봇 x (위가 앞)
    k = 1300.0
    ox, oy = 70, 52

    def P(xy):
        return ox + (y0 - xy[1]) * k, oy + (x1 - xy[0]) * k
    pw, ph = (y0 - y1) * k, (x1 - x0) * k
    s.rect(ox, oy, pw, ph, fill='#fcfcfb', stroke=LINE, sw=1, rx=4)
    for gx in np.arange(0.45, 0.70, 0.05):
        _, py = P([gx, 0])
        s.line(ox, py, ox + pw, py, color='#eceef1', sw=1, arrow=False)
        s.text(ox - 6, py + 4, f'{gx:.2f}', size=11, color=MUTE, anchor='end')
    for gy in np.arange(0.30, -0.25, -0.10):
        px, _ = P([0, gy])
        s.line(px, oy, px, oy + ph, color='#eceef1', sw=1, arrow=False)
        s.text(px, oy + ph + 16, f'{gy:+.1f}', size=11, color=MUTE, anchor='middle')
    s.text(ox + pw / 2, oy + ph + 34, L['xl'], size=11.5, color=MUTE, anchor='middle')
    s.text(ox + pw - 4, oy + ph - 8, L['base'], size=11.5, color=MUTE, anchor='end')
    for e in TR:
        px, py = P(e['tube'])
        s.circle(px, py, 3.2, fill=GRAY, stroke='none', sw=0)
        if e.get('ring'):
            qx, qy = P(e['ring'])
            s.rect(qx - 3.5, qy - 3.5, 7, 7, fill='none', stroke=GRAY, sw=1.3, rx=1)
    lab_off = {1: (0, -15), 3: (-14, -10), 4: (12, -6), 5: (12, 14), 6: (12, -8), 7: (-14, -8), 8: (12, 18), 9: (0, -15), 10: (14, 6), 11: (-14, 22), 12: (-14, -10), 14: (14, 26)}
    for Lr in sorted(LF.values(), key=lambda o: o['L']):
        c = RAMP[int(round(3 * Lr['ns'] / Lr['n']))]
        tp, rp = P(Lr['tube']), P(Lr['ring'])
        s.line(tp[0], tp[1], rp[0], rp[1], color=c, sw=1.3, arrow=False, dash='5 3')
        s.rect(rp[0] - 7, rp[1] - 7, 14, 14, fill='#ffffff', stroke=c, sw=2.6, rx=2)
        s.circle(tp[0], tp[1], 8.5, fill=c, stroke='#ffffff', sw=2)
        if Lr['L'] in lab_off:
            dx, dy = lab_off[Lr['L']]
            s.text(tp[0] + dx, tp[1] + dy, {1: '1·2', 7: '7·13'}.get(Lr['L'], str(Lr['L'])), size=13.5, weight=700,
                   anchor='start' if dx > 0 else ('end' if dx < 0 else 'middle'))
        s.text(rp[0] + 10, rp[1] + 4 + {2: 10, 11: 10, 14: -8}.get(Lr['L'], 0), str(Lr['L']), size=11, color=MUTE)
        if Lr['L'] in (4, 8):
            cp = P(np.array(Lr['tube']) + np.array(Lr['mevec']) / 1000)
            s.line(tp[0], tp[1], cp[0], cp[1], color=INK, sw=1.2, dash='2 3', arrow=True)
            s.path(f'M{cp[0] - 6},{cp[1] - 6} L{cp[0] + 6},{cp[1] + 6} M{cp[0] - 6},{cp[1] + 6} L{cp[0] + 6},{cp[1] - 6}', color=INK, sw=2.4, arrow=False)
    # 범례
    lx, ly = ox + pw + 28, oy + 8
    s.circle(lx + 6, ly, 3.2, fill=GRAY, stroke='none', sw=0); s.text(lx + 18, ly + 4, L['leg'][0], size=12)
    s.rect(lx + 2.5, ly + 22 - 3.5, 7, 7, fill='none', stroke=GRAY, sw=1.3, rx=1); s.text(lx + 18, ly + 26, L['leg'][1], size=12)
    s.circle(lx + 6, ly + 44, 6, fill=RAMP[2], stroke='#ffffff', sw=1.5); s.text(lx + 18, ly + 48, L['leg'][2], size=12)
    s.path(f'M{lx + 1},{ly + 61} L{lx + 11},{ly + 71} M{lx + 1},{ly + 71} L{lx + 11},{ly + 61}', color=INK, sw=2.2, arrow=False)
    s.text(lx + 18, ly + 70, L['leg'][3], size=12)
    s.text(lx, ly + 104, L['legs'], size=12, color=MUTE)
    for i, c in enumerate(RAMP):
        s.rect(lx + i * 46, ly + 114, 38, 18, fill=c, stroke='none', sw=0, rx=3)
        s.text(lx + i * 46 + 19, ly + 127, L['nums'][i], size=12, weight=700, anchor='middle', color=INK if i == 0 else '#ffffff')
    return s


def fig6(lang):
    L = TX[lang]
    W, H = 760, 392
    s = SVG(W, H, lang)
    s.text(14, 28, L['t6'], size=16.5, weight=700)
    ox, oy, pw, ph = 70, 56, 640, 270
    xm = 125.0

    def P(d, n):
        return ox + d / xm * pw, oy + ph - n * (ph - 30) - 14
    for n, nm in ((0, '0%'), (1 / 3, '33%'), (2 / 3, '67%'), (1, '100%')):
        _, py = P(0, n)
        s.line(ox, py, ox + pw, py, color='#eceef1', sw=1, arrow=False)
        s.text(ox - 8, py + 4, nm, size=12, color=MUTE, anchor='end')
    for d in range(0, 126, 25):
        px, _ = P(d, 0)
        s.text(px, oy + ph + 16, str(d), size=11.5, color=MUTE, anchor='middle')
    s.line(ox, oy + ph, ox + pw, oy + ph, color=LINE, sw=1, arrow=False)
    s.text(ox + pw / 2, oy + ph + 36, L['x6'], size=12, color=MUTE, anchor='middle')
    s.text(ox - 46, oy - 12, L['y6'], size=12, color=MUTE)
    px35, _ = P(35, 0)
    s.line(px35, oy, px35, oy + ph, color='#8a919c', sw=1.2, dash='5 4', arrow=False)
    lab = {1: (-10, -10), 2: (-10, -10), 3: (6, 20), 4: (0, -14), 5: (10, 16), 6: (10, -8), 7: (-8, 20), 8: (0, -14), 9: (-12, -12), 10: (10, -8),
           11: (-4, -14), 12: (-10, -10), 13: (-10, -12), 14: (10, -10)}
    for Lr in LF.values():
        px, py = P(Lr['nn_joint'], Lr['ns'] / Lr['n'])
        s.circle(px, py, 7, fill='#9a4d08', stroke='#ffffff', sw=2)
        dx, dy = lab[Lr['L']]
        s.text(px + dx, py + dy, str(Lr['L']) + (' (1/1)' if Lr['n'] == 1 else ''), size=12.5, weight=700, anchor='start' if dx > 0 else ('end' if dx < 0 else 'middle'))
    ax_ = P(56, 0)[0]
    s.text(ax_, oy + 116, L['a6'][0], size=12.5, weight=700)
    s.text(ax_, oy + 136, L['a6'][1], size=12.5, weight=700)
    s.text(ax_, oy + 156, L['a6'][2], size=12, color=MUTE)
    return s


def fig7(lang):
    L = TX[lang]
    W, H = 860, 572
    s = SVG(W, H, lang)
    s.text(14, 28, L['t7'], size=16.5, weight=700)
    k = 9.5
    cx, cy = 440, 270

    def P(a, c):
        return cx + a * k, cy - c * k
    s.rect(cx - 41 * k, cy - 21 * k, 82 * k, 42 * k, fill='#fcfcfb', stroke=LINE, sw=1, rx=4)
    for a in range(-40, 41, 10):
        s.line(*P(a, -21), *P(a, 21), color='#eceef1', sw=1, arrow=False)
        s.text(P(a, -21)[0], P(a, -21)[1] + 16, str(a), size=11, color=MUTE, anchor='middle')
    for c in range(-20, 21, 10):
        s.line(*P(-41, c), *P(41, c), color='#eceef1', sw=1, arrow=False)
        s.text(P(-41, c)[0] - 6, P(-41, c)[1] + 4, str(c), size=11, color=MUTE, anchor='end')
    for a, nm in ((0, L['h0']), (-32, L['h1']), (32, L['h1'])):      # 구멍 중심 표시 (크기는 실제 구멍 아님)
        px, py = P(a, 0)
        s.line(px - 14, py, px + 14, py, color='#6b7280', sw=1.6, arrow=False)
        s.line(px, py - 14, px, py + 14, color='#6b7280', sw=1.6, arrow=False)
        if a != 0:
            s.text(px, py - 22, nm, size=11.5, color=MUTE, anchor='middle')
    s.text(cx, cy + 21 * k + 34, L['x7'], size=12, color=MUTE, anchor='middle')
    s.text(cx - 41 * k - 40, cy - 21 * k - 10, L['y7'], size=12, color=MUTE)
    lab = {'12-1': (-10, 16), '13-1': (12, 16), '13-2': (-10, 18), '13-3': (4, -14), '3-1': (8, 20), '3-2': (12, 2), '3-3': (-8, -10), '7-2': (-8, 14), '7-3': (-12, -10), '9-1': (8, 14), '9-2': (10, 22), '10-1': (8, -10)}
    for r in V:
        if r.get('near_along_mm') is None or not r.get('lift_ok'):
            continue
        a, c = r['near_along_mm'], r['near_across_mm']
        px, py = P(a, c)
        if r['result'] == 'success':
            s.circle(px, py, 6, fill=RAMP[3], stroke='#ffffff', sw=1.8)
        else:
            s.circle(px, py, 7, fill=RAMP[1], stroke='#ffffff', sw=1.8)
            tid = f"{r['layout']}-{r['k']}"
            dx, dy = lab.get(tid, (8, -8))
            s.text(px + dx, py + dy, tid, size=12, weight=700, anchor='start' if dx > 0 else 'end')
    ly = H - 18
    s.circle(40, ly - 4, 6, fill=RAMP[3], stroke='#ffffff', sw=1.5); s.text(52, ly, L['l7'][0], size=12)
    s.circle(250, ly - 4, 7, fill=RAMP[1], stroke='#ffffff', sw=1.5); s.text(262, ly, L['l7'][1], size=12)
    s.line(540, ly - 4, 556, ly - 4, color='#6b7280', sw=1.6, arrow=False); s.line(548, ly - 12, 548, ly + 4, color='#6b7280', sw=1.6, arrow=False)
    s.text(562, ly, L['h0'], size=12)
    return s


if __name__ == '__main__':
    out = sys.argv[2]
    for lang in ('ko', 'en'):
        for f, nm in ((fig5, 'fig5_coverage40'), (fig6, 'fig6_distance_success40'), (fig7, 'fig7_insertion40')):
            sv = f(lang)
            render(sv.svg(), f'{out}/{nm}_{lang}', sv.w, sv.h)
    print('ok')
