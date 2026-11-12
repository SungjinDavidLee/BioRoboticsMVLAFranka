#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_T_1105.py — 11-05 평가 그림: (a) 단계 도달 v1 vs v2, (b) 배치·회차별 결과
  python3 scripts/eval_T_stats.py && python3 scripts/fig_T_1105.py   → figs/fig_T_results_{ko,en}.{png,svg,pdf}
색: 모델 = 기본 범주 색 1·2번(#2a78d6, #eb6834), 결과 = 무채색 순서 단계(실패 #a9a79e < 부분 #75736b < 성공 #33322e)
     둘 다 validate_palette.js 통과 (범주 light / --ordinal light). 결과 칸에는 글자도 같이 씀(색만으로 구분하지 않음)
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from svgkit import SVG, render  # noqa: E402

INK, INK2, MUTE, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
MC = {'v1': '#2a78d6', 'v2': '#eb6834'}
OC = {'fail': '#a9a79e', 'partial': '#75736b', 'success': '#33322e'}
OT = {'fail': INK, 'partial': '#ffffff', 'success': '#ffffff'}
STAGES = ['grasp', 'lift', 'above', 'insert', 'release']
TX = {
    'ko': {'title': '처음 보는 배치 T01–T10에서 v1(100판 학습)과 v2(200판 학습) — 모델마다 20판 (2026-11-05)',
           'a': '(a) 단계까지 간 판 수 (20판 중)', 'b': '(b) 배치·회차별 결과 (실행 순서 v2 → v1 → v1 → v2)',
           'st': ['잡기', '들기', '칸 위', '꽂기*', '성공'], 'm': {'v1': 'v1 (100판)', 'v2': 'v2 (200판)'},
           'row': ['v1 1회', 'v1 2회', 'v2 1회', 'v2 2회'], 'o': {'success': '성', 'partial': '부', 'fail': '실'},
           'ol': {'success': '성공', 'partial': '부분 (쥐었지만 못 꽂음)', 'fail': '실패 (못 쥠)'},
           'tags': {'decoy': '미끼', 'adj': '옆 칸'}, 'sum': '성공',
           'foot': ['* 꽂기에는 자동 판정은 꽂기로 봤지만 사람 판정에서 빨간 칸이 아니었던 판(v1 {w1} · v2 {w2})도 들어감. 단계 = 자동 판정 단계 또는 사람 판정 성공.',
                    '양측 Fisher p — 잡기 {g:.2f} · 칸 위 {a:.2f} · 성공 {s:.2f} (배치 안 순열 {p:.2f}).',
                    '미끼 = 학습에서 빨간 튜브가 자주 있던 튜브 랙 바깥 끝에 노란 튜브. 옆 칸 = 노란 튜브가 빨간 튜브 바로 옆 칸.']},
    'en': {'title': 'Unseen layouts T01–T10: v1 (trained on 100 episodes) vs v2 (200 episodes) — 20 trials each (2026-11-05)',
           'a': '(a) Trials reaching each stage (of 20)', 'b': '(b) Outcome per layout and trial (run order v2 → v1 → v1 → v2)',
           'st': ['grasp', 'lift', 'above slot', 'insert*', 'success'], 'm': {'v1': 'v1 (100 ep.)', 'v2': 'v2 (200 ep.)'},
           'row': ['v1 #1', 'v1 #2', 'v2 #1', 'v2 #2'], 'o': {'success': 'S', 'partial': 'P', 'fail': 'F'},
           'ol': {'success': 'success', 'partial': 'partial (grasped, not inserted)', 'fail': 'failure (no grasp)'},
           'tags': {'decoy': 'decoy', 'adj': 'adj.'}, 'sum': 'success',
           'foot': ['* insert also counts trials auto-judged as inserted that the human judged not in the red slot (v1 {w1} · v2 {w2}). Stage = auto-judged stage or human-judged success.',
                    'Two-sided Fisher p — grasp {g:.2f} · above slot {a:.2f} · success {s:.2f} (within-layout permutation {p:.2f}).',
                    'decoy = yellow tube at the outer end of the tube rack, where the red tube often was in training. adj. = yellow tube right next to the red tube.']},
}


def main():
    S = json.loads((ROOT / 'data' / 'stats_T_1105.json').read_text())
    T = list(csv.DictReader(open(ROOT / 'data' / 'trials_T_1105.csv', encoding='utf-8-sig')))
    L = list(csv.DictReader(open(ROOT / 'data' / 'layouts_T.csv', encoding='utf-8-sig')))
    for lang, X in TX.items():
        W, H = 1180, 530
        s = SVG(W, H, lang)
        s.raw(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fcfcfb"/>')
        s.text(20, 32, X['title'], size=17, weight=700, color=INK)
        # ── (a) 단계 막대 ──
        ax, ay, aw, ah = 70, 120, 430, 280
        s.text(20, 70, X['a'], size=14.5, weight=700, color=INK)
        lx = 20
        for M in ('v1', 'v2'):
            s.rect(lx, 84, 12, 12, fill=MC[M], stroke='none', rx=2)
            s.text(lx + 18, 95, X['m'][M], size=12.5, color=INK2)
            lx += 130
        for v in range(0, 21, 5):
            y = ay + ah - ah * v / 20
            s.raw(f'<line x1="{ax}" y1="{y:.1f}" x2="{ax + aw}" y2="{y:.1f}" stroke="{GRID if v else BASE}" stroke-width="1"/>')
            s.text(ax - 8, y + 4, str(v), size=11.5, color=MUTE, anchor='end')
        gw = aw / len(STAGES)
        bw = 30
        for i, st in enumerate(STAGES):
            cx = ax + gw * (i + 0.5)
            for j, M in enumerate(('v1', 'v2')):
                v = S['models'][M]['stages'][st] if st != 'release' else S['models'][M]['success']
                h = ah * v / 20
                x = cx - bw - 1 + j * (bw + 2)
                y = ay + ah - h
                if h > 0:
                    r = min(4, h)
                    s.raw(f'<path d="M{x},{ay + ah} V{y + r} Q{x},{y} {x + r},{y} H{x + bw - r} Q{x + bw},{y} {x + bw},{y + r} V{ay + ah} Z" fill="{MC[M]}"/>')
                if st == 'release':
                    pct = round(100 * v / 20)
                    s.text(x + bw / 2, y - 7, f'{v}/20', size=12, weight=700, color=INK, anchor='middle')
                    s.text(x + bw / 2, y - 22, f'{pct}%', size=11, color=INK2, anchor='middle')
            s.text(cx, ay + ah + 20, X['st'][i], size=12.5, color=INK2, anchor='middle')
        # ── (b) 배치 칸 ──
        bx, by_, cw, chh = 640, 128, 40, 38
        s.text(560, 70, X['b'], size=14.5, weight=700, color=INK)
        for o, lx in zip(('success', 'partial', 'fail'), (560, 640, 840) if lang == 'ko' else (560, 650, 870)):
            s.rect(lx, 84, 12, 12, fill=OC[o], stroke='none', rx=2)
            s.text(lx + 18, 95, X['ol'][o], size=12.5, color=INK2)
        rows = [('v1', 1), ('v1', 2), ('v2', 1), ('v2', 2)]
        for ri, (M, rep) in enumerate(rows):
            y = by_ + ri * (chh + 2) + (10 if ri >= 2 else 0)
            s.rect(bx - 74, y + 16, 10, 10, fill=MC[M], stroke='none', rx=2)
            s.text(bx - 58, y + 25, X['row'][ri], size=12.5, color=INK2)
            for ci, lay in enumerate(L):
                t = next((u for u in T if u['layout'] == lay['layout'] and u['model'] == M and int(u['rep']) == rep), None)
                x = bx + ci * (cw + 2)
                if t is None:
                    continue
                o = t['result']
                s.rect(x, y, cw, chh, fill=OC[o], stroke='none', rx=4)
                s.text(x + cw / 2, y + chh / 2 + 5, X['o'][o], size=13, weight=700, color=OT[o], anchor='middle')
            if rep == 2:
                k = S['models'][M]['success']
                s.text(bx + 10 * (cw + 2) + 12, y + 4, f"{X['sum']} {k}/20", size=12.5, weight=700, color=INK)
        yb = by_ + 4 * (chh + 2) + 10
        for ci, lay in enumerate(L):
            x = bx + ci * (cw + 2) + cw / 2
            s.text(x, yb + 18, lay['layout'], size=12.5, weight=700, color=INK, anchor='middle')
            tags = []
            if lay['decoy'] == '1':
                tags.append(X['tags']['decoy'])
            if lay['adjacent_yellow'] == '1':
                tags.append(X['tags']['adj'])
            for k_, tg in enumerate(tags):
                s.text(x, yb + 34 + 14 * k_, tg, size=11, color=INK2, anchor='middle')
        # ── 주석 ──
        P = S['stage_fisher_p']
        wr = {M: S['models'][M]['stages']['insert'] - S['models'][M]['success'] for M in ('v1', 'v2')}
        foot = [X['foot'][0].format(w1=wr['v1'], w2=wr['v2']), X['foot'][1].format(g=P['grasp'], a=P['above'], s=S['fisher_p'], p=S['perm']['p_two_sided']), X['foot'][2]]
        s.text(20, H - 62, foot, size=12, color=INK2)
        out = ROOT / 'figs' / f'fig_T_results_{lang}'
        render(s.svg(), str(out), W, H)
        print(f'✔ {out}.png')


if __name__ == '__main__':
    main()
