#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_T_stats.py — 11-05 평가(T01–T10, v1 100판 vs v2 200판)의 숫자를 결과표에서 다시 계산

  python3 scripts/eval_T_stats.py            # data/results_T_1105.csv, data/layouts_T.csv 읽음
  → data/trials_T_1105.csv (유효 40판), data/stats_T_1105.json, 화면에 요약

numpy만 씀 (scipy 없이 Wilson·Fisher·순열·부트스트랩을 직접 계산)
- 유효 판: 결과가 무효(void)가 아니고 메모가 'T<배치>-<모델>-<회차>' 형식인 판
- 단계(잡기·들기·칸 위·꽂기·놓기) 도달 = 스테이션 자동 판정의 단계 표시(폭·손 위치 기준) 또는 사람 판정 성공.
  꽂기·놓기에는 빨간 칸이 아닌 곳에 꽂은 판도 들어감 (빨간 칸에 꽂은 판 = 성공). 결과는 사람 판정(최종)
"""
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'data' / 'results_T_1105.csv'
LAY = ROOT / 'data' / 'layouts_T.csv'
MEMO = re.compile(r'T0?(\d{1,2})-(v[12])-(\d)')
STAGES = ['grasp', 'lift', 'above', 'insert', 'release']
SEED = 20261105


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_two_sided(a, b, c, d):
    """2×2 표 [[a, b], [c, d]] 양측 정확 검정 (확률이 관측값 이하인 표들의 합)"""
    n1, n2, m1 = a + b, c + d, a + c
    n = n1 + n2

    def pr(x):
        return math.comb(n1, x) * math.comb(n2, m1 - x) / math.comb(n, m1)
    p0 = pr(a)
    lo, hi = max(0, m1 - n2), min(n1, m1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def within_layout_permutation(trials):
    """배치마다 4판(v1 2 · v2 2)의 모델 이름을 서로 바꾸는 정확 순열 검정 (양측, 통계량 = v2 성공 − v1 성공)"""
    by = {}
    for t in trials:
        by.setdefault(t['layout'], []).append(t)
    dist = {0: 1}                       # v2 성공 수 합의 분포 (경우의 수)
    obs_v2 = 0
    for L, ts in by.items():
        s = [1 if t['result'] == 'success' else 0 for t in ts]
        n2 = sum(1 for t in ts if t['model'] == 'v2')
        obs_v2 += sum(si for si, t in zip(s, ts) if t['model'] == 'v2')
        loc = {}
        for idx in combinations(range(len(ts)), n2):
            k = sum(s[i] for i in idx)
            loc[k] = loc.get(k, 0) + 1
        new = {}
        for a, ca in dist.items():
            for b, cb in loc.items():
                new[a + b] = new.get(a + b, 0) + ca * cb
        dist = new
    tot_s = sum(1 for t in trials if t['result'] == 'success')
    total = sum(dist.values())
    obs = 2 * obs_v2 - tot_s
    p = sum(c for k, c in dist.items() if abs(2 * k - tot_s) >= abs(obs) - 1e-9) / total
    return obs, p, total


def layout_bootstrap(trials, n_boot=20000):
    by = {}
    for t in trials:
        by.setdefault(t['layout'], {'v1': [0, 0], 'v2': [0, 0]})
        by[t['layout']][t['model']][0] += t['result'] == 'success'
        by[t['layout']][t['model']][1] += 1
    Ls = sorted(by)
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(Ls), len(Ls))
        s1 = sum(by[Ls[i]]['v1'][0] for i in pick)
        n1 = sum(by[Ls[i]]['v1'][1] for i in pick)
        s2 = sum(by[Ls[i]]['v2'][0] for i in pick)
        n2 = sum(by[Ls[i]]['v2'][1] for i in pick)
        out.append(s2 / n2 - s1 / n1)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    lay = {r['layout']: r for r in csv.DictReader(open(LAY, encoding='utf-8-sig'))}
    raw = list(csv.DictReader(open(RES, encoding='utf-8-sig')))
    trials, void = [], []
    for r in raw:
        m = MEMO.search(r.get('note') or '')
        if r['result'] == 'void' or not m:
            void.append({'time': r['time'], 'model': 'v2' if 'v2_' in r['ckpt'] else 'v1', 'result_auto': r['result_auto'],
                         'note': r['note']})
            continue
        L = f'T{int(m.group(1)):02d}'
        t = {'layout': L, 'model': m.group(2), 'rep': int(m.group(3)), 'time': r['time'], 'folder': r['folder'],
             'ckpt': r['ckpt'], 'result': r['result'], 'result_auto': r['result_auto'], 'labeled_by': r['labeled_by'],
             'end': r['end'], 'duration_s': fnum(r['duration_s']), 'chunks': r['chunks'],
             **{s: int(r[s] or 0) for s in STAGES},
             'grip_mm': fnum(r['grip_mm']), 'grasp_err_mm': fnum(r['grasp_err_mm']), 'release_err_mm': fnum(r['release_err_mm']),
             'decoy': int(lay[L]['decoy']), 'adjacent_yellow': int(lay[L]['adjacent_yellow']), 'note': r['note']}
        trials.append(t)
    trials.sort(key=lambda t: t['time'])
    for k, t in enumerate(trials):
        t['order_in_layout'] = sum(1 for u in trials[:k] if u['layout'] == t['layout']) + 1
    keys = list(trials[0])
    with open(ROOT / 'data' / 'trials_T_1105.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(trials)

    S = {'n_valid': len(trials), 'n_void': len(void), 'void': void, 'models': {}}
    for M in ('v1', 'v2'):
        ts = [t for t in trials if t['model'] == M]
        n = len(ts)
        k = sum(t['result'] == 'success' for t in ts)
        S['models'][M] = {
            'n': n, 'success': k, 'partial': sum(t['result'] == 'partial' for t in ts), 'fail': sum(t['result'] == 'fail' for t in ts),
            'rate': k / n, 'wilson95': wilson(k, n),
            'stages_auto': {s: sum(t[s] for t in ts) for s in STAGES},
            # 단계 도달 = 자동 판정 단계 표시 또는 사람 판정 성공 (자동이 놓친 성공 판도 모든 단계에 도달한 것으로 셈)
            'stages': {s: sum(1 for t in ts if t[s] or t['result'] == 'success') for s in STAGES},
            'success_time_s': sorted(t['duration_s'] for t in ts if t['result'] == 'success'),
            'auto_agree': sum(t['result'] == t['result_auto'] for t in ts),
            'by_layout': {L: sum(t['result'] == 'success' for t in ts if t['layout'] == L) for L in sorted(lay)},
            'decoy': [sum(t['result'] == 'success' for t in ts if t['decoy']), sum(1 for t in ts if t['decoy'])],
            'non_decoy': [sum(t['result'] == 'success' for t in ts if not t['decoy']), sum(1 for t in ts if not t['decoy'])],
        }
    a, b = S['models']['v2'], S['models']['v1']
    S['diff_pp'] = 100 * (a['rate'] - b['rate'])
    S['fisher_p'] = fisher_two_sided(a['success'], a['n'] - a['success'], b['success'], b['n'] - b['success'])
    obs, p_perm, n_assign = within_layout_permutation(trials)
    S['perm'] = {'stat_v2_minus_v1': obs, 'p_two_sided': p_perm, 'n_assignments': n_assign}
    S['layout_bootstrap_diff95_pp'] = [100 * x for x in layout_bootstrap(trials)]
    S['stage_fisher_p'] = {s: fisher_two_sided(a['stages'][s], a['n'] - a['stages'][s], b['stages'][s], b['n'] - b['stages'][s]) for s in STAGES}
    dv, nd = a['decoy'], a['non_decoy']
    S['v2_decoy_vs_rest_fisher_p'] = fisher_two_sided(dv[0], dv[1] - dv[0], nd[0], nd[1] - nd[0])
    S['label_changes'] = [{'trial': f"{t['layout']}-{t['model']}-{t['rep']}", 'auto': t['result_auto'], 'final': t['result'], 'note': t['note']}
                          for t in trials if t['result'] != t['result_auto']]
    S['grasp_err_mm_failed'] = {M: [t['grasp_err_mm'] for t in trials if t['model'] == M and t['result'] == 'fail'] for M in ('v1', 'v2')}
    (ROOT / 'data' / 'stats_T_1105.json').write_text(json.dumps(S, ensure_ascii=False, indent=1))

    print(f"유효 {S['n_valid']}판 · 무효 {S['n_void']}판")
    for M in ('v1', 'v2'):
        m = S['models'][M]
        lo, hi = m['wilson95']
        print(f"{M}: 성공 {m['success']}/{m['n']} = {100 * m['rate']:.0f}% (Wilson 95% {100 * lo:.0f}–{100 * hi:.0f}%) · "
              f"부분 {m['partial']} · 실패 {m['fail']} · 단계 {m['stages']} (자동 표시만 {m['stages_auto']}) · 성공 시간 {m['success_time_s']} · 자동=최종 {m['auto_agree']}/{m['n']}")
        print(f"    배치별 성공 {m['by_layout']} · 미끼 배치 {m['decoy']} · 나머지 {m['non_decoy']}")
    print(f"차이 {S['diff_pp']:+.0f}%p · Fisher 양측 p = {S['fisher_p']:.3f} · 배치 안 정확 순열 p = {p_perm:.3f} ({n_assign}가지) · "
          f"배치 부트스트랩 95% {S['layout_bootstrap_diff95_pp'][0]:.0f}–{S['layout_bootstrap_diff95_pp'][1]:.0f}%p")
    print('단계별 Fisher p ' + ', '.join(f'{s} {p:.3f}' for s, p in S['stage_fisher_p'].items()))
    print(f"v2 미끼 배치 vs 나머지 Fisher p = {S['v2_decoy_vs_rest_fisher_p']:.3f}")
    print('자동 판정과 최종 판정이 다른 판: ' + '; '.join(f"{c['trial']} {c['auto']}→{c['final']}" for c in S['label_changes']))
    print(f"실패 판 닫은 자리–빨간 튜브 거리(mm): {S['grasp_err_mm_failed']}")


if __name__ == '__main__':
    main()
