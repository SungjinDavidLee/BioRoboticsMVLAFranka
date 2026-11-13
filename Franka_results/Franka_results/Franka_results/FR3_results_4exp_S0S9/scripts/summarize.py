#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize.py — stage_runs.json → 실험별 S0–S9 요약 (stage_summary.json + 화면)

  python3 scripts/summarize.py <시작 사진 폴더 img/ …>     (eval_deep_*.zip 안의 img/ — 첫 3인칭 화면 밝기용)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_DIRS = [Path(p) for p in sys.argv[1:]]
R = json.load(open(ROOT / 'data' / 'stage_runs.json'))
ORDER = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']


def bright(folder):
    for d in IMG_DIRS:
        p = d / f'{folder}_start_third.jpg'
        if p.exists():
            a = np.asarray(Image.open(p).convert('RGB')).astype(float)
            return round(float(a.mean()), 1), round(float((a.max(axis=2) >= 250).mean() * 100), 1)
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round((c - h) * 100, 1), round((c + h) * 100, 1)


def med(v, nd=1):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return [round(float(np.median(v)), nd), round(float(np.percentile(v, 0)), nd), round(float(np.percentile(v, 100)), nd), len(v)]


out = {}
for e in ('E1', 'E2', 'E3', 'E4'):
    X = [x for x in R if x.get('exp') == e and not x.get('void') and 'S' in x]
    V = [x for x in R if x.get('exp') == e and x.get('void')]
    if not X:
        continue
    n = len(X)
    cum = {}
    for k in ORDER:
        cum[k] = sum(1 for x in X if all(x['S'].get(j) for j in ORDER[:ORDER.index(k) + 1]))
    first = {}
    for x in X:
        first[x['first_fail'] or 'OK'] = first.get(x['first_fail'] or 'OK', 0) + 1
    br = [bright(x['folder']) for x in X]
    reached = lambda k: [x for x in X if all(x['S'].get(j) for j in ORDER[:ORDER.index(k)])]   # 앞 단계까지 통과한 판
    passed = lambda k: [x for x in X if all(x['S'].get(j) for j in ORDER[:ORDER.index(k) + 1])]
    succ = [x for x in X if x['result'] == 'success']
    s = {
        'n': n, 'void': len(V), 'human': {k: sum(1 for x in X if x['result'] == k) for k in ('success', 'partial', 'fail')},
        'succ_ci': wilson(len(succ), n), 'cum': cum, 'first_fail': first,
        'S9_ci': wilson(cum['S9'], n),
        'S0': {'tube_found': sum(1 for x in X if (x.get('S0') or {}).get('tube_found')),
               'bright_third': med([b[0] for b in br if b]), 'sat_third': med([b[1] for b in br if b]),
               'bright_over15': sum(1 for b in br if b and b[0] - 132 > 15)},
        'S2_t': med([x.get('S2_t_s') for x in reached('S2')]),
        'S2_min_d': med([x.get('S2_min_d_mm') for x in reached('S2')]),
        'S3_fb': med([x.get('fb_mm') for x in reached('S3')]),
        'S3_abs_fb': med([abs(x['fb_mm']) for x in reached('S3') if x.get('fb_mm') is not None]),
        'S3_abs_fy': med([abs(x['fy_mm']) for x in reached('S3') if x.get('fy_mm') is not None]),
        'S3_z': med([x.get('close_z_mm') for x in reached('S3')]),
        'S3_v': med([x.get('close_v_mm_s') for x in reached('S3')]),
        'S3_close_t': med([x.get('close_t') for x in reached('S3')]),
        'S4_w3': med([x.get('w3_mm') for x in passed('S4')]),
        'S5_whold': med([x.get('w_hold_mm') for x in passed('S5')]),
        'S6_rise': med([x.get('rise_mm') for x in passed('S6')]),
        'S6_t60': med([x.get('t_lift60_s') for x in passed('S6')]),
        'S6_drift': med([x.get('drift60_mm') for x in passed('S6')]),
        'S6_wdrop': med([x.get('w_drop60_mm') for x in passed('S6')]),
        'S7_clear': med([x.get('clear_mm') for x in passed('S7')]),
        'S8_near': med([x.get('near_hole_mm') for x in reached('S8')]),
        'S8_t': med([x.get('t_transport_s') for x in passed('S8')]),
        'S8_drop': sum(1 for x in reached('S8') if x.get('dropped')),
        'S9_above': med([x.get('ins_above_mm') for x in reached('S9')]),
        'succ_dur': med([x.get('dur') for x in succ]),
        'photo': sum(1 for x in X if x.get('photo')),
    }
    # S0 자세히: 첫 3인칭 화면 포화 ≥ 15% 또는 밝기 > 기준 132 + 15
    bad = [x for x, b in zip(X, br) if b and (b[1] >= 15 or b[0] > 147)]
    s['S0']['bad_light'] = len(bad)
    s['S0']['bad_light_succ'] = sum(1 for x in bad if x['result'] == 'success')
    s['S0']['ok_light_succ'] = [sum(1 for x, b in zip(X, br) if b and not (b[1] >= 15 or b[0] > 147) and x['result'] == 'success'),
                                sum(1 for b in br if b and not (b[1] >= 15 or b[0] > 147))]
    s['S0']['sat15'] = sum(1 for b in br if b and b[1] >= 15)
    if e == 'E4':
        H = [x.get('hy') or {} for x in X]
        s['hy'] = {'grasp_handoff': sum(1 for h in H if h.get('grasp_handoff')),
                   'grasp_handoff_ok': sum(1 for x in X if (x.get('hy') or {}).get('grasp_handoff') and x['S'].get('S5')),
                   'place_handoff': sum(1 for h in H if h.get('place_handoff')),
                   'place_handoff_ok': sum(1 for x in X if (x.get('hy') or {}).get('place_handoff') and x['result'] == 'success'),
                   'no_scene': sum(1 for h in H if any('뚜껑을 못 찾음' in (w or '') for w in h.get('why') or [])),
                   'grip_mm': med([h.get('grip_mm') for h in H]),
                   'snap_mm': med([h.get('snap_mm') for h in H]),
                   'place_snap_mm': med([h.get('place_snap_mm') for h in H]),
                   'ins_above': med([h.get('insert_above_mm') for h in H]),
                   'ins_above_succ': med([h.get('insert_above_mm') for x, h in zip(X, H) if x['result'] == 'success']),
                   'align_abs': med([max(abs(v) for v in h['align_err_mm'][:2]) for h in H if h.get('align_err_mm')]),
                   'arrive_z': med([h['arrive_err_mm'][2] for h in H if h.get('arrive_err_mm')])}
    out[e] = s
    print(f"\n== {e} n={n} void={len(V)} human={s['human']} 성공 {len(succ)}/{n} CI {s['succ_ci']}")
    print('  누적', cum, '| 처음 막힘', first)
    for k, v in s.items():
        if k not in ('n', 'void', 'human', 'cum', 'first_fail'):
            print('  ', k, v)
json.dump(out, open(ROOT / 'data' / 'stage_summary.json', 'w'), ensure_ascii=False, indent=1)
