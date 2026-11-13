#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""score_stages.py — 평가 판을 단계 평가표(S0–S9, FR3 기준값)로 채점 (기록만 읽음)

  python3 scripts/score_stages.py <deep.json 11-05> <deep.json 11-11> <deep.json 11-12>
     [--review data/fail_causes_T_1105_review.csv] [--out data/stage_runs.json]
입력: eval_deep_*.py(로봇 PC, 읽기만)가 만든 deep.json (11-05 v1·v2, 11-11 시뮬 합친, 11-12 VLA + 수식)
출력: stage_runs.json (판마다 지표·단계 통과·처음 막힌 단계), 화면 요약

FR3 기준값
  S2 빨간 뚜껑 수평 25 mm 안 · 손끝 z ≤ 170 mm(학습 닫는 높이 109.6 + 60)
  S3 닫는 순간 손 기준: 앞뒤(손가락 폭 방향, 앞쪽 +) −19 ~ +5 mm, 손가락 방향 |·| ≤ 24 mm, 높이 학습 109.6 ± 15 mm
  S4 닫고 3 s 뒤 폭 20–50 mm
  S5 그 뒤 들기 시작(+2 mm)까지 폭 ≥ 12 mm 유지
  S6 쥔 채(폭 ≥ 12 mm) 닫은 높이 + 60 mm (구멍 깊이 약 50 mm)
  S7 쥔 채 + 60 mm 위 · 잡은 자리에서 수평 40 mm 밖
  S8 쥔 채 빨간 칸 수평 20 mm 안 (사람 판정 성공이면 통과)
  S9 빨간 칸 구멍에 꽂힘 = 사람 판정 성공, 또는 사진 검토로 '꽂았지만 안 놓음'
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

Z_TRAIN = 0.1096          # 학습 200판 닫은 높이 중앙
Z_REL = 0.1174            # 학습 놓은 높이 중앙
FB = (-0.019, 0.005)      # 앞뒤 허용
FY = 0.024                # 손가락 방향 허용
ZTOL = 0.015
PITCH = 0.0372


def norm_note(n):
    m = re.search(r'T0*(\d{1,2})-(v1|v2|mix|hyb)-(\d)', n or '')
    return f'T{int(m.group(1)):02d}-{m.group(2)}-{m.group(3)}' if m else None


def exp_of(t):
    ck = (t.get('ckpt') or '').split('/')[-2] if t.get('ckpt') else ''
    day = t['folder'][8:16]
    h = t.get('cfg_hybrid') or {}
    if h.get('use') and 'grasp_z_mode' in h and day >= '20261112':
        return 'E4'
    if day == '20261105' and ck.startswith('v1_100ep'):
        return 'E1'
    if day == '20261105' and ck.startswith('v2_200ep'):
        return 'E2'
    if day == '20261111' and ck.startswith('v2sim50'):
        return 'E3'
    return None


def first_edge(g, down=True):
    for i in range(1, len(g)):
        if (g[i - 1] > 0.5 and g[i] < 0.5) if down else (g[i - 1] < 0.5 and g[i] > 0.5):
            return i
    return None


def score(t, review=None):
    ts = t['ts']
    T = np.asarray(ts['t'], float)
    P = np.asarray(ts['p'], float)
    W = np.asarray(ts['w'], float)
    G = np.asarray(ts['g'], float)
    YAW = np.asarray(ts['yaw'], float)
    pre = t.get('pre') or {}
    sc = t.get('scene') or {}
    hy = t.get('hybrid') or {}
    r = {'folder': t['folder'], 'note': t.get('note'), 'result': t.get('result'), 'result_auto': t.get('result_auto'),
         'dur': t.get('duration_s')}
    # 빨간 튜브·빨간 칸 위치
    red = pre.get('tube_xy')
    caps = [c for c in (sc.get('caps') or []) if c.get('xy')]
    if hy.get('caps_xy'):
        caps = [{'xy': c, 'red': i == hy.get('red_pick')} for i, c in enumerate(hy['caps_xy'])]
        if hy.get('red_pick') is not None and red is None:
            red = hy['caps_xy'][hy['red_pick']]
    if red is None:
        rc = next((c for c in caps if c.get('red')), None)
        red = rc['xy'] if rc else None
    hole = hy.get('hole') or pre.get('hole_xy')
    r['red_xy'], r['hole_xy'] = red, hole
    # S0
    b = t.get('bright') or {}
    r['S0'] = {'tube_found': pre.get('tube_xy') is not None, 'ring_found': pre.get('ring_px') is not None or pre.get('hole_xy') is not None,
               'bright_third': (b.get('third') or [None])[0], 'sat_third': (b.get('third') or [None, None])[1]}
    ic = first_edge(G, True)
    r['closed'] = ic is not None
    zmax_ok = 0.170
    # S2 접근 (닫기 전, 안 닫았으면 전체)
    end = ic if ic is not None else len(T)
    if red is not None and end > 0:
        d = np.hypot(P[:end, 0] - red[0], P[:end, 1] - red[1])
        okk = np.where((d <= 0.025) & (P[:end, 2] <= zmax_ok))[0]
        r['S2_min_d_mm'] = round(float(d.min()) * 1000, 1)
        r['S2_t_s'] = round(float(T[okk[0]]), 1) if len(okk) else None
        s2 = bool(len(okk))
    else:
        s2 = None
    # 닫는 순간
    if ic is not None:
        p = P[ic]
        yaw = math.radians(YAW[ic])
        hx = np.array([math.cos(yaw), math.sin(yaw)])
        hyv = np.array([-math.sin(yaw), math.cos(yaw)])
        r['close_t'] = round(float(T[ic]), 1)
        r['close_z_mm'] = round(float(p[2]) * 1000, 1)
        k0, k1 = max(0, ic - 2), min(len(T) - 1, ic + 2)
        r['close_v_mm_s'] = round(float(np.linalg.norm(P[k1] - P[k0]) / max(1e-3, T[k1] - T[k0])) * 1000, 1)
        if red is not None:
            dv = p[:2] - np.asarray(red, float)
            r['fb_mm'] = round(float(dv @ hx) * 1000, 1)
            r['fy_mm'] = round(float(dv @ hyv) * 1000, 1)
            r['d_red_mm'] = round(float(np.hypot(*dv)) * 1000, 1)
        others = [c for c in caps if not c.get('red') and red is not None
                  and math.hypot(c['xy'][0] - red[0], c['xy'][1] - red[1]) >= 0.025]
        if others and red is not None:
            do = min(math.hypot(p[0] - c['xy'][0], p[1] - c['xy'][1]) for c in others)
            r['d_other_mm'] = round(do * 1000, 1)
        k3 = min(len(T) - 1, int(np.searchsorted(T, T[ic] + 3.0)))
        r['w3_mm'] = round(float(W[k3]) * 1000, 1)
        # 들기
        z0 = p[2]
        hold = W >= 0.012
        lift30 = next((k for k in range(ic, len(T)) if hold[k] and P[k, 2] - z0 >= 0.002), None)   # 들기 시작(+2 mm)
        lift60 = next((k for k in range(ic, len(T)) if hold[k] and P[k, 2] - z0 >= 0.060), None)
        seg = range(k3, lift30 if lift30 is not None else min(len(T), k3 + 20))
        r['S5_min_w_mm'] = round(float(min([W[k] for k in seg], default=W[k3])) * 1000, 1)
        kk = [k for k in range(ic + 10, min(len(T), ic + 40))]
        r['w_hold_mm'] = round(float(np.median(W[kk])) * 1000, 1) if kk else None
        rise = max([P[k, 2] - z0 for k in range(ic, len(T)) if hold[k]], default=0.0)
        r['rise_mm'] = round(float(rise) * 1000, 1)
        if lift60 is not None:
            r['t_lift60_s'] = round(float(T[lift60] - T[ic]), 1)
            seg6 = range(ic, lift60 + 1)
            r['drift60_mm'] = round(float(max(np.hypot(P[k, 0] - p[0], P[k, 1] - p[1]) for k in seg6)) * 1000, 1)
            r['w_drop60_mm'] = round(float(W[k3] - min(W[k] for k in range(k3, lift60 + 1)) if lift60 >= k3 else 0) * 1000, 1)
        # 수평 이동 시작 높이 (잡은 자리에서 10 mm 벗어날 때 손 높이 − 닫은 높이)
        k10 = next((k for k in range(ic, len(T)) if hold[k] and math.hypot(P[k, 0] - p[0], P[k, 1] - p[1]) >= 0.010), None)
        r['clear_mm'] = round(float(P[k10, 2] - z0) * 1000, 1) if k10 is not None else None
        k40 = next((k for k in range(ic, len(T)) if hold[k] and P[k, 2] - z0 >= 0.060
                    and math.hypot(P[k, 0] - p[0], P[k, 1] - p[1]) >= 0.040), None)
        r['S7_t'] = None if k40 is None else round(float(T[k40]), 1)
        # 이송 · 꽂기 (쥐고 있을 때)
        io = first_edge(G, False)
        r['released'] = io is not None and io > ic
        stop = io if (io is not None and io > ic) else len(T)
        if hole is not None:
            cand = [k for k in range(ic, stop) if hold[k]]
            if cand:
                dd = [math.hypot(P[k, 0] - hole[0], P[k, 1] - hole[1]) for k in cand]
                j = int(np.argmin(dd))
                r['near_hole_mm'] = round(dd[j] * 1000, 1)
                near = [k for k, x in zip(cand, dd) if x <= 0.014]
                if near:
                    r['ins_above_mm'] = round(float(min(P[k, 2] for k in near) - Z_REL) * 1000, 1)
                k20 = next((k for k, x in zip(cand, dd) if x <= 0.020 and (lift60 is None or k >= lift60)), None)
                if k20 is not None and lift60 is not None:
                    r['t_transport_s'] = round(float(T[k20] - T[lift60]), 1)
        # 떨어뜨림 (들고 난 뒤 놓기 명령 전 폭 < 12 mm)
        if lift30 is not None:
            r['dropped'] = any(W[k] < 0.012 for k in range(lift30, stop))
    # ── 단계 판정 ──
    S = {}
    human_ok = r['result'] == 'success'
    # S1: VLA + 수식은 수식이 붙은 뚜껑(빨간 여부) 기록, 아니면 닫은 자리가 빨간 튜브에서 구멍 한 칸(37.2 mm) 안
    #     (11-11 다시 찾은 장면은 밝은 조명에서 가짜 뚜껑이 섞여 '다른 뚜껑' 거리는 판정에 안 씀)
    if (hy.get('grasp') or {}).get('target'):
        S['S1'] = bool(hy['grasp'].get('red'))          # 수식이 넘겨받아 붙은 뚜껑이 빨간 튜브인지
    elif r['closed']:
        # 다른 튜브로 간 증거가 없으면 통과 — 빈 자리로 간 것은 S2(접근)에서 셈.
        # 증거: 사진 검토 G1(11-05) / 수식 기록의 뚜껑 위치로 닫은 자리가 다른 뚜껑 25 mm 안·빨간 뚜껑 25 mm 밖
        S['S1'] = True
        if hy.get('caps_xy') and r.get('d_other_mm') is not None and r.get('d_red_mm') is not None:
            S['S1'] = not (r['d_other_mm'] <= 25 and r['d_red_mm'] > 25)
    else:
        S['S1'] = None
    S['S2'] = s2
    if r['closed'] and r.get('fb_mm') is not None:
        S['S3'] = (FB[0] * 1000 <= r['fb_mm'] <= FB[1] * 1000) and abs(r['fy_mm']) <= FY * 1000 \
            and abs(r['close_z_mm'] - Z_TRAIN * 1000) <= ZTOL * 1000
    elif not r['closed']:
        S['S3'] = False
    else:
        S['S3'] = None
    S['S4'] = bool(r['closed'] and 20 <= r['w3_mm'] <= 50)
    S['S5'] = bool(S['S4'] and r.get('S5_min_w_mm', 0) >= 12)
    S['S6'] = bool(S['S5'] and r.get('rise_mm', 0) >= 60)
    S['S7'] = bool(S['S6'] and r.get('S7_t') is not None)
    S['S8'] = bool(S['S7'] and (human_ok or (r.get('near_hole_mm') is not None and r['near_hole_mm'] <= 20)))
    S['S9'] = bool(human_ok)
    if human_ok:
        S = {k: True for k in S}
    # 뒤 단계가 통과면 앞 단계도 통과 (모르는 값 채움)
    order = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
    for i in range(len(order) - 1, 0, -1):
        if S.get(order[i]) is True and S.get(order[i - 1]) is None:
            S[order[i - 1]] = True
    # 사진 검토 결과(11-05) 우선 — stage_final/cause_final → 처음 막힌 단계
    if review:
        S = apply_review(S, review, r)
        r['review'] = review
    elif red is None and r['closed'] and S.get('S4') is False:
        S = {'S1': True, 'S2': True, 'S3': False, 'S4': False, 'S5': False, 'S6': False, 'S7': False, 'S8': False, 'S9': False}
        r['photo'] = '시작 장면에 튜브 위치 없음 — 3인칭·손목 사진상 빨간 튜브 근처에서 빈손 → 정렬(S3) 실패로 셈'
    elif red is None and S.get('S4'):
        for k in ('S1', 'S2', 'S3'):
            if S.get(k) is None:
                S[k] = True
        r['photo'] = '시작 장면에 튜브 위치 없음 — 쥐었으므로 S1–S3 통과로 셈(사진상 빨간 튜브)'
    S = monotone(S)
    if hy:
        g = hy.get('grasp') or {}
        pl = hy.get('place') or {}
        r['hy'] = {'grasp_handoff': bool(g), 'place_handoff': bool(pl), 'place_skip': hy.get('place_skip'),
                   'grasp_miss': bool(g.get('miss')), 'grip_mm': g.get('grip_mm'), 'z_src': g.get('z_src'),
                   'arrive_err_mm': g.get('arrive_err_mm'), 'settle_s': g.get('settle_s'),
                   'snap_mm': g.get('snap_mm'), 'place_snap_mm': pl.get('snap_mm'), 'trigger': pl.get('trigger'),
                   'align_err_mm': pl.get('align_err_mm'), 'insert_above_mm': (pl.get('insert') or {}).get('above_mm'),
                   'here': pl.get('here'),
                   'why': [e.get('why') for e in (hy.get('events') or []) if e.get('why')]}
    r['S'] = S
    first = None
    for k in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'):
        if S.get(k) is False:
            first = k
            break
    r['first_fail'] = first
    return r


ORDER = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']


def upto(fs):
    """fs 단계에서 처음 막힘: 앞은 통과, fs 실패, 뒤는 못 감(실패)"""
    return {k: (ORDER.index(k) < ORDER.index(fs)) for k in ORDER}


def apply_review(S, rv, r):
    """11-05 사진 검토(처음 막힌 단계·원인)로 맞춤. 잡기 위치 원인(G2–G5)은 잰 값(S2–S5)을 그대로 씀"""
    st, cz = rv['stage_final'], rv['cause_final']
    if st == 'release' and cz == 'R1':                 # 꽂았지만 안 놓음 → 꽂는 순간까지는 통과
        return {k: True for k in ORDER}
    if cz == 'G1':
        return upto('S1')
    if cz in ('G2', 'G3', 'G4', 'G5'):
        out = dict(S)
        for k in ('S6', 'S7', 'S8', 'S9'):
            out[k] = False
        if all(out.get(k) is not False for k in ('S1', 'S2', 'S3', 'S4', 'S5')):
            out['S5'] = False
        return out
    if st == 'grasp':                                  # 판단 못 함(튜브 위치 없음) — 사진상 빨간 튜브 근처에서 빈손·얹힘
        r['photo'] = '시작 장면에 튜브 위치 없음 — 사진상 빨간 튜브 근처에서 빈손 → 정렬(S3) 실패로 셈'
        return upto('S3')
    if cz in ('L1', 'L2', 'L3'):
        return upto('S6')
    if cz in ('A1', 'A2', 'A3'):
        return upto('S8')
    if st == 'insert':
        return upto('S9')
    return S


def monotone(S):
    """뒤 단계가 통과면 앞 단계도 통과 (닫는 순간 잰 위치가 기준 밖이어도 실제로 쥐었으면 통과)"""
    last = max([i for i, k in enumerate(ORDER) if S.get(k) is True], default=-1)
    for i in range(last):
        S[ORDER[i]] = True
    return S


def main():
    args = sys.argv[1:]
    opt = {}
    for k in ('--review', '--out'):
        if k in args:
            i = args.index(k)
            opt[k] = args[i + 1]
            del args[i:i + 2]
    paths = args
    root = Path(__file__).resolve().parent.parent
    rev = {}
    rp = Path(opt.get('--review', root / 'data' / 'fail_causes_T_1105_review.csv'))
    if rp.exists():
        rev = {x['trial']: x for x in csv.DictReader(open(rp, encoding='utf-8'))}
    out = []
    for pth in paths:
        D = json.load(open(pth))
        for t in D['trials']:
            e = exp_of(t)
            if e is None or t.get('result') == 'void':
                if e and t.get('result') == 'void':
                    out.append({'folder': t['folder'], 'exp': e, 'void': True, 'note': t.get('note')})
                continue
            nn = norm_note(t.get('note'))
            rv = rev.get(nn) if e in ('E1', 'E2') and nn else None
            try:
                r = score(t, rv)
            except Exception as ex:
                r = {'folder': t['folder'], 'err': str(ex)}
            r['exp'] = e
            r['tag'] = norm_note(t.get('note')) or {'rollout_20261112_105249': 'T05-hyb-3',
                                                     'rollout_20261111_123358': 'T09-mix-3'}.get(t['folder'])
            r['post_where'] = (t.get('judge_post') or {}).get('where')
            r['post_note'] = (t.get('judge_post') or {}).get('note')
            # 앞의 단계 판정(리뷰 없이) 따로 보관
            if rv:
                r['S_auto'] = score(t, None)['S']
            out.append(r)
    json.dump(out, open(opt.get('--out', root / 'data' / 'stage_runs.json'), 'w'), ensure_ascii=False, indent=1)
    for e in ('E1', 'E2', 'E3', 'E4'):
        R = [x for x in out if x.get('exp') == e and not x.get('void')]
        V = [x for x in out if x.get('exp') == e and x.get('void')]
        if not R:
            continue
        print(f'\n== {e}: {len(R)}판 (무효 {len(V)}) · 사람 판정', {k: sum(1 for x in R if x.get('result') == k) for k in ('success', 'partial', 'fail')})
        for k in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'):
            ok = sum(1 for x in R if all(x['S'].get(j) for j in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'][:['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'].index(k) + 1]))
            unk = sum(1 for x in R if x['S'].get(k) is None)
            print(f'  {k}: 누적 통과 {ok}/{len(R)}' + (f' (판단 못 함 {unk})' if unk else ''))
        from collections import Counter
        print('  처음 막힌 단계:', dict(Counter(x['first_fail'] for x in R)))


if __name__ == '__main__':
    main()
