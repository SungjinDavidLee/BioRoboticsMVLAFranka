#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_report.py — 실물 평가(배치 10개 × 3회) 결과 보고서 재료 만들기 (로봇 PC, 읽기만 — 로봇 안 움직임)

  python3 eval_report.py                      # ~/gello_data/medbio_redslot_v1 → ~/eval_report/ + ~/eval_report.zip

만드는 것
  images/layout_NN_start.jpg   배치마다 시작 장면 (3인칭 | 손목 장면 찾기)
  images/trial_N-K.jpg         시행마다 6장 (시작 3인칭 · 시작 손목 · 닫는 순간 손목 · 닫고 3초 뒤 3인칭 · 칸에 가장 가까이 · 끝 장면)
  data/trials.csv · layouts.csv · analysis.json   시행·배치별 측정값, 학습 데이터와의 거리, 실패 원인 후보
  figs/coverage_draft.png      학습 데이터 배치(회색)와 평가 배치(성공 수별 색) 위에서 본 그림
  README_draft.md              자동 초안 (최종본은 이 zip을 Claude에게 보내 다듬기)

읽는 것: <세션>/_rollouts/rollout_*/ (meta.json · data.npz · third/ · wrist/ · pre_scene.jpg · post_scene.jpg),
        <세션>/episode_*/ (meta.json의 자동 시연 정보 · wrist/000000.jpg), <세션>/auto_calib.json, ~/gello_tools/station.json
"""
import argparse
import csv
import datetime
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

TOOLS = Path.home() / 'gello_tools'
sys.path.insert(0, str(TOOLS))
from gello_to_lerobot import fk          # noqa: E402  수집·실행과 같은 순기구학 (fr3_link0 → fr3_hand_tcp)
import gello_scene as gsc                # noqa: E402  수집·실행과 같은 장면 검출

SLOT_M = 0.032          # 랙 구멍 간격
SAFETY = ('작업 범위', '기울어짐', '움직임 (한도', '역기구학', '관절 한계')   # gello_rollout 안전장치 중단 사유
RES_KO = {'success': '성공', 'partial': '부분', 'fail': '실패', 'void': '무효'}
RES_EN = {'success': 'SUCCESS', 'partial': 'PARTIAL', 'fail': 'FAIL', 'void': 'VOID'}


# ───────────────────────── 공통 ─────────────────────────
def rd(v, n=1):
    if v is None:
        return None
    try:
        if not np.isfinite(v):
            return None
    except TypeError:
        return v
    return round(float(v), n)


def mm(v):
    return None if v is None else rd(float(v) * 1000, 1)


def tcp(q):
    q = np.asarray(q, float)
    return np.full(3, np.nan) if not np.all(np.isfinite(q)) else fk(q)[:3, 3]


def ang_diff(a, b):
    """두 랙 방향(°, 180° 주기) 차이 0~90"""
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def load_colors():
    try:
        c = json.loads((TOOLS / 'station.json').read_text())
        return ((c.get('auto') or {}).get('colors')) or None
    except Exception:
        return None


def load_H(session):
    try:
        c = json.loads((session / 'auto_calib.json').read_text())
    except Exception:
        return None
    if c.get('H') is not None:
        return np.asarray(c['H'], float)
    pts = c.get('points') or []
    if len(pts) >= 4:
        Hm, _rms, _loo, _kind = gsc.fit_homography([p['px'] for p in pts], [p['xy'] for p in pts])
        return None if Hm is None else np.asarray(Hm, float)
    return None


def to_xy(Hm, px):
    if Hm is None or px is None:
        return None
    return [float(v) for v in gsc.apply_h(Hm, [list(px[:2])])[0]]


def scene_geo(img_path, Hm, colors, ring_px=None, tube_px=None):
    """홈 자세 손목 사진 1장 → 랙 2개(빈 랙 = 링이 있는 랙) 방향·중심, 뚜껑들 위치 (로봇 x,y m)"""
    out = {'ok': False}
    img = cv2.imread(str(img_path)) if img_path and Path(img_path).exists() else None
    if img is None:
        return out
    try:
        det = gsc.detect(img, colors)
    except Exception as e:
        out['err'] = str(e)
        return out
    out['ok'] = True
    racks = []
    for r in det.get('racks') or []:
        d = {'px': [r['cx'], r['cy']], 'ang_img': r['angle_deg'] % 180.0, 'len_px': r['len'], 'wid_px': r['wid']}
        if Hm is not None:
            d['xy'] = to_xy(Hm, [r['cx'], r['cy']])
            d['ang'] = math.degrees(gsc.axis_angle_robot(Hm, r['cx'], r['cy'], r['angle_deg'])) % 180.0
        racks.append(d)
    out['n_racks'] = len(racks)

    def nearest(px):
        if px is None or not racks:
            return None
        return int(np.argmin([math.hypot(r['px'][0] - px[0], r['px'][1] - px[1]) for r in racks]))
    i_dst = nearest(ring_px)
    i_src = nearest(tube_px)
    if i_src is not None and i_src == i_dst and len(racks) >= 2:
        i_src = [i for i in range(len(racks)) if i != i_dst][0]
    out['dst'] = racks[i_dst] if i_dst is not None else None
    out['src'] = racks[i_src] if i_src is not None and i_src != i_dst else None
    caps = []
    for i, c in enumerate(det.get('caps') or []):
        caps.append({'px': [c['px'], c['py']], 'xy': to_xy(Hm, [c['px'], c['py']]), 'red': i == det.get('red')})
    out['caps'] = caps
    return out


# ───────────────────────── 학습 데이터 ─────────────────────────
def train_set(session, Hm, colors):
    eps = []
    for d in sorted(session.glob('episode_*')):
        try:
            m = json.loads((d / 'meta.json').read_text())
        except Exception:
            continue
        if (m.get('quality') or {}).get('verdict') == 'bad':
            continue
        A = (m.get('start') or {}).get('auto') or {}
        if A.get('place') not in (None, 'slot'):
            continue
        e = {'name': d.name}
        if A.get('grasp_xyz'):
            e['tube'] = [float(v) for v in A['grasp_xyz'][:2]]
            e['grasp_z'] = float(A['grasp_xyz'][2])
        if A.get('drop_xyz'):
            e['rel'] = [float(v) for v in A['drop_xyz'][:3]]
        if A.get('drop_px'):
            e['ring_px'] = [float(v) for v in A['drop_px'][:2]]
            e['ring'] = to_xy(Hm, e['ring_px'])
        if 'tube' not in e:          # 사람 시연 등: 처음 닫힌 순간 = 잡은 곳, 마지막 = 놓은 곳
            try:
                z = np.load(d / 'data.npz')
                lg, qq = np.asarray(z['leader_grip'], float), np.asarray(z['q'], float)
                j = next((i for i in range(1, len(lg)) if lg[i - 1] > 0.5 and lg[i] < 0.5), None)
                o = next((i for i in range(len(lg) - 1, 0, -1) if lg[i - 1] < 0.5 and lg[i] > 0.5), None)
                if j is not None:
                    p = tcp(qq[j]); e['tube'], e['grasp_z'] = p[:2].tolist(), float(p[2])
                if o is not None:
                    e['rel'] = tcp(qq[o]).tolist()
            except Exception:
                pass
        if 'tube' not in e:
            continue
        tube_px = None
        g = scene_geo(d / 'wrist' / '000000.jpg', Hm, colors, e.get('ring_px'), None)
        if g.get('ok'):
            reds = [c for c in g['caps'] if c['red']]
            if reds:
                tube_px = reds[0]['px']
                g = scene_geo(d / 'wrist' / '000000.jpg', Hm, colors, e.get('ring_px'), tube_px)
            e['src_ang'] = rd((g.get('src') or {}).get('ang'), 1)
            e['dst_ang'] = rd((g.get('dst') or {}).get('ang'), 1)
            e['n_caps'] = len(g['caps'])
        if 'ring' not in e and e.get('rel'):
            e['ring'] = e['rel'][:2]
        eps.append(e)
    return eps


# ───────────────────────── 실행 1회 ─────────────────────────
def trial(F, Hm, colors, z_rel_ref, z_grasp_ref):
    m = json.loads((F / 'meta.json').read_text())
    note = (m.get('note') or '').strip()
    mm_ = re.search(r'(\d+)\s*-\s*(\d+)', note)
    r = {'folder': F.name, 'note': note, 'layout': int(mm_.group(1)) if mm_ else 0, 'k': int(mm_.group(2)) if mm_ else 0,
         'result': m.get('result'), 'result_auto': m.get('result_auto'), 'reason': m.get('reason'),
         'duration_s': m.get('duration_s'), 'latency_ms': rd(m.get('total_ms_median'), 0), 'chunks': m.get('chunks'),
         'started': (m.get('started_at') or '')[11:16], 'mode': m.get('mode') or (m.get('cfg') or {}).get('mode'),
         'exec_steps': m.get('exec_steps'), 'ckpt': str(m.get('ckpt') or '').split('checkpoints/')[-1].split('/')[0]}
    pre = m.get('pre') or {}
    J = m.get('judge') or {}
    r['judge_reason'] = J.get('reason')
    r['stages'] = {s.get('key'): {'ok': s.get('ok'), 'note': s.get('note')} for s in J.get('stages', [])}
    post = m.get('post_scene') or {}
    r['post_where'] = (J.get('post') or {}).get('where')
    r['post_note'] = (J.get('post') or {}).get('note')
    tube = pre.get('grasp_xy') or pre.get('tube_xy')
    hole_raw, hole_plan = pre.get('hole_xy'), pre.get('hole_xy_plan')
    r.update(tube_xy=tube, tube_px=pre.get('tube_px'), ring_px=pre.get('ring_px'), hole_raw=hole_raw, hole_plan=hole_plan,
             post_tube_xy=post.get('tube_xy'))
    g = scene_geo(F / 'wrist' / '000000.jpg', Hm, colors, pre.get('ring_px'), pre.get('tube_px'))
    r['src_ang'] = rd((g.get('src') or {}).get('ang'), 1)
    r['dst_ang'] = rd((g.get('dst') or {}).get('ang'), 1)
    r['dst_rack_xy'] = (g.get('dst') or {}).get('xy')
    r['src_rack_xy'] = (g.get('src') or {}).get('xy')
    r['caps'] = g.get('caps') or []
    z = np.load(F / 'data.npz')
    t, Q, Cm, W, G = (np.asarray(z[k], float) for k in ('t', 'q', 'cmd', 'grip_w', 'grip_cmd'))
    n = len(t)
    P = np.array([tcp(q) for q in Q])
    PC = np.array([tcp(q) for q in Cm])
    r['n'] = n
    closes = [i for i in range(1, n) if G[i - 1] > 0.5 and G[i] < 0.5]
    opens = [i for i in range(1, n) if G[i - 1] < 0.5 and G[i] > 0.5]
    r['n_close'], r['n_open'] = len(closes), len(opens)
    r['idx'] = {}
    if not closes:
        why = str(m.get('reason') or '')
        r['cause'] = '안전장치 중단' if any(k in why for k in SAFETY) else '쥐기 시도 없음'
        r['detail'] = why
        return r, t, P
    ic = closes[0]
    r['idx']['close'] = ic
    r['t_close'] = rd(t[ic] - t[0], 1)
    pc = P[ic]
    r['close_xyz'] = [rd(v, 4) for v in pc]
    k3 = min(n - 1, int(np.searchsorted(t, t[ic] + 3.0)))
    r['idx']['close3'] = k3
    r['w3_mm'] = mm(W[k3])
    r['grasp_ok'] = bool(0.020 <= W[k3] <= 0.050)
    if tube:
        e = pc[:2] - np.asarray(tube[:2])
        r['grasp_err'] = [mm(e[0]), mm(e[1])]
        r['grasp_err_mm'] = mm(np.hypot(*e))
    if z_grasp_ref is not None:
        r['close_dz_mm'] = mm(pc[2] - z_grasp_ref)
    v = np.linalg.norm(P[ic] - P[max(0, ic - 2)]) / max(t[ic] - t[max(0, ic - 2)], 1e-3)
    r['close_v_mm_s'] = mm(v)
    # 닫은 자리에서 가장 가까운 뚜껑 (시작 장면)
    cx = [(c, float(np.hypot(*(pc[:2] - np.asarray(c['xy']))))) for c in r['caps'] if c.get('xy')]
    if cx:
        c0, d0 = min(cx, key=lambda z_: z_[1])
        r['near_cap_red'] = bool(c0['red'])
        r['near_cap_mm'] = mm(d0)
        red = [d for c, d in cx if c['red']]
        r['red_cap_mm'] = mm(red[0]) if red else None
    # 쥔 동안 (닫은 뒤 ~ 처음 연 순간, 손가락 폭 15mm 넘게 남아 있는 칸)
    io = next((i for i in opens if i > ic), None)
    if io is not None:
        r['idx']['open'] = io
        r['t_open'] = rd(t[io] - t[0], 1)
        r['rel_xyz'] = [rd(v, 4) for v in P[io]]
    end = io if io is not None else n - 1
    hold = [i for i in range(ic, end + 1) if W[i] > 0.015 and np.all(np.isfinite(P[i]))]
    r['lift_mm'] = mm(max(P[i][2] for i in hold) - pc[2]) if hold else 0.0
    r['lift_ok'] = bool(r['grasp_ok'] and (r['lift_mm'] or 0) >= 30)
    for key, h in (('raw', hole_raw), ('plan', hole_plan)):
        if not h or not hold:
            continue
        h = np.asarray(h[:2])
        d = np.array([np.hypot(*(P[i][:2] - h)) for i in hold])
        k = int(np.argmin(d))
        r[f'near_{key}_mm'] = mm(d[k])
        r[f'near_{key}_vec'] = [mm(P[hold[k]][0] - h[0]), mm(P[hold[k]][1] - h[1])]
        r[f'near_{key}_idx'] = hold[k]
        if io is not None:
            r[f'rel_err_{key}'] = [mm(P[io][0] - h[0]), mm(P[io][1] - h[1])]
            r[f'rel_err_{key}_mm'] = mm(np.hypot(*(P[io][:2] - h)))
    # 꽂기 시도 수·가장 낮은 높이·막힘 (기준 구멍: 계획 → 없으면 영상)
    h = hole_plan or hole_raw
    if h and hold and z_rel_ref is not None:
        h = np.asarray(h[:2])
        base = [PC[i][2] - P[i][2] for i in hold[:5] if np.all(np.isfinite(PC[i]))]
        b0 = float(np.median(base)) if base else 0.0
        # 꽂기 시도 = 구멍 30mm 안에서 '학습 놓은 높이 + 35mm' 아래로 내려간 횟수. 내려간 뒤 가장 낮은 곳보다 10mm 넘게
        # 다시 올라갔다가 또 내려가면 다시 시도로 셈 (테두리에 막혀 튕기듯 올라갔다 다시 꽂은 경우)
        thr = z_rel_ref + 0.035
        phase, att, low, peak, zmin, blk = 'idle', 0, None, None, None, 0.0
        for i in hold:
            dxy = float(np.hypot(*(P[i][:2] - h)))
            if dxy > 0.030:
                continue
            z_ = P[i][2]
            zmin = z_ if zmin is None else min(zmin, z_)
            if np.all(np.isfinite(PC[i])):
                blk = max(blk, (P[i][2] - PC[i][2]) - (-b0))        # 실제가 명령보다 얼마나 위에 막혀 있었나
            if phase == 'idle' and z_ < thr:
                att, phase, low = 1, 'down', z_
            elif phase == 'down':
                low = min(low, z_)
                if z_ > low + 0.010:
                    phase, peak = 'up', z_
            elif phase == 'up':
                peak = max(peak, z_)
                if z_ < peak - 0.005 and z_ < thr:
                    att, phase, low = att + 1, 'down', z_
        r['attempts'] = att
        r['zmin_dz_mm'] = mm(zmin - z_rel_ref) if zmin is not None else None
        r['block_mm'] = mm(blk) if zmin is not None else None
        # 랙 방향 성분 (옆 칸 쪽이었나)
        if r.get('dst_ang') is not None and r.get('near_plan_idx', r.get('near_raw_idx')) is not None:
            ii = r.get('near_plan_idx', r.get('near_raw_idx'))
            a = math.radians(r['dst_ang'])
            dv = P[ii][:2] - h
            r['near_along_mm'] = mm(dv @ np.array([math.cos(a), math.sin(a)]))
            r['near_across_mm'] = mm(dv @ np.array([-math.sin(a), math.cos(a)]))
    r['cause'], r['detail'] = cause(r, m)
    return r, t, P


def cause(r, m):
    res = r.get('result')
    why = str(m.get('reason') or '')
    safety = any(k in why for k in SAFETY)
    if res == 'void':
        return '무효', (r.get('note') or why)
    if res == 'success':
        bits = []
        if (r.get('attempts') or 0) > 1:
            bits.append(f"꽂기 시도 {r['attempts']}번 (다시 시도 후 성공)")
        if (r.get('block_mm') or 0) >= 6:
            bits.append(f"테두리에 막힘 최대 {r['block_mm']:.0f}mm → 미끄러져 들어감")
        if r.get('n_close', 0) > 1:
            bits.append(f"쥐기 {r['n_close']}번")
        return '성공', ' · '.join(bits) or '한 번에 꽂음'
    e = r.get('grasp_err') or [None, None]
    pos = f"닫은 자리가 빨간 튜브 잡을 자리에서 {r.get('grasp_err_mm')}mm (앞뒤 {e[0]:+.0f}, 좌우 {e[1]:+.0f})" \
        if r.get('grasp_err_mm') is not None else ''
    if r.get('near_cap_red') is False and (r.get('near_cap_mm') or 99) < 20 and (r.get('red_cap_mm') or 0) > 25:
        return '다른 튜브로 감', f"빨간 튜브가 아닌 뚜껑에서 {r['near_cap_mm']:.0f}mm에서 닫음 · 쥔 폭 {r.get('w3_mm')}mm · " + pos \
            + (f' · 끝: {why}' if safety else '')
    if not r.get('grasp_ok'):
        if (r.get('w3_mm') or 0) > 50:
            return '뚜껑 옆에 걸림', f"쥔 폭 {r['w3_mm']:.0f}mm · " + pos
        hi = f" · 학습보다 {r['close_dz_mm']:+.0f}mm 높이에서 닫음" if r.get('close_dz_mm') is not None and abs(r['close_dz_mm']) >= 10 else ''
        return '헛잡음', pos + hi + (f' · 끝: {why}' if safety else '')
    if not r.get('lift_ok'):
        return '못 들어 올림', f"쥔 폭 {r.get('w3_mm')}mm, 최대 {r.get('lift_mm')}mm 올라감" + (f' · 끝: {why}' if safety else '')
    near = r.get('near_plan_mm', r.get('near_raw_mm'))
    along = r.get('near_along_mm')
    side = ''
    if along is not None and abs(along) > 0.6 * SLOT_M * 1000:
        side = f" · 랙 방향으로 {along:+.0f}mm (옆 칸 쪽)"
    elif along is not None:
        side = f" · 랙 방향 {along:+.0f}mm, 수직 {r.get('near_across_mm'):+.0f}mm"
    zinfo = f" · 가장 낮게 학습 놓은 높이 {r['zmin_dz_mm']:+.0f}mm" if r.get('zmin_dz_mm') is not None else ''
    blk = f" · 테두리에 막힘 최대 {r['block_mm']:.0f}mm" if (r.get('block_mm') or 0) >= 6 else ''
    att = f" · 꽂기 시도 {r['attempts']}번" if r.get('attempts') else ''
    if near is not None and near > 25:
        return '칸까지 못 감', f"쥔 채 칸에 가장 가까이 {near:.0f}mm" + side + (f' · 끝: {why}' if safety else '')
    if r.get('post_where') == 'hole':
        return '테두리에 얹힘', '끝 장면은 구멍 위인데 덜 들어감' + zinfo + blk
    if r.get('idx', {}).get('open') is not None:
        pt = f" (끝 장면 튜브가 구멍에서 {r.get('post_note')})" if r.get('post_note') else ''
        return '칸 옆에 놓음', f"놓은 자리 구멍에서 {r.get('rel_err_plan_mm', r.get('rel_err_raw_mm'))}mm" + side + zinfo + blk + pt
    return '칸 위에서 못 꽂음', f"쥔 채 가장 가까이 {near}mm" + side + zinfo + blk + att + f' · 끝: {why}'


# ───────────────────────── 사진 ─────────────────────────
def frame(F, cam, i):
    if i is None:
        return None
    p = F / cam / f'{int(i):06d}.jpg'
    if not p.exists():
        fs = sorted((F / cam).glob('*.jpg'))
        if not fs:
            return None
        p = fs[min(int(i), len(fs) - 1)] if i >= 0 else fs[-1]
    return cv2.imread(str(p))


def label(img, txt, w=400, h=225):
    if img is None:
        img = np.full((h, w, 3), 235, np.uint8)
        cv2.putText(img, 'no image', (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(img, (0, 0), (tw + 10, th + 10), (30, 30, 30), -1)
    cv2.putText(img, txt, (5, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def trial_sheet(F, r, t, out):
    ix = r.get('idx', {})
    ic, k3, io = ix.get('close'), ix.get('close3'), ix.get('open')
    kn = r.get('near_plan_idx', r.get('near_raw_idx'))
    key = io if io is not None else kn
    last = len(sorted((F / 'third').glob('*.jpg'))) - 1
    ts = (lambda i: f' t={t[i] - t[0]:.1f}s' if i is not None and i < len(t) else '')
    post = cv2.imread(str(F / 'post_scene.jpg')) if (F / 'post_scene.jpg').exists() else None
    tiles = [label(frame(F, 'third', 0), '1 start (3rd person)'),
             label(cv2.imread(str(F / 'pre_scene.jpg')) if (F / 'pre_scene.jpg').exists() else frame(F, 'wrist', 0),
                   '2 start (wrist, detection)'),
             label(frame(F, 'wrist', ic), '3 gripper closes (wrist)' + ts(ic)),
             label(frame(F, 'third', k3), '4 close + 3 s (3rd)' + ts(k3)),
             label(frame(F, 'third', key), ('5 release (3rd)' if io is not None else '5 nearest to hole (3rd)') + ts(key)
                   if key is not None else '5 (no hold)'),
             label(post if post is not None else frame(F, 'third', last), '6 end scene' + (' (wrist, home)' if post is not None else ' (3rd, last)'))]
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    head = np.full((34, sheet.shape[1], 3), 250, np.uint8)
    txt = f"layout {r['layout']}-{r['k']}   {RES_EN.get(r.get('result'), r.get('result'))}   {r['folder']}"
    cv2.putText(head, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 82])


def layout_sheet(F, L, out, near_ep=None, near_txt=''):
    a = frame(F, 'third', 0)
    b = cv2.imread(str(F / 'pre_scene.jpg')) if (F / 'pre_scene.jpg').exists() else frame(F, 'wrist', 0)
    tiles = [label(a, f'layout {L} - start (3rd person)', 640, 360), label(b, f'layout {L} - start (wrist at home, detection)', 640, 360)]
    if near_ep is not None:
        tiles.append(label(cv2.imread(str(near_ep / 'wrist' / '000000.jpg')), f'nearest training: {near_ep.name} {near_txt}', 640, 360))
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])


def grid(paths, out, cols=10, w=192, h=108, names=None):
    tiles = []
    for i, p in enumerate(paths):
        im = cv2.imread(str(p)) if p and Path(p).exists() else None
        tiles.append(label(im, (names[i] if names else Path(p).parent.parent.name[-3:]), w, h))
    while len(tiles) % cols:
        tiles.append(np.full((h, w, 3), 255, np.uint8))
    rows_ = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    if rows_:
        cv2.imwrite(str(out), np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 80])


class Tee:
    def __init__(self, *fs):
        self.fs = fs

    def write(self, x):
        for f in self.fs:
            f.write(x)

    def flush(self):
        for f in self.fs:
            f.flush()


def coverage_fig(train, lays, out):
    """위에서 본 그림: x(로봇 앞) 위쪽, y(로봇 왼쪽) 왼쪽. 회색 = 학습, 색 = 평가 배치(성공 수)"""
    W_, H_ = 1000, 820
    x0, x1, y0, y1 = 0.36, 0.74, -0.30, 0.36
    img = np.full((H_, W_, 3), 255, np.uint8)

    def P(xy):
        return (int(60 + (y1 - xy[1]) / (y1 - y0) * (W_ - 120)), int(50 + (x1 - xy[0]) / (x1 - x0) * (H_ - 120)))
    for gx in np.arange(0.40, 0.75, 0.05):
        cv2.line(img, P([gx, y0]), P([gx, y1]), (238, 238, 238), 1)
        cv2.putText(img, f'x {gx:.2f}', (P([gx, y1])[0] - 50, P([gx, y1])[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
    for gy in np.arange(-0.30, 0.36, 0.05):
        cv2.line(img, P([x0, gy]), P([x1, gy]), (238, 238, 238), 1)
        cv2.putText(img, f'{gy:+.2f}', (P([x0, gy])[0] - 18, P([x0, gy])[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
    for e in train:
        if e.get('tube'):
            cv2.circle(img, P(e['tube']), 4, (175, 175, 175), -1)
        if e.get('ring'):
            p = P(e['ring'])
            cv2.rectangle(img, (p[0] - 3, p[1] - 3), (p[0] + 3, p[1] + 3), (175, 175, 175), 1)
    col = {0: (60, 60, 220), 1: (40, 150, 240), 2: (60, 190, 120), 3: (40, 130, 20)}
    for L in lays:
        c = col.get(L['n_success'], (90, 90, 90))
        if L.get('tube') and (L.get('ring_xy') or L.get('hole')):
            cv2.arrowedLine(img, P(L['tube']), P(L.get('ring_xy') or L['hole']), c, 1, cv2.LINE_AA, tipLength=0.03)
        if L.get('tube'):
            cv2.circle(img, P(L['tube']), 9, c, -1)
            cv2.putText(img, str(L['layout']), (P(L['tube'])[0] + 10, P(L['tube'])[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        hp = L.get('ring_xy') or L.get('hole')
        if hp:
            p = P(hp)
            cv2.rectangle(img, (p[0] - 7, p[1] - 7), (p[0] + 7, p[1] + 7), c, 2)
    lg = ['gray dot: training tube (grasp)', 'gray square: training hole (ring)',
          'circle: eval tube, square: eval hole, arrow tube->hole', 'color = successes of 3: red 0, orange 1, green 2, dark green 3']
    for i, s in enumerate(lg):
        cv2.putText(img, s, (W_ - 470, 60 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, 'top view (robot base below), x forward = up, y left = left [m]', (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.imwrite(str(out), img)


# ───────────────────────── 통계 ─────────────────────────
def fit_line(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) < 1e-6:
        return None
    A = np.column_stack([x, np.ones_like(x)])
    (b, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ np.array([b, a])
    r2 = 1 - np.sum((y - yh) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return {'slope': rd(b, 2), 'intercept_m': rd(a, 4), 'r2': rd(r2, 2), 'n': int(len(x))}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 4:
        return None

    def rank(v):
        o = np.argsort(v, kind='mergesort')
        r_ = np.empty(len(v)); r_[o] = np.arange(len(v), dtype=float)
        for u in np.unique(v):               # 같은 값은 평균 순위
            w = v == u
            r_[w] = r_[w].mean()
        return r_
    ra, rb = rank(a), rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return rd(float(np.corrcoef(ra, rb)[0, 1]), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='~/gello_data/medbio_redslot_v1')
    ap.add_argument('--dir', default='', help='실행 기록 폴더 (기본 <세션>/_rollouts)')
    ap.add_argument('--extra', default='~/gello_data/medbio_redslot_v1_rollouts_fixed_extra',
                    help='고정 배치 예비 시행 중 따로 옮긴 기록 (있으면 반복 정밀도 비교에만 씀)')
    ap.add_argument('--out', default='~/eval_report')
    a = ap.parse_args()
    S = Path(a.session).expanduser()
    R = Path(a.dir).expanduser() if a.dir else S / '_rollouts'
    O = Path(a.out).expanduser()
    if O.exists():
        shutil.rmtree(O)
    for sub in ('images', 'data', 'figs'):
        (O / sub).mkdir(parents=True)
    sys.stdout = Tee(sys.__stdout__, open(O / 'data' / 'summary.txt', 'w', encoding='utf-8'))
    Hm = load_H(S)
    colors = load_colors()
    print(f'■ 세션 {S} · 실행 기록 {R} · 보정 H {"있음" if Hm is not None else "없음"}')

    train = train_set(S, Hm, colors)
    zr = [e['rel'][2] for e in train if e.get('rel')]
    zg = [e['grasp_z'] for e in train if e.get('grasp_z') is not None]
    z_rel_ref = float(np.median(zr)) if zr else None
    z_grasp_ref = float(np.median(zg)) if zg else None
    print(f'■ 학습 시행 {len(train)}개 · 놓은 높이 중간 {mm(z_rel_ref)}mm · 잡은 높이 중간 {mm(z_grasp_ref)}mm')

    rows, sheets = [], []
    for F in sorted(R.glob('rollout_*')):
        if not ((F / 'meta.json').exists() and (F / 'data.npz').exists()):
            continue
        try:
            r, t, _P = trial(F, Hm, colors, z_rel_ref, z_grasp_ref)
        except Exception as e:
            print(f'  ⚠ {F.name} 읽기 실패: {e}')
            continue
        r['set'] = 'eval'
        rows.append(r)
        sheets.append((F, r, t))
    extra = []
    X = Path(a.extra).expanduser()
    if X.exists():
        for F in sorted(X.glob('rollout_*')):
            try:
                r, t, _P = trial(F, Hm, colors, z_rel_ref, z_grasp_ref)
                r['set'] = 'fixed_extra'
                extra.append(r)
            except Exception:
                pass
    valid = [r for r in rows if r['result'] in ('success', 'partial', 'fail')]
    print(f'■ 평가 기록 {len(rows)}개 (유효 {len(valid)}) · 고정 배치 추가분 {len(extra)}개')

    # 사진
    for F, r, t in sheets:
        nm = (f"trial_{r['layout']:02d}-{r['k']}" if r['layout'] else f"trial_{r['folder']}") + ('_void' if r['result'] == 'void' else '')
        if (O / 'images' / f'{nm}.jpg').exists():          # 같은 메모가 두 번 (다시 한 판 등)
            nm += '_' + r['folder'][-6:]
        r['img'] = f'images/{nm}.jpg'
        trial_sheet(F, r, t, O / 'images' / f'{nm}.jpg')
    # 배치별
    lay = {}
    for r in valid:
        lay.setdefault(r['layout'], []).append(r)
    T_tube = np.array([e['tube'] for e in train if e.get('tube')]) if train else np.zeros((0, 2))
    T_ring = np.array([e['ring'] if e.get('ring') else [np.nan, np.nan] for e in train]) if train else np.zeros((0, 2))
    T_tube_all = np.array([e['tube'] for e in train]) if train else np.zeros((0, 2))
    T_src = np.array([e.get('src_ang') if e.get('src_ang') is not None else np.nan for e in train], float)
    T_dst = np.array([e.get('dst_ang') if e.get('dst_ang') is not None else np.nan for e in train], float)
    t_mean = T_tube.mean(axis=0) if len(T_tube) else None
    # 학습 시행끼리의 가장 가까운 이웃 거리 (전형적인 학습 데이터 간격)
    nn_train = []
    for i in range(len(train)):
        d = np.hypot(*(T_tube_all - T_tube_all[i]).T) ** 2 + np.nan_to_num(np.hypot(*(T_ring - T_ring[i]).T) ** 2, nan=1.0)
        d[i] = np.inf
        nn_train.append(float(np.sqrt(d.min())))
    nn_train_med = float(np.median(nn_train)) if nn_train else None

    L_rows = []
    prev_tube = None
    for L in sorted(lay):
        rs = sorted(lay[L], key=lambda x: x['k'])
        first = min(rs, key=lambda x: x['k'])
        Lr = {'layout': L, 'n': len(rs), 'n_success': sum(r['result'] == 'success' for r in rs),
              'results': [r['result'] for r in rs], 'causes': [r.get('cause') for r in rs]}

        def med(key):
            v = [r[key] for r in rs if r.get(key)]
            return np.median(np.array(v, float), axis=0).tolist() if v else None
        Lr['tube'] = med('tube_xy')
        Lr['hole_raw'] = med('hole_raw')
        Lr['hole_plan'] = med('hole_plan')
        Lr['hole'] = Lr['hole_plan'] or Lr['hole_raw']
        Lr['ring_xy'] = to_xy(Hm, med('ring_px')) if med('ring_px') else None
        Lr['src_ang'] = first.get('src_ang')
        Lr['dst_ang'] = first.get('dst_ang')
        Lr['n_caps'] = len(first.get('caps') or [])
        # 같은 배치 3회의 닫은 자리 흩어짐 vs 평균 어긋남 (계통 오차인가 우연인가)
        cl = np.array([r['close_xyz'][:2] for r in rs if r.get('close_xyz')], float)
        if len(cl) and Lr['tube']:
            mean_err = cl.mean(axis=0) - np.asarray(Lr['tube'])
            Lr['close_mean_err'] = [mm(mean_err[0]), mm(mean_err[1])]
            Lr['close_mean_err_mm'] = mm(np.hypot(*mean_err))
            Lr['close_spread_mm'] = mm(np.sqrt(np.mean(np.sum((cl - cl.mean(axis=0)) ** 2, axis=1)))) if len(cl) > 1 else None
            if t_mean is not None:
                dvec = t_mean - np.asarray(Lr['tube'])
                Lr['pull_to_train_mean'] = rd(float(mean_err @ dvec / max(dvec @ dvec, 1e-9)), 2)
                Lr['tube_to_train_mean_mm'] = mm(np.hypot(*dvec))
            if prev_tube is not None:
                dvec = np.asarray(prev_tube) - np.asarray(Lr['tube'])
                if np.hypot(*dvec) > 0.01:
                    Lr['pull_to_prev_layout'] = rd(float(mean_err @ dvec / max(dvec @ dvec, 1e-9)), 2)
        prev_tube = Lr['tube'] or prev_tube
        # 학습 데이터와의 거리
        if len(T_tube) and Lr['tube']:
            dt = np.hypot(*(T_tube_all - np.asarray(Lr['tube'])).T)
            Lr['nn_tube_mm'] = mm(dt.min())
            Lr['n_tube_30'] = int(np.sum(dt < 0.03))
            if Lr.get('ring_xy') is not None:
                dh = np.hypot(*(T_ring - np.asarray(Lr['ring_xy'])).T)
                Lr['nn_hole_mm'] = mm(np.nanmin(dh))
                Lr['n_hole_30'] = int(np.nansum(dh < 0.03))
                dj = np.sqrt(dt ** 2 + np.nan_to_num(dh, nan=1.0) ** 2)
                k = int(np.argmin(dj))
                Lr['nn_joint_mm'] = mm(dj[k])
                Lr['nn_joint_ep'] = train[k]['name']
                Lr['n_joint_40'] = int(np.sum((dt < 0.04) & (np.nan_to_num(dh, nan=1.0) < 0.04)))
                if nn_train_med:
                    Lr['nn_joint_vs_train'] = rd(dj[k] / nn_train_med, 2)
            for key, TA in (('src', T_src), ('dst', T_dst)):
                if Lr.get(f'{key}_ang') is not None and np.isfinite(TA).any():
                    da = np.array([ang_diff(Lr[f'{key}_ang'], x) if np.isfinite(x) else np.nan for x in TA])
                    Lr[f'nn_{key}_ang'] = rd(np.nanmin(da), 1)
                    Lr[f'n_{key}_ang_15'] = int(np.nansum(da <= 15))
        # 성공 시행으로 본 판정 기준점 오차 (놓은 손 = 실제 구멍)
        for key in ('raw', 'plan'):
            v = [r.get(f'rel_err_{key}') for r in rs if r['result'] == 'success' and r.get(f'rel_err_{key}')]
            if v:
                Lr[f'ref_err_{key}'] = np.mean(np.array(v, float), axis=0).round(1).tolist()
        L_rows.append(Lr)
        layout_sheet(R / first['folder'], L, O / 'images' / f'layout_{L:02d}_start.jpg',
                     S / Lr['nn_joint_ep'] if Lr.get('nn_joint_ep') else None,
                     f"(tube {Lr.get('nn_tube_mm')} / hole {Lr.get('nn_hole_mm')} mm)" if Lr.get('nn_joint_ep') else '')

    # ── 요약 통계 ──
    A = {'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'session': str(S), 'n_train': len(train),
         'z_rel_ref_mm': mm(z_rel_ref), 'z_grasp_ref_mm': mm(z_grasp_ref), 'nn_train_median_mm': mm(nn_train_med),
         'train_tube_mean': None if t_mean is None else [rd(v, 4) for v in t_mean]}
    if len(T_tube):
        A['train_tube_range'] = {'x': [rd(T_tube[:, 0].min(), 3), rd(T_tube[:, 0].max(), 3)],
                                 'y': [rd(T_tube[:, 1].min(), 3), rd(T_tube[:, 1].max(), 3)]}
    for key, TA in (('src', T_src), ('dst', T_dst)):
        ok = TA[np.isfinite(TA)]
        if len(ok):
            A[f'train_{key}_ang_hist30'] = np.histogram(ok % 180, bins=6, range=(0, 180))[0].tolist()
    # 잡는 자리가 튜브를 얼마나 따라가나 (배치 평균 닫은 자리 vs 튜브)
    pts = [(Lr['tube'], np.asarray(Lr['tube']) + np.asarray(Lr['close_mean_err']) / 1000)
           for Lr in L_rows if Lr.get('tube') and Lr.get('close_mean_err')]
    if len(pts) >= 3:
        tt = np.array([p[0] for p in pts]); cc = np.array([p[1] for p in pts])
        A['track_x'] = fit_line(tt[:, 0], cc[:, 0])
        A['track_y'] = fit_line(tt[:, 1], cc[:, 1])
    # 성공 수와 학습 데이터 거리의 관계
    for key in ('nn_tube_mm', 'nn_hole_mm', 'nn_joint_mm', 'nn_src_ang', 'nn_dst_ang', 'tube_to_train_mean_mm', 'close_mean_err_mm'):
        A[f'spearman_success_vs_{key}'] = spearman([Lr['n_success'] for Lr in L_rows], [Lr.get(key, np.nan) if Lr.get(key) is not None else np.nan for Lr in L_rows])
    good = [Lr for Lr in L_rows if Lr['n_success'] > 0]
    bad = [Lr for Lr in L_rows if Lr['n_success'] == 0]
    for key in ('nn_tube_mm', 'nn_hole_mm', 'nn_joint_mm', 'n_joint_40', 'nn_src_ang', 'nn_dst_ang', 'close_mean_err_mm', 'close_spread_mm'):
        for nm, grp in (('succ', good), ('zero', bad)):
            v = [Lr[key] for Lr in grp if Lr.get(key) is not None]
            A[f'{nm}_median_{key}'] = rd(np.median(v), 1) if v else None
    # 같은 배치 반복 (고정 배치 8회: 배치 1의 3회 + 따로 옮긴 5회)
    fixed = [r for r in valid if r['layout'] == 1] + [r for r in extra if r.get('result') in ('success', 'partial', 'fail')]
    if fixed:
        cl = np.array([r['close_xyz'][:2] for r in fixed if r.get('close_xyz')], float)
        A['fixed'] = {'n': len(fixed), 'n_success': sum(r['result'] == 'success' for r in fixed),
                      'close_spread_mm': mm(np.sqrt(np.mean(np.sum((cl - cl.mean(axis=0)) ** 2, axis=1)))) if len(cl) > 1 else None,
                      'grasp_err_med_mm': rd(np.median([r['grasp_err_mm'] for r in fixed if r.get('grasp_err_mm') is not None]), 1)
                      if any(r.get('grasp_err_mm') is not None for r in fixed) else None}
    ge = [r['grasp_err_mm'] for r in valid if r.get('grasp_err_mm') is not None]
    A['grasp_err_median_all_mm'] = rd(np.median(ge), 1) if ge else None
    A['grasp_err_median_ok_mm'] = rd(np.median([r['grasp_err_mm'] for r in valid if r.get('grasp_ok') and r.get('grasp_err_mm') is not None]), 1) \
        if any(r.get('grasp_ok') and r.get('grasp_err_mm') is not None for r in valid) else None
    A['grasp_err_median_miss_mm'] = rd(np.median([r['grasp_err_mm'] for r in valid if not r.get('grasp_ok') and r.get('grasp_err_mm') is not None]), 1) \
        if any((not r.get('grasp_ok')) and r.get('grasp_err_mm') is not None for r in valid) else None
    A['causes'] = {}
    for r in valid:
        A['causes'][r['cause']] = A['causes'].get(r['cause'], 0) + 1

    # ── 저장 ──
    keep = lambda r: {k: v for k, v in r.items() if k not in ('caps',)}
    (O / 'data' / 'analysis.json').write_text(json.dumps(
        {'summary': A, 'layouts': L_rows, 'trials': [keep(r) for r in rows], 'fixed_extra': [keep(r) for r in extra],
         'train': train}, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    tcols = ['layout', 'k', 'result', 'result_auto', 'cause', 'detail', 'note', 't_close', 'w3_mm', 'grasp_err_mm', 'grasp_err',
             'close_dz_mm', 'near_cap_red', 'near_cap_mm', 'lift_mm', 'near_raw_mm', 'near_plan_mm', 'near_along_mm', 'near_across_mm',
             'rel_err_raw_mm', 'rel_err_plan_mm', 'zmin_dz_mm', 'block_mm', 'attempts', 'post_where', 'duration_s', 'latency_ms',
             'reason', 'folder']
    with open(O / 'data' / 'trials.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(tcols)
        for r in sorted(rows, key=lambda x: (x['layout'], x['k'])):
            w.writerow([r.get(c) for c in tcols])
    lcols = ['layout', 'n_success', 'results', 'causes', 'tube', 'hole', 'src_ang', 'dst_ang', 'n_caps', 'close_mean_err_mm', 'close_spread_mm',
             'pull_to_train_mean', 'pull_to_prev_layout', 'nn_tube_mm', 'n_tube_30', 'nn_hole_mm', 'n_hole_30', 'nn_joint_mm', 'n_joint_40',
             'nn_joint_vs_train', 'nn_src_ang', 'nn_dst_ang', 'ref_err_raw', 'ref_err_plan']
    with open(O / 'data' / 'layouts.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(lcols)
        for Lr in L_rows:
            w.writerow([Lr.get(c) for c in lcols])
    try:
        coverage_fig(train, L_rows, O / 'figs' / 'coverage_draft.png')
    except Exception as e:
        print(f'  ⚠ 그림 실패: {e}')
    draft(O, A, L_rows, rows)

    # ── 화면 요약 (zip을 못 보내면 이 출력을 붙여 주세요) ──
    print(f"\n■ 학습 데이터: 튜브 x {A.get('train_tube_range')} · 학습 시행끼리 가장 가까운 이웃 중간 {A['nn_train_median_mm']}mm")
    print(f"  랙 방향 분포(0~180°, 30° 칸) 튜브 랙 {A.get('train_src_ang_hist30')} · 빈 랙 {A.get('train_dst_ang_hist30')}")
    print(f"■ 잡는 자리가 튜브를 따라가는 정도 (1 = 완전히 따라감): x {A.get('track_x')} · y {A.get('track_y')}")
    print('\n배치 | 성공 | 결과 | 원인 | 닫은자리 평균어긋남/흩어짐 mm | 학습평균쪽 끌림 | 학습 최근접 튜브/구멍/둘다 mm (40mm안 판) | 랙각 최근접 튜브랙/빈랙 °')
    for Lr in L_rows:
        print(f"{Lr['layout']:>3} | {Lr['n_success']}/{Lr['n']} | {'·'.join(RES_KO.get(x, x) for x in Lr['results'])} | "
              f"{'·'.join(str(c) for c in Lr['causes'])} | {Lr.get('close_mean_err_mm')}/{Lr.get('close_spread_mm')} | "
              f"{Lr.get('pull_to_train_mean')} | {Lr.get('nn_tube_mm')}/{Lr.get('nn_hole_mm')}/{Lr.get('nn_joint_mm')} ({Lr.get('n_joint_40')}) | "
              f"{Lr.get('nn_src_ang')}/{Lr.get('nn_dst_ang')} (빈 랙 {Lr.get('dst_ang')}°)")
    print('\n시행 | 결과 | 원인 | 자세히')
    for r in sorted(valid, key=lambda x: (x['layout'], x['k'])):
        print(f"{r['layout']}-{r['k']} | {RES_KO.get(r['result'])} | {r.get('cause')} | {r.get('detail')}")
    print(f"\n■ 성공 수 vs 학습 거리 순위상관: " + ', '.join(f"{k.split('_vs_')[1]} {v}" for k, v in A.items() if k.startswith('spearman') and v is not None))
    if A.get('fixed'):
        print(f"■ 같은 배치 반복: {A['fixed']}")
    grid([S / e['name'] / 'wrist' / '000000.jpg' for e in train], O / 'figs' / 'train_layouts.jpg', names=[e['name'][-3:] for e in train])
    firsts = [min(lay[L_], key=lambda x: x['k']) for L_ in sorted(lay)]
    grid([R / r['folder'] / 'wrist' / '000000.jpg' for r in firsts], O / 'figs' / 'eval_layouts.jpg', cols=5, w=384, h=216,
         names=[f"layout {r['layout']}" for r in firsts])
    sys.stdout.flush()
    zp = O.parent / (O.name + '.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(O.rglob('*')):
            if f.is_file():
                z.write(f, f.relative_to(O.parent))
    print(f'\n✔ {O} 와 {zp} ({zp.stat().st_size / 1e6:.1f} MB) — 이 zip을 Claude 대화에 첨부하세요')


def draft(O, A, L_rows, rows):
    """자동 초안 (숫자·사진 링크만 — 해석은 Claude가 다듬음)"""
    valid = [r for r in rows if r['result'] in ('success', 'partial', 'fail')]
    k = sum(r['result'] == 'success' for r in valid)
    L = ['# 실물 평가 결과 — 초안 (eval_report.py 자동 생성)', '',
         f"- 유효 {len(valid)}회 · 성공 {k} · 생성 {A['generated']}", '', '## 배치별', '']
    for Lr in L_rows:
        L += [f"### 배치 {Lr['layout']} — 성공 {Lr['n_success']}/{Lr['n']}", '',
              f"![배치 {Lr['layout']} 시작](images/layout_{Lr['layout']:02d}_start.jpg)", '',
              '| 회차 | 결과 | 원인 | 자세히 | 메모 |', '|---|---|---|---|---|']
        for r in sorted([r for r in valid if r['layout'] == Lr['layout']], key=lambda x: x['k']):
            L.append(f"| [{r['k']}]({r.get('img', '')}) | {RES_KO.get(r['result'])} | {r.get('cause')} | {r.get('detail')} | {r.get('note')} |")
        L += ['', f"학습 데이터 최근접: 튜브 {Lr.get('nn_tube_mm')}mm · 구멍 {Lr.get('nn_hole_mm')}mm · 둘 다 {Lr.get('nn_joint_mm')}mm "
                  f"(40mm 안 {Lr.get('n_joint_40')}판) · 랙 방향 차이 튜브 랙 {Lr.get('nn_src_ang')}° / 빈 랙 {Lr.get('nn_dst_ang')}°", '']
    (O / 'README_draft.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
