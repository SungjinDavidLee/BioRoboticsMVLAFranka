#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_deep_1105.py — 11-05 평가(T 배치 v1·v2 비교 + 쉬운 배치 E) 실패 원인 분석 재료 모으기 (로봇 PC, 읽기만 — 로봇 안 움직임)
(11-03 eval_deep.py와 같고: 지금 _rollouts만 읽음, 체크포인트·메모 기록, 사진 3장 추가, results.csv 같이 담음)

  python3 eval_deep_1105.py          # ~/gello_data/medbio_redslot_v1 → ~/eval_deep_1105/ + ~/eval_deep_1105.zip

모으는 것
  deep.json
    train[]  학습 판마다: 두 랙(튜브 랙·빈 랙)의 중심·방향·길이, 뚜껑 위치(빨간/다른), 자동 시연이 잡은 자리·그리퍼 방향,
             닫은 순간 손끝 자세(위치·방향·기울기)와 닫기 전 2초 경로
    trials[] 평가 판마다: 같은 장면 정보 + 10Hz 손끝 실제·명령 위치, 그리퍼 방향·기울기, 손가락 폭·그리퍼 명령,
             묶음마다 정책이 낸 행동 16개(위치·그리퍼)
  img/<판>_start_third.jpg   시작 3인칭
  img/<판>_close_wrist.jpg   닫는 순간 손목 사진
  img/<판>_close3_third.jpg  닫고 3초 뒤 3인칭 (들었는지)
  img/<판>_hole_wrist.jpg    쥔 채 구멍에 가장 가까운 순간 손목 사진 (구멍과 튜브가 보임)
  img/<판>_hole_third.jpg    같은 순간 3인칭 사진
  img/<판>_end_third.jpg     끝 3인칭
  img/<판>_strip_wrist.jpg   잡기 전후 손목 사진 6장 (닫기 2초 전 ~ 3초 뒤; 안 닫았으면 가장 낮게 내려간 때 앞뒤)
  img/<판>_strip_third.jpg   판 전체 3인칭 8장 (같은 간격)
  img/<판>_pre_scene.jpg / _post_scene.jpg   스테이션이 그린 시작·끝 장면 검출 그림
  results.csv, layout_plan*.json  (평가 결과표, 배치 계획)
바꾸지 않는 것: 아무것도 안 바꿈 (읽기만). _rollouts 밖의 옛 평가 폴더는 안 읽음
"""
import json
import math
import sys
import zipfile
import shutil
from pathlib import Path

import cv2
import numpy as np

TOOLS = Path.home() / 'gello_tools'
sys.path.insert(0, str(TOOLS))
from gello_to_lerobot import fk      # noqa: E402
import gello_scene as gsc            # noqa: E402

S = Path('~/gello_data/medbio_redslot_v1').expanduser()
O = Path('~/eval_deep_1105').expanduser()


def r4(v):
    return [round(float(x), 4) for x in v]


def pose(q):
    q = np.asarray(q, float)
    if not np.all(np.isfinite(q)):
        return None
    M = fk(q)
    R = M[:3, :3]
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -R[2, 2]))))     # 수직 아래에서 기운 각도
    return M[:3, 3], yaw, tilt


def load_H():
    c = json.loads((S / 'auto_calib.json').read_text())
    if c.get('H') is not None:
        return np.asarray(c['H'], float)
    pts = c['points']
    return np.asarray(gsc.fit_homography([p['px'] for p in pts], [p['xy'] for p in pts])[0], float)


Hm = load_H()
try:
    COLORS = (json.loads((TOOLS / 'station.json').read_text()).get('auto') or {}).get('colors') or None
except Exception:
    COLORS = None


def xy(px):
    return np.asarray(gsc.apply_h(Hm, [list(px[:2])])[0], float)


def geo(img_path, ring_px, red_hint_px=None):
    """손목 홈 사진 → 랙 2개(로봇 x,y·방향·길이), 뚜껑(빨간 여부). 빈 랙 = 빨간 링(ring_px)에 가장 가까운 랙"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    det = gsc.detect(img, COLORS)
    racks = []
    for r in det.get('racks') or []:
        a = math.radians(r['angle_deg'])
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        c = np.array([r['cx'], r['cy']])
        p1, p2 = xy(c - u * r['len'] / 2), xy(c + u * r['len'] / 2)
        w1, w2 = xy(c - v * r['wid'] / 2), xy(c + v * r['wid'] / 2)
        racks.append({'px': r4(c), 'xy': r4(xy(c)), 'ang': round(math.degrees(math.atan2(*(p2 - p1)[::-1])) % 180.0, 1),
                      'len_m': round(float(np.linalg.norm(p2 - p1)), 4), 'wid_m': round(float(np.linalg.norm(w2 - w1)), 4)})
    caps = [{'px': r4([c['px'], c['py']]), 'xy': r4(xy([c['px'], c['py']])), 'red': i == det.get('red')}
            for i, c in enumerate(det.get('caps') or [])]

    def near(px):
        if px is None or not racks:
            return None
        return int(np.argmin([math.hypot(r['px'][0] - px[0], r['px'][1] - px[1]) for r in racks]))
    red = next((c for c in caps if c['red']), None)
    i_dst = near(ring_px)
    i_src = near(red['px'] if red else red_hint_px)
    if i_src is not None and i_src == i_dst and len(racks) > 1:
        i_src = [i for i in range(len(racks)) if i != i_dst][0]
    for i, r in enumerate(racks):
        r['role'] = 'dst' if i == i_dst else ('src' if i == i_src else 'other')
    return {'racks': racks, 'caps': caps, 'ring_xy': r4(xy(ring_px)) if ring_px is not None else None}


def first_close(G):
    return next((i for i in range(1, len(G)) if G[i - 1] > 0.5 and G[i] < 0.5), None)


# ── 학습 판 ──
def train():
    out = []
    for d in sorted(S.glob('episode_*')):
        try:
            m = json.loads((d / 'meta.json').read_text())
        except Exception:
            continue
        if (m.get('quality') or {}).get('verdict') == 'bad':
            continue
        A = (m.get('start') or {}).get('auto') or {}
        e = {'name': d.name, 'source': m.get('source'), 'layout': (A.get('layout') or {}).get('id'),
             'auto': {k: A.get(k) for k in ('grasp_xyz', 'drop_xyz', 'yaw_deg', 'rack_deg', 'drop_px', 'place', 'caps', 'yaw_add_deg')}}
        try:
            e['scene'] = geo(d / 'wrist' / '000000.jpg', A.get('drop_px'))
        except Exception as ex:
            e['scene_err'] = str(ex)
        try:
            z = np.load(d / 'data.npz')
            t = np.asarray(z['t'], float)
            Q = np.asarray(z['q'], float)
            G = np.asarray(z['leader_grip'], float) if 'leader_grip' in z.files else None
            W = np.asarray(z['grip_w'], float) if 'grip_w' in z.files else None
            ic = first_close(G) if G is not None else None
            if ic is not None:
                p, yaw, tilt = pose(Q[ic])
                k3 = min(len(t) - 1, int(np.searchsorted(t, t[ic] + 3.0)))
                path = [r4(pose(Q[i])[0]) for i in range(max(0, ic - 20), ic + 1, 2) if pose(Q[i]) is not None]
                e['close'] = {'t': round(float(t[ic] - t[0]), 2), 'xyz': r4(p), 'yaw': round(yaw, 1), 'tilt': round(tilt, 1),
                              'w3': None if W is None else round(float(W[k3]), 4), 'path': path}
                io = next((i for i in range(ic + 1, len(G)) if G[i - 1] < 0.5 and G[i] > 0.5), None)
                if io is not None:
                    p2, yaw2, tilt2 = pose(Q[io])
                    e['release'] = {'t': round(float(t[io] - t[0]), 2), 'xyz': r4(p2), 'yaw': round(yaw2, 1), 'tilt': round(tilt2, 1)}
        except Exception as ex:
            e['data_err'] = str(ex)
        out.append(e)
    return out


# ── 평가 판 ──
def trial(F, imgdir):
    m = json.loads((F / 'meta.json').read_text())
    pre = m.get('pre') or {}
    J = m.get('judge') or {}
    r = {'folder': F.name, 'note': m.get('note'), 'result': m.get('result'), 'result_auto': m.get('result_auto'), 'ckpt': m.get('ckpt'),
         'labeled_by': m.get('labeled_by'), 'duration_s': m.get('duration_s'), 'closed': m.get('closed'), 'released': m.get('released'),
         'stages': J.get('stages'), 'metrics': J.get('metrics'), 'judge_post': J.get('post'), 'judge_idx': J.get('idx'),
         'judge_reason': J.get('reason'),
         'reason': m.get('reason'), 'pre': {k: pre.get(k) for k in ('tube_xy', 'grasp_xy', 'tube_px', 'ring_px', 'hole_xy', 'hole_xy_plan')},
         'post': m.get('post_scene')}
    try:
        r['scene'] = geo(F / 'wrist' / '000000.jpg', pre.get('ring_px'), pre.get('tube_px'))
    except Exception as ex:
        r['scene_err'] = str(ex)
    z = np.load(F / 'data.npz')
    t, Q, C, W, G = (np.asarray(z[k], float) for k in ('t', 'q', 'cmd', 'grip_w', 'grip_cmd'))
    ts = {'t': [], 'p': [], 'c': [], 'yaw': [], 'tilt': [], 'w': [], 'g': []}
    for i in range(len(t)):
        a, b = pose(Q[i]), pose(C[i])
        if a is None or b is None:
            continue
        ts['t'].append(round(float(t[i] - t[0]), 2))
        ts['p'].append(r4(a[0]))
        ts['c'].append(r4(b[0]))
        ts['yaw'].append(round(a[1], 1))
        ts['tilt'].append(round(a[2], 1))
        ts['w'].append(round(float(W[i]), 4))
        ts['g'].append(round(float(G[i]), 2))
    r['ts'] = ts
    ch = []
    cf = F / 'chunks.jsonl'
    if cf.exists():
        for line in cf.read_text().splitlines():
            try:
                c = json.loads(line)
            except Exception:
                continue
            A = np.asarray(c.get('actions') or [], float)
            if A.ndim != 2 or A.shape[1] < 10:
                continue
            ch.append({'t_obs': c.get('t_obs'), 'state': r4((c.get('state') or [0, 0, 0])[:3]),
                       'xyz': [r4(a[:3]) for a in A], 'grip': [round(float(a[9]), 2) for a in A],
                       'exec': c.get('exec'), 'grip_event': c.get('grip_event')})
    r['chunks'] = ch
    # 사진: 닫는 순간 손목, 쥔 채 구멍에 가장 가까운 순간 손목·3인칭
    ic = first_close(G)
    key = {}
    if ic is not None:
        key['close'] = ic
        hole = pre.get('hole_xy_plan') or pre.get('hole_xy')
        if hole:
            P = [pose(Q[i]) for i in range(ic, len(t))]
            d = [(math.hypot(p[0][0] - hole[0], p[0][1] - hole[1]) if p is not None and W[ic + j] > 0.015 else 9.0) for j, p in enumerate(P)]
            if d and min(d) < 0.08:
                key['hole'] = ic + int(np.argmin(d))
    key['start'] = 0
    if ic is not None:
        key['close3'] = min(len(t) - 1, int(np.searchsorted(t, t[ic] + 3.0)))
    key['end'] = len(t) - 1
    r['key_idx'] = key
    for nm, cam in (('start', 'third'), ('close', 'wrist'), ('close3', 'third'), ('hole', 'wrist'), ('hole', 'third'), ('end', 'third')):
        i = key.get(nm)
        if i is None:
            continue
        im = frame(F, cam, i, 640)
        if im is not None:
            cv2.imwrite(str(imgdir / f'{F.name}_{nm}_{cam}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # 잡기 전후 손목 띠: 닫은 순간 기준 -2, -1, -0.4, 0, +1.5, +3초 / 안 닫았으면 가장 낮게 내려간 순간 기준
    if ic is not None:
        c0, offs = ic, (-2.0, -1.0, -0.4, 0.0, 1.5, 3.0)
    else:
        zs = [p[0][2] if p is not None else 9.0 for p in (pose(Q[i]) for i in range(len(t)))]
        c0, offs = int(np.argmin(zs)), (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0)
        key['low'] = c0
    idx = [min(len(t) - 1, max(0, int(np.searchsorted(t, t[c0] + o)))) for o in offs]
    strip(F, 'wrist', idx, t, t[c0], imgdir / f'{F.name}_strip_wrist.jpg', cols=6)
    idx = [int(round(x)) for x in np.linspace(0, len(t) - 1, 8)]
    strip(F, 'third', idx, t, t[0], imgdir / f'{F.name}_strip_third.jpg', cols=4)
    for nm in ('pre_scene', 'post_scene'):
        if not (F / f'{nm}.jpg').exists():
            continue
        im = cv2.imread(str(F / f'{nm}.jpg'))
        if im is not None:
            im = cv2.resize(im, (640, int(640 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(imgdir / f'{F.name}_{nm}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return r


def frame(F, cam, i, w):
    p = F / cam / f'{i:06d}.jpg'
    if not p.exists():
        return None
    im = cv2.imread(str(p))
    if im is None:
        return None
    return cv2.resize(im, (w, int(w * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)


def strip(F, cam, idx, t, t0, out, cols):
    ims = []
    for i in idx:
        im = frame(F, cam, i, 320)
        if im is None:
            continue
        cv2.putText(im, f'{t[i] - t0:+.1f}s', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, f'{t[i] - t0:+.1f}s', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        ims.append(im)
    if not ims:
        return
    h = ims[0].shape[0]
    ims = [cv2.resize(x, (320, h)) for x in ims]
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack(ims[k:k + cols]) for k in range(0, len(ims), cols)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    if O.exists():
        shutil.rmtree(O)
    (O / 'img').mkdir(parents=True)
    print(f'■ 세션 {S}')
    T = train()
    print(f'■ 학습 판 {len(T)}개 (장면 찾음 {sum(1 for e in T if e.get("scene"))} · 닫은 순간 {sum(1 for e in T if e.get("close"))})')
    R = []
    for base, tag in ((S / '_rollouts', 'eval1105'),):
        if not base.exists():
            continue
        for F in sorted(base.glob('rollout_*')):
            if not (F / 'meta.json').exists() or not (F / 'data.npz').exists():
                continue
            try:
                r = trial(F, O / 'img')
                r['set'] = tag
                R.append(r)
            except Exception as ex:
                print(f'  ⚠ {F.name}: {ex}')
    print(f'■ 평가 판 {len(R)}개 (묶음 기록 있는 판 {sum(1 for r in R if r["chunks"])})')
    try:
        auto_cfg = json.loads((S / 'auto_task.json').read_text())
    except Exception:
        auto_cfg = None
    D = {'H': Hm.tolist(), 'finger': {'open_half': 0.040, 'thick': 0.010, 'width': 0.020, 'cap_r_slot': 0.016},
         'auto_task': auto_cfg, 'train': T, 'trials': R}
    (O / 'deep.json').write_text(json.dumps(D, ensure_ascii=False))
    rc = S / '_rollouts' / 'results.csv'
    if rc.exists():
        shutil.copy2(rc, O / 'results.csv')
    for f in S.glob('layout_plan*.json'):
        shutil.copy2(f, O / f.name)
    zp = O.parent / 'eval_deep_1105.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(O.rglob('*')):
            if f.is_file():
                zf.write(f, f.relative_to(O.parent))
    print(f'✔ {zp} ({zp.stat().st_size / 1e6:.1f} MB) — 이 zip을 Claude 대화에 첨부하세요')


if __name__ == '__main__':
    main()
