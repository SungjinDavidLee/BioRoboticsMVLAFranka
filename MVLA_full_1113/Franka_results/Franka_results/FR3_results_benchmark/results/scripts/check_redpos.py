#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_redpos.py — 빨간 튜브가 튜브 랙의 어느 쪽 끝에 있었나 (학습 100판 vs 평가 배치) — 로봇 PC, 읽기만

  python3 check_redpos.py

'바깥 끝' = 튜브 랙의 튜브들 중 빈 랙에서 가장 먼 것, '안쪽 끝' = 가장 가까운 것, '왼쪽 끝' = 로봇 기준 가장 왼쪽(+y).
정책이 색이 아니라 '늘 빨간 튜브가 있던 자리'로 가는지 확인하려는 것.
"""
import json, math, re, sys
from pathlib import Path
import cv2
import numpy as np
TOOLS = Path.home() / 'gello_tools'
sys.path.insert(0, str(TOOLS))
import gello_scene as gsc

S = Path('~/gello_data/medbio_redslot_v1').expanduser()
try:
    c = json.loads((S / 'auto_calib.json').read_text())
    Hm = np.asarray(c['H'], float) if c.get('H') is not None else gsc.fit_homography([p['px'] for p in c['points']], [p['xy'] for p in c['points']])[0]
except Exception as e:
    sys.exit(f'보정 H 없음: {e}')
try:
    colors = (json.loads((TOOLS / 'station.json').read_text()).get('auto') or {}).get('colors') or None
except Exception:
    colors = None


def xy(px):
    return np.asarray(gsc.apply_h(Hm, [list(px[:2])])[0], float)


def where(img_path, ring_px):
    img = cv2.imread(str(img_path))
    if img is None or ring_px is None:
        return None
    det = gsc.detect(img, colors)
    caps = det.get('caps') or []
    if det.get('red') is None or not det.get('racks'):
        return None
    racks = det['racks']
    i_dst = int(np.argmin([math.hypot(r['cx'] - ring_px[0], r['cy'] - ring_px[1]) for r in racks]))
    red = caps[det['red']]
    i_src = int(np.argmin([math.hypot(r['cx'] - red['px'], r['cy'] - red['py']) for r in racks]))
    if i_src == i_dst:
        others = [i for i in range(len(racks)) if i != i_dst]
        if not others:
            return None
        i_src = others[0]
    src = racks[i_src]
    dst_xy = xy(ring_px)
    sc = xy([src['cx'], src['cy']])
    on = [(i, xy([c_['px'], c_['py']])) for i, c_ in enumerate(caps)]
    on = [(i, p) for i, p in on if np.hypot(*(p - sc)) < 0.085]      # 튜브 랙 위의 뚜껑
    if len(on) < 2 or det['red'] not in [i for i, _ in on]:
        return {'n': len(on), 'pos': '하나뿐'}
    d_dst = {i: float(np.hypot(*(p - dst_xy))) for i, p in on}
    ys = {i: float(p[1]) for i, p in on}
    r = det['red']
    pos = '바깥 끝' if r == max(d_dst, key=d_dst.get) else ('안쪽 끝' if r == min(d_dst, key=d_dst.get) else '가운데')
    lr = '왼쪽 끝' if r == max(ys, key=ys.get) else ('오른쪽 끝' if r == min(ys, key=ys.get) else '가운데')
    return {'n': len(on), 'pos': pos, 'lr': lr}


rows = []
for d in sorted(S.glob('episode_*')):
    try:
        m = json.loads((d / 'meta.json').read_text())
    except Exception:
        continue
    if (m.get('quality') or {}).get('verdict') == 'bad':
        continue
    A = (m.get('start') or {}).get('auto') or {}
    w = where(d / 'wrist' / '000000.jpg', A.get('drop_px'))
    rows.append((d.name, w))
cnt = {}
for nm, w in rows:
    k = (w or {}).get('pos', '못 찾음')
    cnt[k] = cnt.get(k, 0) + 1
cnt2 = {}
for nm, w in rows:
    k = (w or {}).get('lr', '-')
    cnt2[k] = cnt2.get(k, 0) + 1
print(f'■ 학습 {len(rows)}판 — 빨간 튜브 위치 (빈 랙 기준): {cnt} · (로봇 기준 좌우): {cnt2}')
print('  판별: ' + ' '.join(f"{nm[-3:]}:{(w or {}).get('pos', '?')[:2]}" for nm, w in rows))
seen = {}
for F in sorted((S / '_rollouts').glob('rollout_*')):
    try:
        m = json.loads((F / 'meta.json').read_text())
    except Exception:
        continue
    mm = re.search(r'(\d+)\s*-\s*(\d+)', m.get('note') or '')
    if not mm or m.get('result') == 'void':
        continue
    L = int(mm.group(1))
    if L in seen:
        continue
    seen[L] = where(F / 'wrist' / '000000.jpg', (m.get('pre') or {}).get('ring_px'))
print('■ 평가 배치: ' + ' · '.join(f"{L}: {(w or {}).get('pos', '?')}/{(w or {}).get('lr', '?')}" for L, w in sorted(seen.items())))
