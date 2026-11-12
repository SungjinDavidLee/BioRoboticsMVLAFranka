#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""media_export_1112.py — GitHub용 실행 영상·사진 재료 모으기 (로봇 PC, 읽기만 — 로봇 안 움직임, 기록 안 바꿈)

  python3 media_export_1112.py        # → ~/media_1112/ + ~/media_1112.zip

모으는 것 (아래 RUNS의 rollout 폴더를 ~/gello_data 안에서 이름으로 찾음 — _rollouts 아래 하위 폴더에 옮겨 둔 것도 찾음)
  video/<메모>/{third,wrist}/NNNNNN.jpg   영상 재료: 3인칭·손목 전 프레임을 가로 480 px로 줄인 JPEG
  photo/<메모>/<순간>_<카메라>.jpg         원본 해상도 사진: 시작 · 닫기 0.3 s 전 · 닫고 3 s 뒤 · 열기 0.3 s 전 · 열고 1.5 s 뒤 · 끝
  timeline/<메모>.csv                    10 Hz 시각 · 손끝 xyz(관절 실측 FK) · 손가락 폭 · 그리퍼 명령
  picked.json                            고른 판의 메모·판정·체크포인트·사진 순간·수식 기록(hybrid, 있으면)
바꾸지 않는 것: 아무것도 안 바꿈 (읽기·복사만). 원본 폴더는 그대로 둠
"""
import csv
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

HOME = Path.home()
ROOT = HOME / 'gello_data'
TOOLS = HOME / 'gello_tools'
O = HOME / 'media_1112'

# 메모 → rollout 폴더 이름 (평가 기록에서 확인한 이름)
RUNS = {
    'T07-v1-1': 'rollout_20261105_115234',    # E1 실물 100판 — 유일한 성공
    'T06-mix-3': 'rollout_20261111_123141',   # E3 실물 + 시뮬 — 성공
    'T06-hyb-2': 'rollout_20261112_105622',   # E4 VLA + 수식 — 성공 (E2 영상과 같은 배치 T06)
    'T10-hyb-3': 'rollout_20261112_142550',   # E4 — 성공 (조명 정상 구간)
    'T02-hyb-1': 'rollout_20261112_101350',   # E4 — 성공 (사진만)
}
VIDEO = ['T07-v1-1', 'T06-mix-3', 'T06-hyb-2', 'T10-hyb-3']


def find_runs(names):
    """~/gello_data 아래에서 이름이 맞는 rollout 폴더 찾기 (rollout 폴더 안으로는 안 들어감)"""
    found = {}
    for dp, dn, _ in os.walk(ROOT):
        keep = []
        for d in dn:
            if d in names and d not in found:
                found[d] = Path(dp) / d
            if not d.startswith('rollout_'):
                keep.append(d)
        dn[:] = keep
        if len(found) == len(names):
            break
    return found


def tcp(q):
    sys.path.insert(0, str(TOOLS))
    from gello_to_lerobot import fk
    return np.array([fk(qq)[:3, 3] for qq in q])


def at(t, i, dt):
    return int(min(len(t) - 1, max(0, np.searchsorted(t, t[i] + dt))))


def main():
    if O.exists():
        shutil.rmtree(O)
    O.mkdir(parents=True)
    found = find_runs(set(RUNS.values()))
    picked = {}
    for tg, name in RUNS.items():
        F = found.get(name)
        if F is None:
            print(f'  ⚠ {tg}: {name} 폴더를 못 찾음')
            continue
        m = json.loads((F / 'meta.json').read_text())
        z = np.load(F / 'data.npz')
        t = np.asarray(z['t'], float)
        n = len(t)
        gc = np.asarray(z['grip_cmd'], float) if 'grip_cmd' in z.files else None
        ic = io = None
        if gc is not None:
            c = np.where(gc < 0.5)[0]
            if len(c):
                ic = int(c[0])
                o = np.where(gc[ic:] > 0.5)[0]
                io = int(ic + o[0]) if len(o) else None
        key = {'start': 0, 'end': n - 1}
        if ic is not None:
            key['pre_grasp'] = at(t, ic, -0.3)
            key['grasped'] = at(t, ic, 3.0)
        if io is not None:
            key['pre_release'] = at(t, io, -0.3)
            key['released'] = at(t, io, 1.5)
        out = O / 'photo' / tg
        out.mkdir(parents=True, exist_ok=True)
        for nm, i in key.items():
            for cam in ('third', 'wrist'):
                p = F / cam / f'{i:06d}.jpg'
                if p.exists():
                    shutil.copy2(p, out / f'{nm}_{cam}.jpg')
        P = tcp(np.asarray(z['q'], float))
        gw = np.asarray(z['grip_w'], float)
        (O / 'timeline').mkdir(exist_ok=True)
        with open(O / 'timeline' / f'{tg}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t_s', 'x_m', 'y_m', 'z_m', 'grip_w_m', 'grip_cmd'])
            for i in range(n):
                w.writerow([f'{t[i] - t[0]:.3f}', f'{P[i, 0]:.5f}', f'{P[i, 1]:.5f}', f'{P[i, 2]:.5f}', f'{gw[i]:.5f}',
                            f'{gc[i]:.2f}' if gc is not None else ''])
        if tg in VIDEO:
            import cv2
            for cam in ('third', 'wrist'):
                d = O / 'video' / tg / cam
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    src = F / cam / f'{i:06d}.jpg'
                    if not src.exists():
                        continue
                    im = cv2.imread(str(src))
                    im = cv2.resize(im, (480, int(480 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(d / f'{i:06d}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        hy = m.get('hybrid') or {}
        picked[tg] = {'folder': F.name, 'note': m.get('note'), 'result': m.get('result'), 'result_auto': m.get('result_auto'),
                      'ckpt': m.get('ckpt'), 'duration_s': m.get('duration_s'), 'n_frames': n,
                      't0_unix': float(t[0]),
                      'photos': {k: {'idx': int(v), 't_s': round(float(t[v] - t[0]), 2)} for k, v in key.items()},
                      'close_t_s': None if ic is None else round(float(t[ic] - t[0]), 2),
                      'open_t_s': None if io is None else round(float(t[io] - t[0]), 2),
                      'hybrid': {k: hy.get(k) for k in ('use', 'ok', 'events', 'grasp', 'place', 'place_skip')} if hy else None}
        print(f"  {tg}: {m.get('result')} · {n}프레임 · 닫기 {picked[tg]['close_t_s']} s · 열기 {picked[tg]['open_t_s']} s"
              f"{' · 영상 재료' if tg in VIDEO else ''}")
    (O / 'picked.json').write_text(json.dumps(picked, ensure_ascii=False, indent=1, default=str))
    zp = HOME / 'media_1112.zip'
    with zipfile.ZipFile(zp, 'w') as zf:
        for f in sorted(O.rglob('*')):
            if f.is_file():
                zf.write(f, f.relative_to(HOME), compress_type=zipfile.ZIP_STORED if f.suffix == '.jpg' else zipfile.ZIP_DEFLATED)
    print(f'✔ {zp} ({zp.stat().st_size / 1e6:.1f} MB) — 판 {len(picked)}/{len(RUNS)}')


if __name__ == '__main__':
    main()
