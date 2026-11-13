#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fail_report.py — 지금 배치 계획(layout_plan.json)으로 한 자동 시연 전부(저장·휴지통)를 배치별로 정리
(기록 읽기만 — 로봇 안 움직임, 데이터·계획 파일은 바꾸지 않음)

  cd ~/Downloads/fail_1104 && python3 fail_report.py      → ~/fail_report/ + ~/fail_report.zip

담는 것
  - summary.txt / runs.csv : 판마다 배치 번호, 저장/휴지통(이유), 고른 방법, 찾은 뚜껑 수, 잡는 순간 손 오차, 쥔 폭 …
  - 배치별 성공 수와 배치 특징(빨강·노랑 칸, 바로 옆, 랙 방향, 빨간 튜브 자리) — 실패 배치의 공통점 찾기
  - layouts/Lxx.jpg : 판마다 [장면 찾기 화면 | 잡기 직전 손목 | 잡은 뒤 3인칭 | 놓을 때 3인칭]
  - session/ : 세션 폴더의 작은 json(계획·보정)과 station 로그 / code/ : 지금 ~/gello_tools 스테이션 파일(버전 확인용)
"""
import argparse
import csv
import datetime
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import last_insert as li        # noqa: E402  (insert_info·frame·label — ~/gello_tools의 fk 사용)


def tile(img, w, txt):
    if img is None:
        img = np.zeros((480, 848, 3), np.uint8)
    s = w / img.shape[1]
    img = cv2.resize(img, (w, int(round(img.shape[0] * s))))
    return li.label(img, txt)


def held_mm(d, ic):
    try:
        z = np.load(d / 'data.npz')
        G = np.asarray(z['grip_w'], float)
        if ic is None:
            return None
        return round(float(np.nanmedian(G[ic + 3:ic + 13])) * 1000, 1)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(Path.home() / 'gello_data' / 'medbio_redslot_v1'))
    ap.add_argument('--out', default=str(Path.home() / 'fail_report'))
    ap.add_argument('--w', type=int, default=360, help='사진 한 장 너비(px)')
    a_ = ap.parse_args()
    root = Path(a_.data).expanduser()
    out = Path(a_.out).expanduser()
    if out.exists():
        shutil.rmtree(out)
    (out / 'layouts').mkdir(parents=True)

    plan = json.loads((root / 'layout_plan.json').read_text())
    made = plan.get('made') or ''
    L_by = {L['id']: L for L in (plan.get('layouts') or []) + (plan.get('test') or [])}

    runs = []
    for d in li.runs(root):
        try:
            m = json.loads((d / 'meta.json').read_text())
        except Exception:
            continue
        if (m.get('recorded_at') or '') < made:
            continue
        a = (m.get('start') or {}).get('auto') or {}
        if not a:
            continue
        try:
            x = li.insert_info(d) or {'dir': d, 'meta': m, 'auto': a}
        except Exception as e:
            print('읽기 실패', d.name, e)
            x = {'dir': d, 'meta': m, 'auto': a}
        runs.append(x)
    if not runs:
        print('이 계획으로 한 자동 시연 기록이 없습니다:', root, made)
        return

    home_xy = None
    rows, per = [], {}
    for x in runs:
        d, m, a = x['dir'], x['meta'], x['auto']
        if home_xy is None and m.get('home_tcp'):
            ht = np.asarray(m['home_tcp'], float).ravel()
            home_xy = ht[:2] if ht.size in (3, 6, 7) else (ht.reshape(4, 4)[:2, 3] if ht.size == 16 else None)
        q = m.get('quality') or {}
        disc = m.get('discard') or {}
        lay = a.get('layout') or {}
        lid = lay.get('id') if lay.get('plan') in (None, made) else None
        saved = d.parent == root and q.get('verdict') != 'bad'
        tp = a.get('third_pick') or {}
        r = {'episode': d.name, 'where': 'saved' if d.parent == root else 'trash', 'time': m.get('recorded_at'),
             'layout': lid or '?', 'ok': int(saved), 'verdict': q.get('verdict'),
             'why': disc.get('why') or '; '.join((q.get('bad') or []) + (q.get('warn') or []))[:160],
             'pick_src': a.get('pick_src'), 'caps': a.get('caps'), 'third_why': tp.get('why'),
             'rot_deg': a.get('rot_deg'), 'rack_deg': a.get('rack_deg'),
             'grasp_xyz_mm': [round(v * 1000) for v in (a.get('grasp_xyz') or [])[:3]],
             'grasp_err_mm': x.get('grasp_err_mm'), 'grasp_err_hand_mm': x.get('grasp_err_hand_mm'),
             'held_mm': held_mm(d, x.get('ic')), 'rel_err_mm': x.get('rel_err_mm'),
             'drop_px': [round(v) for v in (a.get('drop_px') or [])[:2]],
             'notes': ' | '.join(a.get('notes') or [])[:200]}
        rows.append(r)
        per.setdefault(r['layout'], []).append((r, x))

    # 판 목록
    keys = list(rows[0].keys())
    with open(out / 'runs.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})

    # 배치별
    n_all = 0
    for d in root.glob('episode_*'):
        try:
            if (json.loads((d / 'meta.json').read_text()).get('quality') or {}).get('verdict') != 'bad':
                n_all += 1
        except Exception:
            pass
    lines = [f'세션 {root}', f'저장된 판 전체(나쁨 제외) {n_all}개',
             f'계획 {made} — 배치 {len(plan.get("layouts") or [])}개, 이 계획 뒤 자동 시연 {len(rows)}판 '
             f'(저장 {sum(r["ok"] for r in rows)}, 휴지통·나쁨 {len(rows) - sum(r["ok"] for r in rows)})',
             '배치 번호가 ?인 판 = 자유 배치에서 추천 그림과 "비슷함"이 안 나온 채 실행한 판 (시각 순서로 배치를 맞춰 봄)', '']
    feat_rows = []
    for L in plan.get('layouts') or []:
        lst = per.get(L['id'], [])
        n_ok = sum(r['ok'] for r, _ in lst)
        rx = np.asarray(L.get('red_xy') or [np.nan, np.nan], float)
        dist = float(np.linalg.norm(rx - home_xy)) if home_xy is not None else float('nan')
        fr = {'id': L['id'], 'ok': n_ok, 'tries': len(lst), 'red': L.get('red'), 'yellow': (L.get('yellow') or [None])[0],
              'adj': int(abs((L.get('yellow') or [0])[0] - L.get('red', 0)) == 1), 'decoy': int(bool(L.get('decoy'))),
              'tube_deg': round(float(L['tube']['deg'])), 'red_x_mm': round(rx[0] * 1000), 'red_y_mm': round(rx[1] * 1000),
              'red_from_home_mm': round(dist * 1000) if dist == dist else None, 'rot_tol': L.get('rot_tol_deg')}
        feat_rows.append(fr)
    with open(out / 'layouts.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(feat_rows[0].keys()))
        w.writeheader()
        w.writerows(feat_rows)
    lines.append('배치  성공/시도  빨강칸 노랑칸 바로옆 미끼  튜브랙° 빨간튜브 x,y(mm)  홈에서(mm)')
    for fr in feat_rows:
        lines.append(f"{fr['id']:>4}  {fr['ok']}/{fr['tries']:<3}      {fr['red']}     {fr['yellow']}     {fr['adj']}     {fr['decoy']}   "
                     f"{fr['tube_deg']:>4}   {fr['red_x_mm']:>4},{fr['red_y_mm']:>5}      {fr['red_from_home_mm']}")
    good = [fr for fr in feat_rows if fr['ok'] >= 1]
    bad = [fr for fr in feat_rows if fr['ok'] == 0 and fr['tries'] > 0]
    if good and bad:
        lines.append('')
        for nm, k in (('빨간 튜브 x(앞뒤)', 'red_x_mm'), ('빨간 튜브 y(좌우)', 'red_y_mm'), ('홈에서 거리', 'red_from_home_mm')):
            g_ = [fr[k] for fr in good if fr[k] is not None]
            b_ = [fr[k] for fr in bad if fr[k] is not None]
            if g_ and b_:
                lines.append(f'{nm}: 성공 배치 중앙값 {np.median(g_):.0f}mm (범위 {min(g_)}~{max(g_)}) / '
                             f'실패 배치 중앙값 {np.median(b_):.0f}mm (범위 {min(b_)}~{max(b_)})')
        for nm, k in (('끝 칸(1·4번) 빨강', None), ('바로 옆 노랑', 'adj')):
            if k is None:
                gs = sum(fr['red'] in (1, 4) for fr in good)
                bs = sum(fr['red'] in (1, 4) for fr in bad)
            else:
                gs = sum(fr[k] for fr in good)
                bs = sum(fr[k] for fr in bad)
            lines.append(f'{nm}: 성공 배치 {gs}/{len(good)}, 실패 배치 {bs}/{len(bad)}')
    lines.append('')

    # 판별 자세히 + 사진
    W = a_.w
    for lid, lst in sorted(per.items(), key=lambda kv: kv[0]):
        lines.append(f'■ {lid}')
        tiles = []
        for r, x in lst:
            d = x['dir']
            lines.append(f"  {r['episode']} [{r['where']}] 판정 {r['verdict']} | {r['why']}")
            lines.append(f"    고른 방법 {r['pick_src']} · 뚜껑 {r['caps']}개 · 3인칭 {r['third_why'] or '-'} · 돌림 {r['rot_deg']}° · "
                         f"잡을 자리 {r['grasp_xyz_mm']}mm · 잡는 순간 손−계획 {r['grasp_err_mm']}mm (손 x,y {r['grasp_err_hand_mm']}) · 쥔 폭 {r['held_mm']}mm")
            if r['notes']:
                lines.append(f"    안내: {r['notes']}")
            sc = cv2.imread(str(d / 'auto_scene.jpg'))
            ic, ir = x.get('ic'), x.get('ir')
            nP = len(x.get('P', [])) if x.get('P') is not None else 0
            row = [tile(sc, W, f"{r['episode'][-6:]} {r['where']} scene (red circle = pick)")]
            row.append(tile(cv2.imread(str(d / 'wrist' / f'{max(0, (ic or 0) - 3):06d}.jpg')), W, 'wrist just before grasp'))
            j3 = min((ic or 0) + 25, max(nP - 1, 0)) if ic is not None else 0
            row.append(tile(cv2.imread(str(d / 'third' / f'{j3:06d}.jpg')), W, 'third after grasp'))
            jl = ir if ir is not None else max(nP - 1, 0)
            row.append(tile(cv2.imread(str(d / 'third' / f'{jl:06d}.jpg')), W, 'third at release / end'))
            hmax = max(t.shape[0] for t in row)
            row = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0))) for t in row]
            tiles.append(np.hstack(row))
        lines.append('')
        if tiles:
            cv2.imwrite(str(out / 'layouts' / f"{lid if lid != '?' else 'no_id'}.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])

    # 세션 작은 파일·로그·스테이션 코드 (버전 확인용)
    (out / 'session').mkdir()
    for p in root.glob('*.json'):
        if p.stat().st_size < 3e6:
            shutil.copy2(p, out / 'session' / p.name)
    for p in [root / 'collect_log.jsonl'] + sorted((root / '_station_logs').glob('*')):
        if p.is_file() and p.stat().st_size < 5e6:
            shutil.copy2(p, out / 'session' / p.name)
    (out / 'code').mkdir()
    for p in sorted((Path.home() / 'gello_tools').glob('gello_*.py')) + [Path.home() / 'gello_tools' / 'gello_station_page.html']:
        if p.exists():
            shutil.copy2(p, out / 'code' / p.name)

    txt = '\n'.join(lines)
    (out / 'summary.txt').write_text(txt)
    print(txt)
    zp = out.with_suffix('.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                zf.write(p, p.relative_to(out.parent))
    print(f'\n→ {zp}  ({zp.stat().st_size / 1e6:.1f} MB) — 이 파일을 보내 주세요')


if __name__ == '__main__':
    main()
