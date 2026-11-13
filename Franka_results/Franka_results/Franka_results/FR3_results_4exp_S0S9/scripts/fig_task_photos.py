#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_task_photos.py — 평가표 단계 사진 4장: S0 초기 위치 · S5 파지 · S8 파지 후 이송 · S9 배치 (3인칭 + 손목 한 장씩)

  python3 scripts/fig_task_photos.py <media_1112 폴더> [판 메모]        (기본 T10-hyb-3)
  → figs/task_S0_start.jpg, figs/task_S5_grasp.jpg, figs/task_S8_transport.jpg, figs/task_S9_place.jpg
입력 (scripts/media_export_1112.py 결과):
  <폴더>/photo/<판>/<순간>_<카메라>.jpg   원본 848 × 480 (start · grasped · end)
  <폴더>/video/<판>/<카메라>/<프레임>.jpg  영상용 480 px 프레임 (S8은 원본 사진이 없어 이 프레임을 키워 씀)
  <폴더>/timeline/<판>.csv                  10 Hz 손끝 위치 · 손가락 폭
  <폴더>/picked.json                        사진 순간 · 수식 넘김 기록
  data/stage_runs.csv                       단계 판정 값 (밝기 · 포화 · 쥔 폭)
순간 고르는 법:
  S0 시작 프레임 / S5 닫기 명령 3 s 뒤 (단계 판정이 폭을 재는 순간) /
  S8 쥔 채 빨간 칸 수평 20 mm 안에 처음 들어온 뒤, 수식이 꽂기를 넘겨받기 전까지(VLA가 옮긴 구간) 칸에 가장 가까운 프레임 /
  S9 판 끝 (놓고 위로 약 5 cm 뺀 뒤 — 튜브가 링 구멍에 꽂힌 모습)
계획한 놓는 높이 = 수식이 쓴 z_rel = 잡는 높이 + 10 mm (meta.json hybrid.place.insert.above_mm = 꽂은 순간 손끝 높이 − z_rel)
"""
import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
M = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('media_1112')
TAG = sys.argv[2] if len(sys.argv) > 2 else 'T10-hyb-3'
FR = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
f_t = ImageFont.truetype(FB, 32, index=1)
f_s = ImageFont.truetype(FR, 24, index=1)
f_l = ImageFont.truetype(FB, 22, index=1)
f_c = ImageFont.truetype(FB, 24, index=1)
BG, INK, MUTE = (252, 252, 251), (11, 11, 11), (100, 98, 94)
P = json.loads((M / 'picked.json').read_text())[TAG]
ST = next(r for r in csv.DictReader(open(ROOT / 'data' / 'stage_runs.csv')) if r['run'] == TAG)
TL = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(M / 'timeline' / f'{TAG}.csv'))]
H = P.get('hybrid') or {}
place_ev = next((e for e in H.get('events', []) if e.get('kind') == 'place' and e.get('t') is not None), None)
hole = (H.get('place') or {}).get('target')          # E4: 수식이 쓴 구멍 위치 = 단계 판정 S8 기준 (stage_runs 각주와 같음)
if hole is None:
    raise SystemExit('이 판에는 구멍 위치 기록이 없음 — E4 판을 쓰세요')

# S8: 쥔 채(폭 > 12 mm) 칸 수평 20 mm 안 → 첫 통과 ~ 수식 넘김 전 구간에서 칸에 가장 가까운 프레임
dh = lambda r: math.hypot(r['x_m'] - hole[0], r['y_m'] - hole[1])
t_end = place_ev['t'] if place_ev else P['open_t_s'] - 1.0
win = [r for r in TL if r['t_s'] > P['close_t_s'] + 3 and r['grip_w_m'] > 0.012 and r['t_s'] < t_end and dh(r) <= 0.020]
s8 = min(win, key=dh)
s8_idx = int(round(s8['t_s'] * 10))
s8_first = min(win, key=lambda r: r['t_s'])['t_s']
z_ins = (H.get('place') or {}).get('insert', {}).get('above_mm')

SHOTS = [
    ('S0', 'task_S0_start.jpg', '초기 위치', 'photo', 'start', P['photos']['start']['t_s'],
     f"홈 자세에서 시작 · 시작 장면에서 빨간 튜브와 빨간 링 칸을 찾음 · 첫 3인칭 밝기 {float(ST['start_brightness']):.0f} (기준 132 + 15 이하) · 포화 {float(ST['start_saturation_pct']):.1f}% (기준 15% 미만)"),
    ('S5', 'task_S5_grasp.jpg', '파지', 'photo', 'grasped', P['photos']['grasped']['t_s'],
     f"빨간 튜브 뚜껑을 쥠 · 닫기 명령 3 s 뒤 손가락 폭 {float(ST['w3_mm']):.1f} mm (기준 20–50 mm) · 들기 시작까지 12 mm 이상 유지"),
    ('S8', 'task_S8_transport.jpg', '파지 후 이송', 'video', s8_idx, s8['t_s'],
     f"쥔 채 빨간 칸 위로 옮김 · 칸에서 수평 {dh(s8) * 1000:.0f} mm · 손끝 높이 {s8['z_m'] * 1000:.0f} mm (기준: 수평 20 mm 안, {s8_first:.1f} s에 처음 통과) · 영상 프레임(480 px)을 키운 사진"),
    ('S9', 'task_S9_place.jpg', '배치', 'photo', 'end', P['photos']['end']['t_s'],
     f"빨간 링 구멍에 꽂고 놓은 뒤 위로 약 5 cm 뺀 모습 · 꽂은 순간 손끝 높이 = 계획한 놓는 높이 {z_ins:+.1f} mm (끝까지 들어감) · 사람 판정 성공"
     if z_ins is not None else "빨간 링 구멍에 꽂고 놓은 뒤 위로 뺀 모습 · 사람 판정 성공"),
]


def load(kind, key, cam):
    if kind == 'photo':
        return Image.open(M / 'photo' / TAG / f'{key}_{cam}.jpg').convert('RGB')
    im = Image.open(M / 'video' / TAG / cam / f'{key:06d}.jpg').convert('RGB')
    return im.resize((848, round(848 * im.height / im.width)), Image.LANCZOS)


for sid, fn, name, kind, key, ts, sub in SHOTS:
    a, b = load(kind, key, 'third'), load(kind, key, 'wrist')
    w, h = 848, 480
    gap, top, pad = 16, 108, 20
    im = Image.new('RGB', (pad * 2 + w * 2 + gap, top + h + pad), BG)
    d = ImageDraw.Draw(im)
    chip = f' {sid} '
    cw = int(d.textlength(chip, font=f_c)) + 8
    d.rounded_rectangle((pad, 14, pad + cw, 52), radius=8, fill=INK)
    d.text((pad + 8, 17), chip, font=f_c, fill=(255, 255, 255))
    d.text((pad + cw + 14, 14), f'{name}  ·  E4 VLA + 수식 {TAG} 성공 판 ({ts:.1f} s)', font=f_t, fill=INK)
    d.text((pad, 64), sub, font=f_s, fill=MUTE)
    for k, (src, lab) in enumerate(((a, '3인칭 · third-person'), (b, '손목 · wrist'))):
        x = pad + k * (w + gap)
        im.paste(src.crop((0, 0, w, min(h, src.height))), (x, top))
        bw = int(d.textlength(lab, font=f_l)) + 20
        ov = Image.new('RGBA', (bw, 34), (11, 11, 11, 158))
        im.paste(ov, (x + 10, top + 10), ov)
        d.text((x + 20, top + 13), lab, font=f_l, fill=(255, 255, 255))
    out = ROOT / 'figs' / fn
    im.save(out, quality=88)
    print(f'✔ {out.name} {im.size} ({out.stat().st_size / 1e3:.0f} kB)  t = {ts:.1f} s')
