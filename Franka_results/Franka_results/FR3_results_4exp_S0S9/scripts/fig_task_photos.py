#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_task_photos.py — 과제 사진 3장: 초기 위치 · 파지 · 배치(놓기 직전) — 각각 3인칭 + 손목 원본 사진 한 장씩

  python3 scripts/fig_task_photos.py <사진 폴더> <판 메모> <picked.json> [제목 꼬리]
  예) python3 scripts/fig_task_photos.py media_1112/photo/T10-hyb-3 T10-hyb-3 media_1112/picked.json "E4 VLA + 수식"
  → figs/task_1_start.jpg, figs/task_2_grasp.jpg, figs/task_3_place.jpg
사진 이름: <순간>_<카메라>.jpg 또는 <순간>_<카메라>_<프레임>.jpg (start · grasped · pre_release)
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC, TAG, PICK = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
TAIL = sys.argv[4] if len(sys.argv) > 4 else ''
FR = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
f_t = ImageFont.truetype(FB, 30, index=1)
f_s = ImageFont.truetype(FR, 21, index=1)
f_l = ImageFont.truetype(FB, 22, index=1)
BG, INK, MUTE = (252, 252, 251), (11, 11, 11), (110, 108, 103)
P = json.loads(PICK.read_text())[TAG]
FR_KEY = P.get('photos') or P.get('frames')
SHOTS = [('start', 'task_1_start.jpg', '① 초기 위치', 'Initial position — 홈 자세에서 시작, 빨간 튜브와 빨간 링으로 표시한 빈 칸'),
         ('grasped', 'task_2_grasp.jpg', '② 파지', 'Grasp — 빨간 튜브의 뚜껑을 쥐고 닫기 명령 3 s 뒤'),
         ('pre_release', 'task_3_place.jpg', '③ 배치', 'Place — 빨간 링 칸에 꽂고 놓기 직전 (열기 명령 0.3 s 전)')]


def find(key, cam):
    for p in sorted(SRC.glob(f'{key}_{cam}*.jpg')):
        return Image.open(p).convert('RGB')
    raise SystemExit(f'{SRC}/{key}_{cam}*.jpg 없음')


for key, fn, ttl, sub in SHOTS:
    a, b = find(key, 'third'), find(key, 'wrist')
    w, h = a.size
    gap, top, pad = 16, 96, 20
    im = Image.new('RGB', (pad * 2 + w * 2 + gap, top + h + pad), BG)
    d = ImageDraw.Draw(im)
    ts = FR_KEY[key]['t_s']
    d.text((pad, 14), f'{ttl}  ·  {TAG} ({ts:.1f} s){"  ·  " + TAIL if TAIL else ""}', font=f_t, fill=INK)
    d.text((pad, 56), sub, font=f_s, fill=MUTE)
    for k, (src, lab) in enumerate(((a, '3인칭 · third-person'), (b, '손목 · wrist'))):
        x = pad + k * (w + gap)
        im.paste(src, (x, top))
        bw = int(d.textlength(lab, font=f_l)) + 20
        ov = Image.new('RGBA', (bw, 34), (11, 11, 11, 158))
        im.paste(ov, (x + 10, top + 10), ov)
        d.text((x + 20, top + 13), lab, font=f_l, fill=(255, 255, 255))
    out = ROOT / 'figs' / fn
    im.save(out, quality=88)
    print(f'✔ {out.name} {im.size} ({out.stat().st_size / 1e3:.0f} kB)')
