#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""video_4exp.py — 실험별 성공 판 실행 영상 (3인칭 + 손목 나란히, 실제 속도) + README용 미리보기

  python3 scripts/video_4exp.py <media_1112 폴더> <paper_media_1109 폴더> [E1 E2 E3 E4]
  → media/<실험>_<판>.mp4 (H.264, 10 fps = 기록 주기 그대로), media/<실험>_<판>_x3.gif (3배속), media/<실험>_<판>_poster.jpg
입력: <폴더>/video/<판>/{third,wrist}/NNNNNN.jpg (가로 480 px), <폴더>/timeline/<판>.csv, <폴더>/picked.json
      (media_1112 = scripts/media_export_1112.py, paper_media_1109 = 11-09 재료)
"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
M12 = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('media_1112')
M09 = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('paper_media_1109')
ONLY = set(sys.argv[3:])
RUNS = [  # 실험, 판, 제목, 영문
    ('E1', 'T07-v1-1', M12, 'E1 · 실물 100판 학습 VLA · T07 · 성공', 'GR00T N1.7 fine-tuned on 100 real demos · held-out layout T07 · success'),
    ('E2', 'T06-v2-1', M09, 'E2 · 실물 200판 학습 VLA · T06 · 성공', 'GR00T N1.7 fine-tuned on 200 real demos · held-out layout T06 · success'),
    ('E3', 'T06-mix-3', M12, 'E3 · 실물 200판 + 시뮬 600판 학습 VLA · T06 · 성공', 'GR00T N1.7 co-trained on real + sim demos · held-out layout T06 · success'),
    ('E4', 'T06-hyb-2', M12, 'E4 · VLA(E2 모델) + 수식 · T06 · 성공', 'E2 model + scripted grasp / insert hand-off · held-out layout T06 · success'),
]
FR = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
KR = 1                                       # ttc 안 한국어 글꼴 번호
BG, INK, INK2, MUTE = (252, 252, 251), (11, 11, 11), (82, 81, 78), (137, 135, 129)
SH = [(225, 224, 217), (195, 194, 183)]
VLA_C, FOR_C = (205, 226, 251), (250, 214, 196)     # 막대: VLA 파랑 연하게, 수식 주황 연하게
f_t = ImageFont.truetype(FB, 19, index=KR)
f_s = ImageFont.truetype(FR, 13, index=KR)
f_l = ImageFont.truetype(FB, 14, index=KR)
f_p = ImageFont.truetype(FR, 13, index=KR)
f_pb = ImageFont.truetype(FB, 13, index=KR)
PW, PHH, GAP = 480, 270, 16
W, H = PW * 2 + GAP, 432
Y0 = 66


def phases(exp, tg, M):
    R = list(csv.DictReader(open(M / 'timeline' / f'{tg}.csv')))
    t = [float(r['t_s']) for r in R]
    z = [float(r['z_m']) for r in R]
    g = [float(r['grip_cmd']) for r in R]
    ic = next(i for i, v in enumerate(g) if v < 0.5)
    io = next(i for i in range(ic, len(g)) if g[i] > 0.5)
    il = next(i for i in range(ic, len(z)) if z[i] > z[ic] + 0.005)
    T = t[-1]
    if exp != 'E4':
        ph = [(0, t[ic], '접근 · approach', 0), (t[ic], t[il], '쥐기 · grasp', 1),
              (t[il], t[io], '옮기기 · 꽂기 · carry & insert', 0), (t[io], T, '놓기 · release', 1)]
        foot = f'VLA만 · 닫기 명령 {t[ic]:.1f} s → 들기 시작 {t[il]:.1f} s → 열기 명령 {t[io]:.1f} s  ·  Franka FR3 · 10 Hz record'
        return t, ph, foot
    P = json.loads((M / 'picked.json').read_text())[tg]['hybrid']
    ev = {e['kind']: float(e['t']) for e in P['events'] if e.get('t') is not None and not e.get('why')}
    g0, g1 = ev['grasp'], ev['grasp'] + float(P['grasp']['move_s'])
    p0, p1 = ev['place'], ev['place'] + float(P['place']['move_s'])
    ph = [(0, g0, 'VLA 접근', 'v'), (g0, g1, '수식 잡기·들기', 'f'), (g1, p0, 'VLA 옮기기', 'v'),
          (p0, min(p1, T), '수식 꽂기', 'f')]
    if p1 < T - 0.3:
        ph.append((p1, T, '끝', 0))
    foot = (f'수식이 넘겨받음: 잡기 {g0:.1f} s · 꽂기 {p0:.1f} s  ·  닫기 {t[ic]:.1f} s · 열기 {t[io]:.1f} s'
            f'  ·  Franka FR3 · 10 Hz record')
    return t, ph, foot


def render(exp, tg, M, ttl, sub):
    t, PH, foot = phases(exp, tg, M)
    T_END = t[-1]

    def frame(i):
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((16, 10), ttl, font=f_t, fill=INK)
        d.text((16, 38), sub, font=f_s, fill=MUTE)
        for k, cam, lab in ((0, 'third', '3인칭 · third-person'), (1, 'wrist', '손목 · wrist')):
            x = k * (PW + GAP)
            src = Image.open(M / 'video' / tg / cam / f'{i:06d}.jpg').convert('RGB').crop((0, 0, PW, PHH))
            im.paste(src, (x, Y0))
            bw = d.textlength(lab, font=f_l) + 16
            ov = Image.new('RGBA', (int(bw), 24), (11, 11, 11, 158))
            im.paste(ov, (x + 8, Y0 + 8), ov)
            d.text((x + 16, Y0 + 9), lab, font=f_l, fill=(255, 255, 255))
        ts = f'{t[i]:5.1f} s'
        bw = d.textlength(ts, font=f_l) + 16
        ov = Image.new('RGBA', (int(bw), 24), (11, 11, 11, 158))
        im.paste(ov, (int(PW - 8 - bw), Y0 + 8), ov)
        d.text((PW - bw, Y0 + 9), ts, font=f_l, fill=(255, 255, 255))
        bx, bwid, by = 16, W - 32, Y0 + PHH + 14
        sx = lambda v: bx + bwid * v / T_END  # noqa: E731
        cur = next((k for k, p in enumerate(PH) if t[i] <= p[1]), len(PH) - 1)
        for k, (a, b, lab, c) in enumerate(PH):
            col = VLA_C if c == 'v' else FOR_C if c == 'f' else SH[c]
            d.rectangle([sx(a), by, sx(b) - 1, by + 12], fill=col)
            f = f_pb if k == cur else f_p
            tw = d.textlength(lab, font=f)
            cx = min(max((sx(a) + sx(b)) / 2 - tw / 2, bx), bx + bwid - tw)
            d.text((cx, by + 20), lab, font=f, fill=INK if k == cur else MUTE)
        px = sx(t[i])
        d.rectangle([bx, by + 12, px, by + 14], fill=INK)
        d.polygon([(px - 6, by - 7), (px + 6, by - 7), (px, by + 1)], fill=INK)
        d.text((16, H - 28), foot, font=f_s, fill=MUTE)
        return im

    out = ROOT / 'media'
    out.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    n = len(t)
    hold = 10                                  # 끝 장면 1 s 멈춤
    for i in range(n):
        frame(i).save(tmp / f'{i:06d}.png')
    for k in range(hold):
        shutil.copy(tmp / f'{n - 1:06d}.png', tmp / f'{n + k:06d}.png')
    name = f'{exp}_{tg}'
    mp4 = out / f'{name}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '10', '-i', str(tmp / '%06d.png'),
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(mp4)], check=True)
    gif = out / f'{name}_x3.gif'
    vf = ('select=not(mod(n\\,3)),setpts=N/10/TB,scale=560:-1:flags=lanczos,split[a][b];'
          '[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '10', '-i', str(tmp / '%06d.png'),
                    '-vf', vf, '-r', '10', str(gif)], check=True)
    mid = min(range(n), key=lambda j: abs(t[j] - 0.55 * T_END))
    frame(mid).save(out / f'{name}_poster.jpg', quality=90)
    shutil.rmtree(tmp)
    for p in (mp4, gif):
        print(f'✔ {p.name} ({p.stat().st_size / 1e6:.2f} MB, {T_END:.1f} s)')


if __name__ == '__main__':
    for exp, tg, M, ttl, sub in RUNS:
        if ONLY and exp not in ONLY:
            continue
        if not (M / 'video' / tg).exists():
            print(f'  ⚠ {exp} {tg}: 영상 재료 없음 ({M})')
            continue
        render(exp, tg, M, ttl, sub)
