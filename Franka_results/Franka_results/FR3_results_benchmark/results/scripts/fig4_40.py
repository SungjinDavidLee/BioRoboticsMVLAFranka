"""그림 4 (40회판) — 실물 평가 결과: 처음 30회(배치 1~10 × 3) + 추가 10회(배치 11~13 × 3, 14 × 1)
(a) 단계별 도달 수  (b) 배치별 결과 — 분포 안(학습과 튜브·구멍 위치 35 mm 안) / 분포 밖으로 묶음
색: 결과는 순서형(성공 > 부분 > 실패) — 한 색상의 밝기 3단계 (validate_palette.js --ordinal 통과)"""
from svgkit import SVG, C, INK, MUTE, LINE, render

SUC, PAR, FAI = '#7a3b00', '#c96a06', '#eba25a'
COL = {'S': SUC, 'P': PAR, 'F': FAI}
RES = {1: 'SSS', 2: 'SSS', 3: 'PPP', 4: 'FFF', 5: 'PSS', 6: 'FFS', 7: 'PPP', 8: 'FFF', 9: 'PPF', 10: 'PSS',
       11: 'SSS', 12: 'PSS', 13: 'PPP', 14: 'S'}
IN = [1, 2, 6, 10, 11, 14]
OUT = [3, 4, 5, 7, 8, 9, 12, 13]
N = 40
T = {
 'ko': {
  'a': '(a) 단계별 도달 (40회)',
  'stages': [('시작', '배치 14개 · 40회', 40), ('잡기', '닫고 3 s 뒤 폭 20–50 mm', 31), ('들기', '쥔 채 3 cm 이상', 29), ('꽂고 놓기 = 성공', '끝 장면에서 구멍 안', 17)],
  'loss': ['−9  다른 튜브 쪽 6 · 빨간 튜브 옆 헛잡음 3', '−2  쥔 채 스테이션 제한 시간 중단', '−12  구멍 위 막힘·얹힘 8 · 옆 구멍 3 · 높은 곳에서 놓음 1'],
  'b': '(b) 배치별 결과 (위에서부터 회차)', 'in': '분포 안 — 13/16', 'out': '분포 밖 — 4/24',
  'lay': '배치', 'note': '* 세 번 모두 튜브 랙 반대쪽 끝(다른 튜브 쪽)으로 감 · 배치 11~14는 추가 10회',
  'leg': [('S', '성공'), ('P', '부분 — 쥐었지만 못 꽂음'), ('F', '실패 — 못 쥠')],
  'ab': {'S': '성', 'P': '부', 'F': '실'},
  'sum': ['**분포 안 13/16 = 81%** (Wilson 95% 57–93%) · **분포 밖 4/24 = 17%** (7–36%) · **전체 17/40 = 42.5%** (29–58%)',
          '처음 30회(배치 1~10 × 3) 11/30 + 추가 10회(학습 배치를 보고 재현한 배치 11~13 × 3, 14 × 1) 6/10 · 전체의 배치 단위 부트스트랩 95% 21–65%',
          '분포 안 = 학습 100판 중 튜브·구멍 위치가 함께 35 mm 안인 판이 있는 배치(기준은 처음 30회 뒤에 정함) · 참고: 같은 배치 반복(예비) 8/8'],
 },
 'en': {
  'a': '(a) Stages reached (40 trials)',
  'stages': [('Start', '14 layouts · 40 trials', 40), ('Grasp', 'width 20–50 mm 3 s after closing', 31), ('Lift', '≥ 3 cm while holding', 29), ('Insert & release', 'in the hole in the final scene', 17)],
  'loss': ['−9  went to the other tube 6 · missed the red tube 3', '−2  station gripper timeout while holding', '−12  blocked above the hole 8 · neighbouring hole 3 · released high 1'],
  'b': '(b) Results per layout (trials top to bottom)', 'in': 'in-distribution — 13/16', 'out': 'out-of-distribution — 4/24',
  'lay': 'Layout', 'note': '* all three went to the far end of the tube rack (the other tube) · layouts 11–14 are the added 10 trials',
  'leg': [('S', 'success'), ('P', 'partial — grasped, not inserted'), ('F', 'failure — no grasp')],
  'ab': {'S': 'S', 'P': 'P', 'F': 'F'},
  'sum': ['**In-distribution 13/16 = 81%** (Wilson 95% 57–93%) · **out-of-distribution 4/24 = 17%** (7–36%) · **overall 17/40 = 42.5%** (29–58%)',
          'First 30 trials (layouts 1–10 × 3) 11/30 + 10 added trials (training layouts recreated by eye: 11–13 × 3, 14 × 1) 6/10 · layout bootstrap 95% (overall) 21–65%',
          'In-distribution = a training episode within 35 mm in tube + hole position (rule set after the first 30 trials) · reference: repeated fixed layout (pilot) 8/8'],
 }}


def main(lang):
    L = T[lang]
    W, H = 1320, 400
    s = SVG(W, H, lang)
    s.text(14, 28, L['a'], size=16.5, weight=700)
    x0, y0, bw, bh, gap = 196, 52, 270, 24, 54
    for k in (0, 10, 20, 30, 40):
        xx = x0 + bw * k / N
        s.line(xx, y0 - 8, xx, y0 + 3 * gap + bh + 6, color='#e6e8eb', sw=1, arrow=False)
        s.text(xx, y0 + 3 * gap + bh + 22, str(k), size=11.5, color=MUTE, anchor='middle')
    s.text(x0 + bw / 2, y0 + 3 * gap + bh + 40, ('시행 수' if lang == 'ko' else 'trials'), size=11.5, color=MUTE, anchor='middle')
    for i, (nm, sub, v) in enumerate(L['stages']):
        y = y0 + i * gap
        s.text(x0 - 12, y + 12, nm, size=13.8, weight=700, anchor='end')
        s.text(x0 - 12, y + 30, sub, size=11.4, color=MUTE, anchor='end')
        s.rect(x0, y, bw, bh, fill='#f3f4f6', stroke='none', sw=0, rx=4)
        s.rect(x0, y, bw * v / N, bh, fill=C['run'][0] if i else '#9aa1ac', stroke='none', sw=0, rx=4)
        s.text(x0 + bw * v / N + 8, y + 17, f'{v}/{N}' + (f' ({100 * v / N:.0f}%)' if i else ''), size=13, weight=700)
        if i < len(L['loss']):
            s.text(x0 + bw * L['stages'][i + 1][2] / N + 6, y + bh + 17, '↓ ' + L['loss'][i], size=11.6, color=MUTE)
    bx0 = 640
    s.text(bx0, 28, L['b'], size=16.5, weight=700)
    sq, g, cw = 30, 4, 42
    cx0, cy0 = bx0 + 44, 84
    s.text(cx0 - 10, cy0 + 3 * (sq + g) + 18, L['lay'], size=12, color=MUTE, anchor='end')
    x = cx0
    for grp, title in ((IN, L['in']), (OUT, L['out'])):
        gx0 = x
        for lay in grp:
            r = RES[lay]
            s.text(x + sq / 2, cy0 - 8, f"{r.count('S')}/{len(r)}", size=11.8, weight=700, anchor='middle')
            for t, c in enumerate(r):
                y = cy0 + t * (sq + g)
                s.rect(x, y, sq, sq, fill=COL[c], stroke='#ffffff', sw=2, rx=4)
                s.text(x + sq / 2, y + sq / 2 + 5, L['ab'][c], size=11.8, weight=700, anchor='middle', color=INK if c == 'F' else '#ffffff')
            s.text(x + sq / 2, cy0 + 3 * (sq + g) + 18, str(lay) + ('*' if lay in (4, 8) else ''), size=12, anchor='middle')
            x += cw
        s.line(gx0, cy0 - 30, x - cw + sq, cy0 - 30, color='#8a919c', sw=1.2, arrow=False)
        s.text(gx0, cy0 - 36, title, size=12.4, weight=700)
        x += 22
    s.text(cx0, cy0 + 3 * (sq + g) + 42, L['note'], size=11.4, color=MUTE)
    ly = cy0 + 3 * (sq + g) + 68
    lx = cx0
    for (r, nm), wd in zip(L['leg'], (70, 196, 0) if lang == 'ko' else (86, 228, 0)):
        s.rect(lx, ly - 12, 16, 16, fill=COL[r], stroke='none', sw=0, rx=3)
        s.text(lx + 22, ly + 1, nm, size=12, color=INK)
        lx += wd
    s.line(14, 304, W - 14, 304, color='#e3e6ea', sw=1, arrow=False)
    s.text(14, 330, L['sum'], size=13, color=INK, lh=1.55)
    return s


if __name__ == '__main__':
    for lang in ('ko', 'en'):
        sv = main(lang)
        render(sv.svg(), f'evalfigs40/fig4_results40_{lang}', sv.w, sv.h)
    print('ok')
