#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_summary.py — ⑤ 모델 실행 평가(배치 10개 × 3회) 결과 정리 (로봇 PC, 읽기만 — 로봇 안 움직임)

  python3 eval_summary.py                                  # ~/gello_data/medbio_redslot_v1/_rollouts
  python3 eval_summary.py --dir <폴더> > ~/eval_result.md

- 결과는 사람이 확인·고친 판정(meta.json의 result)을 씀. 무효는 뺌
- 배치·회차는 메모의 '배치-회차'(예: 3-2)에서 읽음
- 단계(잡기·들기)는 자동 판정 기록에서, 실패 유형은 메모 + 자동 판정 이유에서
"""
import argparse
import collections
import csv
import json
import math
import re
from pathlib import Path

KO = {'success': '성공', 'partial': '부분', 'fail': '실패', 'void': '무효'}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ftype(m):
    """실패·부분 유형 (메모 우선, 없으면 자동 판정 이유로)"""
    note = (m.get('note') or '')
    J = m.get('judge') or {}
    why = (J.get('reason') or '') + ' ' + (m.get('reason') or '')
    txt = note + ' ' + why
    rules = [('다른 튜브', '다른 튜브로 감'), ('헛잡', '헛잡음'), ('걸림', '뚜껑·옆 튜브에 걸림'), ('떨어뜨', '옮기다 떨어뜨림'),
             ('얹', '테두리에 얹힘(덜 꽂힘)'), ('다른 곳', '칸이 아닌 곳에 놓음'), ('아닌 곳', '칸이 아닌 곳에 놓음'),
             ('못 꽂', '칸에 못 꽂음'), ('시간 안에 못 놓', '꽂았지만 시간 안에 못 놓음'), ('안 놓', '꽂았지만 시간 안에 못 놓음'),
             ('작업 범위', '안전장치 중단(범위·속도·기울기)'), ('움직임 (한도', '안전장치 중단(범위·속도·기울기)'),
             ('기울어짐', '안전장치 중단(범위·속도·기울기)'), ('못 들어 올림', '쥐었지만 못 들어 올림'), ('최대 시간', '시간 초과'),
             ('쥐기 시도 없음', '쥐기 시도 없음'), ('멈춤', '사람이 멈춤'), ('컨트롤러', '컨트롤러 꺼짐')]
    for key, name in rules:
        if key in txt:
            return name
    return '기타 (' + (why.strip() or note)[:40] + ')'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='~/gello_data/medbio_redslot_v1/_rollouts')
    a = ap.parse_args()
    R = Path(a.dir).expanduser()
    ms = []
    for f in sorted(R.glob('rollout_*/meta.json')):
        try:
            m = json.loads(f.read_text())
            m['folder'] = f.parent.name
            ms.append(m)
        except Exception:
            pass
    valid = [m for m in ms if m.get('result') in ('success', 'partial', 'fail')]
    void = [m for m in ms if m.get('result') == 'void']
    nores = [m for m in ms if m.get('result') not in ('success', 'partial', 'fail', 'void')]
    n = len(valid)
    k = sum(1 for m in valid if m['result'] == 'success')
    lo, hi = wilson(k, n)
    ck = collections.Counter(str(m.get('ckpt') or (m.get('server') or {}).get('ckpt') or '?').split('checkpoints/')[-1].split('/')[0]
                             for m in valid)
    cond = collections.Counter(f"{m.get('exec_steps') or (m.get('cfg') or {}).get('exec_steps')}칸·"
                               f"{float(m.get('max_s') or (m.get('cfg') or {}).get('max_s') or 0):.0f}초·"
                               f"{'연속' if (m.get('mode') or (m.get('cfg') or {}).get('mode')) == 'async' else '정지'}" for m in valid)
    print(f'# 평가 결과 ({R})\n')
    print(f'- 체크포인트: {dict(ck)} · 실행 조건: {dict(cond)}')
    print(f'- 유효 {n}회 · 무효 {len(void)}회 (통계 제외)' + (f' · 판정 없음 {len(nores)}회' if nores else ''))
    print(f'- **성공 {k}/{n} = {100 * k / max(n, 1):.0f}% (Wilson 95% 구간 {100 * lo:.0f}–{100 * hi:.0f}%)** · '
          f"부분 {sum(1 for m in valid if m['result'] == 'partial')} · 실패 {sum(1 for m in valid if m['result'] == 'fail')}")
    retry = sum(1 for m in valid if m['result'] == 'success' and '재시도' in (m.get('note') or ''))
    if retry:
        print(f'- 그중 재시도 후 성공 {retry}회 → 한 번에 성공 {k - retry}/{n}')

    # 단계별 (잡기·들기는 기록 기준 — 폭·높이라 기준점 오차와 무관, 꽂기는 사람 판정의 성공)
    def stg(m, key):
        for s in (m.get('judge') or {}).get('stages', []):
            if s.get('key') == key:
                return s.get('ok')
        return None
    g = sum(1 for m in valid if stg(m, 'grasp') is True)
    li = sum(1 for m in valid if stg(m, 'lift') is True)
    print(f'\n## 단계별\n\n| 단계 | 된 횟수 | 비율 |\n|---|---|---|')
    print(f'| 잡기 (닫고 3초 뒤 폭 20~50mm) | {g}/{n} | {100 * g / max(n, 1):.0f}% |')
    print(f'| 들기 (쥔 채 3cm↑) | {li}/{n} | {100 * li / max(n, 1):.0f}% |')
    print(f'| 꽂고 놓기 (최종 성공) | {k}/{n} | {100 * k / max(n, 1):.0f}% |')
    if li:
        print(f'\n- 들어 올린 뒤 꽂기까지 성공: {k}/{li} ({100 * k / li:.0f}%) · 잡기 성공 중 들기: {li}/{max(g, 1)}')

    # 배치별
    lay = collections.defaultdict(list)
    for m in valid:
        mm = re.search(r'(\d+)\s*-\s*(\d+)', m.get('note') or '')
        lay[int(mm.group(1)) if mm else 0].append((int(mm.group(2)) if mm else 0, m))
    print(f'\n## 배치별\n\n| 배치 | 결과 | 회차별 | 메모 |\n|---|---|---|---|')
    full = 0
    for L in sorted(lay):
        rows = sorted(lay[L], key=lambda x: x[0])
        s = sum(1 for _, m in rows if m['result'] == 'success')
        full += int(s == len(rows) == 3)
        name = f'{L}' if L else '(메모 없음)'
        print(f"| {name} | {s}/{len(rows)} | {' · '.join(KO[m['result']] for _, m in rows)} | "
              f"{'; '.join((m.get('note') or '').strip() for _, m in rows if (m.get('note') or '').strip())} |")
    n_lay = len([L for L in lay if L])
    print(f'\n- 3회 모두 성공한 배치 {full}/{n_lay} · 한 번이라도 성공한 배치 '
          f"{sum(1 for L in lay if L and any(m['result'] == 'success' for _, m in lay[L]))}/{n_lay} · "
          f"모두 실패·부분인 배치 {sum(1 for L in lay if L and not any(m['result'] == 'success' for _, m in lay[L]))}/{n_lay}")
    if 0 in lay:
        print(f'- ⚠ 메모에 배치-회차가 없는 판 {len(lay[0])}개 — ⑤ 탭에서 메모를 채우면 배치별로 묶임')

    # 실패 유형
    bad = [m for m in valid if m['result'] != 'success']
    if bad:
        c = collections.Counter(ftype(m) for m in bad)
        print(f'\n## 성공 못 한 {len(bad)}회의 유형\n\n| 유형 | 횟수 |\n|---|---|')
        for t, v in c.most_common():
            print(f'| {t} | {v} |')
    if void:
        print(f"\n- 무효 사유: {'; '.join(((m.get('note') or m.get('reason') or '?')[:40]) for m in void)}")

    # 시간·자동 판정 일치
    dur = sorted(m.get('duration_s') for m in valid if m['result'] == 'success' and m.get('duration_s'))
    if dur:
        med = dur[len(dur) // 2] if len(dur) % 2 else (dur[len(dur) // 2 - 1] + dur[len(dur) // 2]) / 2
        print(f'\n## 기타\n\n- 성공 판 소요 시간: 중간 {med:.1f}초 (범위 {dur[0]:.1f}–{dur[-1]:.1f})')
    lat = sorted(m.get('total_ms_median') for m in valid if m.get('total_ms_median'))
    if lat:
        print(f'- 서버 왕복 지연(판별 중간값): 중간 {lat[len(lat) // 2]:.0f}ms')
    hum = [m for m in valid if m.get('result_auto')]
    agree = sum(1 for m in hum if m['result'] == m['result_auto'])
    if hum:
        print(f'- 자동 판정과 최종(사람 확인) 판정 일치: {agree}/{len(hum)} ({100 * agree / len(hum):.0f}%)')
        dif = collections.Counter(f"{KO.get(m['result_auto'])}→{KO.get(m['result'])}" for m in hum if m['result'] != m['result_auto'])
        if dif:
            print(f'  - 고친 것: {dict(dif)}')
    # csv
    out = R / 'eval_summary.csv'
    try:
        with open(out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['folder', 'time', 'layout', 'trial', 'result', 'result_auto', 'grasp', 'lift', 'duration_s', 'type', 'note'])
            for L in sorted(lay):
                for t, m in sorted(lay[L], key=lambda x: x[0]):
                    w.writerow([m['folder'], (m.get('started_at') or '')[11:16], L or '', t or '', m['result'], m.get('result_auto'),
                                stg(m, 'grasp'), stg(m, 'lift'), m.get('duration_s'),
                                '' if m['result'] == 'success' else ftype(m), m.get('note', '')])
        print(f'\n(표: {out})')
    except Exception as e:
        print(f'(csv 저장 실패: {e})')


if __name__ == '__main__':
    main()
