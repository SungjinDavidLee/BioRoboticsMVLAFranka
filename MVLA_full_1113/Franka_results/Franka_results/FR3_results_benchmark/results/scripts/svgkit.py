"""작은 SVG 도우미 + Playwright 렌더 (PNG 3배, 벡터 PDF)"""
import html

INK, MUTE, LINE = '#1f2328', '#5b6370', '#c3c9d2'
C = {  # 단계 색 (진한 색, 옅은 바탕)
    'gen': ('#1d5fd1', '#edf3ff'), 'data': ('#0b8073', '#e7f6f3'),
    'train': ('#6f3cc4', '#f3edfc'), 'run': ('#c96a06', '#fff4e5'), 'gray': ('#5b6370', '#f4f5f7'),
}
FONT = {'ko': "'Noto Sans CJK KR','Noto Sans KR',sans-serif", 'en': "'Inter','Noto Sans CJK KR',sans-serif"}
MONO = "'DejaVu Sans Mono',monospace"


class SVG:
    def __init__(self, w, h, lang='ko'):
        self.w, self.h, self.lang, self.items = w, h, lang, []

    def add(self, s):
        self.items.append(s)

    def rect(self, x, y, w, h, fill='#fff', stroke=LINE, sw=1.2, rx=8, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        o = f' fill-opacity="{opacity}"' if opacity is not None else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{o} stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=15, weight=400, color=INK, anchor='start', mono=False, italic=False, lh=1.32, spacing=None):
        lines = s if isinstance(s, (list, tuple)) else [s]
        fam = MONO if mono else FONT[self.lang]
        st = ' font-style="italic"' if italic else ''
        ls = f' letter-spacing="{spacing}"' if spacing is not None else ''
        out = [f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{st}{ls}>']
        for i, ln in enumerate(lines):
            out.append(f'<tspan x="{x}" dy="{0 if i == 0 else size * lh:.1f}">{self._rich(ln)}</tspan>')
        out.append('</text>')
        self.add(''.join(out))

    @staticmethod
    def _rich(ln):
        # **굵게**, _{아래첨자}, ^{위첨자}
        import re
        s = html.escape(ln, quote=False)
        s = re.sub(r'\*\*(.+?)\*\*', r'<tspan font-weight="700">\1</tspan>', s)
        s = re.sub(r'_\{(.+?)\}', r'<tspan baseline-shift="sub" font-size="72%">\1</tspan>', s)
        s = re.sub(r'\^\{(.+?)\}', r'<tspan baseline-shift="super" font-size="72%">\1</tspan>', s)
        return s

    def line(self, x1, y1, x2, y2, color=INK, sw=1.6, dash=None, arrow=True, head='a'):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#{head}{color.strip("#")})"' if arrow else ''
        self._marker(color, head)
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def path(self, d, color=INK, sw=1.6, dash=None, arrow=True, fill='none', head='a', start_arrow=False):
        da = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#{head}{color.strip("#")})"' if arrow else ''
        if start_arrow:
            self._marker(color, 's')
            m += f' marker-start="url(#s{color.strip("#")})"'
        if arrow:
            self._marker(color, head)
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{da}{m} stroke-linejoin="round" stroke-linecap="round"/>')

    _markers = None

    def _marker(self, color, head):
        if self._markers is None:
            self._markers = {}
        key = head + color.strip('#')
        if key in self._markers:
            return
        if head == 's':
            self._markers[key] = (f'<marker id="{key}" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                                  f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
        else:
            self._markers[key] = (f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                                  f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')

    def circle(self, cx, cy, r, fill='none', stroke=INK, sw=1.4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def raw(self, s):
        self.add(s)

    def svg(self):
        defs = '<defs>' + ''.join((self._markers or {}).values()) + '</defs>'
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f'<rect width="100%" height="100%" fill="#ffffff"/>{defs}' + '\n'.join(self.items) + '</svg>')


def render(svg_text, base, w, h):
    """svg → base.svg / base.png(3배) / base.pdf(벡터)"""
    from playwright.sync_api import sync_playwright
    open(base + '.svg', 'w', encoding='utf-8').write(svg_text)
    page_html = f'<html><head><style>html,body{{margin:0;padding:0;background:#fff}} @page{{size:{w}px {h}px;margin:0}}</style></head><body>{svg_text}</body></html>'
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={'width': w, 'height': h}, device_scale_factor=3)
        pg.set_content(page_html)
        pg.wait_for_timeout(300)
        pg.screenshot(path=base + '.png', clip={'x': 0, 'y': 0, 'width': w, 'height': h})
        pg.pdf(path=base + '.pdf', width=f'{w}px', height=f'{h}px', print_background=True)
        b.close()
