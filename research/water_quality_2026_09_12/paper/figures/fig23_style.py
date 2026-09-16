"""Python-only publication styling and traceability for Figures 2 and 3.

Uses the current Figure 1 semantic palette; does not mutate paperfig defaults.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paperfig as pf

INK, GREY, MID, LIGHT = '#30363D', '#707A84', '#AEB7BF', '#D6DCE1'
CYAN, TEAL, GOLD = '#56B4E9', '#009E73', '#E69F00'
VERM, PURPLE, MAGENTA, NAVY = '#D55E00', '#7B61A8', '#CC79A7', '#2F5D7C'
PALE_CYAN, PALE_GOLD, PALE_PURPLE, PALE_TEAL = '#EFF8FC', '#FFF7E5', '#F5F0F8', '#EAF7F2'
DASH, DOT = (0, (3, 1.8)), (0, (1, 1.6))
SOURCES = {}


def setup():
    pf.setup()
    mpl.rcParams.update({'axes.grid': False, 'axes.linewidth': .55,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white',
        'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7.2,
        'ytick.labelsize': 7.2, 'legend.fontsize': 7.2, 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def load(path):
    path = Path(path)
    SOURCES[path.relative_to(pf.RES).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))


def record_file(path):
    path = Path(path)
    SOURCES[path.relative_to(pf.REPO).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()


def axis(fig, x, y, w, h):
    W, H = fig.get_size_inches() * 25.4
    return fig.add_axes([x/W, y/H, w/W, h/H])


def text(fig, x, y, value, size=7.5, color=INK, weight='normal', **kw):
    W, H = fig.get_size_inches() * 25.4
    return fig.text(x/W, y/H, value, fontsize=size, color=color, fontweight=weight,
                    va=kw.pop('va', 'center'), **kw)


def panel(fig, x, y, letter, title):
    text(fig, x, y, letter, 9, weight='bold')
    text(fig, x+5, y, title, 8, weight='bold')


def style(ax, pale=None, grid='y'):
    ax.set_facecolor(pale or 'white')
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GREY)
    ax.tick_params(length=2, width=.5, pad=2)
    if grid:
        ax.grid(axis=grid, which='major', color='#E4E9ED', lw=.4)
    ax.set_axisbelow(True)


def key(fig, x, y, label, color=GREY, marker=None, ls='-', filled=False, size=7.2):
    W, H = fig.get_size_inches()*25.4
    fig.add_artist(Line2D([x/W, (x+4)/W], [y/H, y/H], transform=fig.transFigure,
        color=color, lw=.9, ls=ls, marker=marker, markevery=[1], ms=3.7,
        mfc=color if filled else 'white', mec=color, mew=.7))
    text(fig, x+5.5, y, label, size, color=GREY if color in (LIGHT, MID) else color)


def source_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (pf.OUT/name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def finish(fig, name, contract, extra=None):
    """Fixed-size export; report visible text outside the page, glyph size and resolution."""
    import fitz
    from PIL import Image
    import xml.etree.ElementTree as ET
    W, H = fig.get_size_inches()*25.4
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(pf.OUT/f'{name}.{ext}', dpi=600)
    pf._png_to_tif(pf.OUT/f'{name}.png', pf.OUT/f'{name}.tif')
    fig.savefig(pf.OUT/f'{name}_preview.png', dpi=220)
    with fitz.open(pf.OUT/f'{name}.pdf') as doc:
        page = doc[0]
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                 for line in b['lines'] for s in line['spans']]
        outside = [s['text'] for s in spans if s['bbox'][0] < 0 or s['bbox'][1] < 0
                   or s['bbox'][2] > page.rect.width or s['bbox'][3] > page.rect.height]
        dimensions = [round(page.rect.width/72*25.4, 3), round(page.rect.height/72*25.4, 3)]
    with Image.open(pf.OUT/f'{name}.tif') as im:
        pixels, dpi = im.size, [float(v) for v in im.info['dpi']]
    qa = {'dimensions_mm': dimensions, 'minimum_glyph_pt': min(s['size'] for s in spans),
          'text_outside_page': outside, 'svg_live_text_nodes': len(ET.parse(pf.OUT/f'{name}.svg').findall('.//{http://www.w3.org/2000/svg}text')),
          'tiff_pixels': pixels, 'tiff_dpi': dpi, 'preview_inspection_required': True}
    manifest = {'contract': contract, 'source_sha256': SOURCES,
                'source_roots': {'result_files': str(pf.RES), 'networks_prefix': str(pf.REPO)},
                'qa': qa, **(extra or {})}
    (pf.OUT/f'{name}_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (pf.OUT/f'{name}_qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa))
    assert not outside, outside
    assert qa['minimum_glyph_pt'] >= 4.9
    assert W <= 177.8 and H <= 220
    assert all(abs(float(d)-600) < 1e-6 for d in dpi)
    plt.close(fig)
