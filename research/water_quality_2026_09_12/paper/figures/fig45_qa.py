"""Read-only scientific/export checks, with machine-readable QA reports for Figures 4 and 5."""
from pathlib import Path
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import fitz
from PIL import Image

OUT = Path(__file__).resolve().parent / 'out'
for number, stem in [(4, 'fig4_exp2'), (5, 'fig5_exp3')]:
    manifest = json.loads((OUT / f'fig{number}_manifest.json').read_text(encoding='utf-8'))
    rows = {}
    for key, expected in manifest['data_rows'].items():
        with (OUT / f'fig{number}_source_{key}.csv').open(encoding='utf-8', newline='') as handle:
            rows[key] = sum(1 for _ in csv.DictReader(handle))
        assert rows[key] == expected
    for source, digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
    with fitz.open(OUT / f'{stem}.pdf') as doc:
        assert len(doc) == 1
        page = doc[0]
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        outside = [s['text'] for s in spans if s['bbox'][0] < -.1 or s['bbox'][1] < -.1 or
                   s['bbox'][2] > page.rect.width + .1 or s['bbox'][3] > page.rect.height + .1]
        dimensions = [page.rect.width / 72 * 25.4, page.rect.height / 72 * 25.4]
        minimum = min(s['size'] for s in spans)
        assert not outside, outside
        assert minimum >= 4.9 - .01, minimum
        assert abs(dimensions[0] - 176) < .1 and dimensions[1] <= 220
        page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(OUT / f'{stem}_qa_preview.png')
    svg = ET.parse(OUT / f'{stem}.svg')
    text_nodes = len(svg.findall('.//{http://www.w3.org/2000/svg}text'))
    assert text_nodes > 40
    with Image.open(OUT / f'{stem}.tif') as im:
        dpi = [float(v) for v in im.info['dpi']]
        assert dpi == [600, 600]
        pixels = list(im.size)
    report = {'dimensions_mm': [round(v, 2) for v in dimensions], 'minimum_glyph_size_pt': minimum,
              'text_outside_page': outside, 'svg_text_nodes': text_nodes, 'tiff_dpi': dpi,
              'tiff_pixels': pixels, 'data_rows': rows, 'source_hashes_verified': True,
              'encoding': manifest['encoding']}
    (OUT / f'fig{number}_qa.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'figure': number, **report}, ensure_ascii=False))
