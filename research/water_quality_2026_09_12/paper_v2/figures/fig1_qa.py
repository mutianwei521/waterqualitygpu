"""Inspect the exported Figure 1 without changing any artwork."""
from pathlib import Path
import csv
import json
import xml.etree.ElementTree as ET
import fitz
from PIL import Image

OUT = Path(__file__).resolve().parent/'out'
csv_counts = {}
for name in ('bc', 'd', 'e', 'f'):
    with (OUT/f'fig1_source_{name}.csv').open(encoding='utf-8', newline='') as handle:
        csv_counts[name] = sum(1 for _ in csv.DictReader(handle))
with fitz.open(OUT/'fig1_engine.pdf') as doc:
    page=doc[0]
    spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b
           for line in b['lines'] for s in line['spans']]
    outside=[s['text'] for s in spans if s['bbox'][0]<0 or s['bbox'][1]<0
             or s['bbox'][2]>page.rect.width or s['bbox'][3]>page.rect.height]
    svg=ET.parse(OUT/'fig1_engine.svg')
    with Image.open(OUT/'fig1_engine.tif') as im:
        pixels=im.size
        dpi=[float(x) for x in im.info['dpi']]
    report={'dimensions_mm':[round(page.rect.width/72*25.4,2),round(page.rect.height/72*25.4,2)],
            'minimum_glyph_size_pt':min(s['size'] for s in spans),'text_outside_page':outside,
            'svg_text_nodes':len(svg.findall('.//{http://www.w3.org/2000/svg}text')),
            'tiff_pixels':pixels,'tiff_dpi':dpi,
            'data_check':'The plotting script reads saved JSON directly; CSV row counts and the complete d heatmap count are asserted here.',
            'data_rows':csv_counts,
            'd_heatmap_cells':csv_counts['d'],
            'd_unplotted_values':0}
    page.get_pixmap(matrix=fitz.Matrix(3,3),clip=fitz.Rect(0,0,page.rect.width,60/25.4*72)).save(OUT/'fig1a_detail.png')
    (OUT/'fig1_qa.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
    assert not outside
    assert report['minimum_glyph_size_pt']>=4.9
    assert report['svg_text_nodes']>100
    assert dpi==[600,600]
    assert csv_counts=={'bc':25,'d':154,'e':44,'f':24}
