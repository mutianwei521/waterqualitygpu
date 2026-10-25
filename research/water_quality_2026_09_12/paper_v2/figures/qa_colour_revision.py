"""Cross-figure export QA for the Figure 1--5 and Figure S1 colour revision."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import fitz
from PIL import Image


HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RENDER = OUT / "qa_colour_revision"
RENDER.mkdir(exist_ok=True)
BASES = [
    "fig1_engine",
    "fig2_field",
    "fig3_exp1",
    "fig4_exp2",
    "fig5_exp3",
    "figS1_flow",
]
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
report = {}

for base in BASES:
    pdf_path = OUT / f"{base}.pdf"
    svg_path = OUT / f"{base}.svg"
    tif_path = OUT / f"{base}.tif"
    png_path = OUT / f"{base}.png"
    for path in (pdf_path, svg_path, tif_path, png_path):
        assert path.is_file(), path
    with fitz.open(pdf_path) as document:
        assert len(document) == 1
        page = document[0]
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
        ]
        outside = [
            {"text": span["text"], "bbox": span["bbox"]}
            for span in spans
            if span["bbox"][0] < -0.5
            or span["bbox"][1] < -0.5
            or span["bbox"][2] > page.rect.width + 0.5
            or span["bbox"][3] > page.rect.height + 0.5
        ]
        unembedded = sorted({font[3] for font in page.get_fonts(full=True) if font[0] == 0})
        dimensions = [
            round(page.rect.width / 72 * 25.4, 2),
            round(page.rect.height / 72 * 25.4, 2),
        ]
        minimum_glyph = min(span["size"] for span in spans)
        page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False).save(RENDER / f"{base}.png")
    svg = ET.parse(svg_path)
    text_nodes = len(svg.findall(".//svg:text", SVG_NS))
    with Image.open(tif_path) as image:
        tif_pixels = list(image.size)
        tif_dpi = [round(float(value), 2) for value in image.info.get("dpi", (0, 0))]
    report[base] = {
        "dimensions_mm": dimensions,
        "minimum_glyph_size_pt": minimum_glyph,
        "text_outside_page": outside,
        "unembedded_fonts": unembedded,
        "svg_text_nodes": text_nodes,
        "tiff_pixels": tif_pixels,
        "tiff_dpi": tif_dpi,
    }
    assert dimensions[0] <= 183.0 and dimensions[1] <= 220.0
    assert minimum_glyph >= 4.9
    assert not outside
    assert not unembedded
    assert text_nodes > 0
    assert tif_dpi == [600.0, 600.0]

(OUT / "qa_colour_revision.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
