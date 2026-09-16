"""Submission QA for every main-text and supplementary figure export.

The checks are intentionally format-level: page size, readable type, embedded PDF
fonts, live SVG text, 600-dpi TIFF export, and text remaining inside the PDF page.
They complement (rather than replace) visual inspection of the rendered previews.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import fitz
from PIL import Image


HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RENDER = OUT / "qa_full_figure_set"
RENDER.mkdir(exist_ok=True)
BASES = [
    "fig1_engine",
    "fig2_field",
    "fig3_exp1",
    "fig4_exp2",
    "fig5_exp3",
    "figS1_flow",
    "figS2_sweep",
    "figS3_cost",
    "figS4_calib",
    "figS5_booster",
    "figS6_source",
    "figS7_ident",
]
REVISED_BASES = {"fig3_exp1", "figS1_flow", "figS2_sweep", "figS3_cost", "figS4_calib",
                 "figS5_booster", "figS6_source", "figS7_ident"}
SUBMISSION_WIDTH_MM = 165.0
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def inspect(base: str) -> dict:
    paths = {suffix: OUT / f"{base}.{suffix}" for suffix in ("pdf", "svg", "png", "tif")}
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 0, path

    with fitz.open(paths["pdf"]) as document:
        assert len(document) == 1, paths["pdf"]
        page = document[0]
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if span.get("text", "").strip()
        ]
        assert spans, paths["pdf"]
        outside = [
            {"text": span["text"], "bbox": [round(v, 2) for v in span["bbox"]]}
            for span in spans
            if span["bbox"][0] < -0.5
            or span["bbox"][1] < -0.5
            or span["bbox"][2] > page.rect.width + 0.5
            or span["bbox"][3] > page.rect.height + 0.5
        ]
        unembedded = sorted({font[3] for font in page.get_fonts(full=True) if font[0] == 0})
        dimensions = [round(page.rect.width / 72 * 25.4, 2), round(page.rect.height / 72 * 25.4, 2)]
        minimum_glyph = round(min(span["size"] for span in spans), 3)
        page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False).save(RENDER / f"{base}.png")

    svg = ET.parse(paths["svg"])
    text_nodes = len(svg.findall(".//svg:text", SVG_NS))
    with Image.open(paths["tif"]) as image:
        tif_pixels = list(image.size)
        tif_dpi = [round(float(value), 2) for value in image.info.get("dpi", (0, 0))]

    scaled_minimum = round(minimum_glyph * SUBMISSION_WIDTH_MM / dimensions[0], 3)
    result = {
        "dimensions_mm": dimensions,
        "minimum_glyph_size_pt": minimum_glyph,
        "minimum_glyph_at_165mm_pt": scaled_minimum,
        "text_outside_page": outside,
        "unembedded_fonts": unembedded,
        "svg_live_text_nodes": text_nodes,
        "tiff_pixels": tif_pixels,
        "tiff_dpi": tif_dpi,
    }
    assert dimensions[0] <= 183.0 and dimensions[1] <= 220.0, result
    assert minimum_glyph >= 4.9, result
    assert not outside, result
    assert not unembedded, result
    assert text_nodes > 0, result
    assert tif_dpi == [600.0, 600.0], result
    if base in REVISED_BASES:
        assert scaled_minimum >= 5.0, result
    return result


report = {base: inspect(base) for base in BASES}
destination = OUT / "qa_full_figure_set.json"
destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
