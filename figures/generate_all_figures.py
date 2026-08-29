#!/usr/bin/env python3
"""Generate PLOS-ready SVG sources and LZW-compressed RGB TIFF figures."""

from __future__ import annotations

import csv
import hashlib
import html
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "S1_Data.csv"
OUT = ROOT / "figures"
SOURCE = OUT / "source"
FONT = "Arial, Helvetica, sans-serif"
INK = "#1F2933"
MUTED = "#52606D"
GRID = "#CBD2D9"
BLUE = "#236A8A"
ORANGE = "#D07922"
GREEN = "#2F855A"
RED = "#B6493A"
LIGHT = "#EEF2F6"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_start(width: int, height: int) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">'
        f'<path d="M0,0 L12,6 L0,12 z" fill="{INK}"/></marker></defs>',
    ]


MIN_TEXT_SIZE = 27  # keeps rendered text at or above 8 pt at the 7.5 in publication width


def text(x: float, y: float, value: object, size: int = 30, weight: int = 400,
         anchor: str = "start", fill: str = INK, style: str = "") -> str:
    size = max(size, MIN_TEXT_SIZE)
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}" {style}>{esc(value)}</text>')


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = INK,
         width: float = 3, dash: str = "", marker: bool = False) -> str:
    attrs = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{attrs}{marker_attr}/>' )


def rect(x: float, y: float, width: float, height: float, fill: str = "#FFFFFF",
         stroke: str = INK, stroke_width: float = 2, radius: float = 10) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')


def circle(x: float, y: float, radius: float, fill: str, stroke: str = "#FFFFFF") -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'


def square(x: float, y: float, radius: float, fill: str, stroke: str = "#FFFFFF") -> str:
    return (f'<rect x="{x-radius:.1f}" y="{y-radius:.1f}" width="{2*radius:.1f}" height="{2*radius:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>')


def triangle(x: float, y: float, radius: float, fill: str, stroke: str = "#FFFFFF") -> str:
    points = f"{x:.1f},{y-radius:.1f} {x-radius:.1f},{y+radius:.1f} {x+radius:.1f},{y+radius:.1f}"
    return f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'


def read_data() -> list[dict[str, str]]:
    with DATA_FILE.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_svg(name: str, parts: list[str]) -> Path:
    SOURCE.mkdir(parents=True, exist_ok=True)
    path = SOURCE / f"{name}_source.svg"
    path.write_text("\n".join(parts + ["</svg>", ""]), encoding="utf-8")
    return path


def render_tiff(svg_path: Path, output_path: Path, pixel_width: int = 2250) -> None:
    with tempfile.TemporaryDirectory(prefix="plos_fig_") as temp_dir:
        png_path = Path(temp_dir) / "render.png"
        subprocess.run([
            "rsvg-convert", "--background-color", "white", "--keep-aspect-ratio",
            "--width", str(pixel_width), "--output", str(png_path), str(svg_path)
        ], check=True)
        with Image.open(png_path) as image:
            rgb = image.convert("RGB")
            rgb.save(output_path, format="TIFF", compression="tiff_lzw", dpi=(300, 300))


def make_fig1() -> tuple[Path, Path]:
    global MIN_TEXT_SIZE
    MIN_TEXT_SIZE = 27
    w, h = 1800, 780
    p = svg_start(w, h)
    boxes = [(70, 110, 330, 145, "Teacher logits", "frozen"),
             (710, 110, 380, 145, "Active-label", "projection"),
             (1390, 110, 330, 145, "Student logits", "trainable")]
    for x, y, bw, bh, top, bottom in boxes:
        p.append(rect(x, y, bw, bh, fill="#F8FAFC", radius=12))
        p.append(text(x + bw/2, y + 62, top, 34, 600, "middle"))
        p.append(text(x + bw/2, y + 110, bottom, 27, 400, "middle", MUTED))
    p.append(line(400, 182, 700, 182, marker=True))
    p.append(text(550, 150, "teacher class logits", 25, 400, "middle", MUTED))
    p.append(line(1090, 182, 1380, 182, marker=True))
    p.append(text(1235, 150, "class KD", 27, 500, "middle"))
    p.append(rect(500, 420, 800, 145, fill="#FFFFFF", radius=12))
    p.append(text(900, 480, "Frozen view schedule", 33, 600, "middle"))
    p.append(text(900, 528, "epoch 1 ... 150", 28, 400, "middle", MUTED))
    p.append(line(900, 420, 900, 265, marker=True))
    p.append(text(940, 360, "active view", 25, 400, "start", MUTED))
    p.append(f'<path d="M235 255 L235 665 Q235 710 280 710 L1510 710 Q1555 710 1555 665 L1555 265" '
             f'fill="none" stroke="{INK}" stroke-width="4" stroke-dasharray="18,14" marker-end="url(#arrow)"/>')
    p.append(text(895, 695, "box KD (class projection bypassed)", 29, 500, "middle"))
    svg = save_svg("Fig1", p)
    tif = OUT / "Fig1.tif"
    render_tiff(svg, tif)
    return svg, tif


def make_fig2(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    segments = [r for r in rows if r["record_type"] == "schedule_segment"]
    exposure = {r["condition"]: float(r["value"]) for r in rows if r["record_type"] == "schedule_exposure"}
    segments.sort(key=lambda r: float(r["ci95_low"]))
    global MIN_TEXT_SIZE
    MIN_TEXT_SIZE = 27
    w, h = 1800, 750
    p = svg_start(w, h)
    left, right, top = 170, 1680, 110
    colors = ["#DCEAF2", "#B9D5E4", "#8FBBD0", "#5F9AB8"]
    xscale = lambda epoch: left + (float(epoch) / 150.0) * (right - left)
    for idx, row in enumerate(segments):
        start = float(row["ci95_low"]) - 1
        end = float(row["ci95_high"])
        x1, x2 = xscale(start), xscale(end)
        p.append(rect(x1, top, x2-x1, 150, fill=colors[idx], stroke="#FFFFFF", stroke_width=2, radius=0))
        p.append(text((x1+x2)/2, top+62, row["condition"].replace("_", " "), 31, 600, "middle"))
        count = int(float(row["value"]))
        noun = "coordinate" if count == 1 else "coordinates"
        p.append(text((x1+x2)/2, top+110, f"{count} active {noun}", 24, 400, "middle", MUTED))
    p.append(line(left, top+170, right, top+170, width=3))
    for epoch in (0, 30, 70, 110, 150):
        x = xscale(epoch)
        p.append(line(x, top+162, x, top+180, width=2))
        p.append(text(x, top+220, epoch, 25, 400, "middle", MUTED))
    p.append(text((left+right)/2, top+265, "Epoch", 28, 500, "middle"))
    p.append(text(left, 470, "Nominal coordinate exposure", 30, 600))
    max_exp = 150.0
    for i, cls in enumerate(("knife", "gun", "stick")):
        y = 535 + i*62
        value = exposure[cls]
        p.append(text(left, y+27, cls, 26, 500))
        p.append(rect(left+145, y, 1200, 34, fill="#EFF3F6", stroke="#D8DEE4", stroke_width=1, radius=0))
        p.append(rect(left+145, y, 1200*value/max_exp, 34, fill=[BLUE, ORANGE, GREEN][i], stroke="none", stroke_width=0, radius=0))
        p.append(text(left+1370, y+27, f"{int(value)} epochs", 25, 500))
    svg = save_svg("Fig2", p)
    tif = OUT / "Fig2.tif"
    render_tiff(svg, tif)
    return svg, tif


def make_fig3(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    values = {r["record_id"]: float(r["value"]) for r in rows if r["value"] not in ("", None)}
    global MIN_TEXT_SIZE
    MIN_TEXT_SIZE = 27
    w, h = 1800, 1040
    p = svg_start(w, h)
    y_min, y_max = -0.03, 0.015
    top, bottom = 150, 880
    def y(v: float) -> float:
        return bottom - (v-y_min)/(y_max-y_min)*(bottom-top)
    panels = [(150, 840), (1030, 1650)]
    for left, right in panels:
        p.append(line(left, top, left, bottom, stroke=MUTED, width=2))
        p.append(line(left, bottom, right, bottom, stroke=MUTED, width=2))
        for tick in (-0.03, -0.02, -0.01, 0.0, 0.01):
            p.append(line(left, y(tick), right, y(tick), stroke=INK if tick == 0 else GRID, width=3 if tick == 0 else 1.5))
    for tick in (-0.03, -0.02, -0.01, 0.0, 0.01):
        p.append(text(125, y(tick)+9, f"{tick:+.3f}" if tick else "0", 24, 400, "end", MUTED))
    p.append(text(42, (top+bottom)/2, "C4-M minus C4MixFT-M (mAP50-95)", 27, 500, "middle", INK,
                  'transform="rotate(-90 42 515)"'))
    p.append(text(150, 95, "A. Development validation", 34, 700))
    p.append(text(1030, 95, "B. Evaluation-composition sensitivity", 34, 700))
    seed_data = [("42", values["VAL_H2_42"], BLUE, "circle"),
                 ("1337", values["VAL_H2_1337"], ORANGE, "square"),
                 ("20260703", values["VAL_H2_20260703"], GREEN, "triangle")]
    xs = [280, 500, 720]
    for x, (seed, val, color, shape) in zip(xs, seed_data):
        p.append({"circle": circle, "square": square, "triangle": triangle}[shape](x, y(val), 15, color))
        p.append(text(x, y(val)-28, f"{val:+.6f}", 24, 600, "middle"))
        p.append(text(x, 930, f"seed {seed}", 24, 500, "middle"))
        if seed == "20260703":
            p.append(text(x, 966, "post hoc", 21, 400, "middle", MUTED))
    eval_points = {
        "42": (values["original_full_evaluation_H2_42"], values["source_excluded_evaluation_H2_42"], BLUE, circle),
        "1337": (values["original_full_evaluation_H2_1337"], values["source_excluded_evaluation_H2_1337"], ORANGE, square),
    }
    x1, x2 = 1170, 1510
    for seed, (v1, v2, color, shape) in eval_points.items():
        p.append(line(x1, y(v1), x2, y(v2), stroke=color, width=4))
        p.append(shape(x1, y(v1), 15, color))
        p.append(shape(x2, y(v2), 15, color))
        p.append(text(x1, y(v1)-28 if v1 >= 0 else y(v1)-28, f"{v1:+.6f}", 23, 600, "middle"))
        p.append(text(x2, y(v2)+43 if v2 < 0 else y(v2)+43, f"{v2:+.6f}", 23, 600, "middle"))
    p.append(text(x1, 925, "Original", 24, 500, "middle"))
    p.append(text(x1, 958, "2,553 images", 21, 400, "middle", MUTED))
    p.append(text(x2, 925, "Source-excluded", 24, 500, "middle"))
    p.append(text(x2, 958, "2,019 images", 21, 400, "middle", MUTED))
    p.append(circle(1070, 1010, 11, BLUE)); p.append(text(1095, 1018, "seed 42", 21, 400))
    p.append(square(1280, 1010, 11, ORANGE)); p.append(text(1305, 1018, "seed 1337", 21, 400))
    svg = save_svg("Fig3", p)
    tif = OUT / "Fig3.tif"
    render_tiff(svg, tif)
    return svg, tif


def make_fig4(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    selected = {r["record_id"]: r for r in rows if r["record_id"].startswith("FOLLOWUP_")}
    names = ["order_axis_composite", "tail_axis_composite", "difference_of_differences_composite"]
    labels = ["Order-axis\ncomposite", "Tail-axis\ncomposite", "Difference-of-\ndifferences"]
    global MIN_TEXT_SIZE
    MIN_TEXT_SIZE = 24
    w, h = 1600, 1000
    p = svg_start(w, h)
    left, right, top, bottom = 170, 1470, 120, 820
    y_min, y_max = -0.03, 0.03
    def y(v: float) -> float:
        return bottom - (v-y_min)/(y_max-y_min)*(bottom-top)
    p.append(f'<rect x="{left}" y="{y(0.01):.1f}" width="{right-left}" height="{y(-0.01)-y(0.01):.1f}" fill="{LIGHT}"/>')
    for tick in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03):
        p.append(line(left, y(tick), right, y(tick), stroke=INK if tick == 0 else GRID, width=3 if tick == 0 else 1.5))
        p.append(text(left-25, y(tick)+9, f"{tick:+.3f}" if tick else "0", 24, 400, "end", MUTED))
    p.append(line(left, top, left, bottom, stroke=MUTED, width=2))
    p.append(line(left, bottom, right, bottom, stroke=MUTED, width=2))
    p.append(text(48, 470, "Composite contrast (mAP50-95)", 27, 500, "middle", INK,
                  'transform="rotate(-90 48 470)"'))
    xs = [420, 820, 1220]
    seed_shapes = [("42", BLUE, circle, -28), ("1337", ORANGE, square, 0), ("20260703", GREEN, triangle, 28)]
    for x, name, label in zip(xs, names, labels):
        summary = selected[f"FOLLOWUP_{name}_summary"]
        estimate = float(summary["value"])
        lo, hi = float(summary["ci95_low"]), float(summary["ci95_high"])
        p.append(line(x, y(lo), x, y(hi), stroke=RED, width=5))
        p.append(line(x-16, y(lo), x+16, y(lo), stroke=RED, width=5))
        p.append(line(x-16, y(hi), x+16, y(hi), stroke=RED, width=5))
        p.append(square(x, y(estimate), 14, RED, stroke=RED))
        for seed, color, shape, offset in seed_shapes:
            val = float(selected[f"FOLLOWUP_{name}_{seed}"]["value"])
            p.append(shape(x+offset, y(val), 12, color))
        first, second = label.split("\n")
        p.append(text(x, 875, first, 26, 500, "middle"))
        p.append(text(x, 908, second, 26, 500, "middle"))
    p.append(text(1270, y(0.008), "retrospective +/-0.01 region", 18, 400, "start", MUTED))
    p.append(circle(250, 960, 10, BLUE)); p.append(text(272, 968, "seed 42", 20, 400))
    p.append(square(440, 960, 10, ORANGE)); p.append(text(462, 968, "seed 1337", 20, 400))
    p.append(triangle(670, 960, 10, GREEN)); p.append(text(692, 968, "seed 20260703", 20, 400))
    p.append(square(980, 960, 10, RED, stroke=RED)); p.append(text(1002, 968, "mean and descriptive 95% interval", 20, 400))
    svg = save_svg("Fig4", p)
    tif = OUT / "Fig4.tif"
    render_tiff(svg, tif)
    return svg, tif


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_map(outputs: list[tuple[str, Path, Path, str, str]]) -> None:
    fields = ["Figure", "panel", "input_data_file", "columns_used", "transformation", "script",
              "source_file", "output_file", "output_sha256"]
    with (OUT / "FIGURE_DATA_MAP.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for figure, svg, tif, columns, transform in outputs:
            writer.writerow({"Figure": figure, "panel": "all", "input_data_file": str(DATA_FILE.relative_to(ROOT)),
                             "columns_used": columns, "transformation": transform,
                             "script": "figures/generate_all_figures.py",
                             "source_file": str(svg.relative_to(ROOT)), "output_file": str(tif.relative_to(ROOT)),
                             "output_sha256": sha256(tif)})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_data()
    outputs = []
    svg, tif = make_fig1()
    outputs.append(("Fig1", svg, tif, "schematic; no quantitative columns", "vector diagram of documented method flow"))
    svg, tif = make_fig2(rows)
    outputs.append(("Fig2", svg, tif, "record_type,condition,value,ci95_low,ci95_high", "schedule segments and nominal exposure"))
    svg, tif = make_fig3(rows)
    outputs.append(("Fig3", svg, tif, "record_id,seed,value,n_images", "plot exact fixed-pair contrasts by evidence stage"))
    svg, tif = make_fig4(rows)
    outputs.append(("Fig4", svg, tif, "record_id,seed,value,ci95_low,ci95_high", "plot seed-indexed composite contrasts and descriptive intervals"))
    write_map(outputs)
    print("Generated Fig1-Fig4 SVG sources and PLOS TIFF files")


if __name__ == "__main__":
    main()
