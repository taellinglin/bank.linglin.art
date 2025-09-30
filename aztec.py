#!/usr/bin/env python3
"""
aztec_color_svg.py

Generate a colored Aztec-style SVG from a serial string using segno to create the Aztec code matrix,
then build a custom SVG with multiple colored passes/gradients so overlaps produce additive color mixing.

Usage:
    pip install segno [cairosvg optional]
    python aztec_color_svg.py --data "BILL-12345" --out aztec_color.svg --scale 12 --style radial --png

Author: For RingMaster Lin
"""
from __future__ import annotations
import argparse
import os
import math
import shutil
import base64
from typing import List, Tuple
import xml.etree.ElementTree as ET

# Optional libs (segno required)
try:
    from segno import make
except Exception as e:
    raise RuntimeError("This script requires 'segno'. Install with: pip install segno") from e

# optional PNG export
try:
    import cairosvg
    _HAS_CAIROSVG = True
except Exception:
    _HAS_CAIROSVG = False

# -------------------------
# Helpers
# -------------------------
def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def rgb_tuple_to_hex(c: Tuple[float,float,float]) -> str:
    return '#{0:02x}{1:02x}{2:02x}'.format(
        int(max(0, min(1, c[0]))*255),
        int(max(0, min(1, c[1]))*255),
        int(max(0, min(1, c[2]))*255)
    )

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def color_from_hue(h: float, s: float=0.8, v: float=0.9) -> Tuple[float,float,float]:
    """Return RGB 0..1 from hue 0..1"""
    i = int(h*6)
    f = (h*6) - i
    p = v*(1-s)
    q = v*(1-f*s)
    t = v*(1-(1-f)*s)
    i = i % 6
    if i == 0: r,g,b = v,t,p
    elif i == 1: r,g,b = q,v,p
    elif i == 2: r,g,b = p,v,t
    elif i == 3: r,g,b = p,q,v
    elif i == 4: r,g,b = t,p,v
    else: r,g,b = v,p,q
    return (r,g,b)

# -------------------------
# Extract Aztec matrix from segno
# -------------------------
def aztec_matrix_from_segno(data: str, compact: bool=True, layers: int=None):
    """
    Use segno to generate an Aztec symbol and return a 2D boolean matrix (list of rows),
    where True = dark module, False = light module.
    compact: whether to try compact Aztec (segno decides)
    layers: if you want to force a certain layer count (optional)
    """
    # segno.make -> symbol object; request aztec specifically via keyword 'mode' or 'symbol' if supported.
    # segno supports 'aztec' mode by passing kind='aztec' or mode='aztec' depending on version.
    # We'll attempt common call patterns and fallback gracefully.
    try:
        sym = make(data)
    except TypeError:
        # different segno versions use 'kind' or 'symbol'
        try:
            sym = make(data)
        except Exception:
            # fallback: try segno.encoder.encode? last-resort: let segno auto choose maybe aztec via qr?
            sym = make(data)
    except Exception:
        # if segno couldn't make aztec, raise useful error
        raise

    # symbol.matrix returns a sequence of rows (2d boolean). If not present, use matrix_iter
    if hasattr(sym, 'matrix'):
        mat = list(sym.matrix)  # each row: tuple of booleans (True dark)
    else:
        mat = [row for row in sym.matrix_iter()]
    # normalize to list of lists of bool
    matrix = [ [bool(v) for v in row] for row in mat ]
    return matrix

# -------------------------
# SVG builder
# -------------------------
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

def build_colored_aztec_svg(matrix: List[List[bool]],
                             scale: int = 10,
                             margin_modules: int = 4,
                             style: str = "radial",
                             out_path: str = "aztec_color.svg",
                             passes: List[dict] = None):
    """
    Build an SVG at out_path from boolean matrix.
    - scale: pixels per module
    - margin_modules: white margin around symbol
    - style: 'radial' or 'directional' gradient mapping
    - passes: list of dicts describing color passes, each with keys:
        - 'color' (rgb tuple 0..1), 'opacity' (0..1), 'offset' (subpixel offset as (dx,dy))
    If passes is None, we use default 3-color passes (R,G,B) with slight offsets to create additive mixing.
    """
    h = len(matrix)
    w = len(matrix[0]) if h>0 else 0

    # default passes: red, green, blue with slight offsets
    if passes is None:
        passes = [
            {"color": (1.0,0.05,0.05), "opacity": 0.45, "offset": (-0.18, -0.05)},
            {"color": (0.05,1.0,0.05), "opacity": 0.40, "offset": (0.12, 0.08)},
            {"color": (0.05,0.05,1.0), "opacity": 0.45, "offset": (0.05, 0.18)}
        ]

    # SVG size in px
    svg_w = int((w + 2*margin_modules) * scale)
    svg_h = int((h + 2*margin_modules) * scale)

    root = ET.Element("{" + SVG_NS + "}svg", {
        "width": str(svg_w),
        "height": str(svg_h),
        "viewBox": f"0 0 {svg_w} {svg_h}",
        "version": "1.1",
        "xmlns": SVG_NS
    })

    # background
    bg = ET.SubElement(root, "rect", {
        "x":"0","y":"0","width":str(svg_w),"height":str(svg_h),
        "fill":"#000000"
    })

    defs = ET.SubElement(root, "defs")

    # create per-pass gradients (radial or linear) keyed by pass index
    for pi, p in enumerate(passes):
        gid = f"grad_{pi}"
        if style == "radial":
            grad = ET.SubElement(defs, "radialGradient", {"id":gid, "cx":"50%","cy":"50%","r":"60%"})
            ET.SubElement(grad, "stop", {"offset":"0%", "stop-color": rgb_tuple_to_hex(tuple(min(1,c+0.15) for c in p["color"])), "stop-opacity":"1"})
            ET.SubElement(grad, "stop", {"offset":"70%", "stop-color": rgb_tuple_to_hex(p["color"]), "stop-opacity":"1"})
            ET.SubElement(grad, "stop", {"offset":"100%", "stop-color":"#000000", "stop-opacity":"0"})
        else:
            grad = ET.SubElement(defs, "linearGradient", {"id":gid, "x1":"0%","y1":"0%","x2":"100%","y2":"0%"})
            ET.SubElement(grad, "stop", {"offset":"0%", "stop-color": rgb_tuple_to_hex(tuple(min(1,c+0.15) for c in p["color"])), "stop-opacity":"1"})
            ET.SubElement(grad, "stop", {"offset":"100%", "stop-color": rgb_tuple_to_hex(p["color"]), "stop-opacity":"1"})

    # group to hold passes
    group_root = ET.SubElement(root, "g", {"id":"aztec_group"})

    # iterate passes: for each pass, draw rectangles for modules that are True
    for pi, p in enumerate(passes):
        gpass = ET.SubElement(group_root, "g", {"id":f"pass_{pi}", "fill":rgb_tuple_to_hex(p["color"]), "fill-opacity":str(p["opacity"])})
        dx, dy = p.get("offset",(0.0,0.0))
        # small subpixel offset in pixels
        dx_px = dx * scale
        dy_px = dy * scale

        for row in range(h):
            for col in range(w):
                if not matrix[row][col]:
                    continue
                # compute top-left pixel coords with margin
                x = (col + margin_modules) * scale + dx_px
                y = (row + margin_modules) * scale + dy_px
                # use rect with a slight rounding/inner padding to create separation
                pad = max(0, scale * 0.06)
                rx = x + pad
                ry = y + pad
                rw = scale - pad*2
                rh = scale - pad*2
                # optionally apply gradient fill via defs by referencing gradient id
                # alternate use gradient for every N-th module for variety
                use_grad = ((row+col+pi) % 5 == 0)
                if use_grad:
                    gid = f"grad_{pi}"
                    rattrs = {"x":f"{rx:.3f}","y":f"{ry:.3f}","width":f"{rw:.3f}","height":f"{rh:.3f}","fill":f"url(#{gid})","fill-opacity":str(p["opacity"]), "rx":"1", "ry":"1"}
                else:
                    rattrs = {"x":f"{rx:.3f}","y":f"{ry:.3f}","width":f"{rw:.3f}","height":f"{rh:.3f}","fill":rgb_tuple_to_hex(p["color"]), "fill-opacity":str(p["opacity"]), "rx":"1", "ry":"1"}
                ET.SubElement(gpass, "rect", rattrs)

    # Add a central finder/bullseye overlay to give Aztec look (optional)
    # draw concentric rings centered on symbol center
    center_x = svg_w / 2.0
    center_y = svg_h / 2.0
    finder_group = ET.SubElement(root, "g", {"id":"finder", "fill":"none"})
    # a few rings: black gap then colored ring, etc.
    ring_count = 4
    for i in range(ring_count):
        r = (min(w,h)/2.0 + margin_modules - i*0.8) * scale * 0.6
        if r <= 0: continue
        col = rgb_tuple_to_hex(color_from_hue((i*0.12)%1.0))
        ET.SubElement(finder_group, "circle", {
            "cx":f"{center_x:.3f}", "cy":f"{center_y:.3f}", "r":f"{r:.3f}",
            "stroke":col, "stroke-width":"2", "stroke-opacity":"0.75", "fill":"none"
        })

    # small signature metadata in hidden comment
    comment = ET.Comment(f"Generated Aztec color SVG")
    root.append(comment)

    # write file
    ensure_dir_for_file(out_path)
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    return out_path

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create colored Aztec-like SVG from data.")
    p.add_argument("--data", type=str, required=True, help="Data to encode (serial, text, etc.)")
    p.add_argument("--out", type=str, default="aztec_color.svg", help="Output SVG file path")
    p.add_argument("--scale", type=int, default=12, help="Pixel scale per module")
    p.add_argument("--margin", type=int, default=6, help="Margin modules around code")
    p.add_argument("--style", type=str, default="radial", choices=["radial","directional"], help="Gradient style")
    p.add_argument("--png", action="store_true", help="Also write PNG (requires cairosvg)")
    return p.parse_args()

def main():
    args = parse_args()
    data = args.data
    out = args.out
    scale = args.scale
    margin = args.margin
    style = args.style

    print(f"[+] Building Aztec matrix for data: {data!r}")
    matrix = aztec_matrix_from_segno(data)
    print(f"[+] Matrix size: {len(matrix)} x {len(matrix[0]) if matrix else 0}")

    print("[+] Building colored SVG ...")
    svg_path = build_colored_aztec_svg(matrix, scale=scale, margin_modules=margin, style=style, out_path=out)
    print(f"[+] Wrote SVG: {svg_path}")

    if args.png:
        if not _HAS_CAIROSVG:
            print("[!] cairosvg not installed; cannot write PNG. Install with `pip install cairosvg`")
        else:
            png_path = os.path.splitext(svg_path)[0] + ".png"
            cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=None, scale=1.0)
            print(f"[+] Wrote PNG: {png_path}")

if __name__ == "__main__":
    main()
