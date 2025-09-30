#!/usr/bin/env python3
"""
fantasy_banknote.py — Enhanced procedural fantasy banknote generator
----------------------------------------------------------------------
Creates a stylized, clearly-marked "banknote" from an input image.
Outputs an SVG (vector) and optional PNG preview.
Requires: Pillow, svgwrite. Optional: fontTools for glyph paths.

Author: RingMaster Lin
"""

import os
import sys
import math
import io
import base64
import argparse
import hashlib
from typing import Tuple, List
import requests
from PIL import Image, ImageOps

try:
    import svgwrite
except Exception:
    print("[!] svgwrite required: pip install svgwrite")
    raise

USE_FONTTOOLS = True
try:
    from fontTools.ttLib import TTFont
    from fontTools.pens.svgPathPen import SVGPathPen
except Exception:
    USE_FONTTOOLS = False

# ----------------------
# Utility / conversions
# ----------------------
MM_TO_PX = 300.0 / 25.4
def mm_to_px(mm: float, dpi: float = 300.0) -> int:
    return int(round(mm * dpi / 25.4))

def sha256_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()

def image_to_datauri_png(img: Image.Image) -> str:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

# ----------------------
# Load fonts (optional)
# ----------------------
def load_fonts(font_dir="./fonts"):
    fonts = {}
    if not os.path.isdir(font_dir):
        return fonts
    for fn in os.listdir(font_dir):
        if fn.lower().endswith((".otf", ".ttf")):
            try:
                font_name = os.path.splitext(fn)[0]
                fonts[font_name] = TTFont(os.path.join(font_dir, fn))
                print(f"[+] Loaded font: {fn}")
            except Exception as e:
                print(f"[!] Could not load font {fn}: {e}")
    return fonts

from svgwrite import path

def add_smooth_triangle(dwg, pts, color, opacity=0.25):
    """
    Draws a triangle with curved edges using cubic Bezier interpolation.
    pts: list of 3 (x,y) tuples
    """
    p0, p1, p2 = pts

    # Control points are at 1/3 distance toward next point for gentle curves
    def ctrl(p_from, p_to):
        return (p_from[0] + (p_to[0]-p_from[0])/3, p_from[1] + (p_to[1]-p_from[1])/3)

    d = f"M{p0[0]},{p0[1]} "
    d += f"C{ctrl(p0,p1)[0]},{ctrl(p0,p1)[1]} {ctrl(p1,p0)[0]},{ctrl(p1,p0)[1]} {p1[0]},{p1[1]} "
    d += f"C{ctrl(p1,p2)[0]},{ctrl(p1,p2)[1]} {ctrl(p2,p1)[0]},{ctrl(p2,p1)[1]} {p2[0]},{p2[1]} "
    d += f"C{ctrl(p2,p0)[0]},{ctrl(p2,p0)[1]} {ctrl(p0,p2)[0]},{ctrl(p0,p2)[1]} {p0[0]},{p0[1]} Z"

    dwg.add(dwg.path(d=d, fill=color, opacity=opacity))
def moire_fishnet_background(
    dwg: svgwrite.Drawing,
    seed: bytes,
    width: int,
    height: int,
    rows: int = 512,              # number of horizontal "threads"
    warp_amp: float = 512.0,      # vertical warp amplitude
    steps_per_curve: int = 24,    # resolution along each curve
    input_image: Image.Image = None,
    line_opacity: float = 0.5,
    stroke_width: float = 0.5,
    dithering_density: float = 0.5  # fraction of pattern cells filled
):
    import random
    import math
    random.seed(int.from_bytes(seed, "big"))

    # --- optional image for coloring ---
    if input_image:
        img = input_image.convert("RGB").resize((width, height), Image.LANCZOS)
        pixels = img.load()
    else:
        pixels = None

    # --- horizontal "threads" (beziers that cross fully left→right) ---
    y_spacing = height / (rows + 1)
    for i in range(rows):
        y_base = y_spacing * (i + 1)
        points = []
        for step in range(steps_per_curve + 1):
            x = width * step / steps_per_curve
            # consistent warp across curve
            phase = (i * 0.37 + step / steps_per_curve * 2 * math.pi)
            warp = math.sin(phase) * warp_amp * random.uniform(0.4, 1.0)
            y = y_base + warp
            points.append((x, y))

        # Color selection
        if pixels:
            px, py = map(int, points[steps_per_curve//2])
            r,g,b = pixels[min(px,width-1), min(py,height-1)]
            color = f"rgb({r},{g},{b})"
        else:
            cidx = i % len(seed)
            color = f"rgb({(seed[cidx]*37)%256},{(seed[(cidx+1)%len(seed)]*59)%256},{(seed[(cidx+2)%len(seed)]*83)%256})"

        # --- smooth bezier-like path ---
        path_d = f"M{points[0][0]},{points[0][1]}"
        for j in range(1, len(points)-1):
            x0,y0 = points[j-1]
            x1,y1 = points[j]
            x2,y2 = points[j+1]
            cx = (x1 + x2)/2
            cy = (y1 + y2)/2
            path_d += f" Q{x1},{y1} {cx},{cy}"
        # last point to right edge
        path_d += f" T{points[-1][0]},{points[-1][1]}"
        dwg.add(dwg.path(d=path_d, stroke=color, fill="none", stroke_width=stroke_width, opacity=line_opacity))

    # --- subtle QR-like dithering pattern (geometric, not image-based) ---
    cell = 6
    cols = math.ceil(width / cell)
    rows_d = math.ceil(height / cell)

    for r in range(rows_d):
        for c in range(cols):
            # Deterministic checker/XOR pattern
            pattern_val = (r % 4) ^ (c % 4)   # repeating block of 4x4
            pattern_norm = pattern_val / 3.0  # normalize to 0–1

            if pattern_norm < dithering_density:
                x = c * cell
                y = r * cell
                s = cell * 0.6
                idx = (r*cols + c) % len(seed)
                v = seed[idx]
                color = f"rgb({(v*53)%256},{(v*97)%256},{(v*67)%256})"
                dwg.add(dwg.rect(
                    insert=(x + (cell-s)/2, y + (cell-s)/2),
                    size=(s, s),
                    fill=color,
                    opacity=0.2
                ))


# ----------------------
# Embed font in SVG (Windows-safe)
# ----------------------
def embed_font(dwg: svgwrite.Drawing, ttfont: TTFont, font_name: str):
    import tempfile
    import os

    tmp_path = os.path.join(tempfile.gettempdir(), f"{font_name}.woff")
    ttfont.flavor = "woff"
    ttfont.save(tmp_path)

    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    style = f"@font-face {{ font-family: '{font_name}'; src: url('data:font/woff;base64,{b64}') format('woff'); }}"
    dwg.defs.add(dwg.style(style))

# ----------------------
# Tiled background
# ----------------------
def add_fancy_tiled_background(
    dwg: svgwrite.Drawing,
    seed: bytes,
    width: int,
    height: int,
    tile: int = 8,
    input_image: Image.Image = None
):
    import random, math
    random.seed(int.from_bytes(seed, "big"))

    # --- Image for color & warp ---
    if input_image:
        img = input_image.convert("RGB").resize((width, height), Image.LANCZOS)
        pixels = img.load()
    else:
        pixels = None

    # --- Grid resolution ---
    nx = max(4, width // tile)
    ny = max(4, height // tile)

    # --- Build warped grid ---
    grid = []
    for iy in range(ny + 1):
        row = []
        for ix in range(nx + 1):
            x = ix * width / nx
            y = iy * height / ny

            # --- Seed-based random offsets ---
            angle_seed = random.uniform(0, 2*math.pi)
            amp_seed = random.uniform(0.3, 1.0) * tile

            # --- Image-based warp (optional) ---
            if pixels:
                px = min(int(x), width-1)
                py = min(int(y), height-1)
                r, g, b = pixels[px, py]
                brightness = (r + g + b) / 3 / 255  # 0..1
                warp_amp = amp_seed * brightness
                warp_angle = angle_seed + brightness * math.pi
            else:
                warp_amp = amp_seed
                warp_angle = angle_seed

            # Apply warp
            x += warp_amp * math.cos(warp_angle)
            y += warp_amp * math.sin(warp_angle)
            row.append((x, y))
        grid.append(row)

    # --- Draw horizontal lines ---
    for row in grid:
        d_str = f"M{row[0][0]},{row[0][1]}"
        for x, y in row[1:]:
            d_str += f" L{x},{y}"
        # Color
        if pixels:
            px, py = map(int, row[0])
            r, g, b = pixels[min(px, width-1), min(py, height-1)]
            color = f"rgb({r},{g},{b})"
        else:
            v = seed[len(row) % len(seed)]
            color = f"rgb({(v*37)%256},{(v*59)%256},{(v*83)%256})"
        dwg.add(dwg.path(d=d_str, stroke=color, fill="none", stroke_width=0.4, opacity=0.15))

    # --- Draw vertical lines ---
    for ix in range(nx + 1):
        d_str = f"M{grid[0][ix][0]},{grid[0][ix][1]}"
        for iy in range(1, ny + 1):
            x, y = grid[iy][ix]
            d_str += f" L{x},{y}"
        # Color
        if pixels:
            px, py = map(int, grid[0][ix])
            r, g, b = pixels[min(px, width-1), min(py, height-1)]
            color = f"rgb({r},{g},{b})"
        else:
            v = seed[ix % len(seed)]
            color = f"rgb({(v*83)%256},{(v*59)%256},{(v*37)%256})"
        dwg.add(dwg.path(d=d_str, stroke=color, fill="none", stroke_width=0.4, opacity=0.15))

# ----------------------
# QR-like border
# ----------------------
def add_qr_like_border(dwg: svgwrite.Drawing, seed: bytes, width: int, height: int, cell=8, margin_cells=3):
    cols = int(math.ceil(width / cell))
    rows = int(math.ceil(height / cell))
    for r in range(rows):
        for c in range(cols):
            if r < margin_cells or r >= rows - margin_cells or c < margin_cells or c >= cols - margin_cells:
                idx = (r * cols + c) % len(seed)
                v = seed[idx]
                if (v + ((r+c)*13) ) % 256 > 150:
                    x = c * cell
                    y = r * cell
                    s = 1 if (v % 3 == 0) else (0.6 if (v % 3 == 1) else 0.35)
                    w = max(1, int(cell * s))
                    h = max(1, int(cell * s))
                    dwg.add(dwg.rect(insert=(x + (cell-w)/2, y + (cell-h)/2), size=(w, h), fill="#111", opacity=0.9))

# ----------------------
# Guilloché / spiral generator
# ----------------------
def polar_mandala(seed: bytes, px_radius: float, layers=6, sym=12, points=800,
                  spiral_tightness=0.02, swell_base=0.01) -> List[List[Tuple[float,float]]]:
    curves = []
    max_r = px_radius
    for L in range(layers):
        pts = []
        phase = (seed[(L*3) % len(seed)] / 255.0) * 2*math.pi
        fold_count = 8 + (seed[(L*5) % len(seed)] % 16)
        swell_amp = swell_base + (seed[(L*7) % len(seed)] / 255.0) * 0.02
        twist = ((seed[(L*11) % len(seed)] / 255.0) - 0.5) * spiral_tightness
        base_r_norm = 0.22 + 0.05 * L / max(1, layers-1)
        for i in range(points):
            theta = 2*math.pi*i/points
            swell = swell_amp * math.sin(theta * fold_count + phase)
            spiral_increment = 0.0 + (theta / (2*math.pi)) * 0.07 * (L / max(1,layers))
            r_norm = base_r_norm + swell + spiral_increment
            r_norm = max(0.02, min(0.98, r_norm))
            rad = r_norm * max_r
            t = theta + twist * math.sin(theta * fold_count * 0.5 + phase)
            x = rad * math.cos(t)
            y = rad * math.sin(t)
            pts.append((x,y))
        curves.append(pts)
    return curves

# ----------------------
# Image embed & prepare
# ----------------------
def load_and_prepare_image(path: str, max_dim: int = 600, circular: bool = True) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    im.thumbnail((max_dim, max_dim), Image.LANCZOS)
    if circular:
        size = im.size
        mask = Image.new("L", size, 0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(mask)
        r = min(size)//2
        cx, cy = size[0]//2, size[1]//2
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=255)
        im = ImageOps.fit(im, (2*r, 2*r), centering=(0.5,0.5))
        im.putalpha(mask.crop((cx-r, cy-r, cx+r, cy+r)))
    return im

def add_center_seal(dwg: svgwrite.Drawing, im: Image.Image, cx: float, cy: float, size_px: float, frame=True):
    datauri = image_to_datauri_png(im)
    insert = (cx - size_px/2, cy - size_px/2)
    dwg.add(dwg.image(href=datauri, insert=insert, size=(size_px, size_px), opacity=1.0))
    if frame:
        dwg.add(dwg.circle(center=(cx, cy), r=size_px/2 + 8, fill="none", stroke="#111", stroke_width=2.0, opacity=1))
        dwg.add(dwg.circle(center=(cx, cy), r=size_px/2 + 16, fill="none", stroke="#111", stroke_width=1.2, opacity=1))

# ----------------------
# Text seal with optional datetime
# ----------------------
from datetime import datetime
def add_text_seal(dwg, cx, cy, radius, text, inner_text=None, include_datetime=False):
    dwg.add(dwg.circle(center=(cx,cy), r=radius, fill="none", stroke="#111", stroke_width=2))
    dwg.add(dwg.circle(center=(cx,cy), r=radius*0.9, fill="none", stroke="#111", stroke_width=1))
    if inner_text:
        dwg.add(dwg.text(inner_text, insert=(cx, cy+radius*0.15),
                         text_anchor="middle", font_size=int(radius*0.5),
                         font_family="serif", fill="#111"))
    dwg.add(dwg.text(text, insert=(cx, cy-radius-4), text_anchor="middle",
                     font_size=int(radius*0.12), font_family="monospace", fill="#111", opacity=1))
    if include_datetime:
        dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
        for i, char in enumerate(dt_string):
            angle = 2 * math.pi * i / len(dt_string)
            r = radius * 0.65
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            dwg.add(dwg.text(char, insert=(x, y), font_size=int(radius*0.08),
                             fill="#111", font_family="Daemon Full Working", text_anchor="middle", opacity=1))

# ----------------------
# Secondary ring
# ----------------------
def add_secondary_ring(dwg: svgwrite.Drawing, cx: float, cy: float, radius: float, seed: bytes, segments: int = 360):
    import random
    random.seed(int.from_bytes(seed, 'big'))
    for i in range(segments):
        angle = 2*math.pi*i/segments
        r = radius * (0.95 + 0.1*random.random())
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        dwg.add(dwg.circle(center=(x,y), r=1.0, fill="#111", opacity=1))

# ----------------------
# Math corner patterns
# ----------------------
def add_math_patterns(dwg: svgwrite.Drawing, cx: float, cy: float, seed: bytes, width: int, height: int):
    def lissajous(a, b, delta, points=400, scale=1.0, offset=(0,0)):
        pts = []
        for t in range(points):
            tt = 2*math.pi*t/points
            x = scale * math.sin(a*tt + delta) + offset[0]
            y = scale * math.sin(b*tt) + offset[1]
            pts.append((x,y))
        return pts

    corner_coords = [(width*0.1, height*0.1), (width*0.9, height*0.1),
                     (width*0.1, height*0.9), (width*0.9, height*0.9)]
    for idx, (ox, oy) in enumerate(corner_coords):
        a = 2 + (seed[idx] % 5)
        b = 3 + (seed[(idx+2)%len(seed)] % 5)
        delta = (seed[(idx*3) % len(seed)] / 255.0) * 2*math.pi
        scale = min(width, height)*0.08
        curve = lissajous(a,b,delta, points=600, scale=scale, offset=(ox,oy))
        path = dwg.path(d=f"M{curve[0][0]},{curve[0][1]}", stroke="#111", fill="none", stroke_width=1.0, opacity=0.25)
        for x,y in curve[1:]:
            path.push(f"L{x},{y}")
        dwg.add(path)

# ----------------------
# Add corner micro-patterns based on denomination
# ----------------------
def add_value_security(dwg, value: int, x0: float, y0: float, cell: int = 4, size=12):
    bin_str = bin(value)[2:].zfill(16)
    for i, b in enumerate(bin_str):
        row = i // 4
        col = i % 4
        cx = x0 + col * cell
        cy = y0 + row * cell
        if b == "1":
            dwg.add(dwg.circle(center=(cx, cy), r=size*0.35, fill="#111", opacity=0.7))
        else:
            dwg.add(dwg.circle(center=(cx, cy), r=size*0.2, fill="#111", opacity=0.2))

# ----------------------
# Image selector
# ----------------------
import glob
def select_images(num=6, folder="."):
    exts = ["*.png","*.jpg","*.jpeg","*.bmp","*.webp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(f"{folder}/{ext}"))
    if not files:
        return []
    files.sort()
    return files[:num]
def add_corner_denominations(dwg: svgwrite.Drawing, W:int, H:int, denomination:str, font_family="Daemon Full Working", font_size=32):
    """Add big denomination texts to the corners, similar to US bills."""
    positions = [
        (W*0.08, H*0.12),
        (W*0.92, H*0.12),
        (W*0.08, H*0.88),
        (W*0.92, H*0.88),
    ]
    anchor = ["start","end","start","end"]
    for (x,y), anchor_side in zip(positions, anchor):
        dwg.add(dwg.text(denomination,
                         insert=(x,y),
                         font_size=font_size,
                         font_family=font_family,
                         fill="#111",
                         text_anchor=anchor_side,
                         alignment_baseline="middle",
                         opacity=0.85))
        
def add_corner_denominations_split(
    dwg: svgwrite.Drawing,
    W: int, H: int,
    number: str,
    font_numeric="Karamuruh",
    font_chinese="FengGuangMingRui",
    font_size_number=72,
    font_size_chinese=24
):
    """
    Add corner denominations:
    - Number (with commas) in Daemon Full Working.otf, big
    - Always append 卢纳币 in Chinese font
    """
    chinese_name = "卢纳币"  # always append

    from PIL import ImageFont

    # Format number with commas
    try:
        num_value = int(number)
        number_str = f"{num_value:,}"  # 1000000 -> "1,000,000"
    except ValueError:
        number_str = number  # fallback, in case it's already a string like "SPECIMEN"

    # Try to load the numeric font properly
    try:
        fn_numeric = ImageFont.truetype(f"./fonts/{font_numeric}", font_size_number)
        font_family_numeric = os.path.basename(fn_numeric).replace(".ttf", "")

    except Exception:
        fn_numeric = None
        font_family_numeric = "monospace"

    positions = [
        (W*0.08, H*0.12),   # top-left
        (W*0.92, H*0.12),   # top-right
        (W*0.08, H*0.88),   # bottom-left
        (W*0.92, H*0.88),   # bottom-right
    ]
    anchors_number = ["start", "end", "start", "end"]
    anchors_chinese = ["start", "end", "start", "end"]

    for (x, y), anc_num, anc_chi in zip(positions, anchors_number, anchors_chinese):
        # Draw number (with commas)
        dwg.add(dwg.text(
            number_str,
            insert=(x, y),
            font_size=font_size_number,
            font_family=font_numeric,
            fill="#111",
            text_anchor=anc_num,
            alignment_baseline="middle",
            opacity=1
        ))

        # Approximate width for positioning Chinese label
        num_width = font_size_number * len(number_str) * 0.6
        try:
            if fn_numeric:
                bbox = fn_numeric.getbbox(number_str)
                num_width = bbox[2] - bbox[0]
        except Exception:
            pass

        if anc_num == "end":  # right aligned
            chinese_x = x - num_width - 4
        else:  # left aligned
            chinese_x = x + num_width + 4

        dwg.add(dwg.text(
            chinese_name,
            insert=(chinese_x, y),
            font_size=font_size_chinese,
            font_family=font_chinese,
            fill="#111",
            text_anchor=anc_chi,
            alignment_baseline="middle",
            opacity=1
        ))



def add_treasury_and_slogan(dwg: svgwrite.Drawing, W:int, H:int,
                            title="灵国国库", slogan="灵之意志，天下共识",
                            font_title="FengGuangMingRui", font_slogan="FengGuangMingRui"):
    # Title (top center)
    dwg.add(dwg.text(title,
                     insert=(W/2, H*0.12),
                     font_size=int(H*0.12),
                     font_family=font_title,
                     fill="#111",
                     text_anchor="middle",
                     alignment_baseline="middle",
                     opacity=0.9))
    # Slogan (bottom center)
    dwg.add(dwg.text(slogan,
                     insert=(W/2, H*0.88),
                     font_size=int(H*0.07),
                     font_family=font_slogan,
                     fill="#111",
                     text_anchor="middle",
                     alignment_baseline="middle",
                     opacity=0.7))

def add_chinese_microprint(dwg: svgwrite.Drawing, cx:int, cy:int, radius:int, text="壹佰 卢纳币",
                           repetitions=1, font_family="FengGuangMingRui", font_size=8):
    """Add Chinese microprint around a small circle as a security feature."""
    import math
    n = repetitions
    for i in range(n):
        angle = 2*math.pi*i/n
        x = cx + radius*math.cos(angle)
        y = cy + radius*math.sin(angle)
        rotation = math.degrees(angle) + 90
        dwg.add(dwg.text(text,
                         insert=(x,y),
                         font_size=font_size,
                         font_family=font_family,
                         fill="#111",
                         opacity=1,
                         text_anchor="middle",
                         alignment_baseline="middle",
                         transform=f"rotate({rotation},{x},{y})"))

# ----------------------
# Generate fantasy banknote
# ----------------------
# ----------------------
# Iris-style triangular Bezier background
# ----------------------
def add_iris_background(
    dwg: svgwrite.Drawing,
    seed: bytes,
    width: int,
    height: int,
    triangle_size: int = 32,
    warp_amp: float = 12.0,
    layers: int = 6,
    input_image: Image.Image = None,
    line_opacity: float = 0.12,
    stroke_width: float = 0.35
):
    """
    Draws an iris-style triangular tessellation background with Bezier curves.
    - triangle_size: approximate pixel size of each triangle
    - layers: number of color-cycled layers
    - warp_amp: maximum warp displacement
    """
    import random
    random.seed(int.from_bytes(seed, 'big'))

    # Optional image pixels for color sampling
    if input_image:
        img = input_image.convert("RGB").resize((width, height), Image.LANCZOS)
        pixels = img.load()
    else:
        pixels = None

    # --- Create triangular grid points ---
    nx = max(4, width // triangle_size)
    ny = max(4, height // triangle_size)
    dx = width / nx
    dy = height / ny

    points = []
    for iy in range(ny + 1):
        row = []
        for ix in range(nx + 1):
            x = ix * dx + (iy % 2) * dx/2  # stagger for triangles
            y = iy * dy
            # warp displacement
            angle = random.uniform(0, 2*math.pi)
            amp = warp_amp * random.uniform(0.3, 1.0)
            if pixels:
                px = min(int(x), width-1)
                py = min(int(y), height-1)
                r, g, b = pixels[px, py]
                brightness = (r + g + b)/3/255
                amp *= brightness
                angle += brightness * math.pi
            x += amp * math.cos(angle)
            y += amp * math.sin(angle)
            row.append((x, y))
        points.append(row)

    # --- Color cycling function (CMYK-inspired) ---
    def cmyk_color(layer_idx, total_layers):
        phase = 2 * math.pi * layer_idx / max(1, total_layers)
        # CMYK-like cycle: cyan, magenta, yellow, blue (mix)
        r = int(127 + 127 * math.sin(phase))
        g = int(127 + 127 * math.sin(phase + 2.0))
        b = int(127 + 127 * math.sin(phase + 4.0))
        return f"rgb({r},{g},{b})"

    # --- Draw triangles with Bezier edges ---
    for iy in range(ny):
        for ix in range(nx):
            # Determine neighboring points (triangles)
            try:
                p0 = points[iy][ix]
                p1 = points[iy+1][ix]
                p2 = points[iy][ix+1]
                p3 = points[iy+1][ix+1]
            except IndexError:
                continue

            triangles = [(p0,p1,p2), (p2,p1,p3)]
            for t_idx, tri in enumerate(triangles):
                # Bezier path
                path = f"M{tri[0][0]},{tri[0][1]}"
                path += f" Q{tri[1][0]},{tri[1][1]} {tri[2][0]},{tri[2][1]}"
                path += f" Z"
                color = cmyk_color(random.randint(0, layers-1), layers)
                dwg.add(dwg.path(d=path, stroke=color, fill="none", stroke_width=stroke_width, opacity=line_opacity))

    # --- Optional center warp accent ---
    cx, cy = width/2, height/2
    for l in range(layers):
        rad = min(width, height)/2 * (0.2 + 0.1*l)
        steps = 12 + l*4
        for s in range(steps):
            angle = 2*math.pi*s/steps
            x = cx + rad * math.cos(angle)
            y = cy + rad * math.sin(angle)
            path = f"M{cx},{cy} Q{(cx+x)/2},{(cy+y)/2} {x},{y}"
            color = cmyk_color(l, layers)
            dwg.add(dwg.path(d=path, stroke=color, fill="none", stroke_width=stroke_width*0.6, opacity=line_opacity))
def number_to_chinese(num: int) -> str:
    numerals = {
        0:"零", 1:"壹", 2:"贰", 3:"叁", 4:"肆",
        5:"伍", 6:"陆", 7:"柒", 8:"捌", 9:"玖"
    }
    units = ["", "拾", "佰", "仟", "万", "拾万", "佰万", "仟万", "亿"]
    s = str(num)
    result = ""
    for i, digit in enumerate(s[::-1]):
        n = int(digit)
        if n != 0:
            result = numerals[n] + units[i] + result
        elif not result.startswith("零"):
            result = "零" + result
    return result.rstrip("零")

def generate_fantasy_banknote(seed_text: str, input_image_path: str, outfile_svg: str,
                               width_mm: float = 160.0, height_mm: float = 60.0,
                               title: str = "灵国国库", subtitle: str = "灵之意志，天下共识",
                               denomination: str = "100 卢纳币", specimen: bool = True,
                               fonts = {}):
    font_main = "FengGuangMingRui"
    font_numeric = "Karamuruh"

    W = mm_to_px(width_mm)
    H = mm_to_px(height_mm)
    dwg = svgwrite.Drawing(outfile_svg, size=(W,H), viewBox=f"0 0 {W} {H}")
    # After dwg = svgwrite.Drawing(...)
    from fontTools.ttLib import TTFont

    embed_font(dwg, TTFont("../../fonts/gunmetl.ttf"), "Karamuruh")
    embed_font(dwg, TTFont("../../fonts/Daemon Full Working.otf"), "Daemon Full Working")

    embed_font(dwg, TTFont("../../fonts/FengGuangMingRui Regular.ttf"), "FengGuangMingRui")

    seed = sha256_bytes(seed_text)

    im = None
    if input_image_path and os.path.exists(input_image_path):
        im = Image.open(input_image_path).convert("RGB")
    #add_fancy_tiled_background(dwg, seed, W, H, 16, im)
    # Use iris-style triangular Bezier background
    moire_fishnet_background(
        dwg=dwg,
        seed=seed,
        width=W,
        height=H,
        warp_amp=256,
        input_image=im,
        line_opacity=0.5,
        stroke_width=0.5,
        dithering_density=0.5
    )

    add_qr_like_border(dwg, seed, W, H, cell=10, margin_cells=3)

    left_cx = int(W*0.18)
    right_cx = int(W*0.82)
    cy = H//2
    small_radius = min(W,H)*0.25

    add_text_seal(dwg, left_cx, cy, small_radius*0.65, "灵国国库", "日", include_datetime=True)
    add_text_seal(dwg, right_cx, cy, small_radius*0.65, "灵国国库", "月", include_datetime=True)

    center_px_radius = int(min(W,H)*0.32)
    center_curves = polar_mandala(seed, center_px_radius, layers=7, sym=12, points=800)
    cx = W//2
    cy = H//2


    # Load and sample colors from input image
    im = load_and_prepare_image(input_image_path, max_dim=center_px_radius)
    palette = get_palette_from_image(im, num_colors=7)
    #add_image_palette_background(dwg, W, H, palette, tile=8)
    add_number_boxes(dwg, W, H, palette, box_count=32, max_size=16)

    add_center_seal(dwg, im, cx, cy, center_px_radius*1.2)
    add_secondary_ring(dwg, cx, cy, radius=center_px_radius*0.88, seed=seed, segments=256)
    #add_math_patterns(dwg, cx, cy, seed, W, H)
    
    denom_map = {"1":1, "10":10, "100":100, "1000":1000, "10000":10000}
    num_value = denom_map.get(denomination.replace(" 卢纳币",""), 100)
    corner_offset = 8
    #add_value_security(dwg, num_value, corner_offset, corner_offset)
    #add_value_security(dwg, num_value, W - 32, corner_offset)
    #add_value_security(dwg, num_value, corner_offset, H - 32)
    #add_value_security(dwg, num_value, W - 32, H - 32)
    # Add Treasury title and slogan
    add_treasury_and_slogan(dwg, W, H, title="灵国国库", slogan="灵之意志，天下共识",
                            font_title="FengGuangMingRui", font_slogan="FengGuangMingRui")

    # Parse denomination number
    num_value = int(denomination.replace(" 卢纳币", ""))

    # Chinese version
    chinese_value = number_to_chinese(num_value)

    # Add corner denominations (Arabic numerals)
    add_corner_denominations_split(
        dwg, W, H, number=str(num_value),
        font_numeric="Daemon Full Working", font_chinese="FengGuangMingRui"
    )

    # Add Chinese microprint with correct denomination
    add_chinese_microprint(
        dwg, cx, cy, radius=int(center_px_radius*0.7),
        text=f"{chinese_value} 卢纳币",
        repetitions=16
    )




    if specimen:
        dwg.add(dwg.text("SPECIMEN", insert=(W*0.5,H*0.92),
                         font_size=int(H*0.08), fill="#333", font_family="monospace", text_anchor="middle", opacity=0.75))

    dwg.save()
    print(f"[+] Saved: {outfile_svg}")
from PIL import ImageStat

def get_palette_from_image(img: Image.Image, num_colors: int = 6) -> list:
    """Return a list of RGB tuples from the most prominent colors of an image."""
    small = img.resize((64,64))
    paletted = small.convert("P", palette=Image.ADAPTIVE, colors=num_colors)
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    colors = []
    for count, idx in color_counts[:num_colors]:
        r,g,b = palette[idx*3:idx*3+3]
        colors.append((r,g,b))
    return colors

def add_image_palette_background(dwg: svgwrite.Drawing, width: int, height: int, palette: list, tile: int = 64):
    """
    Fill the background with colored triangles from the palette.
    Triangles are arranged in a grid, optionally alternating orientation.
    """
    import random
    import math

    cols = math.ceil(width / tile)
    rows = math.ceil(height / tile)

    for r in range(rows):
        for c in range(cols):
            # Center of the current tile
            cx = c * tile + tile / 2
            cy = r * tile + tile / 2
            # Triangle size
            s = tile * 0.5

            # Alternate orientation for variation
            if (r + c) % 2 == 0:
                pts = [(cx, cy - s/2), (cx - s/2, cy + s/2), (cx + s/2, cy + s/2)]
            else:
                pts = [(cx, cy + s/2), (cx - s/2, cy - s/2), (cx + s/2, cy - s/2)]

            color = f"rgb{random.choice(palette)}"
            dwg.add(dwg.polygon(points=pts, fill=color, opacity=0.25))



def add_number_boxes(dwg: svgwrite.Drawing, width:int, height:int,
                     palette: list, box_count:int = 50, max_size:int=12):
    import random
    for _ in range(box_count):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(4, max_size)
        color = random.choice(palette)
        number = str(random.randint(0,9))
        dwg.add(dwg.rect(insert=(x,y), size=(size,size), fill=f"rgb{color}", opacity=0.35))
        dwg.add(dwg.text(number, insert=(x+size*0.2, y+size*0.8),
                         font_size=int(size*0.8), fill="#000", font_family="monospace"))

if __name__ == "__main__":
    import argparse
    import os
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Fantasy banknote generator")
    parser.add_argument("seed_text", type=str, help="Seed text or name for the note")
    parser.add_argument("input_image", type=str, help="Input image path")
    parser.add_argument("--outfile", type=str, default="banknote.svg", help="Base output SVG file")
    parser.add_argument("--specimen", action="store_true", help="Add SPECIMEN overlay")
    parser.add_argument("--copies", type=int, default=9, help="Number of distinct notes to generate")
    parser.add_argument("--yen_model", action="store_true", help="Use 1-100,000,000 denominations")
    args = parser.parse_args()

    fonts = load_fonts("./fonts")
    
    # Generate denominations
    if args.yen_model:
        base_denoms = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
        denominations = base_denoms[:9]  # top 9 denominations
    else:
        denominations = [100 * (i + 1) for i in range(9)]  # default 9 denominations

    import time

    for i in tqdm(range(args.copies), desc="Generating banknotes"):
        new_seed = args.seed_text  # no _i prefix in filenames

        for denom in denominations:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(args.outfile)
            # Filename format: seed_denomination_datetime.svg
            outfile_svg = f"{new_seed}_{denom}_{timestamp}{ext}"

            denomination_str = f"{denom} 卢纳币"

            generate_fantasy_banknote(
                seed_text=f"{new_seed}_{i}",  # keep unique seed for generation
                input_image_path=args.input_image,
                outfile_svg=outfile_svg,
                specimen=args.specimen,
                denomination=denomination_str,
                fonts=fonts
            )

