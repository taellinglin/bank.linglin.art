#!/usr/bin/env python3
"""
banknote_backside_batch_color.py

Generates nine symmetric backside SVGs for denominations with Red/Blue colors.

Author: RingMaster Lin
"""
from io import BytesIO
import os
import math
import argparse
import svgwrite
from typing import List, Tuple
import base64
import colorsys
from skimage import color, segmentation, measure, util
import numpy as np
import random
import hashlib
import secrets
from datetime import datetime
import time
import re
import io
import requests

# At the top of your module
bg_image = None  # initially empty

def load_background(path):
    global bg_image
    from PIL import Image
    bg_image = Image.open(path).convert("RGB")

# Optional PNG conversion
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

# ----------------------
# Utilities
# ----------------------
MM_TO_PX = 300.0 / 25.4

def mm_to_px(mm: float, dpi: float = 300.0) -> int:
    return int(round(mm * dpi / 25.4))

# ----------------------
# Spirograph / hypotrochoid
# ----------------------
def spiro_points(R: float, r: float, d: float, steps: int = 2000) -> List[Tuple[float, float]]:
    pts = []
    g = math.gcd(int(round(R)), int(round(r))) if (R>0 and r>0) else 1
    lcm_period = r // g if g != 0 else 1
    total_t = 2 * math.pi * lcm_period
    for i in range(steps):
        t = total_t * i / steps
        x = (R - r) * math.cos(t) + d * math.cos(((R - r) / r) * t)
        y = (R - r) * math.sin(t) - d * math.sin(((R - r) / r) * t)
        pts.append((x, y))
    return pts

def pts_to_path(pts: List[Tuple[float, float]], translate=(0,0), scale=1.0, smooth=True) -> str:
    if not pts:
        return ""
    tx, ty = translate
    def p(i):
        x, y = pts[i]
        return (x*scale + tx, y*scale + ty)
    d = f"M{p(0)[0]:.3f},{p(0)[1]:.3f}"
    if smooth and len(pts) > 2:
        for i in range(1, len(pts)-1):
            xm, ym = p(i)
            xn, yn = p(i+1)
            midx, midy = (xm + xn)/2, (ym + yn)/2
            d += f" Q{xm:.3f},{ym:.3f} {midx:.3f},{midy:.3f}"
        xl, yl = p(len(pts)-1)
        d += f" T{xl:.3f},{yl:.3f}"
    else:
        for i in range(1, len(pts)):
            x, y = p(i)
            d += f" L{x:.3f},{y:.3f}"
    return d

# ----------------------
# Fonts
# ----------------------
CHINESE_FONT = "./fonts/FengGuangMingRui.ttf"
NUMBER_FONT  = "./fonts/Daemon Full Working.otf"

def embed_font(dwg, font_path: str, font_name: str):
    with open(font_path, "rb") as f:
        font_data = f.read()
    font_b64 = base64.b64encode(font_data).decode("ascii")
    style = f"""
    @font-face {{
        font-family: '{font_name}';
        src: url(data:font/ttf;base64,{font_b64}) format('truetype');
    }}
    """
    dwg.defs.add(dwg.style(style))

# ----------------------
# Artwork elements
# ----------------------
def add_corner_denoms(dwg, W: int, H: int, denom_str: str):
    """
    Draws denomination numbers in all four corners with white outline 0.05cm behind
    fill and fill opacity 0.9. Bottom ones remain aligned as before.
    """

    # Format denomination with commas
    try:
        denom_formatted = f"{int(denom_str):,}"
    except ValueError:
        denom_formatted = denom_str

    first_digit = denom_formatted[0]
    rest_digits = denom_formatted[1:]

    # Sizes
    BIG_FONT = 128
    SMALL_FONT = 72

    # Padding
    PADDING = int(0.5 * 30 * 3.78)

    # Stroke thickness in pixels (0.05 cm at 300 DPI)
    STROKE_WIDTH = 0.05 * 300 / 2.54

    # Colors for corners
    COLORS = ["red", "green", "blue", "black"]

    # Helper to add text with stroke behind
    def add_text_with_outline(x, y, text, font_size, color, anchor, baseline):
        # Stroke first
        dwg.add(dwg.text(
            text,
            insert=(x, y),
            font_size=font_size,
            font_family="Daemon Full Working",
            fill="none",
            stroke="#FFF",
            stroke_width=STROKE_WIDTH,
            text_anchor=anchor,
            alignment_baseline=baseline,
            opacity=0.9
        ))
        # Fill on top
        dwg.add(dwg.text(
            text,
            insert=(x, y),
            font_size=font_size,
            font_family="Daemon Full Working",
            fill=color,
            stroke="none",
            text_anchor=anchor,
            alignment_baseline=baseline,
            opacity=0.9
        ))

    # --- Top-left ---
    add_text_with_outline(PADDING, PADDING, first_digit, BIG_FONT, COLORS[0], "start", "hanging")
    offset_x = PADDING + BIG_FONT * 0.6
    add_text_with_outline(offset_x, PADDING, rest_digits, SMALL_FONT, COLORS[0], "start", "hanging")

    # --- Top-right ---
    add_text_with_outline(W - PADDING, PADDING, rest_digits, SMALL_FONT, COLORS[1], "end", "hanging")
    offset_x = W - PADDING - SMALL_FONT * len(rest_digits) * 0.55
    add_text_with_outline(offset_x, PADDING, first_digit, BIG_FONT, COLORS[1], "end", "hanging")

    # --- Bottom-left ---
    add_text_with_outline(PADDING, H - PADDING, first_digit, BIG_FONT, COLORS[2], "start", "baseline")
    offset_x = PADDING + BIG_FONT * 0.6
    add_text_with_outline(offset_x, H - PADDING, rest_digits, SMALL_FONT, COLORS[2], "start", "baseline")

    # --- Bottom-right ---
    add_text_with_outline(W - PADDING, H - PADDING, rest_digits, SMALL_FONT, COLORS[3], "end", "baseline")
    offset_x = W - PADDING - SMALL_FONT * len(rest_digits) * 0.55
    add_text_with_outline(offset_x, H - PADDING, first_digit, BIG_FONT, COLORS[3], "end", "baseline")

def hexagon(cx, cy, radius):
    """Return points of a hexagon centered at (cx, cy)"""
    return [(cx + radius * math.cos(math.radians(angle)),
             cy + radius * math.sin(math.radians(angle)))
            for angle in range(0, 360, 60)]

def tesselated_hex(dwg, x, y, size, rows=16, cols=16, stroke_color="#000000"):
    hex_r = float(size) / (cols * 2)   # size is numeric
    dx = 1.5 * hex_r
    dy = (3 ** 0.5) * hex_r / 2

    def hexagon(cx, cy, r):
        return [(cx + r * math.cos(math.pi/3 * i),
                 cy + r * math.sin(math.pi/3 * i)) for i in range(6)]

    for row in range(rows):
        for col in range(cols):
            cx = x + col * dx * 2 + (row % 2) * dx
            cy = y + row * dy * 2
            dwg.add(dwg.polygon(points=hexagon(cx, cy, hex_r),
                                fill="none", stroke=stroke_color, stroke_width=1))

def tesselated_triangles(dwg, x, y, size, rows=8, cols=8, stroke_color="#000000"):
    """Draw a tessellation of equilateral triangles starting from (x,y)."""
    tri_h = (math.sqrt(3) / 2) * size  # height of an equilateral triangle

    for row in range(rows):
        for col in range(cols):
            # Alternate upright vs inverted triangles
            if (row + col) % 2 == 0:
                points = [
                    (x + col*size/2, y + row*tri_h),
                    (x + col*size/2 + size/2, y + row*tri_h + tri_h),
                    (x + col*size/2 - size/2, y + row*tri_h + tri_h),
                ]
            else:
                points = [
                    (x + col*size/2, y + row*tri_h + tri_h),
                    (x + col*size/2 + size/2, y + row*tri_h),
                    (x + col*size/2 - size/2, y + row*tri_h),
                ]
            dwg.add(dwg.polygon(points=points,
                                fill="none",
                                stroke=stroke_color,
                                stroke_width=1))

def add_functional_corner_decorations(dwg, W, H, denom, timestamp, serial_id,
                                      size=100, padding=75, stroke_width=1):
    # Main + highlight colors per corner
    COLORS = [
        ("#D80027", "#FF5555", "#FF69B4"),  # top-left (red + pink)
        ("#009E60", "#55FFAA", "#FFD700"),  # top-right (green + yellow)
        ("#0052B4", "#55AAFF", "#FF69B4"),  # bottom-left (blue + pink)
        ("#222222", "#AAAAAA", "#FFD700"),  # bottom-right (black/gray + yellow)
    ]

    def micro_text_pattern(x, y, text, rows=12, cols=12, spacing=10,
                           c_main="#000", c_highlight="#FF69B4"):
        """Repeating microtext grid with alternating highlight color."""
        for row in range(rows):
            for col in range(cols):
                color = c_main if (row+col) % 3 else c_highlight
                dwg.add(dwg.text(text,
                                 insert=(x + col*spacing, y + row*spacing),
                                 font_size=6, font_family="Daemon Full Working",
                                 fill=color, opacity=0.25))

    def tesselated_triangles(dwg, x, y, s, rows=8, cols=8,
                             c_main="#000", c_highlight="#FFD700"):
        """Draw tessellated upright + inverted triangles with mixed colors."""
        h = s * (3 ** 0.5) / 2
        for row in range(rows):
            for col in range(cols):
                x0 = x + col * s
                y0 = y + row * h
                if (row + col) % 2 == 0:
                    pts = [(x0, y0 + h), (x0 + s/2, y0), (x0 + s, y0 + h)]
                else:
                    pts = [(x0, y0), (x0 + s, y0), (x0 + s/2, y0 + h)]
                stroke_color = c_main if (row+col) % 4 else c_highlight
                dwg.add(dwg.polygon(points=pts, fill="none",
                                    stroke=stroke_color,
                                    stroke_width=0.6, opacity=0.7))

    def top_left(x, y, denom):
        main, secondary, highlight = COLORS[0]
        for i in range(3):
            offset = i*size*0.18
            stroke_c = main if i % 2 == 0 else highlight
            dwg.add(dwg.rect(insert=(x+offset, y+offset),
                             size=(size-2*offset, size-2*offset),
                             rx=8, ry=8, fill="none",
                             stroke=stroke_c, stroke_width=stroke_width))
        dwg.add(dwg.text(denom, insert=(x+size/2, y+size/2),
                         font_size=22, text_anchor="middle",
                         alignment_baseline="middle",
                         font_family="Daemon Full Working", fill=secondary))
        micro_text_pattern(x+12, y+12, denom, c_main=secondary, c_highlight=highlight)

    def top_right(x, y, denom):
        main, secondary, highlight = COLORS[1]
        tesselated_triangles(dwg, x - size, y, size/6, rows=12, cols=12,
                             c_main=main, c_highlight=highlight)
        dwg.add(dwg.text(denom, insert=(x - size/2, y + size/2),
                         font_size=20, text_anchor="middle",
                         alignment_baseline="middle",
                         font_family="Daemon Full Working", fill=random.choice([secondary, highlight])))

    def bottom_left(x, y, denom):
        main, secondary, highlight = COLORS[2]
        tesselated_triangles(dwg, x, y - size, size/6, rows=12, cols=12,
                             c_main=main, c_highlight=highlight)
        dwg.add(dwg.text(denom, insert=(x + size/2, y - size/2),
                         font_size=20, text_anchor="middle",
                         alignment_baseline="middle",
                         font_family="Daemon Full Working", fill=random.choice([secondary, highlight])))

    def bottom_right(x, y, denom, timestamp):
        main, secondary, highlight = COLORS[3]
        for i in range(4):
            offset = i*size*0.18
            stroke_c = main if i % 2 else highlight
            dwg.add(dwg.rect(insert=(x - size + offset, y - size + offset),
                             size=(size - 2*offset, size - 2*offset),
                             rx=10, ry=10, fill="none",
                             stroke=stroke_c, stroke_width=stroke_width))
        dwg.add(dwg.text(denom, insert=(x - size/2, y - size/2),
                         font_size=22, text_anchor="middle",
                         alignment_baseline="middle",
                         font_family="Daemon Full Working", fill=random.choice([secondary, highlight])))
        micro_text_pattern(x - size + 5, y - size + 5, f"{denom} {timestamp}",
                           c_main=secondary, c_highlight=highlight)

    # Apply all four corners
    top_left(padding, padding, denom)
    top_right(W - padding, padding, denom)
    bottom_left(padding, H - padding, denom)
    bottom_right(W - padding, H - padding, denom, timestamp)

import math
def add_decorative_border(dwg, W:int, H:int, border_info:dict, denom_value: int, timestamp_ms:int):
    """
    Adds multi-band border around diamond area.
    Each band encodes parts of the timestamp (year, month, etc.)
    and the pattern is influenced by denom_value.
    
    Shapes used:
        0 → filled diamond
        1 → empty diamond
        2 → filled square
        3 → empty square
        4 → X / stitch
    """

    import datetime
    if isinstance(timestamp_ms, dict):
        timestamp_ms = timestamp_ms.get("timestamp_ms", 0)

    # Convert timestamp_ms to datetime
    ts = datetime.datetime.fromtimestamp(timestamp_ms / 1000.0)
    bands = [
        ("year", ts.year % 100, 0.25 + 0.025),           # ¼ cm (largest)
        ("month", ts.month, 0.1875 + 0.025),             # 3/16 cm
        ("day", ts.day, 0.125 + 0.025),                  # ⅛ cm
        ("hour", ts.hour, 0.1875 + 0.025),               # 3/16 cm (repeats at hour level)
        ("minute", ts.minute, 0.09375 + 0.025),          # 3/32 cm (half of hour)
        ("second", ts.second, 0.046875 + 0.025),         # 3/64 cm (half of minute)
        ("microsecond", ts.microsecond // 1000, 0.0234375 + 0.025)  # 3/128 cm (half of second)
    ]

    # Get diamond area from border_info
    start_x = float(border_info.get("diamond_start_x", 0))
    start_y = float(border_info.get("diamond_start_y", 0))
    width   = float(border_info.get("diamond_width", W))
    height  = float(border_info.get("diamond_height", H))

    # Convert cm to pixels (assuming 96 dpi)
    cm_to_px = lambda cm: float(cm * 96.0 / 2.54)

    pad_base = cm_to_px(0.25)  # ¼ cm padding
    inset = -0.75
    denom_value = denom_value or 0  # default if None
    # Add opacity to both fill and stroke
    fill_opacity = 1
    stroke_opacity = 1
    # --- shape drawing helpers
    def draw_shape(g, x, y, size, kind, band_index):
        half = size / 2.0
    
        # Alternate by band - even bands: dark fill/light stroke, odd bands: light fill/dark stroke
        fill_black = band_index % 2 == 0
        
        if fill_black:
            fill_color = "#000"  # Dark gray
            stroke_color = "#FFFFFF"  # Light gray
            stroke_opacity = 1/(band_index+0.01)
            fill_opacity = 1
        else:
            fill_color = "#FFF"  # Light gray
            stroke_color = "#000000"  # Dark gray
            fill_opacity = band_index/1
            stroke_opacity = 1
        
        # 50% transparency
        stroke_width = max(0.5, size * 0.025)
        
        if kind == 0:  # filled diamond
            pts = [(x+half, y), (x+size, y+half), (x+half, y+size), (x, y+half)]
            g.add(dwg.polygon(points=pts, fill=fill_color, fill_opacity=fill_opacity, 
                            stroke=stroke_color, stroke_opacity=stroke_opacity, stroke_width=stroke_width))
        elif kind == 1:  # empty diamond
            pts = [(x+half, y), (x+size, y+half), (x+half, y+size), (x, y+half)]
            g.add(dwg.polygon(points=pts, fill="none", 
                            stroke=stroke_color, stroke_opacity=stroke_opacity, stroke_width=stroke_width))
        elif kind == 2:  # filled square
            g.add(dwg.rect(insert=(x, y), size=(size, size), 
                        fill=fill_color, fill_opacity=fill_opacity,
                        stroke=stroke_color, stroke_opacity=stroke_opacity, stroke_width=stroke_width))
        elif kind == 3:  # empty square
            g.add(dwg.rect(insert=(x, y), size=(size, size), fill="none",
                        stroke=stroke_color, stroke_opacity=stroke_opacity, stroke_width=stroke_width))
        elif kind == 4:  # X / stitch
            g.add(dwg.line(start=(x, y), end=(x+size, y+size), 
                        stroke=stroke_color, stroke_opacity=stroke_opacity, stroke_width=stroke_width))
            g.add(dwg.line(start=(x+size, y), end=(x, y+size), 
                        stroke=stroke_color, stroke_opacity=stroke_opacity, stroke_width=stroke_width))

    for band_index, (band_name, value, band_cm) in enumerate(bands):
        band_size = cm_to_px(band_cm)
        g = dwg.g()

        # number of tiles along edges
        num_cols = int((width - 2*pad_base - 2*inset) // band_size)
        num_rows = int((height - 2*pad_base - 2*inset) // band_size)

        offset = value + (denom_to_int(denom_value) % 97)

        # --- top border
        y = start_y + pad_base + inset
        for c in range(num_cols):
            x = start_x + pad_base + inset + c * band_size
            kind = (c + offset) % 5
            draw_shape(g, x, y, band_size, kind, band_index)  # Added band_index

        # --- bottom border
        y = start_y + height - pad_base - inset - band_size
        for c in range(num_cols):
            x = start_x + pad_base + inset + c * band_size
            kind = (c + offset + 1) % 5
            draw_shape(g, x, y, band_size, kind, band_index)  # Added band_index

        # --- left border
        x = start_x + pad_base + inset
        for r in range(num_rows):
            y = start_y + pad_base + inset + r * band_size
            kind = (r + offset + 2) % 5
            draw_shape(g, x, y, band_size, kind, band_index)  # Added band_index

        # --- right border
        x = start_x + width - pad_base - inset - band_size
        for r in range(num_rows):
            y = start_y + pad_base + inset + r * band_size
            kind = (r + offset + 3) % 5
            draw_shape(g, x, y, band_size, kind, band_index)  # Added band_index

        dwg.add(g)
        inset += band_size

import math
import random
import svgwrite

def add_hightech_hologram_seals(dwg, W:int, H:int, radius:int=64, stroke_w:float=1.4):
    """
    Adds high-tech Ghost-in-the-Shell style holographic seals
    instead of floral spirograph motifs.
    """
    def add_hologram_circle(group, cx, cy, base_r, layers=5):
        """Draw concentric circuit-like arcs and ticks"""
        for i in range(layers):
            r = base_r * (0.5 + i*0.15)
            # alternating colors
            color = ["#00FFE0", "#FF0080", "#00FF88", "#D80027", "#0052B4"][i % 5]
            opacity = 0.55 if i % 2 == 0 else 0.8

            # broken circular arc
            arc_path = []
            steps = 36
            for k in range(steps):
                ang1 = (k * 360/steps) * math.pi/180
                ang2 = ((k+0.6) * 360/steps) * math.pi/180
                x1, y1 = cx + r*math.cos(ang1), cy + r*math.sin(ang1)
                x2, y2 = cx + r*math.cos(ang2), cy + r*math.sin(ang2)
                arc_path.append(f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}")

            group.add(dwg.path(d=" ".join(arc_path),
                               stroke=color, fill="none",
                               stroke_width=stroke_w*0.8, opacity=opacity))

            # radial ticks (like a microchip encoder ring)
            for j in range(0, 360, 15):
                a = math.radians(j)
                x1, y1 = cx + (r-3)*math.cos(a), cy + (r-3)*math.sin(a)
                x2, y2 = cx + (r+3)*math.cos(a), cy + (r+3)*math.sin(a)
                group.add(dwg.line((x1,y1),(x2,y2),
                                   stroke=color, stroke_width=0.7, opacity=0.6))

    # left hologram seal
    lx, ly = int(W*0.18), int(H*0.5)
    g_left = dwg.g(opacity=0.95)
    add_hologram_circle(g_left, lx, ly, radius, layers=6)
    g_left.add(dwg.text("天圆", insert=(lx, ly+4),
                        font_size=int(radius*0.22), text_anchor="middle",
                        font_family="FengGuangMingRui",
                        fill="#00FFE0", opacity=1))
    dwg.add(g_left)

    # mirrored hologram seal
    cx = W/2
    mirrored_rx = cx + (cx - lx)
    g_mirror = dwg.g(transform=f"translate({cx},0) scale(-1,1) translate({-cx},0)")
    add_hologram_circle(g_mirror, lx, ly, radius, layers=6)
    dwg.add(g_mirror)

    # mirrored text, rotated upside down
    g_mirror_text = dwg.g(transform=f"rotate(180 {mirrored_rx} {ly})")
    g_mirror_text.add(dwg.text("地方", insert=(mirrored_rx, ly+4),
                               font_size=int(radius*0.22), text_anchor="middle",
                               font_family="FengGuangMingRui",
                               fill="#FF0080", opacity=1))
    dwg.add(g_mirror_text)


def add_central_spiro_and_background(dwg, W:int, H:int, denom_exponent:int):
    cx, cy = W/2, H/2
    base_R = min(W,H) * (0.35 + 0.02 * denom_exponent)
    base_r = max(4, int(base_R * (0.08 + 0.02 * (denom_exponent % 5))))
    base_d = int(base_r * (0.7 + 0.25 * ((denom_exponent+1) % 4)))

    g_bg = dwg.g(opacity=0.8)
    step = int(min(W,H) * (0.015 + 0.002 * denom_exponent))
    diamond_size = max(2, step//2)
    offset = (denom_exponent * 7) % step

    for x in range(0, W//2, step):
        for y in range(0, H//2, step):
            px = x + offset
            py = y + offset
            pts = [
                (px + diamond_size/2, py),
                (px + diamond_size, py + diamond_size/2),
                (px + diamond_size/2, py + diamond_size),
                (px, py + diamond_size/2),
            ]
            g_bg.add(dwg.polygon(points=pts, fill="#0052B4"))

    dwg.add(g_bg)
    dwg.add(dwg.g(transform=f"translate({cx},0) scale(-1,1) translate({-cx},0)").add(g_bg))
    dwg.add(dwg.g(transform=f"translate(0,{cy}) scale(1,-1) translate(0,{-cy})").add(g_bg))
    dwg.add(dwg.g(transform=f"translate({cx},{cy}) scale(-1,-1) translate({-cx},{-cy})").add(g_bg))

    bands = [
        {"R_mult": 1.0, "r_mult": 1.0, "d_mult": 1.0},
        {"R_mult": 0.7, "r_mult": 0.8, "d_mult": 0.9},
        {"R_mult": 0.45, "r_mult": 0.6, "d_mult": 0.7},
    ]

    palette_options = [
        ["#D80027", "#FF7F50", "#FFA500"],
        ["#0052B4", "#00BFFF", "#87CEFA"],
        ["#009E60", "#00FF7F", "#32CD32"],
        ["#800080", "#DA70D6", "#FF00FF"],
        ["#FF0000", "#00FF00", "#0000FF"],
    ]
    colors = palette_options[denom_exponent % len(palette_options)]

    for band_index, band in enumerate(bands):
        R = base_R * band["R_mult"]
        r = max(3, int(base_r * band["r_mult"]))
        d = int(base_d * band["d_mult"])
        steps = 1500 + band_index*300
        theta_offset = math.radians(denom_exponent * 12 + band_index * 7)

        pts = []
        total_t = 2*math.pi
        for i in range(steps):
            t = total_t * i / steps
            x = (R - r) * math.cos(t + theta_offset) + d * math.cos(((R - r)/r) * t)
            y = (R - r) * math.sin(t + theta_offset) - d * math.sin(((R - r)/r) * t)
            pts.append((x, y))

        pd = pts_to_path(pts, translate=(cx,cy), scale=1.0, smooth=True)
        color = colors[band_index % len(colors)]
        dwg.add(dwg.path(d=pd, stroke=color, stroke_width=0.6, fill="none", opacity=0.55))

        mirror_x = dwg.g(transform=f"translate({cx},0) scale(-1,1) translate({-cx},0)")
        mirror_x.add(dwg.path(d=pd, stroke=color, stroke_width=0.5, fill="none", opacity=0.5))
        dwg.add(mirror_x)
        mirror_y = dwg.g(transform=f"translate(0,{cy}) scale(1,-1) translate(0,{-cy})")
        mirror_y.add(dwg.path(d=pd, stroke=color, stroke_width=0.5, fill="none", opacity=0.5))
        dwg.add(mirror_y)

def hsl_to_rgb_string(h, s, l):
    h = h / 360.0
    s = s / 100.0
    l = l / 100.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

def cm_to_px(cm, dpi=300.0):
    return cm * dpi / 2.54
import hashlib
import json
import base64
import zlib
from datetime import datetime

def encode_banknote_metadata(title_text, phrase_text, serial_id, timestamp_ms, denomination):
    """
    Encode banknote metadata into a structured prompt seed
    """
    # Create a structured dictionary
    metadata = {
        "title": title_text,
        "phrase": phrase_text,
        "serial": serial_id,
        "timestamp": timestamp_ms,
        "denomination": denomination,
        "theme": generate_theme_from_metadata(title_text, phrase_text, denomination)
    }
    
    # Convert to JSON and compress
    json_str = json.dumps(metadata, ensure_ascii=False)
    compressed = zlib.compress(json_str.encode('utf-8'))
    
    # Base64 encode for readability
    encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')
    
    return encoded

def decode_banknote_metadata(encoded_seed):
    """
    Decode the seed back into metadata
    """
    try:
        decoded = base64.urlsafe_b64decode(encoded_seed)
        decompressed = zlib.decompress(decoded)
        metadata = json.loads(decompressed.decode('utf-8'))
        return metadata
    except:
        return None

def generate_theme_from_metadata(title_text, phrase_text, denomination):
    """
    Generate a thematic description based on the metadata
    """
    # Theme mapping based on denomination
    denomination_themes = {
        "1": "foundational, basic, elemental",
        "10": "growth, development, progression",
        "100": "harmony, balance, unity",
        "1000": "prosperity, abundance, wealth",
        "10000": "tradition, heritage, legacy",
        "100000": "power, authority, sovereignty",
        "1000000": "mystery, enlightenment, wisdom",
        "10000000": "divine, celestial, eternal",
        "100000000": "imperial, majestic, supreme"
    }
    
    # Get theme based on denomination
    denom_value = str(denomination).split()[0] if isinstance(denomination, str) else str(denomination)
    theme = denomination_themes.get(denom_value, "noble, elegant, prestigious")
    
    # Add elements based on title and phrase
    title_words = title_text.lower().split()
    phrase_words = phrase_text.lower().split()
    
    if "灵" in title_text or "spirit" in title_text.lower():
        theme += ", spiritual, ethereal"
    if "国" in title_text or "kingdom" in title_text.lower():
        theme += ", regal, governmental"
    if "国库" in title_text or "treasury" in title_text.lower():
        theme += ", financial, economic"
    
    if "意志" in phrase_text or "will" in phrase_text.lower():
        theme += ", determined, resolute"
    if "天下" in phrase_text or "world" in phrase_text.lower():
        theme += ", universal, global"
    if "共识" in phrase_text or "consensus" in phrase_text.lower():
        theme += ", harmonious, united"
    
    return theme

def create_background_prompt_from_seed(encoded_seed, name=""):
    """
    Create a background prompt from encoded metadata
    """
    metadata = decode_banknote_metadata(encoded_seed)
    
    if not metadata:
        # Fallback to default prompt
        default_prompt = read_prompt_file(
            "background_prompt.txt",
            "abstract ornamental pattern, intricate design, currency background, banknote pattern"
        )
        return default_prompt.format(name=name) if "{name}" in default_prompt else default_prompt
    
    # Build sophisticated prompt from metadata
    prompt_parts = []
    
    # Base pattern description
    prompt_parts.append("intricate ornamental pattern for banknote currency")
    
    # Add theme
    prompt_parts.append(f"theme: {metadata['theme']}")
    
    # Add elements based on title
    if "灵" in metadata['title']:
        prompt_parts.append("spiritual symbols, ethereal elements")
    if "国" in metadata['title']:
        prompt_parts.append("national emblems, governmental seals")
    if "国库" in metadata['title']:
        prompt_parts.append("financial motifs, treasure symbols")
    
    # Add elements based on phrase
    if "意志" in metadata['phrase']:
        prompt_parts.append("determined patterns, resolute designs")
    if "天下" in metadata['phrase']:
        prompt_parts.append("universal symbols, global patterns")
    if "共识" in metadata['phrase']:
        prompt_parts.append("harmonious patterns, united elements")
    
    # Add denomination-based elements
    denom = str(metadata['denomination'])
    if denom in ["100", "500", "1000"]:
        prompt_parts.append("precious metal accents, gold and silver filigree")
    if denom in ["500", "1000", "5000"]:
        prompt_parts.append("complex security patterns, anti-counterfeit elements")
    
    # Add technical specifications
    prompt_parts.append("vector art style, clean lines, professional banknote design")
    prompt_parts.append("subtle colors, elegant financial aesthetic")
    
    # Combine all parts
    full_prompt = ", ".join(prompt_parts)
    
    if name:
        full_prompt = f"{name} themed, " + full_prompt
    
    return full_prompt

def create_portrait_prompt_from_seed(encoded_seed, name=""):
    """
    Create a portrait prompt from encoded metadata
    """
    metadata = decode_banknote_metadata(encoded_seed)
    
    if not metadata:
        # Fallback to default prompt
        default_prompt = read_prompt_file(
            "portrait_prompt.txt",
            "portrait of {name}, elegant character, official portrait, banknote portrait"
        )
        return default_prompt.format(name=name) if "{name}" in default_prompt else default_prompt
    
    # Build sophisticated portrait prompt
    prompt_parts = []
    
    # Base portrait description
    prompt_parts.append("official banknote portrait")
    
    # Add character traits based on metadata
    prompt_parts.append("elegant, dignified character")
    
    # Add elements based on title and theme
    if "灵" in metadata['title']:
        prompt_parts.append("spiritual leader, wise appearance")
    if "国" in metadata['title']:
        prompt_parts.append("national figure, authoritative presence")
    
    # Add style elements based on denomination
    denom = str(metadata['denomination'])
    if denom in ["100", "500", "1000"]:
        prompt_parts.append("regal attire, formal clothing")
    if denom in ["500", "1000", "5000"]:
        prompt_parts.append("imperial accessories, prestigious appearance")
    
    # Add technical specifications
    prompt_parts.append("photorealistic, high detail, serious expression")
    prompt_parts.append("masterpiece, professional currency art")
    
    # Combine all parts
    full_prompt = ", ".join(prompt_parts)
    
    if name:
        full_prompt = f"portrait of {name}, " + full_prompt
    else:
        full_prompt = "portrait of important figure, " + full_prompt
    
    return full_prompt

# Updated background generation function

# Updated portrait generation function
def generate_character_portrait_from_metadata(encoded_seed, name="", width: int = 512, height: int = 512, 
                                            save_path: str = "./portraits"):
    """
    Generate portrait using metadata-encoded prompt
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Create prompt from metadata
    prompt = create_portrait_prompt_from_seed(encoded_seed, name)
    negative_prompt = read_prompt_file("negative_prompt.txt", "ugly, deformed, blurry, low quality")
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": random.randint(0, 2**32 - 1),
        "steps": 30,
        "cfg_scale": 8,
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,
        "n_iter": 1,
        "restore_faces": True,
        "tiling": False,
        "enable_hr": True,
        "hr_scale": 1.5,
        "hr_upscaler": "ESRGAN_4x",
    }
    
    try:
        response = requests.post("http://127.0.0.1:3014/sdapi/v1/txt2img", json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        images = result.get('images', [])
        
        if images:
            image_data = base64.b64decode(images[0])
            image = Image.open(BytesIO(image_data))
            
            # Generate filename with metadata hash
            metadata_hash = hashlib.md5(encoded_seed.encode()).hexdigest()[:8]
            clean_name = re.sub(r'[^\w\-_]', '_', name) if name else "unknown"
            filename = f"portrait_{clean_name}_{metadata_hash}.png"
            filepath = os.path.join(save_path, filename)
            
            image.save(filepath)
            print(f"[+] Generated metadata-based portrait: {filepath}")
            print(f"[+] Prompt: {prompt}")
            return filepath
        
    except Exception as e:
        print(f"[!] Error generating portrait: {e}")
        return None
def add_subtle_frame_and_microgrid(dwg, W: int, H: int, border_info: dict, denomination: int, timestamp_ms: int, seed_hash: bytes):
    """
    Adds multi-band frame and microgrid INSIDE the QR border area.
    Uses deterministic patterns based on denomination, timestamp, and seed.
    """
    # Get the inner diamond area from border_info
    diamond_start_x = border_info['diamond_start_x'] + 0.25
    diamond_start_y = border_info['diamond_start_y'] + 0.25
    diamond_width = border_info['diamond_width'] - 0.25
    diamond_height = border_info['diamond_height'] - 0.25
    
    # Use deterministic values instead of random
    denom_seed = denomination % 100
    time_seed = timestamp_ms % 10000
    hash_seed = sum(seed_hash) % 256 if seed_hash else 0
    
    # Calculate padding relative to the diamond area
    pad = int(min(diamond_width, diamond_height) * 0.03)
    
    # --- Multi-band frame system (deterministic based on inputs) ---
    frame_layers = []
    for i in range(6):
        # Deterministic layer properties based on inputs
        stroke_hue = (denom_seed * 37 + time_seed * 13 + hash_seed * 7 + i * 59) % 360
        
        # Use fractional cm measurements: 1/2 cm, 1/4 cm, 1/8 cm, etc.
        width_base_cm = [0.25, 0.125, 0.125/2, 0.125/2, 0.25/2, 0.5/2][i]  # 1/2, 1/4, 1/8, 1/8, 1/4, 1/2 cm
        width_base = cm_to_px(width_base_cm)
        
        dash_patterns = [None, [6, 6], [1, 4], None, [1, 4], [12, 8]]
        
        frame_layers.append({
            "stroke": hsl_to_rgb_string(stroke_hue, 100, 50),
            "width": width_base * (0.6 + (denom_seed % 3) * 0.3),
            "dash": dash_patterns[i],
            "opacity": 0.5 + (time_seed % 100) * 0.00000005
        })

    for i, style in enumerate(frame_layers):
        inset = pad + i * 3
        rect_params = dict(
            insert=(diamond_start_x + inset, diamond_start_y + inset),
            size=(diamond_width - 2 * inset, diamond_height - 2 * inset),
            fill="none",
            stroke=style["stroke"],
            stroke_width=style["width"],
            stroke_linejoin="miter",   # square corners
            stroke_miterlimit=4,       # controls sharpness, 4 is safe
            opacity=style["opacity"]
        )
        if style["dash"]:
            rect_params["stroke_dasharray"] = style["dash"]
        dwg.add(dwg.rect(**rect_params))


    # --- Deterministic microdots based on inputs ---
    base_cell = 3
    cols = math.ceil(diamond_width / base_cell)
    rows = math.ceil(diamond_height / base_cell)

    g = dwg.g(opacity=0.25)
    for r in range(rows):
        for c in range(cols):
            # Deterministic decision to draw dot
            dot_value = (denom_seed * r * 17 + time_seed * c * 23 + hash_seed * 29) % 100
            if dot_value < 40:  # 40% density
                x = diamond_start_x + c * base_cell
                y = diamond_start_y + r * base_cell
                
                # Deterministic color from inputs
                color_hue = (denom_seed * c * 41 + time_seed * r * 31 + hash_seed * 19) % 360
                color = hsl_to_rgb_string(color_hue, 85, 55)

                
                # Deterministic size and position
                size_seed = (denom_seed * r * 7 + time_seed * c * 11) % 100
                radius = 0.5 + (size_seed / 100) * 1.0
                
                pos_seed = (denom_seed * c * 13 + time_seed * r * 17) % 100
                jitter_x = ((pos_seed / 100) * 2.4) - 1.2
                jitter_y = (((pos_seed * 7) % 100 / 100) * 2.4) - 1.2
                
                opacity_seed = (denom_seed * r * 3 + time_seed * c * 5) % 100
                opacity = 0.1 + (opacity_seed / 100) * 0.7
                
                g.add(dwg.circle(
                    center=(x + base_cell/2 + jitter_x, y + base_cell/2 + jitter_y),
                    r=radius,
                    fill=color,
                    opacity=opacity
                ))
    dwg.add(g)

    # --- Mirror layer (deterministic opacity) ---
    mirror_opacity = 0.04 + ((denom_seed + time_seed) % 100) * 0.0004
    mirror = dwg.g(
        transform=f"translate({diamond_start_x + diamond_width/2},0) scale(-1,1) translate({-diamond_start_x - diamond_width/2},0)", 
        opacity=mirror_opacity
    )
    for elem in g.elements:
        mirror.add(elem.copy())
    dwg.add(mirror)

def add_circular_qr_continuous(dwg, cx, cy, text, inner_radius=0, outer_radius=256,
                               segments=360, colors=None, opacity=0.75):
    if colors is None:
        colors = ["#D80027", "#0052B4", "#009E60"]

    import qrcode
    qr = qrcode.QRCode(
        version=4,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=1,
        border=0
    )
    qr.add_data(str(text))
    qr.make(fit=True)
    qr_matrix = qr.get_matrix()
    qr_size = len(qr_matrix)

    for i in range(segments):
        theta_start = 2 * math.pi * i / segments
        theta_end = 2 * math.pi * (i + 1) / segments

        for j in range(inner_radius, outer_radius):
            qr_x = int(i / segments * qr_size) % qr_size
            qr_y = int((j - inner_radius) / (outer_radius - inner_radius) * qr_size) % qr_size

            if qr_matrix[qr_y][qr_x]:
                x1 = cx + j * math.cos(theta_start)
                y1 = cy + j * math.sin(theta_start)
                x2 = cx + j * math.cos(theta_end)
                y2 = cy + j * math.sin(theta_end)
                color = colors[(i+j) % len(colors)]
                dwg.add(dwg.line(start=(x1,y1), end=(x2,y2), stroke=color, stroke_width=1.2, opacity=opacity))

def generate_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M")

import hashlib
import secrets
from datetime import datetime
import base64

def generate_serial_id_combined():
    """
    Generate a unique, compact serial ID combining:
    - Timestamp (milliseconds)
    - Random cryptographic salt
    - SHA3-256 hash (more secure than SHA256)
    - Base62 encoding (more compact than base36)
    """
    # 1. Get precise timestamp with microseconds
    ts = int(datetime.now().timestamp() * 1000000)  # microseconds for more precision
    
    # 2. Generate random salt
    salt = secrets.token_bytes(4)  # 4 bytes for better randomness
    
    # 3. Combine and hash with SHA3-256
    raw = f"{ts}-".encode() + salt
    h = hashlib.sha3_256(raw).digest()
    
    # 4. Use base64 URL-safe encoding (more compact than base36)
    serial_b64 = base64.urlsafe_b64encode(h[:12]).decode('ascii')  # First 12 bytes → 16 chars
    
    # Remove padding and take first 12 characters for clean format
    serial_clean = serial_b64.replace('=', '')[:12]
    
    # 5. Format with prefix and groups for readability
    return f"SN-{serial_clean[:4]}-{serial_clean[4:8]}-{serial_clean[8:12]}"

# Alternative version with checksum for validation
def generate_serial_id_with_checksum():
    """
    Generate serial ID with built-in checksum for validation
    """
    ts = int(datetime.now().timestamp() * 1000000)
    salt = secrets.token_bytes(3)
    raw = f"{ts}-".encode() + salt
    h = hashlib.sha3_256(raw).digest()
    
    # Take first 10 bytes for serial
    serial_bytes = h[:10]
    serial_b64 = base64.urlsafe_b64encode(serial_bytes).decode('ascii').replace('=', '')[:14]
    
    # Add checksum (last 2 bytes of hash)
    checksum_bytes = h[-2:]
    checksum_b64 = base64.urlsafe_b64encode(checksum_bytes).decode('ascii').replace('=', '')[:3]
    
    return f"SN-{serial_b64}-{checksum_b64}"

def validate_serial_id(serial_id):
    """
    Validate a serial ID format and checksum (if using checksum version)
    """
    if not serial_id.startswith("SN-"):
        return False
    
    # Basic format validation
    parts = serial_id.split('-')
    if len(parts) != 4:  # SN-XXXX-XXXX-XXXX format
        return False
    
    # Check if all parts are valid base64 URL-safe
    import re
    pattern = r'^[A-Za-z0-9_-]+$'
    for part in parts[1:]:
        if not re.match(pattern, part):
            return False
    
    return True

def add_center_text(dwg, W: int, H: int, title: str, phrase: str, denom_color: str):
    # Define padding in pixels
    TOP_PADDING = int(0.5 * 30 * 4)      # ~0.5 cm
    BOTTOM_PADDING = int(0.5 * 30 * 4)   # ~0.5 cm

    # Stroke thickness in pixels (0.05 cm at 300 DPI)
    STROKE_WIDTH = 0.05 * 300 / 2.54

    # Helper to add text with outline
    def add_text_with_outline(x, y, text, font_size, fill_color, stroke_color, baseline, denom_color):
        # Stroke first
        dwg.add(dwg.text(
            text,
            insert=(x, y),
            font_size=font_size,
            font_family="FengGuangMingRui",
            fill=fill_color,
            stroke="white",
            stroke_width=STROKE_WIDTH,
            text_anchor="middle",
            alignment_baseline=baseline,
            opacity=0.5
        ))
        # Fill on top
        dwg.add(dwg.text(
            text,
            insert=(x, y),
            font_size=font_size,
            font_family="FengGuangMingRui",
            fill=fill_color,
            stroke=stroke_color,
            text_anchor="middle",
            alignment_baseline=baseline,
            opacity=1
        ))

    # Title near the top
    add_text_with_outline(x=(W/2), y=TOP_PADDING, text=title, font_size=int(H*0.12), fill_color="black", stroke_color=denom_color, baseline="hanging", denom_color=denom_color)

    # Phrase near the bottom
    add_text_with_outline(x=(W/2), y=(H - BOTTOM_PADDING), text=phrase, font_size=int(H*0.08), fill_color="black", stroke_color=denom_color, baseline="baseline", denom_color=denom_color)


def add_security_background(
    dwg: svgwrite.Drawing,
    W: int,
    H: int,
    denomination: int,
    seed: bytes = None,
    serial_id: str = None,
    margin: int = 60,
    base_triangle_size: int = 16,
    hierarchy_levels: int = 2
):
    seed_hash = hashlib.sha3_512(serial_id.encode("utf-8")).digest() if serial_id else seed
    seed_len = len(seed_hash)
    seed_i = 0

    def byte_to_saturation(byte, min_sat=0.25, max_sat=0.75):
        return min_sat + (byte / 255.0) * (max_sat - min_sat)

    def rgb_to_hex(rgb):
        return "#{:02X}{:02X}{:02X}".format(*rgb)

    if denomination > 0:
        hierarchy_levels = max(1, int(math.log10(denomination)))
    else:
        hierarchy_levels = 1

    def draw_triangle_svg(x0, y0, size, level=1):
        nonlocal seed_i
        scale_byte = seed_hash[seed_i % seed_len]
        size *= 0.5 + (scale_byte / 255.0)
        seed_i += 1

        if level > hierarchy_levels:
            tri_up = [(x0, y0), (x0 + size, y0), (x0 + size/2, y0 + math.sqrt(3)/2*size)]
            tri_down = [(x0, y0), (x0 + size/2, y0 - math.sqrt(3)/2*size), (x0 + size, y0)]
            for tri in [tri_up, tri_down]:
                sat_byte = seed_hash[seed_i % seed_len]
                saturation = byte_to_saturation(sat_byte)
                seed_i += 1
                hue = (saturation * 360 + denomination) % 360
                r, g, b = colorsys.hsv_to_rgb(hue/360, 0.2, 1.0)
                hex_color = rgb_to_hex((int(r*255), int(g*255), int(b*255)))

                op_byte = seed_hash[seed_i % seed_len]
                opacity = 0.1 + (op_byte / 255.0) * 0.25
                seed_i += 1

                dwg.add(dwg.polygon(points=tri, fill=hex_color, fill_opacity=opacity))
        else:
            step = size / 3
            for dy in range(3):
                for dx in range(3):
                    draw_triangle_svg(x0 + dx*step, y0 + dy*step, step, level+1)

    h = math.sqrt(3)/2 * base_triangle_size
    for y in range(0, int(H + h), int(base_triangle_size)):
        offset = 0 if (y // h) % 2 == 0 else base_triangle_size // 2
        for x in range(-base_triangle_size, int(W + base_triangle_size), base_triangle_size):
            draw_triangle_svg(x + offset, y, base_triangle_size)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_str = f"Denom Exp: {denomination}   Date: {date_str}"
    dwg.add(dwg.text(
        text_str,
        insert=(margin, H - margin),
        font_size=int(H*0.035),
        fill="#000000",
        fill_opacity=0.05,
        font_family="Daemon Full Working"
    ))

    print(f"[+] Added security background with triangles, denomination exponent, and date.")
    return seed_hash

def add_roygbiv_qr_style(dwg: svgwrite.Drawing, W: int, H: int, url: str = "https://linglin.art",
                         stamp_width: int = 40, stamp_height: int = 40, rows: int = 3, side: str = "both"):
    colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]
    n_colors = len(colors)
    
    hash_bytes = hashlib.sha3_512(url.encode("utf-8")).digest()
    
    cols = stamp_width // (stamp_width // rows)
    bar_w = stamp_width / cols
    bar_h = stamp_height / rows

    bar_colors = [colors[b % n_colors] for b in hash_bytes]

    def draw_stamp(x_offset: int):
        idx = 0
        for row in range(rows):
            for col in range(cols):
                color = bar_colors[idx % len(bar_colors)]
                dwg.add(dwg.rect(
                    insert=(x_offset + col*bar_w, (H - stamp_height)/2 + row*bar_h),
                    size=(bar_w, bar_h),
                    fill=color
                ))
                idx += 1

    if side in ("left", "both"):
        draw_stamp(0)
    if side in ("right", "both"):
        draw_stamp(W - stamp_width)

    print(f"[+] Added ROYGBIV QR-style stamps ({rows} rows) pointing to {url}")

def generate_sd_background(seed_text: str, width: int, height: int, 
                          save_path: str = "./backgrounds"):
    """
    Generate a background image using Stable Diffusion API with prompts from file
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Read prompts from files
    def read_prompt_file(filename, default_prompt=""):
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            return default_prompt
        except:
            return default_prompt
    
    background_prompt = read_prompt_file(
        "background_prompt.txt",
        "abstract ornamental pattern, intricate design, currency background, banknote pattern, {name} theme, gold filigree, subtle textures, elegant financial design, subtle colors, professional banknote background"
    )
    negative_prompt = read_prompt_file(
        "negative_prompt.txt",
        "text, words, letters, numbers, people, faces, animals, blurry, low quality, watermark, signature"
    )
    
    # Format the prompt with the seed_text
    formatted_prompt = background_prompt.format(name=seed_text)
    
    payload = {
        "prompt": formatted_prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": random.randint(0, 2**32 - 1),
        "steps": 20,
        "cfg_scale": 7,
        "sampler_name": "Euler a",
        "batch_size": 1,
        "n_iter": 1,
        "restore_faces": False,
        "tiling": True,
        "enable_hr": False,
    }
    
    try:
        response = requests.post("http://127.0.0.1:3014/sdapi/v1/txt2img", json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        images = result.get('images', [])
        
        if images:
            image_data = base64.b64decode(images[0])
            image = Image.open(BytesIO(image_data))
            
            # Generate filename
            clean_name = re.sub(r'[^\w\-_]', '_', seed_text)
            timestamp = int(time.time())
            filename = f"bg_{clean_name}_{timestamp}.png"
            filepath = os.path.join(save_path, filename)
            
            image.save(filepath)
            print(f"[+] Generated background: {filepath}")
            return filepath
        
    except Exception as e:
        print(f"[!] Error generating background: {e}")
        return None
from PIL import Image

    

def denom_to_int(denom_str: str) -> int:
    match = re.search(r'\d+', denom_str)
    if match:
        return int(match.group())
    raise ValueError(f"No numeric part found in denomination '{denom_str}'")

def make_qr_seed(denom: str, serial_id: str, timestamp: str = "") -> bytes:
    combined = f"{denom}|{serial_id}|{timestamp}"
    seed = hashlib.sha256(combined.encode("utf-8")).digest()
    return seed

def to_bytes(data, encoding='utf-8'):
    if isinstance(data, bytes):
        return data
    elif isinstance(data, str):
        return data.encode(encoding)
    elif isinstance(data, int):
        length = (data.bit_length() + 7) // 8 or 1
        return data.to_bytes(length, byteorder='big', signed=True)
    elif isinstance(data, float):
        import struct
        return struct.pack('>d', data)
    else:
        raise TypeError(f"Cannot convert type {type(data)} to bytes")

def add_qr_like_border(dwg: svgwrite.Drawing, seed: str, width: int, height: int, serial_id=None, timestamp_ms=None):
    inset_px = mm_to_px(0.5)
    border_thickness_px = mm_to_px(3)
    
    qr_border_start_x = inset_px
    qr_border_start_y = inset_px
    qr_border_end_x = width - inset_px
    qr_border_end_y = height - inset_px
    
    qr_border_inner_start_x = inset_px + border_thickness_px
    qr_border_inner_start_y = inset_px + border_thickness_px
    qr_border_inner_end_x = width - inset_px - border_thickness_px
    qr_border_inner_end_y = height - inset_px - border_thickness_px
    
    cell = max(2, border_thickness_px // 8)
    
    qr_border_width = qr_border_end_x - qr_border_start_x
    qr_border_height = qr_border_end_y - qr_border_start_y
    cols = int(math.ceil(qr_border_width / cell))
    rows = int(math.ceil(qr_border_height / cell))
    
    seed_bytes = to_bytes(make_qr_seed(seed, serial_id, str(timestamp_ms) if timestamp_ms else None))
    
    for r in range(rows):
        for c in range(cols):
            x = qr_border_start_x + c * cell
            y = qr_border_start_y + r * cell
            
            if (qr_border_inner_start_x <= x < qr_border_inner_end_x and
                qr_border_inner_start_y <= y < qr_border_inner_end_y):
                continue
            
            idx = (r * cols + c) % len(seed_bytes)
            v = seed_bytes[idx]

            red = (v * 3) % 256
            green = (v * 7 + r * 5) % 256
            blue = (v * 13 + c * 11) % 256
            color = f"rgb({red},{green},{blue})"

            s = 1 if (v % 3 == 0) else (0.6 if (v % 3 == 1) else 0.35)
            w = max(1, int(cell * s))
            h = max(1, int(cell * s))

            dwg.add(dwg.rect(
                insert=(x + (cell - w) / 2, y + (cell - h) / 2),
                size=(w, h),
                fill=color,
                opacity=1
            ))
    
    return {
        'diamond_start_x': qr_border_inner_start_x,
        'diamond_start_y': qr_border_inner_start_y,
        'diamond_width': qr_border_inner_end_x - qr_border_inner_start_x,
        'diamond_height': qr_border_inner_end_y - qr_border_inner_start_y,
        'image_start_x': qr_border_inner_start_x + border_thickness_px,
        'image_start_y': qr_border_inner_start_y + border_thickness_px,
        'image_width': qr_border_inner_end_x - qr_border_inner_start_x - 2 * border_thickness_px,
        'image_height': qr_border_inner_end_y - qr_border_inner_start_y - 2 * border_thickness_px
    }

def generate_timestamp_ms():
    return int(time.time() * 1000)

def generate_timestamp_ms_precise():
    now = datetime.now()
    return int(now.timestamp() * 1000) + now.microsecond // 1000

def generate_timestamp_ms_formatted():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"

def sha3_512_salted(s: str, salt: str = None) -> bytes:
    hash_obj = hashlib.sha3_512()
    if salt is not None:
        hash_obj.update(str(salt).encode("utf-8"))
    hash_obj.update(str(s).encode("utf-8"))
    return hash_obj.digest()

def generate_security_pattern(bg_input, output_path=None, seed_data=None, font_path=None, pattern_density=0.1):
    """
    FIXED: Handle both file paths and actual image data
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Load background - handle both file paths and image data
    if isinstance(bg_input, (bytes, bytearray)):
        print(f"Opening background from bytes: {len(bg_input)} bytes")
        bg = Image.open(BytesIO(bg_input)).convert("RGBA")
    elif isinstance(bg_input, str) and os.path.exists(bg_input):
        print(f"Opening background from file: {bg_input}")
        bg = Image.open(bg_input).convert("RGBA")
    else:
        # Create a default background if input is invalid
        print("Creating default background")
        bg = Image.new("RGBA", (800, 600), (255, 255, 255, 255))
    
    width, height = bg.size

    # Generate seed from data using SHA3-512
    if seed_data is None:
        seed_data = datetime.now().isoformat()
    seed_hash = sha3_512_salted(str(seed_data))
    seed_int = int.from_bytes(seed_hash[:8], "big")

    # Deterministic RNG
    class DeterministicRandom:
        def __init__(self, seed):
            self.state = seed
        def random(self):
            self.state = (self.state * 1103515245 + 12345) & 0x7fffffff
            return self.state / 0x7fffffff
        def randint(self, a, b):
            return a + int(self.random() * (b - a + 1))

    det_random = DeterministicRandom(seed_int)

    # Create overlay
    overlay = Image.new("RGBA", bg.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # Microtext option
    font, text = None, None
    if font_path and os.path.exists(font_path):
        try:
            font_size = max(6, int(min(width, height) * 0.02))
            font = ImageFont.truetype(font_path, font_size)
            text = str(seed_data)
        except Exception as e:
            print(f"[!] Font load failed: {e}")

    # Draw pattern
    for x in range(0, width, 5):
        for y in range(0, height, 5):
            if det_random.random() < pattern_density:
                color_seed = (x * y + seed_int) % 0xffffff
                r, g, b = (color_seed >> 16) & 0xff, (color_seed >> 8) & 0xff, color_seed & 0xff
                color = (r, g, b, det_random.randint(50, 100))
                if font and text:
                    draw.text((x, y), text, font=font, fill=color)
                else:
                    draw.point((x, y), fill=color)

    # Combine
    result = Image.alpha_composite(bg, overlay)

    # Handle output path
    if output_path is None:
        output_path = "pattern_output.png"
    elif os.path.isdir(output_path):
        base = "security_pattern"
        output_path = os.path.join(output_path, f"{base}_{int(time.time())}.png")

    result.save(output_path)
    print(f"[+] Saved patterned image → {output_path}")
def denomination_color(denom: int) -> str:
    """
    Returns a light ROYGBIV hex color based on the denomination.
    Maps 1 → Red, 100,000,000 → Violet on a log scale.
    """
    # Clamp between 1 and 100,000,000
    denom = max(1, min(100_000_000, denom_to_int(denom)))

    # Normalize exponent (log10 scale)
    exp = math.log10(denom) / math.log10(100_000_000)  # 0.0 → 1.0

    # ROYGBIV palette
    roygbiv = [
        (255, 0, 0),       # Red
        (255, 165, 0),     # Orange
        (255, 255, 0),     # Yellow
        (0, 128, 0),       # Green
        (0, 0, 255),       # Blue
        (75, 0, 130),      # Indigo
        (143, 0, 255)      # Violet
    ]

    # Find segment in ROYGBIV
    idx = int(exp * (len(roygbiv) - 1))
    frac = exp * (len(roygbiv) - 1) - idx

    # Interpolate between two colors
    c1 = roygbiv[idx]
    c2 = roygbiv[min(idx + 1, len(roygbiv) - 1)]
    r = int(c1[0] + (c2[0] - c1[0]) * frac)
    g = int(c1[1] + (c2[1] - c1[1]) * frac)
    b = int(c1[2] + (c2[2] - c1[2]) * frac)

    # Light tint: blend 70% white + 30% color
    r = int(0.7 * 255 + 0.3 * r)
    g = int(0.7 * 255 + 0.3 * g)
    b = int(0.7 * 255 + 0.3 * b)

    return f"#{r:02X}{g:02X}{b:02X}"
def add_rainbow_microseal(
    dwg: svgwrite.Drawing,
    cx: int,
    cy: int,
    radius: int,
    symbol: str = None,
    repetitions: int = 64,
    font_family: str = "Daemon Full Working",
    font_size: int = 8
):
    """
    Add a rainbow-encoded microprint seal around a circle using transparency + ROYGBIV.
    Each character gets color cycling and varying opacity for a holographic/mosaic effect.
    """

    # Default symbol = datetime stamp
    if symbol is None:
        symbol = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # ROYGBIV palette
    colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]

    n = repetitions
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        rotation = math.degrees(angle) + 90

        # Pick color + opacity per index
        color = colors[i % len(colors)]
        opacity = 0.25 + 0.75 * ((i % len(colors)) / (len(colors) - 1))  # fade mosaic style

        dwg.add(dwg.text(
            symbol,
            insert=(x, y),
            font_size=font_size,
            font_family=font_family,
            fill=color,
            opacity=opacity,
            text_anchor="middle",
            alignment_baseline="middle",
            transform=f"rotate({rotation},{x},{y})"
        ))
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
import hashlib
import json
import base64
import zlib
from datetime import datetime

def encode_banknote_metadata(title_text, phrase_text, serial_id, timestamp_ms, denomination):
    """
    Encode banknote metadata into a structured prompt seed
    """
    metadata = {
        "title": title_text,
        "phrase": phrase_text,
        "serial": serial_id,
        "timestamp": timestamp_ms,
        "denomination": denomination,
        # Theme is simplified — always derived from denomination only
        "theme": generate_theme_from_metadata(denomination)
    }
    
    json_str = json.dumps(metadata, ensure_ascii=False)
    compressed = zlib.compress(json_str.encode("utf-8"))
    encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")
    return encoded


def decode_banknote_metadata(encoded_seed):
    """
    Decode the seed back into metadata
    """
    try:
        decoded = base64.urlsafe_b64decode(encoded_seed)
        decompressed = zlib.decompress(decoded)
        metadata = json.loads(decompressed.decode("utf-8"))
        return metadata
    except:
        return None

def generate_theme_from_metadata(denomination):
    """
    Generate a thematic kawaii scene description based on the denomination.
    Focused on cute dolls, pastel scenery, playful animals, and girly East Asian mural motifs.
    """
    denom_themes = {
        "1": "tiny kawaii villages with pastel houses, gentle rivers, smiling animals, soft morning light, cute paper lanterns",
        "10": "charming spring gardens, sakura petals floating, kawaii koi ponds, playful foxes, pastel butterflies, adorable dolls playing",
        "100": "balanced East Asian temple courtyards, bamboo groves, cheerful cranes, lotus flowers, sweet little dolls with ribbons, pastel clouds",
        "500": "regal pagodas, mountain vistas, friendly dragons, glowing lanterns, kawaii girls in traditional dresses, whimsical clouds",
        "1000": "prosperous festival scenes, bustling bridges, flocks of koi and cranes, animated skies, dolls holding tiny parasols, candy-colored banners",
        "10000": "heritage landscapes, ancient castles, serene lakes, floating lanterns, mystical mountains, cute girls with fans and parasols, pastel blossoms",
        "100000": "imperial gardens, majestic temples, celestial animals like foxes and cranes, harmonious kawaii scenes, tiny dolls in elegant attire",
        "1000000": "mystical moonlit mountains, glowing waterfalls, playful mythical creatures, ethereal night skies, dolls twirling under lantern light, sparkling stars",
        "10000000": "celestial floating islands, dragons curling in fluffy clouds, fantastical flora and fauna, magical skies, cute girls dancing and playing with animals",
        "100000000": "grand imperial murals, epic kawaii landscapes, dragons and phoenixes with smiling faces, celestial palaces, adorable dolls in luxurious attire, divine playful ambiance"
    }

    denom_value = str(denomination).split()[0] if isinstance(denomination, str) else str(denomination)
    return denom_themes.get(denom_value, "joyful kawaii scenery, playful animals, pastel flowers, cute dolls, whimsical landscapes")


def create_background_prompt_from_seed(encoded_seed, name=""):
    """
    Create a kawaii mural/scenery prompt from encoded metadata.
    Focuses on East Asian landscapes, animals, and cultural elements,
    with denomination influencing theme and scene richness.
    """
    metadata = decode_banknote_metadata(encoded_seed)
    
    # Base scene prompt
    prompt_parts = [
        "grand kawaii mural scenery in East Asian tradition",
        "temples, rivers, mountains, cherry blossoms, bamboo forests, and lanterns",
        "playful animals like cranes, koi, foxes, and dragons",
        "whimsical, painterly, Studio Ghibli-inspired, highly detailed, joyful atmosphere"
    ]
    
    if metadata:
        # Apply denomination-based scene theme
        theme = generate_theme_from_metadata(metadata['denomination'])
        prompt_parts.append(theme)
    
    # Include name if provided
    if name:
        prompt_parts.insert(0, f"{name}-themed kawaii mural")
    
    return ", ".join(prompt_parts)



def create_portrait_prompt_from_seed(encoded_seed, name=""):
    """
    Create a portrait within mural scenery,
    treating the figure as part of an East Asian narrative scene.
    """
    metadata = decode_banknote_metadata(encoded_seed)
    
    # Base portrait-in-scenery prompt
    prompt_parts = [
        "heroic mural figure within grand East Asian scenery",
        "integrated into temples, rivers, mountains, or celestial skies",
        "ornate hanfu, kimono, or hanbok attire, dignified presence",
        "Studio Ghibli style, painterly, vibrant and harmonious",
        "composition emphasizes scenery and human clarity equally"
    ]
    
    if metadata:
        denom = str(metadata['denomination'])
        denom_mods = {
            "1": "pastel dawn colors, serenity",
            "10": "spring blossoms and lively palette",
            "100": "jade and gold harmony",
            "1000": "rich autumn atmosphere",
            "10000": "deep crimson heritage mood",
            "100000": "regal gold accents",
            "1000000": "cosmic indigo glow",
            "10000000": "celestial white-blue ethereal light",
            "100000000": "imperial grandeur, golden-red splendor"
        }
        if denom in denom_mods:
            prompt_parts.append(denom_mods[denom])
    
    if name:
        prompt_parts.insert(0, f"portrait of {name}")
    else:
        prompt_parts.insert(0, "portrait of legendary East Asian figure")
    
    return ", ".join(prompt_parts)



import os


def read_prompt_file(filepath: str, default: str = "") -> str:
    """
    Read a prompt file and return its contents as a single string.
    If the file doesn't exist, return the provided default string.
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        return default.strip()

def generate_kawaii_mural_from_background(denomination, filename="background_prompt.txt"):
    """
    Generate a kawaii East Asian mural/scenery prompt based on a base background prompt,
    and append a color palette derived from the denomination.
    """
    # Read base prompt from file
    base_prompt = read_prompt_file(filename)
    
    # Generate denomination-based color palette
    palette = denomination_to_color(denomination)  # e.g., "pastel pinks and blues"
    
    # Combine into final prompt
    prompt = (
        f"{base_prompt}, kawaii hand-drawn oekaki style, playful animals and dolls, "
        f"Studio Ghibli-inspired, whimsical, painterly, soft textures, "
        f"use a palette dominated by {palette} in the style of Chinese DMT Studio Ghibli"
    )
    
    return prompt
# Updated background generation function

    


def add_holographic_seals(dwg, W:int, H:int, serial_id:str, denomination:int, radius:int=64):
    """
    Left: Organic hexagon filled with overlapping circle outlines (blue), vertically centered.  
    Right: Dense nested mandala (red), gradient & layers encode data.
    """

    def data_hash(serial_id, denomination):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        data = f"{serial_id}-{denomination}-{now}"
        return hashlib.sha256(data.encode()).hexdigest()

    def create_gradient(dwg, grad_id, colors):
        grad = dwg.defs.add(dwg.linearGradient(id=grad_id, x1="0%", y1="0%", x2="100%", y2="100%"))
        n = len(colors)
        for i,c in enumerate(colors):
            grad.add_stop_color(offset=i/(n-1), color=c)
        return f"url(#{grad_id})"

    # ---- Left Organic Hex Pattern ----
    # ---- Left Organic Circular Macro Pattern ----
    # ---- Left Symmetrical Hex Star Pattern ----
# ---- Left Symmetrical Hex Star Pattern (Centered) ----
    def draw_blue_hexagon(group, cx, cy, size, data):
        grad_fill = create_gradient(dwg, "blue_grad", ["#7B00FF", "#002AFF", "#00A6FF"])
        thickness = 1.8
        hex_r = size / 6  # smaller so the star fits

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        idx = 0

        # Directions in a hex grid (axial coords)
        directions = [
            (1, 0), (0, 1), (-1, 1),
            (-1, 0), (0, -1), (1, -1)
        ]

        # Function to convert axial coords to pixel coords
        def hex_to_pixel(q, r, scale):
            x = scale * (3/2 * q)
            y = scale * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
            return x, y

        # Generate concentric rings (center + 2 rings = star-like)
        max_ring = 2
        coords = [(0, 0)]  # center hex

        for ring in range(1, max_ring + 1):
            q, r = ring, 0
            for d in range(6):  # 6 directions
                dq, dr = directions[d]
                for _ in range(ring):
                    coords.append((q, r))
                    q -= dq
                    r -= dr

        # Place circles at each hex coordinate
        for q, r in coords:
            px, py = hex_to_pixel(q, r, hex_r * 1.2)
            hex_cx = cx + px
            hex_cy = cy + py

            # Update bounding box
            min_x, max_x = min(min_x, hex_cx), max(max_x, hex_cx)
            min_y, max_y = min(min_y, hex_cy), max(max_y, hex_cy)

            # Data-driven circle count
            count = int(data[idx:idx+2], 16) % 4 + 2
            idx = (idx + 2) % len(data)

            for k in range(count):
                r_small = hex_r * 0.3 * (k + 1)
                circle_cy_top = hex_cy - r_small
                circle_cy_bottom = hex_cy + r_small
                min_y = min(min_y, circle_cy_top)
                max_y = max(max_y, circle_cy_bottom)
                group.add(dwg.circle(center=(hex_cx, hex_cy), r=r_small,
                                    fill="none", stroke=grad_fill,
                                    stroke_width=thickness, opacity=1))

        # Compute bounding box center
        emblem_width = max_x - min_x
        emblem_height = max_y - min_y
        emblem_cx = min_x + emblem_width / 2
        emblem_cy = min_y + emblem_height / 2

        # Shift so emblem center = (cx, cy)
        dx = cx - emblem_cx
        dy = cy - emblem_cy
        group.translate(dx, dy)




    # ---- Right Dense Mandala ----
    def draw_red_mandala(group, cx, cy, size, data):
        layers = [
            ("circle", size),
            ("square", size*0.7),
            ("diamond", size*0.5),
            ("circle", size*0.3)
        ]
        num_radial = 8
        thickness = 1
        grad_colors = ["#FF0044","#FF0000","#FF5757","#FF9100"]
        grad_fill = create_gradient(dwg, "red_grad", grad_colors)

        for l_idx, (shape, r) in enumerate(layers):
            val = int(data[(l_idx*4)%len(data):(l_idx*4+4)%len(data)],16)/0xFFFF
            for k in range(num_radial):
                angle = (2*math.pi/num_radial)*k + val*math.pi
                ox = cx + math.cos(angle)*r*0.2
                oy = cy + math.sin(angle)*r*0.2
                if shape == "circle":
                    group.add(dwg.circle(center=(ox,oy), r=r*val+0.5, stroke=grad_fill,
                                         fill="none", stroke_width=thickness, opacity=1))
                elif shape == "square":
                    half = r*val
                    pts = [(ox-half, oy-half),(ox+half, oy-half),(ox+half, oy+half),(ox-half, oy+half)]
                    group.add(dwg.polygon(pts, stroke=grad_fill, fill="none", stroke_width=thickness, opacity=1))
                elif shape == "diamond":
                    half = r*val
                    pts = [(ox,oy-half),(ox+half,oy),(ox,oy+half),(ox-half,oy)]
                    group.add(dwg.polygon(pts, stroke=grad_fill, fill="none", stroke_width=thickness, opacity=1))

        for k in range(num_radial):
            angle = k*2*math.pi/num_radial
            x1 = cx + layers[1][1]*math.cos(angle)
            y1 = cy + layers[1][1]*math.sin(angle)
            x2 = cx + layers[2][1]*math.cos(angle)
            y2 = cy + layers[2][1]*math.sin(angle)
            group.add(dwg.line((x1,y1),(x2,y2), stroke=grad_fill, stroke_width=thickness, opacity=1))

    # ---- Main ----
    data = data_hash(serial_id, denomination)

    # Left blue organic, vertically centered
    lx = int(W*0.18)
    g_left = dwg.g()
    draw_blue_hexagon(g_left, lx, H/2, radius, data)
    dwg.add(g_left)

    # Right red dense mandala
    rx, ry = int(W*0.82), int(H*0.5)
    g_right = dwg.g()
    draw_red_mandala(g_right, rx, ry, radius, data[::-1])
    dwg.add(g_right)

    # Add mirrored text
    dwg.add(dwg.text("天圆", insert=(lx, H/2+4),
                     font_size=int(radius*0.2), text_anchor="middle",
                     font_family="FengGuangMingRui",
                     fill="#0095C7", stroke="#FFF", stroke_width=1.0))

    g_right_text = dwg.g(transform=f"rotate(180 {rx} {ry})")
    g_right_text.add(dwg.text("地方", insert=(rx, ry+4),
                              font_size=int(radius*0.2), text_anchor="middle",
                              font_family="FengGuangMingRui",
                              fill="#FF0033", stroke="#000", stroke_width=1.0))
    dwg.add(g_right_text)


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
                        fill="#000",
                        opacity=1,
                        text_anchor="middle",
                        alignment_baseline="middle",
                        transform=f"rotate({rotation},{x},{y})"))
def generate_backside_svg(outfile: str, denomination: int, title_text: str, phrase_text: str, size_px: Tuple[int,int], 
                         serial_id: str = None, timestamp_ms: str = None, seed_text: str = ""):
    W, H = size_px
    denom_exp = int(math.log10(denom_to_int(denomination))) if denom_to_int(denomination) > 0 else 0
    timestamp = timestamp_ms or generate_timestamp_ms_precise()
    serial_id = serial_id or generate_serial_id_combined()
    denom_exponent = int(round(math.log10(denom_to_int(denomination)))) if denom_to_int(denomination) > 0 else 0
    denom_value = denomination  # Already an int
    
    # Encode all metadata into a seed for consistent theming
    encoded_seed = encode_banknote_metadata(
        title_text=title_text,
        phrase_text=phrase_text,
        serial_id=serial_id,
        timestamp_ms=timestamp,
        denomination=denom_value
    )
    
    print(f"[+] Backside metadata seed: {encoded_seed[:30]}...")
    
    dwg = svgwrite.Drawing(outfile, size=(W,H), viewBox=f"0 0 {W} {H}")
    embed_font(dwg, CHINESE_FONT, "FengGuangMingRui")
    embed_font(dwg, NUMBER_FONT, "Daemon Full Working")
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill=denomination_color(denom=denom_value)))


    # Replace the entire block with this single function call:
    add_vectorized_background(
        dwg,
        W=W,
        H=H,
        seed_text=seed_text,
        bg_dir="./backgrounds",
        n_segments=1024,
        denomination=denomination
    )
    cx, cy = W//2, H//2
    
    # Add functional elements
    add_functional_corner_decorations(dwg, W, H, denomination, timestamp, serial_id)
    
    # Create circular QR with metadata-consistent colors
    qr_colors = generate_theme_colors_from_seed(encoded_seed)
    add_circular_qr_continuous(
        dwg,
        cx, cy,
        text=str(denom_exp),  # Use denomination exponent as QR data
        inner_radius=int(min(W,H)*0.0),
        outer_radius=int(min(W,H)*0.360),
        segments=4,
        colors=qr_colors,
        opacity=0.5
    )

    # Add border with metadata
    border_info = add_qr_like_border(dwg, str(denomination), W, H, serial_id=serial_id, timestamp_ms=timestamp)
    denom_color=denomination_to_color(denom_exponent)
    # Add design elements
    add_holographic_seals(
        dwg,
        W, H,
        serial_id=serial_id,
        denomination=denom_exponent,
        radius=int(min(W,H)*0.25)
    )
    #add_hightech_hologram_seals(dwg, W, H, radius=int(min(W,H)*0.12), stroke_w=1)
    add_subtle_frame_and_microgrid(dwg, W, H, border_info, denom_to_int(denomination), timestamp, to_bytes(serial_id))
    add_decorative_border(dwg, W, H, border_info, denom_value, timestamp)
    add_center_text(dwg, W, H, title_text, phrase_text, denom_color)
    add_corner_denoms(dwg, W, H, str(denomination))
    
    qr_url=f"https://bank.linglin.art/verify/{serial_id}"
    # Add ROYGBIV QR style with metadata-based theming
    add_roygbiv_qr_style(dwg, W=W, H=H, url=qr_url, stamp_width=60, stamp_height=60, rows=6)
    matrix = segno.make(qr_url).matrix  # matrix is a list of lists of booleans

    add_colored_aztec_to_canvas(
        dwg,
        cx=cx-360,
        cy=cy-0,
        matrix=matrix,
        scale=3,  # distance left/right from center
        denom_exponent=denom_exponent,
        rotation=0,
        border_opacity=0.5
    )
    add_colored_aztec_to_canvas(
        dwg,
        cx=cx+360,
        cy=cy-0,
        matrix=matrix,
        scale=3,  # distance left/right from center
        denom_exponent=denom_exponent,
        rotation=180,
        border_opacity=0.5
    )

    
    
    # Add verification text
    add_verification_text(dwg, W, H, serial_id, timestamp)
    # After successfully generating a bill and saving to DB:


    dwg.save()
    print(f"[+] Saved {outfile}")
import qrcode
from PIL import Image, ImageDraw
import math
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
# ROYGBIV palette
COLORS = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]
import segno
from aztec import aztec_matrix_from_segno, build_colored_aztec_svg
import tempfile
import base64
import svgwrite

# Example denomination_to_color function
def denomination_to_color(denom_exponent: int) -> str:
    """Map denomination exponent (0-8) to a color in a 9-color spectrum."""
    spectrum = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF", "#4B0082", "#8F00FF", "#FF00FF"]
    # Clamp to 0-8
    idx = max(0, min(denom_exponent, len(spectrum)-1))
    return spectrum[idx]

def add_colored_aztec_to_canvas(
    dwg, cx, cy, matrix, denom_exponent,
    scale=12, border=12, rotation=0, border_opacity=0.5
):
    """
    Draw an Aztec QR code centered at (cx, cy) with:
    - denomination-colored semi-transparent border,
    - white background,
    - black modules,
    - optional rotation (degrees).
    """
    nrows = len(matrix)
    ncols = len(matrix[0])

    qr_size = ncols * scale  # actual QR size in px (square)
    border_color = denomination_to_color(denom_exponent)

    # Group everything together so it rotates around the QR's center
    qr_group = dwg.g(transform=f"rotate({rotation},{cx},{cy})")

    # 1. Colored border (semi-transparent)
    qr_group.add(dwg.rect(
        insert=(cx - (qr_size/2 + border), cy - (qr_size/2 + border)),
        size=(qr_size + 2*border, qr_size + 2*border),
        fill=border_color,
        opacity=border_opacity
    ))

    # 2. White background
    qr_group.add(dwg.rect(
        insert=(cx - qr_size/2, cy - qr_size/2),
        size=(qr_size, qr_size),
        fill="white",
        opacity=border_opacity
    ))

    # 3. Black modules
    for r in range(nrows):
        for c in range(ncols):
            if matrix[r][c]:  # filled module
                x = cx - qr_size/2 + c * scale
                y = cy - qr_size/2 + r * scale
                qr_group.add(dwg.rect(insert=(x, y), size=(scale, scale), fill="black"))

    dwg.add(qr_group)
    return dwg



def add_colored_aztecs(
    dwg: svgwrite.Drawing,
    W: int,
    H: int,
    cx: int,
    cy: int,
    qr_url: str,
    scale: int = 12,
    margin: int = 6,
    style: str = "radial",
    offset: int = 300,
    size: int = 150
) -> svgwrite.Drawing:
    # Build matrix and temporary SVG for Aztec
    matrix = aztec_matrix_from_segno(qr_url)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
        tmp_svg_path = tmp_file.name

    build_colored_aztec_svg(matrix, scale=scale, margin_modules=margin,
                            style=style, out_path=tmp_svg_path)

    # Load Aztec SVG and encode to base64
    with open(tmp_svg_path, "rb") as f:
        svg_bytes = f.read()
    svg_data_url = "data:image/svg+xml;base64," + base64.b64encode(svg_bytes).decode("ascii")

    # Left Aztec
    dwg.add(dwg.image(
        href=svg_data_url,
        insert=(cx - offset - size/2, cy - size/2),
        size=(size, size)
    ))

    # Right Aztec
    dwg.add(dwg.image(
        href=svg_data_url,
        insert=(cx + offset - size/2, cy - size/2),
        size=(size, size)
    ))

    return dwg

def to_bytes(data, encoding='utf-8'):
    """
    Convert different types of data to bytes.

    Parameters:
        data: str, int, float, or bytes
        encoding: str, encoding to use if data is a string

    Returns:
        bytes representation of the input
    """
    if isinstance(data, bytes):
        return data
    elif isinstance(data, str):
        return data.encode(encoding)
    elif isinstance(data, int):
        # Convert int to bytes (big-endian, minimum number of bytes)
        length = (data.bit_length() + 7) // 8 or 1
        return data.to_bytes(length, byteorder='big', signed=True)
    elif isinstance(data, float):
        import struct
        return struct.pack('>d', data)  # 8-byte double, big-endian
    else:
        raise TypeError(f"Cannot convert type {type(data)} to bytes")
def add_roygbiv_qr_style_aztec(dwg, W, H, url, stamp_width=60, stamp_height=60, rows=6):
    """
    Create an Aztec-style QR code with ROYGBIV colors and add it to the SVG drawing.
    
    Args:
        dwg: The svgwrite Drawing object
        W: Width of the canvas
        H: Height of the canvas
        url: The URL to encode in the QR code
        stamp_width: Width of the QR code
        stamp_height: Height of the QR code
        rows: Number of rows in the Aztec pattern
    """
    if not QR_AVAILABLE:
        print("QR code generation not available. Skipping QR code addition.")
        return
    
    # Create QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=2,
        border=1,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    # Create QR code image (black on white)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img = qr_img.resize((stamp_width, stamp_height), Image.NEAREST)
    
    # Convert QR code to SVG elements
    qr_array = np.array(qr_img.convert('L'))
    qr_array = (qr_array < 128).astype(np.uint8)  # Binary mask
    
    # ROYGBIV colors
    roygbiv_colors = [
        "#FF0000",      # Red
        "#FFA500",      # Orange
        "#FFFF00",      # Yellow
        "#00FF00",      # Green
        "#0000FF",      # Blue
        "#4B0082",      # Indigo
        "#EE82EE"       # Violet
    ]
    
    # Calculate positions (120px from center on both sides)
    center_x = W // 2
    center_y = H // 2
    
    # Left position
    left_x = center_x - 120 - stamp_width
    left_y = center_y - stamp_height // 2
    
    # Right position
    right_x = center_x + 120
    right_y = center_y - stamp_height // 2
    
    # Draw QR codes at both positions
    for pos_x, pos_y in [(left_x, left_y), (right_x, right_y)]:
        # Draw black background
        dwg.add(dwg.rect(
            insert=(pos_x, pos_y),
            size=(stamp_width, stamp_height),
            fill="#000000"
        ))
        
        # Calculate cell size
        cell_height = stamp_height // rows
        cells_per_row = stamp_width // cell_height
        
        # Draw Aztec-style pattern
        for y in range(rows):
            color_idx = y % len(roygbiv_colors)
            color = roygbiv_colors[color_idx]
            
            for x in range(cells_per_row):
                # Calculate position in QR code
                qr_x = int(x * qr_array.shape[1] / cells_per_row)
                qr_y = int(y * qr_array.shape[0] / rows)
                
                # If this part of the QR code is black, draw the pattern element
                if qr_array[qr_y, qr_x] == 1:
                    # Draw a square with a circular cutout (Aztec style)
                    center_x_px = pos_x + x * cell_height + cell_height // 2
                    center_y_px = pos_y + y * cell_height + cell_height // 2
                    radius = cell_height // 2 - 2
                    
                    # Draw the colored square
                    dwg.add(dwg.rect(
                        insert=(pos_x + x * cell_height, pos_y + y * cell_height),
                        size=(cell_height, cell_height),
                        fill=color
                    ))
                    
                    # Draw a circular cutout in the center
                    dwg.add(dwg.circle(
                        center=(center_x_px, center_y_px),
                        r=radius // 2,
                        fill="#000000"
                    ))
def create_aztec_style_qr(qr_img, width, height, rows=6):
    """
    Convert a standard QR code to an Aztec-style pattern with ROYGBIV colors.
    
    Args:
        qr_img: PIL Image of the QR code
        width: Output width
        height: Output height
        rows: Number of rows in the pattern
    
    Returns:
        PIL Image with Aztec-style QR code
    """
    # Convert to numpy array for processing
    qr_array = np.array(qr_img.convert('L'))
    qr_array = (qr_array < 128).astype(np.uint8)  # Binary mask
    
    # Create output image with black background
    output = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(output)
    
    # ROYGBIV colors
    roygbiv_colors = [
        (255, 0, 0),      # Red
        (255, 165, 0),    # Orange
        (255, 255, 0),    # Yellow
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (75, 0, 130),     # Indigo
        (238, 130, 238)   # Violet
    ]
    
    # Calculate cell size based on rows
    cell_height = height // rows
    cells_per_row = width // cell_height
    
    # Draw Aztec-style pattern
    for y in range(rows):
        color_idx = y % len(roygbiv_colors)
        color = roygbiv_colors[color_idx]
        
        for x in range(cells_per_row):
            # Calculate position in QR code
            qr_x = int(x * qr_array.shape[1] / cells_per_row)
            qr_y = int(y * qr_array.shape[0] / rows)
            
            # If this part of the QR code is black, draw the pattern element
            if qr_array[qr_y, qr_x] == 1:
                # Draw a square with a circular cutout (Aztec style)
                center_x = x * cell_height + cell_height // 2
                center_y = y * cell_height + cell_height // 2
                radius = cell_height // 2 - 2
                
                # Draw the colored square
                draw.rectangle(
                    [x * cell_height, y * cell_height, 
                     (x + 1) * cell_height, (y + 1) * cell_height],
                    fill=color
                )
                
                # Draw a circular cutout in the center
                draw.ellipse(
                    [center_x - radius // 2, center_y - radius // 2,
                     center_x + radius // 2, center_y + radius // 2],
                    fill='black'
                )
    
    return output
# Helper function to generate theme colors from seed
def generate_theme_colors_from_seed(encoded_seed):
    """
    Generate consistent colors based on the metadata seed
    """
    metadata = decode_banknote_metadata(encoded_seed)
    if not metadata:
        return [
            "#FF0000", "#FF7F00", "#FFFF00", "#00FF00", 
            "#0000FF", "#4B0082", "#8B00FF"
        ]
    
    # Generate colors based on denomination and theme
    denom = str(metadata['denomination'])
    theme = metadata['theme']
    
    color_schemes = {
        "1": ["#E8E8E8", "#D0D0D0", "#B8B8B8", "#A0A0A0"],  # Monochrome
        "10": ["#FFEBCD", "#DEB887", "#CD853F", "#A0522D"],  # Earth tones
        "100": ["#87CEEB", "#4682B4", "#1E90FF", "#0000CD"],  # Blues
        "1000": ["#98FB98", "#32CD32", "#228B22", "#006400"],  # Greens
        "10000": ["#DAA520", "#B8860B", "#CD853F", "#8B4513"],  # Gold/brown
        "100000": ["#FFD700", "#DAA520", "#B8860B", "#8B4513"],  # Gold
        "1000000": ["#9370DB", "#8A2BE2", "#4B0082", "#483D8B"],  # Purples
        "10000000": ["#FF69B4", "#FF1493", "#C71585", "#8B008B"],  # Pinks
        "100000000": ["#DC143C", "#B22222", "#8B0000", "#800000"]   # Reds
    }
    
    # Default to high denomination colors
    base_colors = color_schemes.get(denom, color_schemes[denom])
    
    # Add accent colors based on theme keywords
    if "spiritual" in theme:
        base_colors.extend(["#FFFFFF", "#E6E6FA"])  # White, lavender
    if "regal" in theme:
        base_colors.extend(["#000080", "#191970"])  # Navy blues
    if "financial" in theme:
        base_colors.extend(["#008000", "#006400"])  # Dark greens
    
    return base_colors[:7]  # Return up to 7 colors

def add_verification_text(dwg, W, H, serial_id, timestamp):
    """Add verification text to the backside with serial above and issued date/time below"""
    # Handle different timestamp formats
    if isinstance(timestamp, int):
        # Check if timestamp is in milliseconds (typical for JavaScript/other systems)
        if timestamp > 1000000000000:  # If timestamp is > year 2001 in milliseconds
            timestamp = timestamp / 1000  # Convert milliseconds to seconds
        try:
            from datetime import datetime
            formatted_timestamp = f"发行: {datetime.fromtimestamp(timestamp).strftime('%m/%d/%Y %H:%M:%S')}"
        except (OSError, ValueError):
            # If timestamp conversion fails, use a fallback format
            formatted_timestamp = f"发行: {timestamp}"
    elif hasattr(timestamp, 'strftime'):
        # It's a datetime object
        formatted_timestamp = f"发行: {timestamp.strftime('%m/%d/%Y %H:%M:%S')}"
    else:
        # Fallback - just convert to string
        formatted_timestamp = f"发行: {str(timestamp)}"
    
    # Calculate center position
    center_x = W / 2
    verification_serial =  "序列号: " + str(serial_id)
    # Add serial number (top line) with Daemon Full Working font
    dwg.add(dwg.text(
        verification_serial,
        insert=(center_x-500, H - 120),  # Position for top line
        font_size=16,
        fill="#00FF5E",
        font_family="FengGuangMingRui",
        text_anchor="middle",
        opacity=1
    ))
    
    # Add issued timestamp (bottom line) with FengGuangMingRui font
    dwg.add(dwg.text(
        formatted_timestamp,
        insert=(center_x+500, H - 120),  # Position for bottom line
        font_size=16,
        fill="#FF0000",
        font_family="FengGuangMingRui",
        text_anchor="middle",
        opacity=1
    ))
import glob
def add_vectorized_background(dwg, W, H, seed_text="", bg_dir="./backgrounds", margin=60, n_segments=1024, background_prompt="", denomination=None):
    """
    Enhanced version that generates background using prompt from background_prompt.txt
    """
    import os
    import glob
    import hashlib
    import random
    import numpy as np
    from PIL import Image
    from skimage import color, segmentation, measure
    import svgwrite
    
    background_path = None
    
    # Read background prompt from file
    prompt_file = "./background_prompt.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            background_prompt = f.read().strip()
        print(f"[+] Using prompt from file: {background_prompt}")
    else:
        background_prompt = "kawaii oekaki Chinese DMT Studio Ghibli style banknote background"
        print(f"[!] Prompt file not found, using default: {background_prompt}")
    
    # Generate background using the prompt
    background_path = generate_sd_background(
        prompt=background_prompt,
        width=W - 2*margin,
        height=H - 2*margin,
        save_path=bg_dir,
        seed_text=seed_text,
        denomination=denomination
    )
    
    # Fallback to random background if generation failed
    if not background_path or not os.path.exists(background_path):
        files = [f for f in os.listdir(bg_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if files:
            background_path = os.path.join(bg_dir, random.choice(files))
            print(f"[+] Using random background: {background_path}")
        else:
            print("[!] No background files found.")
            return
    
    # Continue with the original vectorization logic
    img = Image.open(background_path).convert("RGB")
    img = img.resize((W - 2*margin, H - 2*margin), Image.LANCZOS)
    
    # convert to np array
    arr = np.array(img)
    arr_lab = color.rgb2lab(arr)

    # segment into superpixels
    segments = segmentation.slic(arr_lab, n_segments=n_segments, compactness=20, start_label=1)

    # extract contours of each segment
    group = dwg.g(opacity=0.7)  # Group for all background elements
    
    for seg_val in np.unique(segments):
        mask = (segments == seg_val).astype(float)
        contours = measure.find_contours(mask, 0.5)

        for contour in contours:
            # rescale contour to SVG coords (add margin)
            contour = contour[:, ::-1]  # (y, x) → (x, y)
            contour[:, 0] += margin
            contour[:, 1] += margin

            # build path string
            path_data = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in contour) + " Z"

            # get average color for this region
            avg_col = np.mean(arr[segments == seg_val], axis=0).astype(int)
            fill = svgwrite.rgb(int(avg_col[0]), int(avg_col[1]), int(avg_col[2]))

            group.add(dwg.path(d=path_data, fill=fill, stroke="none"))

    dwg.add(group)
    print(f"[+] Vectorized background with {len(np.unique(segments))} segments")
    return group

def generate_sd_background(prompt, width=512, height=512, save_path="./backgrounds", seed_text="", denomination=None):
    """
    Generate background using Stable Diffusion API with the given prompt
    """
    import os
    import requests
    import base64
    import hashlib
    import time
    from io import BytesIO
    from PIL import Image
    import random
    
    os.makedirs(save_path, exist_ok=True)
    
    # Read negative prompt from file or use default
    negative_prompt_file = "negative_prompt.txt"
    if os.path.exists(negative_prompt_file):
        with open(negative_prompt_file, 'r') as f:
            negative_prompt = f.read().strip()
    else:
        negative_prompt = "text, words, blurry, low quality, watermark, signature"
    
    print(f"[+] Generating background with prompt: {prompt}")
    print(f"[+] Negative prompt: {negative_prompt}")
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": random.randint(0, 2**32 - 1),
        "steps": 20,
        "cfg_scale": 7.5,
        "sampler_name": "Euler a",
        "batch_size": 1,
        "n_iter": 1,
        "restore_faces": False,
        "tiling": True,
        "enable_hr": False,
    }
    
    try:
        response = requests.post("http://127.0.0.1:3014/sdapi/v1/txt2img", json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        images = result.get('images', [])
        
        if images:
            image_data = base64.b64decode(images[0])
            image = Image.open(BytesIO(image_data))
            
            # Generate UNIQUE filename with denomination and prompt hash
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:6]
            denom_str = f"d{denomination}" if denomination else "node"
            filename = f"bg_{denom_str}_{prompt_hash}_{int(time.time())}.png"
            filepath = os.path.join(save_path, filename)
            
            image.save(filepath)
            print(f"[+] Generated background: {filepath}")
            return filepath
        
    except Exception as e:
        print(f"[!] Error generating background: {e}")
        return None

def generate_sd_background_from_metadata(encoded_seed, name="", width: int = 512, height: int = 512, 
                                       save_path: str = "./backgrounds", denom: int = None):
    """
    Generate background using metadata-encoded prompt
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Create prompt from metadata
    prompt = generate_kawaii_mural_from_background(denomination=denom_to_int(denom))
    negative_prompt = read_prompt_file("negative_prompt.txt", "text, words, blurry, low quality")
    
    # Print the prompts being used
    print(f"[+] Generating background with prompt: {prompt}")
    print(f"[+] Negative prompt: {negative_prompt}")
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": random.randint(0, 2**32 - 1),
        "steps": 20,
        "cfg_scale": 7.5,
        "sampler_name": "Euler a",
        "batch_size": 1,
        "n_iter": 1,
        "restore_faces": False,
        "tiling": True,
        "enable_hr": False,
    }
    
    try:
        response = requests.post("http://127.0.0.1:3014/sdapi/v1/txt2img", json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        images = result.get('images', [])
        
        if images:
            image_data = base64.b64decode(images[0])
            image = Image.open(BytesIO(image_data))
            
            # Generate UNIQUE filename with denomination and metadata hash
            metadata_hash = hashlib.md5(encoded_seed.encode()).hexdigest()[:6]
            denom_str = f"d{denom}" if denom else "node"
            filename = f"bg_{denom_str}_meta_{metadata_hash}_{int(time.time())}.png"
            filepath = os.path.join(save_path, filename)
            
            image.save(filepath)
            print(f"[+] Generated metadata-based background: {filepath}")
            return filepath
        
    except Exception as e:
        print(f"[!] Error generating background: {e}")
        return None
def add_security_pattern_overlay(dwg, pattern_path, W, H, margin):
    """
    Add security pattern as an overlay to the SVG
    """
    try:
        # Load the security pattern image
        pattern_img = Image.open(pattern_path).convert("RGBA")
        pattern_img = pattern_img.resize((W - 2*margin, H - 2*margin), Image.LANCZOS)
        
        # Convert to numpy array for processing
        pattern_arr = np.array(pattern_img)
        
        # Create a group for security pattern elements
        pattern_group = dwg.g(opacity=0.3)  # Semi-transparent overlay
        
        # Simple approach: convert pattern to SVG paths
        # For complex patterns, you might want to use a different approach
        alpha_threshold = 128  # Only include pixels with sufficient opacity
        
        # Sample points from the pattern (simplified approach)
        height, width = pattern_arr.shape[:2]
        step = max(1, width // 50)  # Adjust sampling density
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                if pattern_arr[y, x, 3] > alpha_threshold:  # Check alpha channel
                    # Add a small shape at this position
                    color = pattern_arr[y, x, :3]
                    fill = svgwrite.rgb(int(color[0]), int(color[1]), int(color[2]))
                    
                    # Add a small circle or other shape
                    cx = x + margin
                    cy = y + margin
                    pattern_group.add(dwg.circle(
                        center=(cx, cy),
                        r=2,  # Small radius
                        fill=fill,
                        stroke="none"
                    ))
        
        dwg.add(pattern_group)
        
    except Exception as e:
        print(f"[!] Error adding security pattern overlay: {e}")
        # Fallback: add a simple pattern
        add_fallback_security_pattern(dwg, W, H, margin)

def add_fallback_security_pattern(dwg, W, H, margin):
    """
    Fallback security pattern if the main pattern fails
    """
    pattern_group = dwg.g(opacity=0.15, fill="none", stroke="#ff0000", stroke_width=0.5)
    
    # Add some simple geometric patterns
    spacing = 20
    for x in range(margin, W - margin, spacing):
        pattern_group.add(dwg.line(
            start=(x, margin),
            end=(x, H - margin),
            stroke_dasharray="2,4"
        ))
    
    for y in range(margin, H - margin, spacing):
        pattern_group.add(dwg.line(
            start=(margin, y),
            end=(W - margin, y),
            stroke_dasharray="4,2"
        ))
    
    dwg.add(pattern_group)
def fractal_stamp(
    dwg: svgwrite.Drawing,
    width: int,
    height: int,
    denom: str = "100",
    timestamp: str = None,
    font_family: str = "Daemon Full Working",
    base_font_size: int = 20,
    depth: int = 3
):
    """
    Create a fractal-like denomination microprint background + rainbow timestamp stripe.
    - denom: fractal mosaic background
    - timestamp: rainbow encoded stripe
    """

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        timestamp = str(timestamp)  # <--- fix here

    # Colors for ROYGBIV rainbow stripe
    rainbow = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]

    # --- Step 1: Draw fractal <denom> background ---
    def recursive_denom(x, y, size, level):
        if level <= 0:
            return
        dwg.add(dwg.text(
            denom,
            insert=(x, y),
            font_size=size,
            font_family=font_family,
            fill="#000000",
            opacity=0.05 * level,  # fading layers
            text_anchor="middle",
            alignment_baseline="middle"
        ))
        # Recursively branch smaller denom around
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            recursive_denom(x + dx * size, y + dy * size, size * 0.5, level - 1)

    # Scatter recursive fractal denominations
    step = base_font_size * 3
    for i in range(0, width, step):
        for j in range(0, height, step):
            if random.random() < 0.2:  # sparse fractal seeds
                recursive_denom(i, j, base_font_size, depth)

    # --- Step 2: Rainbow timestamp stripe ---
    stripe_height = base_font_size * 2
    for idx, char in enumerate(timestamp):
        color = rainbow[idx % len(rainbow)]
        opacity = 0.3 + 0.7 * (idx % len(rainbow)) / (len(rainbow)-1)
        x = (width / len(timestamp)) * idx + (width/len(timestamp))/2
        y = height / 2

        dwg.add(dwg.text(
            char,
            insert=(x, y),
            font_size=base_font_size * 1.5,
            font_family=font_family,
            fill=color,
            opacity=opacity,
            text_anchor="middle",
            alignment_baseline="middle"
        ))

    # Optional second rainbow stripe diagonally
    for idx, char in enumerate(timestamp):
        color = rainbow[idx % len(rainbow)]
        opacity = 0.2 + 0.8 * (math.sin(idx) * 0.5 + 0.5)
        x = (width / len(timestamp)) * idx
        y = height * 0.75 + math.sin(idx*0.5) * 20

        dwg.add(dwg.text(
            char,
            insert=(x, y),
            font_size=base_font_size,
            font_family=font_family,
            fill=color,
            opacity=opacity,
            text_anchor="middle",
            alignment_baseline="middle",
            transform=f"rotate(-15,{x},{y})"
        ))

def run_single_denomination(outdir: str = ".", base_name: str = "banknote", denomination: int = 1, 
                           width_mm: float = 160.0, height_mm: float = 60.0,
                           title_text: str = "灵国国库", phrase_text: str = "灵之意志，天下共识", seed_text: str = "Username", serial_id: str = "SNB-", timestamp: str = None,
                           png: bool = False):
    W = mm_to_px(width_mm)
    H = mm_to_px(height_mm)
    os.makedirs(outdir, exist_ok=True)
    
    fname = f"{base_name}.svg"
    path = os.path.join(outdir, fname)
    generate_backside_svg(path, denomination, title_text, phrase_text, (W,H), serial_id, timestamp, seed_text)
    
    if png:
        if not CAIROSVG_AVAILABLE:
            print("[!] cairosvg not installed — skipping PNG for", path)
        else:
            png_path = os.path.splitext(path)[0] + ".png"
            try:
                cairosvg.svg2png(url=path, write_to=png_path, output_width=W, output_height=H)
                print(f"[+] Saved {png_path}")
            except Exception as e:
                print("[!] Failed to convert to PNG:", e)

# Then modify the argument parsing to accept a denomination parameter
def run_batch(outdir: str = ".", base_name: str = "banknote", width_mm: float = 160.0, height_mm: float = 60.0,
              title_text: str = "灵国国库", phrase_text: str = "灵之意志，天下共识", seed_text: str = "Username", serial_id: str = "FRONT", timestamp: str = None,
              png: bool = False):
    denoms = [10**i for i in range(0,9)]
    W = mm_to_px(width_mm)
    H = mm_to_px(height_mm)
    os.makedirs(outdir, exist_ok=True)
    for d in denoms:
        # Include denomination in the filename to avoid overwriting
        fname = f"{base_name}_{d}.svg"  # Add denomination to filename
        path = os.path.join(outdir, fname)
        generate_backside_svg(path, d, title_text, phrase_text, seed_text, serial_id, timestamp, (W,H))
        
        if png:
            if not CAIROSVG_AVAILABLE:
                print("[!] cairosvg not installed — skipping PNG for", path)
            else:
                png_path = os.path.splitext(path)[0] + ".png"
                try:
                    cairosvg.svg2png(url=path, write_to=png_path, output_width=W, output_height=H)
                    print(f"[+] Saved {png_path}")
                except Exception as e:
                    print("[!] Failed to convert to PNG:", e)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Red/Blue symmetric banknotes")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("--basename", type=str, default="banknote", help="Base filename")
    parser.add_argument("--denomination", type=int, help="Single denomination to generate (if not specified, generates all)")
    parser.add_argument("--width-mm", type=float, default=160.0, help="Width in mm")
    parser.add_argument("--height-mm", type=float, default=60.0, help="Height in mm")
    parser.add_argument("--title", type=str, default="灵国国库", help="Center title text")
    parser.add_argument("--phrase", type=str, default="灵之意志，天下共识", help="Phrase under the title")
    parser.add_argument("--seed_text", type=str, default="Name", help="Seed Text, usually a Username")
    parser.add_argument("--serial_id", type=str, default="Name", help="serial_id")
    parser.add_argument("--timestamp", type=int, help="Datetime Stamp precisely on the microsecond")
    parser.add_argument("--png", action="store_true", help="Attempt to output PNGs (requires cairosvg)")
    args = parser.parse_args()

    if args.denomination:
        run_single_denomination(outdir=args.outdir, base_name=args.basename, denomination=args.denomination,
                               width_mm=args.width_mm, height_mm=args.height_mm,
                               title_text=args.title, phrase_text=args.phrase,seed_text=args.seed_text, serial_id=args.serial_id, timestamp=args.timestamp, png=args.png)
    else:
        run_batch(outdir=args.outdir, base_name=args.basename, width_mm=args.width_mm, height_mm=args.height_mm,
                  title_text=args.title, phrase_text=args.phrase,seed_text=args.seed_text, serial_id=args.serial_id, timstamp=args.timestamp, png=args.png)