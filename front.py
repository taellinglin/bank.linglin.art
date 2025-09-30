#!/usr/bin/env python3
"""
fantasy_banknote.py — Enhanced procedural fantasy banknote generator
----------------------------------------------------------------------
Creates a stylized, clearly-marked "banknote" from an input image.
Outputs an SVG (vector) and optional PNG preview.
Requires: Pillow, svgwrite. Optional: fontTools for glyph paths.

Author: RingMaster Lin
"""
from io import BytesIO
import os
import sys
import math
import io
import base64
import argparse
import hashlib
from typing import Tuple, List
import binascii
from PIL import Image, ImageOps
import numpy as np
from sklearn.cluster import KMeans
import requests
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
import os
import time

def generate_security_pattern(bg_path, output_path=None, seed_data=None, font_path=None, pattern_density=0.1):
    """
    Overlay a colored pattern on a background image, using SHA3-512 hashing for deterministic patterns.
    If output_path is a directory, auto-generate a filename inside it.
    If output_path is None, overwrite the background.
    """

    # Load background
    bg = Image.open(bg_path).convert("RGBA")
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
    if font_path:
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
        output_path = bg_path  # overwrite
    elif os.path.isdir(output_path):
        base = os.path.splitext(os.path.basename(bg_path))[0]
        output_path = os.path.join(output_path, f"{base}_pattern_{int(time.time())}.png")

    result.save(output_path)
    print(f"[+] Saved patterned image → {output_path}")

def sha3_512_salted(s: str, salt: str = None) -> bytes:
    """
    Returns SHA3-512(salt + s) where salt is optional.
    If salt is None, only s is hashed.
    """
    hash_obj = hashlib.sha3_512()
    if salt is not None:
        hash_obj.update(str(salt).encode("utf-8"))
    hash_obj.update(str(s).encode("utf-8"))
    return hash_obj.digest()

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
def make_qr_seed(seed: str, serial_id: str = None, timestamp: str = None) -> str:
    """
    Simple deterministic seed combiner for QR patterns.
    """
    parts = []
    if seed:
        parts.append(seed)
    if serial_id:
        parts.append(serial_id)
    if timestamp:
        parts.append(timestamp)
    
    return "_".join(parts) if parts else "default_seed"
def add_qr_like_border(dwg: svgwrite.Drawing, seed: str, width: int, height: int, serial_id=None, timestamp_ms=None):
    """
    Adds a QR-like border with colored "pixels" derived from a seed.
    Border is positioned 1/8 cm from the edge and 1/2 cm thick.
    Returns dimensions for subsequent layers.
    """
    # Convert measurements to pixels (300 DPI)
    inset_px = mm_to_px(0.5)  # 1/8 cm
    border_thickness_px = mm_to_px(3)  # 1/2 cm

    # Border outer coordinates
    qr_border_start_x = float(inset_px)
    qr_border_start_y = float(inset_px)
    qr_border_end_x = float(width - inset_px)
    qr_border_end_y = float(height - inset_px)

    # Border inner coordinates
    qr_border_inner_start_x = qr_border_start_x + border_thickness_px
    qr_border_inner_start_y = qr_border_start_y + border_thickness_px
    qr_border_inner_end_x = qr_border_end_x - border_thickness_px
    qr_border_inner_end_y = qr_border_end_y - border_thickness_px

    # Cell size
    cell = max(2, border_thickness_px // 8)

    # Grid dimensions
    qr_border_width = qr_border_end_x - qr_border_start_x
    qr_border_height = qr_border_end_y - qr_border_start_y
    cols = int(math.ceil(qr_border_width / cell))
    rows = int(math.ceil(qr_border_height / cell))

    # Seed bytes
    seed_bytes = to_bytes(make_qr_seed(seed, serial_id, str(timestamp_ms) if timestamp_ms else None))

    # Draw border
    for r in range(rows):
        for c in range(cols):
            x = float(qr_border_start_x + c * cell)
            y = float(qr_border_start_y + r * cell)

            # Skip inner area
            if (qr_border_inner_start_x <= x < qr_border_inner_end_x and
                qr_border_inner_start_y <= y < qr_border_inner_end_y):
                continue

            idx = (r * cols + c) % len(seed_bytes)
            v = seed_bytes[idx]

            # Color
            red = int((v * 3) % 256)
            green = int((v * 7 + r * 5) % 256)
            blue = int((v * 13 + c * 11) % 256)
            color = f"rgb({red},{green},{blue})"

            # Size scaling
            s = 1.0 if (v % 3 == 0) else (0.6 if (v % 3 == 1) else 0.35)
            w = float(max(1, int(cell * s)))
            h = float(max(1, int(cell * s)))

            dwg.add(dwg.rect(
                insert=(x + (cell - w) / 2, y + (cell - h) / 2),
                size=(w, h),
                fill=color,
                fill_opacity=1.0
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

# ------------------------
# Center seal as concentric colored dots
# ------------------------
def add_center_seal(dwg: svgwrite.Drawing, im: Image.Image, cx: float, cy: float, size_px: float, frame=True, step=4):
    im = im.convert("RGB").resize((int(size_px), int(size_px)), Image.LANCZOS)
    pixels = im.load()
    radius = size_px/2

    for row in range(0, im.height, step):
        for col in range(0, im.width, step):
            dx = col - radius
            dy = row - radius
            if dx*dx + dy*dy > radius*radius:
                continue  # omit dots outside circle
            r, g, b = pixels[col, row]
            dwg.add(dwg.circle(
                center=(cx - radius + col, cy - radius + row),
                r=step/2,
                fill=svgwrite.rgb(r, g, b),
                stroke="none",
                opacity=1.0
            ))

    if frame:
        dwg.add(dwg.circle(center=(cx, cy), r=radius+8, fill="none", stroke="#000", stroke_width=2.0))
        dwg.add(dwg.circle(center=(cx, cy), r=radius+16, fill="none", stroke="#000", stroke_width=1.0))
def add_mixed_font_text(dwg, text, insert_pos, text_anchor="middle", font_size=12, 
                       chinese_font="FengGuangMingRui", english_font="Daemon Full Working",
                       padding=2, fill_color="currentColor"):
    """
    Add mixed Chinese/English text with proper alignment and padding
    
    Parameters:
    dwg: svgwrite Drawing object
    text: input text containing mixed Chinese and English
    insert_pos: (x, y) position to insert text
    text_anchor: text anchor position ("start", "middle", "end")
    font_size: base font size
    chinese_font: font family for Chinese characters
    english_font: font family for English characters
    padding: space between language segments
    fill_color: text color
    """
    
    # Pattern to match Chinese characters
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+'
    
    x, y = insert_pos
    segments = []
    last_end = 0
    
    # Split text into Chinese and non-Chinese segments
    for match in re.finditer(chinese_pattern, text):
        # Add non-Chinese text before the match
        if match.start() > last_end:
            segments.append(('english', text[last_end:match.start()]))
        
        # Add Chinese text
        segments.append(('chinese', match.group()))
        last_end = match.end()
    
    # Add remaining non-Chinese text
    if last_end < len(text):
        segments.append(('english', text[last_end:]))
    
    # Calculate total width and segment positions
    segment_widths = []
    total_width = 0
    
    for lang_type, segment_text in segments:
        if segment_text.strip():
            # Different width multipliers for Chinese vs English
            char_width = font_size * 0.85 if lang_type == 'chinese' else font_size * 0.55
            width = len(segment_text) * char_width
            segment_widths.append(width)
            total_width += width + padding
        else:
            segment_widths.append(0)
    
    total_width -= padding  # Remove last padding
    
    # Determine starting position based on text anchor
    if text_anchor == "middle":
        current_x = x - total_width / 2
    elif text_anchor == "end":
        current_x = x - total_width
    else:  # "start"
        current_x = x
    
    # Add each segment with appropriate font and positioning
    for i, ((lang_type, segment_text), seg_width) in enumerate(zip(segments, segment_widths)):
        if segment_text.strip():
            font_family = chinese_font if lang_type == 'chinese' else english_font
            
            # Adjust vertical position for better centering
            # Chinese characters often need slight vertical adjustment
            vertical_offset = 0
            if lang_type == 'chinese':
                vertical_offset = font_size * 0.1  # Slight upward adjustment for Chinese
            
            dwg.add(dwg.text(segment_text, 
                            insert=(current_x, y + vertical_offset),
                            text_anchor="start",
                            font_size=font_size,
                            font_family=font_family,
                            fill=fill_color,
                            alignment_baseline="middle"))
            
            # Update current position with padding
            current_x += seg_width + padding


# Alternative version for more precise control with text elements
def add_mixed_font_text_precise(dwg, text, insert_pos, text_anchor="middle", font_size=12, 
                               chinese_font="FengGuangMingRui", english_font="Daemon Full Working",
                               chinese_padding=3, english_padding=1, fill_color="currentColor", stroke: str = "#000", stroke_width: float = 1 ):
    """
    More precise version with different padding for Chinese and English
    """
    
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+'
    x, y = insert_pos
    segments = []
    last_end = 0
    
    # Split text
    for match in re.finditer(chinese_pattern, text):
        if match.start() > last_end:
            segments.append(('english', text[last_end:match.start()]))
        segments.append(('chinese', match.group()))
        last_end = match.end()
    
    if last_end < len(text):
        segments.append(('english', text[last_end:]))
    
    # Calculate positions
    segment_data = []
    total_width = 0
    
    for lang_type, segment_text in segments:
        if segment_text.strip():
            char_width = font_size * (0.85 if lang_type == 'chinese' else 0.55)
            width = len(segment_text) * char_width
            padding = chinese_padding if lang_type == 'chinese' else english_padding
            segment_data.append({
                'type': lang_type,
                'text': segment_text,
                'width': width,
                'padding': padding,
                'font': chinese_font if lang_type == 'chinese' else english_font,
                'offset': font_size * (0.01 if lang_type == 'chinese' else 0)
            })
            total_width += width + padding
        else:
            segment_data.append({'width': 0, 'padding': 0})
    
    if segment_data:
        total_width -= segment_data[-1]['padding']  # Remove last padding
    
    # Determine starting position
    if text_anchor == "middle":
        current_x = x - total_width / 2
    elif text_anchor == "end":
        current_x = x - total_width
    else:
        current_x = x
    
    # Render segments
    for i, segment in enumerate(segment_data):
        if segment.get('text'):
            dwg.add(dwg.text(segment['text'],
                            insert=(current_x, y + segment['offset'] - 20),
                            text_anchor="start",
                            font_size=font_size,
                            font_family=segment['font'],
                            fill=fill_color,
                            stroke=stroke,
                            stroke_width=stroke_width,
                            alignment_baseline="middle"))
            
            current_x += segment['width'] + segment['padding']
def add_mixed_font_text(dwg, text, insert_pos, text_anchor="middle", font_size=12, 
                       chinese_font="FengGuangMingRui", english_font="Daemon Full Working"):
    """
    Add text with mixed Chinese and English fonts to SVG drawing
    
    Parameters:
    dwg: svgwrite Drawing object
    text: input text containing mixed Chinese and English
    insert_pos: (x, y) position to insert text
    text_anchor: text anchor position
    font_size: base font size
    chinese_font: font family for Chinese characters
    english_font: font family for English characters
    """
    
    # Pattern to match Chinese characters (CJK Unified Ideographs)
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+'
    
    x, y = insert_pos
    segments = []
    last_end = 0
    
    # Split text into Chinese and non-Chinese segments
    for match in re.finditer(chinese_pattern, text):
        # Add non-Chinese text before the match
        if match.start() > last_end:
            segments.append(('english', text[last_end:match.start()]))
        
        # Add Chinese text
        segments.append(('chinese', match.group()))
        last_end = match.end()
    
    # Add remaining non-Chinese text
    if last_end < len(text):
        segments.append(('english', text[last_end:]))
    
    # Calculate total text width for centering
    total_width = 0
    for lang_type, segment_text in segments:
        # Simple width estimation (adjust multiplier as needed)
        char_width = font_size * 0.6 if lang_type == 'chinese' else font_size * 0.5
        total_width += len(segment_text) * char_width
    
    # Starting position based on text anchor
    current_x = x
    if text_anchor == "middle":
        current_x = x - total_width / 2
    elif text_anchor == "end":
        current_x = x - total_width
    
    # Add each segment with appropriate font
    for lang_type, segment_text in segments:
        if segment_text.strip():  # Skip empty segments
            font_family = chinese_font if lang_type == 'chinese' else english_font
            
            dwg.add(dwg.text(segment_text, 
                            insert=(current_x, y),
                            text_anchor="start",
                            font_size=font_size,
                            font_family=font_family,
                            fill="currentColor"))
            
            # Update current position (simple width estimation)
            char_width = font_size * 0.6 if lang_type == 'chinese' else font_size * 0.6
            current_x += len(segment_text) * char_width
            
            
            

def clean_string(s: str) -> str:
    """
    Removes numbers, underscores, and punctuation from the string.
    If the string contains Chinese characters, return their Unicode code points concatenated.
    Example: 'ling lin 灵林' -> 'linglin70756797'
    """
    # Regex for Chinese characters (CJK Unified Ideographs)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', s)
    
    if chinese_chars:
        # Convert Chinese chars into their codepoints (decimal, no \u)
        chinese_unicode = ''.join(str(ord(ch)) for ch in chinese_chars)
        # Remove numbers, underscores, punctuation from the rest
        latin_part = re.sub(r'[\d\W_]+', '', s, flags=re.UNICODE)
        # Drop the Chinese chars from latin_part so they don't duplicate
        latin_part = ''.join(ch for ch in latin_part if ch not in chinese_chars)
        return latin_part + chinese_unicode
    else:
        # For non-Chinese text, just remove digits and punctuation
        return re.sub(r'[\d\W_]+', '', s, flags=re.UNICODE)
# ----------------------
# Text seal with optional datetime
# ----------------------
from datetime import datetime
import math
from datetime import datetime

import math
from datetime import datetime

def add_text_seal(
    dwg, cy, radius, text_left, text_right, denom_color,
    inner_text=None, include_datetime=False,
    seed_text=None, serial_id=None,
    canvas_width=1600
):
    """
    Draws two seals at 1/4 and 3/4 of canvas width.
      - Left: 0.5 white bg, black text/circles, denom_color border
      - Right: 0.5 black bg, white text
    Optionally encodes seed_text/serial_id into rotary star patterns.
    """

    cx_left = canvas_width * 0.15
    cx_right = canvas_width * 0.85

    # --- Background Circles ---
    dwg.add(dwg.circle(center=(cx_left, cy), r=radius,
                       fill="white", fill_opacity=0.5,
                       stroke="black", stroke_width=2, stroke_opacity=1))

    dwg.add(dwg.circle(center=(cx_right, cy), r=radius-0.2,
                       fill="black", fill_opacity=0.5,
                       stroke="white", stroke_width=0.5, stroke_opacity=1))

   # --- Inner Decorative Circles ---
    for cx, txtcol, fillcol, symbol in [
        (cx_left, "black", "white", u"日"),  # Left = Sun
        (cx_right, "white", "black", u"月")  # Right = Moon
    ]:
        dwg.add(dwg.circle(center=(cx, cy), r=radius*0.95,
                        fill="none", stroke=denom_color, stroke_width=2))
        dwg.add(dwg.circle(center=(cx, cy), r=radius*0.85,
                        fill="none", stroke=denom_color, stroke_width=1))

        # Center symbol (Sun or Moon)
        dwg.add(dwg.text(symbol, insert=(cx, cy+radius*0.15),
                        text_anchor="middle", font_size=int(radius*0.5),
                        font_family="FengGuangMingRui", fill=txtcol))

    # Title text with precise mixed font alignment
    add_mixed_font_text_precise(dwg, clean_string(text_left), (cx_left, cy-radius-6.1), 
                            text_anchor="middle", 
                            font_size=int(radius*0.2),
                            chinese_font="FengGuangMingRui", 
                            english_font="Daemon Full Working",
                            chinese_padding=32,    # More padding for Chinese segments
                            english_padding=32,
                            stroke="#FFF",
                            stroke_width=0.5,
                            fill_color="#000")

    add_mixed_font_text_precise(dwg, text_right, (cx_right, cy-radius-6.1),
                            text_anchor="middle",
                            font_size=int(radius*0.2),
                            chinese_font="FengGuangMingRui", 
                            english_font="Daemon Full Working",
                            chinese_padding=32,
                            english_padding=32,
                            stroke="#000",
                            stroke_width=0.5,
                            fill_color="#FFF")


    # --- Rotary Dial Numbers ---
    if include_datetime:
        dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
        n = len(dt_string)
        for cx, txtcol, code in [
            (cx_left, "black", seed_text),
            (cx_right, "white", serial_id)
        ]:
            points = []
            for i, char in enumerate(dt_string):
                angle = 2 * math.pi * i / n
                r = radius * 0.65
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                dwg.add(dwg.text(char, insert=(x, y),
                                 font_size=int(radius*0.08), fill=txtcol,
                                 font_family="Daemon Full Working",
                                 text_anchor="middle", opacity=1))
                if code and char in code:
                    dwg.add(dwg.circle(center=(x, y), r=radius*0.05,
                                       fill=txtcol, opacity=0.6))
                    points.append((x, y))
            if len(points) > 1:
                for i in range(len(points)):
                    x1, y1 = points[i]
                    x2, y2 = points[(i+1) % len(points)]
                    dwg.add(dwg.line(start=(x1, y1), end=(x2, y2),
                                     stroke=txtcol, stroke_width=1, opacity=0.7))

def add_secondary_ring(dwg: svgwrite.Drawing, cx: float, cy: float, radius: float, seed: bytes, segments: int = 360, d_color: str = None):
    """
    Creates a geometric mandala-style ring with seed-colored shapes and black outlines.
    Features colored datapoints with black borders and adjusted ring thickness.
    """
    import random
    random.seed(int.from_bytes(seed, 'big'))
    
    # Use seed bytes for deterministic pattern generation
    seed_values = [b for b in seed]
    
    # Define ring thickness for geometric patterns
    ring_thickness = radius * 0.15
    inner_radius = radius - ring_thickness / 2
    outer_radius = radius + ring_thickness / 2
    
    # Color palette based on seed values
    def get_color_from_seed(seed_val):
        # Create deterministic colors from seed values
        hue = (seed_val * 137) % 360  # Golden ratio inspired
        saturation = 80 + (seed_val % 20)
        lightness = 40 + (seed_val % 15)
        
        # Convert HSL to HEX
        return hsl_to_hex(hue, saturation, lightness)

    def hsl_to_hex(h, s, l):
        """Convert HSL color values to HEX format"""
        h = h % 360
        s = max(0, min(100, s)) / 100
        l = max(0, min(100, l)) / 100
        
        # HSL to RGB conversion
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Convert RGB to HEX
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    dwg.add(dwg.circle(center=(cx, cy), r=outer_radius, fill=d_color, 
                    stroke="#000000", stroke_width=ring_thickness * 0.14, fill_opacity=0.5))  # Thicker
    dwg.add(dwg.circle(center=(cx, cy), r=inner_radius, fill="#FFF", 
                    stroke="#000000", stroke_width=ring_thickness * 0.06, fill_opacity=0.5))  # Thinner
    dwg.add(dwg.circle(center=(cx, cy), r=(inner_radius-40), fill="#000", 
                    stroke="#000000", stroke_width=ring_thickness * 0.03, fill_opacity=0.5))  # Thicker
    # Create geometric patterns with radial alignment
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        
        # Use seed for deterministic pattern spacing
        spacing_seed = seed_values[i % len(seed_values)]
        
        # Only draw elements at specific radial intervals based on seed
        should_draw = (spacing_seed % 8) < 3  # 3/8 elements drawn
        
        if should_draw:
            # Seed-based radial positioning
            position_seed = seed_values[(i + 17) % len(seed_values)]
            radial_position = 0.2 + 0.6 * (position_seed / 255)
            r = inner_radius + (outer_radius - inner_radius) * radial_position
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            
            # Geometric elements only - seed-based selection
            shape_seed = seed_values[(i + 7) % len(seed_values)]
            shape_type = shape_seed % 4
            
            # Get color from seed
            color_seed = seed_values[(i + 11) % len(seed_values)]
            shape_color = get_color_from_seed(color_seed)
            
            # Calculate rotation angle (90 degrees to center - radial alignment)
            rotation_angle = math.degrees(angle) + 90
            
            # Determine if shape should be solid or knocked-out
            is_solid = (shape_seed % 2 == 0)
            
            if shape_type == 0:
                # Square
                size_base = ring_thickness * 0.22
                size_var = 0.3 * (shape_seed / 255)
                size = size_base * (0.7 + size_var)
                
                if is_solid:
                    # Solid colored square with black border
                    dwg.add(dwg.rect(insert=(x-size/2, y-size/2), 
                                    size=(size, size), fill=shape_color, 
                                    stroke="#000000", stroke_width=size*0.08,
                                    transform=f"rotate({rotation_angle},{x},{y})"))
                else:
                    # Square with knocked-out center and colored/black border
                    hole_size = size * 0.6
                    path_data = f"M {x-size/2} {y-size/2} " \
                              f"L {x+size/2} {y-size/2} " \
                              f"L {x+size/2} {y+size/2} " \
                              f"L {x-size/2} {y+size/2} Z " \
                              f"M {x-hole_size/2} {y-hole_size/2} " \
                              f"L {x+hole_size/2} {y-hole_size/2} " \
                              f"L {x+hole_size/2} {y+hole_size/2} " \
                              f"L {x-hole_size/2} {y+hole_size/2} Z"
                    dwg.add(dwg.path(d=path_data, fill="#ffffff", fill_rule="evenodd", 
                                    stroke=shape_color, stroke_width=size*0.12,
                                    transform=f"rotate({rotation_angle},{x},{y})"))
                
            elif shape_type == 1:
                # Diamond
                size_base = ring_thickness * 0.26
                size_var = 0.4 * (shape_seed / 255)
                size = size_base * (0.6 + size_var)
                
                if is_solid:
                    # Solid colored diamond with black border
                    points = [
                        (x, y-size/2), (x+size/2, y), 
                        (x, y+size/2), (x-size/2, y)
                    ]
                    dwg.add(dwg.polygon(points=points, fill=shape_color, 
                                       stroke="#000000", stroke_width=size*0.08,
                                       transform=f"rotate({rotation_angle},{x},{y})"))
                else:
                    # Diamond with knocked-out center and colored/black border
                    hole_size = size * 0.5
                    outer_points = [
                        (x, y-size/2), (x+size/2, y), 
                        (x, y+size/2), (x-size/2, y)
                    ]
                    inner_points = [
                        (x, y-hole_size/2), (x+hole_size/2, y), 
                        (x, y+hole_size/2), (x-hole_size/2, y)
                    ]
                    path_data = f"M {outer_points[0][0]} {outer_points[0][1]} " \
                              f"L {outer_points[1][0]} {outer_points[1][1]} " \
                              f"L {outer_points[2][0]} {outer_points[2][1]} " \
                              f"L {outer_points[3][0]} {outer_points[3][1]} Z " \
                              f"M {inner_points[0][0]} {inner_points[0][1]} " \
                              f"L {inner_points[1][0]} {inner_points[1][1]} " \
                              f"L {inner_points[2][0]} {inner_points[2][1]} " \
                              f"L {inner_points[3][0]} {inner_points[3][1]} Z"
                    dwg.add(dwg.path(d=path_data, fill="#ffffff", fill_rule="evenodd", 
                                    stroke=shape_color, stroke_width=size*0.12,
                                    transform=f"rotate({rotation_angle},{x},{y})"))
                
            elif shape_type == 2:
                # Circle
                size_base = ring_thickness * 0.18
                size_var = 0.5 * ((shape_seed + 11) % 256 / 255)
                size = size_base * (0.5 + size_var)
                
                if is_solid:
                    # Solid colored circle with black border
                    dwg.add(dwg.circle(center=(x, y), r=size, fill=shape_color, 
                                      stroke="#000000", stroke_width=size*0.08))
                else:
                    # Circle with knocked-out center and colored/black border
                    hole_size = size * 0.6
                    path_data = f"M {x+size} {y} " \
                              f"A {size} {size} 0 1 1 {x-size} {y} " \
                              f"A {size} {size} 0 1 1 {x+size} {y} Z " \
                              f"M {x+hole_size} {y} " \
                              f"A {hole_size} {hole_size} 0 1 1 {x-hole_size} {y} " \
                              f"A {hole_size} {hole_size} 0 1 1 {x+hole_size} {y} Z"
                    dwg.add(dwg.path(d=path_data, fill="#ffffff", fill_rule="evenodd", 
                                    stroke=shape_color, stroke_width=size*0.12))
                
            else:
                # Triangle
                size_base = ring_thickness * 0.24
                size_var = 0.6 * (shape_seed / 255)
                size = size_base * (0.4 + size_var)
                flip = 1 if (shape_seed % 2 == 0) else -1
                
                if is_solid:
                    # Solid colored triangle with black border
                    points = [
                        (x, y - flip * size/2), 
                        (x + size/2, y + flip * size/2), 
                        (x - size/2, y + flip * size/2)
                    ]
                    dwg.add(dwg.polygon(points=points, fill=shape_color, 
                                       stroke="#000000", stroke_width=size*0.08,
                                       transform=f"rotate({rotation_angle},{x},{y})"))
                else:
                    # Triangle with knocked-out center and colored/black border
                    hole_size = size * 0.5
                    outer_points = [
                        (x, y - flip * size/2), 
                        (x + size/2, y + flip * size/2), 
                        (x - size/2, y + flip * size/2)
                    ]
                    inner_points = [
                        (x, y - flip * hole_size/2), 
                        (x + hole_size/2, y + flip * hole_size/2), 
                        (x - hole_size/2, y + flip * hole_size/2)
                    ]
                    path_data = f"M {outer_points[0][0]} {outer_points[0][1]} " \
                              f"L {outer_points[1][0]} {outer_points[1][1]} " \
                              f"L {outer_points[2][0]} {outer_points[2][1]} Z " \
                              f"M {inner_points[0][0]} {inner_points[0][1]} " \
                              f"L {inner_points[1][0]} {inner_points[1][1]} " \
                              f"L {inner_points[2][0]} {inner_points[2][1]} Z"
                    dwg.add(dwg.path(d=path_data, fill="#ffffff", fill_rule="evenodd", 
                                    stroke=shape_color, stroke_width=size*0.12,
                                    transform=f"rotate({rotation_angle},{x},{y})"))
            
            # Add black spoke from inner radius to glyph element
            spoke_seed = seed_values[(i + 29) % len(seed_values)]
            if spoke_seed % 5 == 0:
                inner_x = cx + inner_radius * math.cos(angle)
                inner_y = cy + inner_radius * math.sin(angle)
                spoke_width = ring_thickness * 0.018  # Slightly thinner
                dwg.add(dwg.line(start=(inner_x, inner_y), end=(x, y), 
                                stroke="#000000", stroke_width=spoke_width, opacity=0.8))

    # Add solid black inner and outer ring borders with adjusted thickness
    # Thinner inner border, thicker outer border
    
    

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
        path = dwg.path(d=f"M{curve[0][0]},{curve[0][1]}", stroke="#000", fill="none", stroke_width=1.0, opacity=0.5)
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
            dwg.add(dwg.circle(center=(cx, cy), r=size*0.35, fill="#000", opacity=0.75))
        else:
            dwg.add(dwg.circle(center=(cx, cy), r=size*0.2, fill="#000", opacity=0.5))

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
                         fill="#000",
                         text_anchor=anchor_side,
                         alignment_baseline="middle",
                         opacity=0.85))
        
def add_corner_denominations_split(
    dwg: svgwrite.Drawing,
    W: int, H: int,
    number: str,
    font_numeric="Daemon Full Working",
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
            fill="#000",
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
                     fill="#000",
                     text_anchor="middle",
                     alignment_baseline="middle",
                     opacity=1.0))
    # Slogan (bottom center)
    dwg.add(dwg.text(slogan,
                     insert=(W/2, H*0.88),
                     font_size=int(H*0.07),
                     font_family=font_slogan,
                     fill="#000",
                     text_anchor="middle",
                     alignment_baseline="middle",
                     opacity=1.0))

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
from datetime import datetime
import random
import hashlib
import secrets
def cm_to_px(cm, dpi=300.0):
    """
    Convert centimeters to pixels.
    
    Parameters:
        cm: float - measurement in centimeters
        dpi: float - dots per inch (default: 300 DPI)
    
    Returns:
        float - measurement in pixels
    """
    return cm * dpi / 2.54
def add_subtle_frame_and_microgrid(dwg, W: int, H: int, border_info: dict, denomination: int, timestamp_ms: int, seed_hash: bytes):
    """
    Adds multi-band frame and microgrid INSIDE the QR border area.
    Uses deterministic patterns based on denomination, timestamp, and seed.
    """
    # Get the inner diamond area from border_info
    diamond_start_x = border_info['diamond_start_x'] + 0.03
    diamond_start_y = border_info['diamond_start_y'] + 0.03
    diamond_width = border_info['diamond_width'] - 0.03
    diamond_height = border_info['diamond_height'] - 0.03
    
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
            "opacity": 0 + (time_seed % 100) * 1
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

    g = dwg.g(opacity=1)
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
# --- Dynamic data generation ---
def generate_timestamp():
    """Return current date-time as YYYYMMDD-HHMM string"""
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
from PIL import Image, ImageDraw, ImageFont


def add_random_background_png(
    dwg: svgwrite.Drawing,
    W: int,
    H: int,
    seed: bytes,
    serial_id: str = None,
    margin: int = 60,
    triangle_size=32,  # size of largest triangle
    hierarchy_levels=3  # depth of subdivision
):
    """
    Create a fully vectorized interlocking triangle background inside dwg.
    Colors, saturation, and opacity are derived deterministically from seed or serial_id.
    Triangles are sampled from the source image for color info.
    Pattern forms an interlocking / \ / \ / \ up-down-up-down pattern.
    """
    import hashlib
    import colorsys
    from PIL import Image

    bg_dir = "./backgrounds"
    files = [f for f in os.listdir(bg_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("[!] No background files found in ./backgrounds/")
        return
    chosen_file = random.choice(files)
    path = os.path.join(bg_dir, chosen_file)

    # Load image
    img = Image.open(path).convert("RGB")
    img = img.resize((W - 2*margin, H - 2*margin), Image.LANCZOS)
    pixels = img.load()
    img_w, img_h = img.size

    # Deterministic seed
    if serial_id:
        seed_hash = hashlib.sha3_512(serial_id.encode("utf-8")).digest()
    else:
        seed_hash = seed
    seed_len = len(seed_hash)
    seed_i = 0

    def byte_to_saturation(byte, min_sat=0.25, max_sat=0.75):
        return min_sat + (byte / 255.0) * (max_sat - min_sat)

    def draw_triangle_vector(x0, y0, size, level=1, is_up=True):
        nonlocal seed_i
        if level > hierarchy_levels:
            # Create either upward or downward facing triangle
            if is_up:
                tri = [(x0, y0), (x0 + size, y0), (x0 + size/2, y0 + math.sqrt(3)/2*size)]
            else:
                tri = [(x0, y0 + math.sqrt(3)/2*size), (x0 + size/2, y0), (x0 + size, y0 + math.sqrt(3)/2*size)]
            
            # Sample pixels inside triangle for average color
            min_x = max(int(min(p[0] for p in tri)), 0)
            max_x = min(int(max(p[0] for p in tri)), img_w-1)
            min_y = max(int(min(p[1] for p in tri)), 0)
            max_y = min(int(max(p[1] for p in tri)), img_h-1)
            
            r_total = g_total = b_total = count = 0
            for px in range(min_x, max_x+1):
                for py in range(min_y, max_y+1):
                    r, g, b = pixels[px, py]
                    r_total += r
                    g_total += g
                    b_total += b
                    count += 1
            avg_color = (r_total//count, g_total//count, b_total//count) if count else (200,200,200)

            # Saturation & HLS
            sat_byte = seed_hash[seed_i % seed_len]
            saturation = byte_to_saturation(sat_byte)
            r, g, b = [v/255.0 for v in avg_color]
            h_val, l_val, _ = colorsys.rgb_to_hls(r, g, b)
            r_new, g_new, b_new = colorsys.hls_to_rgb(h_val, l_val, saturation)
            tri_color = f"rgb({int(r_new*255)},{int(g_new*255)},{int(b_new*255)})"

            # Opacity
            op_byte = seed_hash[(seed_i+1) % seed_len]
            opacity = 0.0 + (op_byte / 255.0) * 0.25
            seed_i += 2

            dwg.add(dwg.polygon(points=tri, fill=tri_color, fill_opacity=opacity, stroke="none"))
        else:
            # Subdivide triangle into smaller interlocking triangles
            step = size / 2
            if is_up:
                # For upward triangle, create three smaller triangles (two down, one up)
                draw_triangle_vector(x0, y0, step, level+1, False)  # Bottom-left (down)
                draw_triangle_vector(x0 + step, y0, step, level+1, False)  # Bottom-right (down)
                draw_triangle_vector(x0 + step/2, y0 + math.sqrt(3)/2*step, step, level+1, True)  # Top (up)
            else:
                # For downward triangle, create three smaller triangles (two up, one down)
                draw_triangle_vector(x0, y0 + math.sqrt(3)/2*step, step, level+1, True)  # Top-left (up)
                draw_triangle_vector(x0 + step, y0 + math.sqrt(3)/2*step, step, level+1, True)  # Top-right (up)
                draw_triangle_vector(x0 + step/2, y0, step, level+1, False)  # Bottom (down)

    # Calculate triangle height
    triangle_height = math.sqrt(3)/2 * triangle_size
    
    # Create interlocking pattern across the grid
    for row in range(0, H - 2*margin + int(triangle_height), int(triangle_height)):
        # Alternate starting pattern for each row
        is_up_first = (row // int(triangle_height)) % 2 == 0
        
        for col in range(0, W - 2*margin + triangle_size, triangle_size):
            x = col + margin
            y = row + margin
            
            # Alternate between upward and downward triangles
            is_up = is_up_first if col % (2 * triangle_size) == 0 else not is_up_first
            
            draw_triangle_vector(x, y, triangle_size, 1, is_up)

    print(f"[+] Embedded interlocking vector triangle-pattern background: {chosen_file}")
    return path, seed_hash

def add_fractal_security_pattern(
    dwg: svgwrite.Drawing,
    W: int,
    H: int,
    seed: bytes,
    serial_id: str = None,
    margin: int = 60,
    base_size: int = 128,
    levels: int = 4
):
    """
    Create a recursive fractal security pattern (Sierpiński-inspired).
    At each recursion level, a new shape is chosen deterministically
    (triangle, circle, diamond, hex, square).
    Shapes shrink in size and nest inside each other.
    Colors are sampled from seed/serial encoding.

    Returns:
        (str, bytes): (path identifier string, seed_hash)
    """
    import hashlib, math, colorsys, os

    # Deterministic seed
    if serial_id:
        seed_hash = hashlib.sha3_512(serial_id.encode("utf-8")).digest()
        chosen_file = f"fractal_security_{serial_id}.svg"
    else:
        seed_hash = seed
        chosen_file = "fractal_security_seed.svg"

    # Path placeholder (no real external image like backgrounds, but consistent with your other API)
    path = os.path.join("./backgrounds", chosen_file)

    seed_len = len(seed_hash)
    seed_i = 0

    def next_byte():
        nonlocal seed_i
        b = seed_hash[seed_i % seed_len]
        seed_i += 1
        return b

    def choose_shape():
        shapes = ["triangle", "circle", "diamond", "hex", "square"]
        return shapes[next_byte() % len(shapes)]

    def color_from_seed():
        r, g, b = next_byte(), next_byte(), next_byte()
        sat = 0.5 + (next_byte() / 255.0) * 0.4
        light = 0.4 + (next_byte() / 255.0) * 0.4
        h, _, _ = colorsys.rgb_to_hls(r/255, g/255, b/255)
        rr, gg, bb = colorsys.hls_to_rgb(h, light, sat)
        return f"rgb({int(rr*255)},{int(gg*255)},{int(bb*255)})"

    def draw_shape(cx, cy, size, shape, opacity):
        if shape == "triangle":
            h = math.sqrt(3)/2 * size
            pts = [(cx, cy - 2/3*h), (cx - size/2, cy + h/3), (cx + size/2, cy + h/3)]
            dwg.add(dwg.polygon(points=pts, fill=color_from_seed(), stroke="none", fill_opacity=opacity))
        elif shape == "circle":
            dwg.add(dwg.circle(center=(cx, cy), r=size/2,
                               fill=color_from_seed(), stroke="none", fill_opacity=opacity))
        elif shape == "diamond":
            pts = [(cx, cy - size/2), (cx - size/2, cy), (cx, cy + size/2), (cx + size/2, cy)]
            dwg.add(dwg.polygon(points=pts, fill=color_from_seed(), stroke="none", fill_opacity=opacity))
        elif shape == "hex":
            pts = [(cx + size/2*math.cos(2*math.pi*i/6),
                    cy + size/2*math.sin(2*math.pi*i/6)) for i in range(6)]
            dwg.add(dwg.polygon(points=pts, fill=color_from_seed(), stroke="none", fill_opacity=opacity))
        elif shape == "square":
            pts = [(cx - size/2, cy - size/2), (cx + size/2, cy - size/2),
                   (cx + size/2, cy + size/2), (cx - size/2, cy + size/2)]
            dwg.add(dwg.polygon(points=pts, fill=color_from_seed(), stroke="none", fill_opacity=opacity))

    def recursive_draw(cx, cy, size, level):
        if level > levels or size < 8:
            return
        shape = choose_shape()
        opacity = 0.15 + (next_byte() / 255.0) * 0.35
        draw_shape(cx, cy, size, shape, opacity)
        recursive_draw(cx, cy, size * 0.6, level + 1)

    # Draw central fractal
    recursive_draw(W/2, H/2, base_size, 1)

    print(f"[+] Fractal security pattern embedded: {chosen_file}")
    return path, seed_hash


# -----------------------------
# Helper to hash seed
# -----------------------------
def seed_from_denom_date(denomination: int, date_str: str):
    """
    Generate deterministic seed from denomination exponent + date string.
    """
    seed_input = f"{denomination}_{date_str}".encode("utf-8")
    return int(hashlib.sha256(seed_input).hexdigest(), 16) % (10**8)

# -----------------------------
# Add security background
# -----------------------------
import math, hashlib, colorsys
from datetime import datetime
import svgwrite
def add_vectorized_overlay_from_image(
    dwg: svgwrite.Drawing,
    W: int,
    H: int,
    input_image_path: str,
    seed: bytes,
    serial_id: str = None,
    margin: int = 0,
    pattern_type: str = "hex",  # "hex" or "triangle"
    hex_size: int = 32,
    triangle_size: int = 32,
    opacity_max: float = 0.25
):
    """
    Overlay a fully vectorized pattern (hex or triangle) derived from a background image.
    Colors are sampled from the image, saturation and opacity determined from seed.
    """
    import hashlib, math, colorsys
    from PIL import Image
    import random

    # Load image
    img = Image.open(input_image_path).convert("RGB")
    img = img.resize((W - 2*margin, H - 2*margin), Image.LANCZOS)
    pixels = img.load()
    img_w, img_h = img.size

    # Deterministic seed
    if serial_id:
        seed_hash = hashlib.sha3_512(serial_id.encode("utf-8")).digest()
    else:
        seed_hash = seed
    seed_len = len(seed_hash)
    seed_i = 0

    def byte_to_saturation(byte, min_sat=0.25, max_sat=0.75):
        return min_sat + (byte / 255.0) * (max_sat - min_sat)

    def sample_pixel_color(cx, cy):
        px = min(max(int(cx - margin), 0), img_w-1)
        py = min(max(int(cy - margin), 0), img_h-1)
        r, g, b = pixels[px, py]
        # Slight variation from seed
        if seed_hash:
            v = seed_hash[seed_i % seed_len]
            r = min(255, max(0, r + (v-128)//4))
            g = min(255, max(0, g + (v-128)//4))
            b = min(255, max(0, b + (v-128)//4))
        return r, g, b

    def draw_hex(cx, cy, size):
        nonlocal seed_i
        points = []
        for i in range(6):
            angle = math.pi/3 * i
            x = cx + size * math.cos(angle)
            y = cy + size * math.sin(angle)
            points.append((x, y))
        r, g, b = sample_pixel_color(cx, cy)
        fill_color = f"rgb({r},{g},{b})"
        op_byte = seed_hash[(seed_i+1) % seed_len]
        opacity = (op_byte / 255.0) * opacity_max
        seed_i += 2
        dwg.add(dwg.polygon(points=points, fill=fill_color, stroke="none", fill_opacity=opacity))

    def draw_triangle(cx, cy, size, is_up=True):
        nonlocal seed_i
        h = math.sqrt(3)/2 * size
        if is_up:
            tri = [(cx, cy), (cx + size, cy), (cx + size/2, cy + h)]
        else:
            tri = [(cx, cy + h), (cx + size/2, cy), (cx + size, cy + h)]
        r, g, b = sample_pixel_color(cx, cy)
        fill_color = f"rgb({r},{g},{b})"
        op_byte = seed_hash[(seed_i+1) % seed_len]
        opacity = (op_byte / 255.0) * opacity_max
        seed_i += 2
        dwg.add(dwg.polygon(points=tri, fill=fill_color, stroke="none", fill_opacity=opacity))

    if pattern_type == "hex":
        # Hex grid
        w_step = 3/2 * hex_size
        h_step = math.sqrt(3) * hex_size
        for row in range(0, int(H - 2*margin + h_step), int(h_step)):
            offset = 0 if (row // int(h_step)) % 2 == 0 else hex_size * 3/4
            for col in range(0, int(W - 2*margin + w_step), int(w_step)):
                draw_hex(col + offset + margin, row + margin, hex_size)
    else:
        # Triangle grid
        tri_h = math.sqrt(3)/2 * triangle_size
        for row in range(0, H - 2*margin + int(tri_h), int(tri_h)):
            is_up_first = (row // int(tri_h)) % 2 == 0
            for col in range(0, W - 2*margin + triangle_size, triangle_size):
                is_up = is_up_first if col % (2 * triangle_size) == 0 else not is_up_first
                draw_triangle(col + margin, row + margin, triangle_size, is_up=is_up)

    print(f"[+] Added vector overlay from {input_image_path}")
    return seed_hash

def add_roygbiv_qr_style(dwg: svgwrite.Drawing, W: int, H: int, url: str = "https://linglin.art",
                         stamp_width: int = 40, stamp_height: int = 40, rows: int = 3, side: str = "both"):
    """
    Adds a ROYGBIV pseudo-QR stamp with 'rows' rows, filling stamp_width x stamp_height,
    on the left, right, or both sides of the SVG.
    The color of each bar is deterministic based on the URL.
    """
    colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]
    n_colors = len(colors)
    
    # Deterministic hash from URL
    hash_bytes = hashlib.sha3_512(url.encode("utf-8")).digest()
    
    # Compute columns to fill width
    cols = stamp_width // (stamp_width // rows)
    bar_w = stamp_width / cols
    bar_h = stamp_height / rows

    # Build the grid as a 1D sequence of bars
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
    """
    Adds a hierarchical triangle security pattern with denomination exponent and date text.
    All colors and opacity are derived deterministically from seed or serial_id.
    Triangles are drawn behind the text.
    """
    # --- Deterministic seed ---
    seed_hash = hashlib.sha3_512(serial_id.encode("utf-8")).digest() if serial_id else seed
    seed_len = len(seed_hash)
    seed_i = 0

    # --- Helpers ---
    def byte_to_saturation(byte, min_sat=0.25, max_sat=0.75):
        return min_sat + (byte / 255.0) * (max_sat - min_sat)

    def rgb_to_hex(rgb):
        return "#{:02X}{:02X}{:02X}".format(*rgb)

    # --- Hierarchy level from denomination exponent ---
    if denomination > 0:
        hierarchy_levels = max(1, int(math.log10(denomination)))
    else:
        hierarchy_levels = 1

    # --- Draw triangles recursively ---
    def draw_triangle_svg(x0, y0, size, level=1):
        nonlocal seed_i
        scale_byte = seed_hash[seed_i % seed_len]
        size *= 0.5 + (scale_byte / 255.0)  # 0.5x - 1.5x
        seed_i += 1

        if level > hierarchy_levels:
            tri_up = [(x0, y0), (x0 + size, y0), (x0 + size/2, y0 + math.sqrt(3)/2*size)]
            tri_down = [(x0, y0), (x0 + size/2, y0 - math.sqrt(3)/2*size), (x0 + size, y0)]
            for tri in [tri_up, tri_down]:
                # Color from seed
                sat_byte = seed_hash[seed_i % seed_len]
                saturation = byte_to_saturation(sat_byte)
                seed_i += 1
                hue = (saturation * 360 + denomination) % 360  # slight tint from denomination
                r, g, b = colorsys.hsv_to_rgb(hue/360, 0.2, 1.0)  # light tint
                hex_color = rgb_to_hex((int(r*255), int(g*255), int(b*255)))

                # Opacity
                op_byte = seed_hash[seed_i % seed_len]
                opacity = 0.1 + (op_byte / 255.0) * 0.25
                seed_i += 1

                dwg.add(dwg.polygon(points=tri, fill=hex_color, fill_opacity=opacity))
        else:
            step = size / 3
            for dy in range(3):
                for dx in range(3):
                    draw_triangle_svg(x0 + dx*step, y0 + dy*step, step, level+1)

    # --- Loop over grid to cover full background ---
    h = math.sqrt(3)/2 * base_triangle_size
    for y in range(0, int(H + h), int(base_triangle_size)):
        offset = 0 if (y // h) % 2 == 0 else base_triangle_size // 2
        for x in range(-base_triangle_size, int(W + base_triangle_size), base_triangle_size):
            draw_triangle_svg(x + offset, y, base_triangle_size)

    # --- Add denomination exponent and date text in background ---
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_str = f"Denom Exp: {denomination}   Date: {date_str}"
    dwg.add(dwg.text(
        text_str,
        insert=(margin, H - margin),
        font_size=int(H*0.035),
        fill="#000000",
        fill_opacity=0.25,
        font_family="Daemon Full Working"
    ))

    print(f"[+] Added security background with triangles, denomination exponent, and date.")
    return seed_hash

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


from PIL import Image
import time
from datetime import datetime
def generate_triangle_overlay(W, H, denom: int, seed_hash: bytes, margin=60, base_size=128, levels=4, out_dir="./security_patterns"):
    """
    Generate a fractal triangle overlay SVG for security purposes.
    - W, H: canvas size
    - denom: denomination (used for stroke encoding)
    - seed_hash: bytes object used for deterministic colors
    - margin: optional margin around edges
    - base_size: starting size of the largest triangle
    - levels: recursion depth
    Returns the path of the saved SVG.
    """
    # Make output directory if needed
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Create SVG drawing
    dwg = svgwrite.Drawing(size=(W, H))

    seed_len = len(seed_hash)
    seed_i = 0

    def next_byte():
        nonlocal seed_i
        b = seed_hash[seed_i % seed_len]
        seed_i += 1
        return b

    def choose_shape():
        """Always use triangle for this overlay."""
        return "triangle"

    def color_from_seed():
        """Generate pastel fill color from seed."""
        r = next_byte()
        g = next_byte()
        b = next_byte()
        sat = 0.5 + (next_byte() / 255.0) * 0.4
        light = 0.4 + (next_byte() / 255.0) * 0.4
        h, _, _ = colorsys.rgb_to_hls(r/255, g/255, b/255)
        rr, gg, bb = colorsys.hls_to_rgb(h, light, sat)
        return f"rgb({int(rr*255)},{int(gg*255)},{int(bb*255)})"

    def stroke_from_seed_and_denom():
        """Encode denom + seed hash as stroke color."""
        val = int.from_bytes(seed_hash[:3], "big") + denom
        r = (val >> 16) & 0xFF
        g = (val >> 8) & 0xFF
        b = val & 0xFF
        return f"rgb({r},{g},{b})"

    def draw_triangle(cx, cy, size, opacity=0.2):
        h = math.sqrt(3)/2 * size
        pts = [(cx, cy - 2/3*h), (cx - size/2, cy + h/3), (cx + size/2, cy + h/3)]
        dwg.add(dwg.polygon(points=pts,
                            fill=color_from_seed(),
                            stroke=stroke_from_seed_and_denom(),
                            fill_opacity=opacity,
                            stroke_opacity=0.5,
                            stroke_width=1))

    def recursive_draw(cx, cy, size, level):
        if level > levels or size < 8:
            return
        draw_triangle(cx, cy, size, opacity=0.15 + next_byte()/255*0.35)
        recursive_draw(cx, cy, size * 0.6, level + 1)

    # Draw fractal at center
    recursive_draw(W/2, H/2, base_size, 1)

    # Safe filename
    safe_seed_str = binascii.hexlify(seed_hash).decode("ascii")
    outfile = os.path.join(out_dir, f"triangle_overlay_{denom}_{safe_seed_str}.svg")

    # Save SVG
    dwg.saveas(outfile)
    print(f"[+] Triangle overlay saved: {outfile}")
    return outfile
def generate_timestamp_ms():
    """
    Generate current timestamp in milliseconds with microsecond precision.
    Returns integer representing milliseconds since epoch.
    """
    return int(time.time() * 1000)

# Alternative version that includes microseconds for even more precision:
def generate_timestamp_ms_precise():
    """
    Generate timestamp with microsecond precision.
    Returns integer representing milliseconds.microseconds.
    """
    now = datetime.now()
    return int(now.timestamp() * 1000) + now.microsecond // 1000

# If you want a formatted string version:
def generate_timestamp_ms_formatted():
    """
    Generate formatted timestamp string with milliseconds precision.
    Returns string in format: YYYYMMDD-HHMMSS-SSS
    """
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"
def load_and_prepare_image(path: str, max_dim: int) -> Image.Image:
    """
    Load an image, convert to RGB, and resize to a square of max_dim x max_dim.
    Preserves aspect ratio with white padding.
    """
    from PIL import Image

    im = Image.open(path).convert("RGB")
    orig_w, orig_h = im.size

    scale = min(max_dim / orig_w, max_dim / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
    final_im = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    offset_x = (max_dim - new_w) // 2
    offset_y = (max_dim - new_h) // 2
    final_im.paste(im_resized, (offset_x, offset_y))

    return final_im
import colorsys

def hsl_to_rgb_string(h, s, l):
    """Convert HSL (0–360, 0–100, 0–100) to rgb(r,g,b) CSS string."""
    h = h / 360.0
    s = s / 100.0
    l = l / 100.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

def denomination_color(denom: int) -> str:
    """
    Returns a light ROYGBIV hex color based on the denomination.
    Maps 1 → Red, 100,000,000 → Violet on a log scale.
    """
    # Clamp between 1 and 100,000,000
    denom = max(1, min(100_000_000, denom))

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

def get_random_background(bg_dir="./backgrounds"):
    files = [
        os.path.join(bg_dir, f)
        for f in os.listdir(bg_dir)
        if os.path.isfile(os.path.join(bg_dir, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not files:
        raise FileNotFoundError(f"No background images found in {bg_dir}")
    return random.choice(files)

def generate_security_pattern():
    bg_path = get_random_background("./backgrounds")
    print(f"[+] Using background → {bg_path}")
    bg = Image.open(bg_path).convert("RGBA")
    return bg

def vectorize_image_by_color(img_path, max_colors=64):
    """
    Vectorize image by color clustering (like Inkscape Trace Bitmap).
    Returns a list of (color, polygon_points) groups.
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)

    # KMeans clustering to reduce color complexity
    kmeans = KMeans(n_clusters=max_colors, random_state=42)
    labels = kmeans.fit_predict(flat)
    palette = kmeans.cluster_centers_.astype(int)

    traced_polys = []
    label_img = labels.reshape(h, w)

    for idx, color in enumerate(palette):
        mask = (label_img == idx).astype(np.uint8) * 255
        pil_mask = Image.fromarray(mask)
        contours = pil_mask.getbbox()
        if not contours:
            continue

        # Approx polygon by grid (simplified)
        ys, xs = np.where(label_img == idx)
        pts = list(zip(xs.tolist(), ys.tolist()))
        if len(pts) > 50:  # reduce density
            pts = pts[::len(pts)//50]

        traced_polys.append(((color[0], color[1], color[2]), pts))

    return traced_polys, (w, h)

from skimage import color, segmentation, measure, util
import re
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
        response = requests.post("https://192.168.0.3:3014/sdapi/v1/txt2img", json=payload, timeout=120)
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
def generate_sd_background_from_metadata(encoded_seed, name="", width: int = 512, height: int = 512, 
                                       save_path: str = "./backgrounds", denom: int = None):
    """
    Generate background using metadata-encoded prompt
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
        "steps": 25,
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
            
            # Generate filename with metadata hash
            metadata_hash = hashlib.md5(encoded_seed.encode()).hexdigest()[:8]
            filename = f"bg_{metadata_hash}_{int(time.time())}.png"
            filepath = os.path.join(save_path, filename)
            
            image.save(filepath)
            print(f"[+] Generated metadata-based background: {filepath}")
            return filepath
        
    except Exception as e:
        print(f"[!] Error generating background: {e}")
        return None
# Modified add_vectorized_background to accept encoded seed
def add_vectorized_background(dwg, W, H, seed_text="", bg_dir="./backgrounds", margin=60, n_segments=1024, background_prompt=""):
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
        seed_text=seed_text
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


def generate_sd_background(prompt, width=512, height=512, save_path="./backgrounds", seed_text=""):
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
        "steps": 25,
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
            
            # Generate filename with prompt hash
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            filename = f"bg_{prompt_hash}_{int(time.time())}.png"
            filepath = os.path.join(save_path, filename)
            
            image.save(filepath)
            print(f"[+] Generated background: {filepath}")
            return filepath
        
    except Exception as e:
        print(f"[!] Error generating background: {e}")
        return None



def add_random_background_vectorized(dwg, W:int, H:int, margin:int=60, max_colors:int=12):
    """
    Load a random background from ./backgrounds/, vectorize it,
    scale to fit with margin, and overlay as SVG polygons.
    """
    bg_dir = "./backgrounds"
    files = [f for f in os.listdir(bg_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("[!] No background files found in ./backgrounds/")
        return

    chosen = random.choice(files)
    path = os.path.join(bg_dir, chosen)

    print(f"[+] Vectorizing background: {chosen}")
    polys, (w, h) = vectorize_image_by_color(path, max_colors=max_colors)

    scale_x = (W - 2*margin) / w
    scale_y = (H - 2*margin) / h
    scale = min(scale_x, scale_y)

    group = dwg.g(opacity=0.6)  # semi-transparent overlay
    for color, pts in polys:
        if len(pts) < 3:
            continue
        scaled_pts = [(x*scale+margin, y*scale+margin) for (x,y) in pts]
        fill = svgwrite.rgb(color[0], color[1], color[2])
        group.add(dwg.polygon(points=scaled_pts, fill=fill, stroke="none"))

    dwg.add(group)
    return group
import qrcode
from PIL import Image, ImageDraw
import math
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    print("QR code generation not available. Install qrcode[pil] package.")
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
def generate_fantasy_banknote(seed_text: str, input_image_path: str, outfile_svg: str,
                               width_mm: float = 160.0, height_mm: float = 60.0,
                               title: str = "灵国国库", subtitle: str = "天圆地方", serial_id: str = "SERIALID", timestamp:str = "TIMESTAMP",
                               denomination: str = "100 卢纳币", specimen: bool = True,
                               fonts = {}):
    font_main = "FengGuangMingRui"
    font_numeric = "Karamuruh"
    timestamp_ms = timestamp or generate_timestamp_ms()
    serial_id = serial_id or generate_serial_id_with_checksum()
    W = mm_to_px(width_mm)
    H = mm_to_px(height_mm)
    dwg = svgwrite.Drawing(outfile_svg, size=(W,H), viewBox=f"0 0 {W} {H}")
    
    # Embed fonts
    #from fontTools.ttLib import TTFont
    #embed_font(dwg, TTFont("./fonts/Daemon Full Working.otf"), "Daemon Full Working")
    #embed_font(dwg, TTFont("./fonts/FengGuangMingRui.ttf"), "FengGuangMingRui")
    
    denom_value = denom_to_int(denomination)
    denom_exponent = int(round(math.log10(denom_value))) if denom_value > 0 else 0
    seed_text=seed_text
    # Generate seed and background
    seed_hash = sha3_512_salted(seed_text, serial_id)
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill=denomination_color(denom=denom_value)))
    
    # Add QR border first and get border info for other layers
    border_info = add_qr_like_border(dwg, seed_text, W, H, serial_id, timestamp_ms)
    
    #background = generate_triangle_overlay(W=W, H=H,denom=denom_value, seed_hash=seed_hash, margin=60, base_size=128, levels=4, out_dir="./security_overlays")
    add_vectorized_background(dwg=dwg, W=W, H=H, seed_text=seed_text, bg_dir="./backgrounds", margin=60, n_segments=1024, 
                              background_prompt=generate_kawaii_mural_from_background(denomination=denom_exponent, filename="background_prompt.txt"))
    #add_random_background_vectorized(dwg=dwg, seed_text=seed_text, width_mm=W, height_mm=H, serial_id=serial_id, time=timestamp_ms)
    print("Generated:", path)

    # Add background image using the returned dimensions
    
    
    #background3 = add_random_background_png(dwg, W, H, seed=seed_hash, serial_id=serial_id, margin=60, triangle_size=128, hierarchy_levels=1)
    #background4 = add_random_background_png(dwg, W, H, seed=seed_hash, serial_id=serial_id, margin=60, triangle_size=256, hierarchy_levels=1)

    
    
    seed_hex = to_bytes(denom_exponent).hex()
    #generate_security_pattern()

    #add_security_background(dwg,W=W, H=H, denomination=denom_exponent, seed=seed_hash, serial_id=serial_id, margin=60, base_triangle_size=256, hierarchy_levels=4)

    # Add frame and microgrid with deterministic encoding
    add_subtle_frame_and_microgrid(dwg, W, H, border_info, denom_value, timestamp_ms, to_bytes(seed_hash))
    
    # Add diamond border using the returned dimensions
    add_decorative_border(dwg, W, H, border_info, denom_value, timestamp_ms)

    # Load and process center image
    im = None
    if input_image_path and os.path.exists(input_image_path):
        im = Image.open(input_image_path).convert("RGB")
    
    center_px_radius = int(min(W,H)*0.32)
    cx, cy = W//2, H//2
    left_cx = int(W*0.16)
    right_cx = int(W*0.84)
    cy = H//2
    d_color = denomination_to_color(denom_exponent)
    small_radius = min(W,H)*0.25
    text_left = f"{str(seed_text)}"
    text_right= f"{str(serial_id)}"
    add_text_seal(dwg, cy=cy, radius=small_radius*0.65, text_left=text_left, text_right=text_right, denom_color=d_color,inner_text="日", include_datetime=True, seed_text=seed_text, serial_id=serial_id, canvas_width=W)
    #add_text_seal(dwg, cx=right_cx, cy=cy, radius=small_radius*0.65, text=text_right, denom_color=d_color,inner_text="月", include_datetime=True, seed_text=seed_text, serial_id=serial_id)
    add_secondary_ring(dwg, cx, cy, radius=center_px_radius*0.88, seed=to_bytes(seed_hash), segments=360, d_color=d_color)

    # Add center elements
    add_center_seal(dwg, im, cx, cy, center_px_radius*1.2)
    
    # Add text elements
    add_center_text(dwg, W, H, title, subtitle, denom_color=denomination_to_color(denom_exponent))
    
    # Add corner elements
    add_functional_corner_decorations(dwg, W, H, denomination, timestamp_ms, serial_id)
    add_corner_denoms(dwg, W, H, str(denom_value))
    
    # Add Chinese microprint
    chinese_value = number_to_chinese(denom_value)
    add_chinese_microprint(
        dwg, cx, cy, radius=int(center_px_radius*0.7),
        text=f"{chinese_value} 卢纳币",
        repetitions=16
    )
    qr_url=f"https://bank.linglin.art/verify/{serial_id}"
    
    add_roygbiv_qr_style(dwg, W=W, H=H, url=qr_url, stamp_width=60, stamp_height=60, rows=6)
    
    matrix = segno.make(qr_url).matrix  # matrix is a list of lists of booleans

    # Add the Aztec/QR directly onto the canvas
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
    # Add specimen text if needed
    if specimen:
        dwg.add(dwg.text("SPECIMEN", insert=(W*0.5,H*0.92),
                         font_size=int(H*0.08), fill="#333", font_family="monospace", 
                         text_anchor="middle", opacity=0.75))
    

    dwg.save()
    print(f"[+] Saved: {outfile_svg}")
from PIL import ImageStat
# ROYGBIV palette
COLORS = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]
from aztec import aztec_matrix_from_segno, build_colored_aztec_svg
import segno
import tempfile
import svgwrite
import segno

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
    """
    Add two colored Aztec SVGs to an existing svgwrite.Drawing.
    
    - dwg: existing svgwrite.Drawing
    - W, H: canvas width & height
    - cx, cy: portrait center
    - qr_url: URL / data to encode
    - scale, margin, style: parameters for Aztec generation
    - offset: horizontal distance from portrait center
    - size: target size of QR in px
    """
    # Build matrix and temporary SVG for Aztec
    matrix = aztec_matrix_from_segno(qr_url)
    tmp_svg_path = tempfile.mktemp(suffix=".svg")
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
def denom_to_int(denom_str: str) -> int:
    """
    Convert a denomination string like "100 yuan" to an integer 100.
    Ignores non-digit characters.
    """
    import re
    match = re.search(r'\d+', denom_str)
    if match:
        return int(match.group())
    raise ValueError(f"No numeric part found in denomination '{denom_str}'")
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

import svgwrite
import math
    
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
import math

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

# Add border
# ----------------------
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

        offset = value + (denom_value % 97)

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
from stable_diffussion_api import StableDiffusionClient
def add_random_background_vectorized(dwg, W:int, H:int, margin:int=60, max_colors:int=12):
    """
    Generate a new background using Stable Diffusion,
    save it in ./backgrounds/, vectorize it,
    scale to fit with margin, and overlay as SVG polygons.
    """
    bg_dir = "./backgrounds"
    os.makedirs(bg_dir, exist_ok=True)

    # Generate a new background
    client = StableDiffusionClient()
    client.generate_background()  # saves into ./backgrounds/background_0.png

    # Pick the most recent background file
    files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("[!] No background files found in ./backgrounds/")
        return

    chosen = max(files, key=os.path.getctime)  # latest generated
    print(f"[+] Vectorizing background: {os.path.basename(chosen)}")

    # Vectorize
    polys, (w, h) = vectorize_image_by_color(chosen, max_colors=max_colors)

    # Scale to fit
    scale_x = (W - 2*margin) / w
    scale_y = (H - 2*margin) / h
    scale = min(scale_x, scale_y)

    # Draw polygons
    group = dwg.g(opacity=0.6)  # semi-transparent overlay
    for color, pts in polys:
        if len(pts) < 3:
            continue
        scaled_pts = [(x*scale+margin, y*scale+margin) for (x,y) in pts]
        fill = svgwrite.rgb(color[0], color[1], color[2])
        group.add(dwg.polygon(points=scaled_pts, fill=fill, stroke="none"))

    dwg.add(group)
    return group
if __name__ == "__main__":
    import argparse
    import os
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Fantasy banknote generator")
    parser.add_argument("seed_text", type=str, help="Seed text or name for the note")
    parser.add_argument("input_image", type=str, help="Input image path")
    parser.add_argument("--outfile", type=str, default="banknote.svg", help="Base output SVG file")
    parser.add_argument("--specimen", action="store_true", help="Add SPECIMEN overlay")
    parser.add_argument("--copies", type=int, default=1, help="Number of distinct notes to generate")
    parser.add_argument("--yen_model", action="store_true", help="Use 1-100,000,000 denominations")
    parser.add_argument("--serial_id", type=str, help="Serial ID")
    parser.add_argument("--timestamp", type=str, help="Timestamp String")
    
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
            timestamp = args.timestamp
            base, ext = os.path.splitext(args.outfile)
            # Filename format: seed_denomination_datetime.svg
            # For FRONT
            outfile_svg = f"./images/{new_seed}/{denom}/{new_seed}_-_{denom}_-_{timestamp}_FRONT.svg"
            outfile_dir = os.path.dirname(outfile_svg)
            os.makedirs(outfile_dir, exist_ok=True)





            denomination_str = f"{denom} 卢纳币"

            generate_fantasy_banknote(
                seed_text=f"{new_seed}_{i}",  # keep unique seed for generation
                input_image_path=args.input_image,
                outfile_svg=outfile_svg,
                specimen=args.specimen,
                denomination=denomination_str,
                fonts=fonts,
                serial_id=args.serial_id,
                timestamp=timestamp
                
            )

