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
import os
import time
import tqdm
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

    hash_obj = hashlib.sha3_512()
    if salt is not None:
        hash_obj.update(str(salt).encode("utf-8"))
    hash_obj.update(str(s).encode("utf-8"))
    return hash_obj.digest()




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
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', s)
    if chinese_chars:
        chinese_unicode = ''.join(str(ord(ch)) for ch in chinese_chars)
        latin_part = re.sub(r'[\d\W_]+', '', s, flags=re.UNICODE)
        latin_part = ''.join(ch for ch in latin_part if ch not in chinese_chars)
        return latin_part + chinese_unicode
    else:
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
    time_seed = int(timestamp_ms) % 10000
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

from PIL import Image, ImageDraw, ImageFont







# -----------------------------
# Add security background
# -----------------------------
import math, hashlib, colorsys
from datetime import datetime
import svgwrite


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


import re

# Modified add_vectorized_background to accept encoded seed
def add_vectorized_background(dwg, W, H, seed_text="", bg_dir="./backgrounds", margin=60, n_segments=1024, background_prompt="", denomination=None):
    """
    Enhanced version that generates background using prompt from background_prompt.txt
    with retry logic for file synchronization
    """
    import os
    import glob
    import hashlib
    import random
    import numpy as np
    from PIL import Image
    from skimage import color, segmentation, measure
    import svgwrite
    import time
    
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
    
    # --- ADD RETRY LOGIC HERE ---
    max_retries = 10
    retry_delay = 0.5  # seconds
    img = None
    
    for attempt in range(max_retries):
        try:
            print(f"[+] Attempt {attempt + 1}/{max_retries} to load background: {background_path}")
            
            # Try to open and verify the image
            with Image.open(background_path) as test_img:
                test_img.verify()  # This checks if the image is complete
            
            # If verify succeeded, reopen for actual use
            img = Image.open(background_path).convert("RGB")
            break  # Success!
            
        except (IOError, SyntaxError, Exception) as e:
            if attempt == max_retries - 1:
                print(f"[!] Failed to load background after {max_retries} attempts: {e}")
                # Fallback to simple background
                dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill='#f0f0f0'))
                return
            else:
                print(f"[!] Background not ready (attempt {attempt + 1}), waiting {retry_delay}s...")
                time.sleep(retry_delay)
    
    # If we still don't have an image after retries, use fallback
    if img is None:
        print("[!] Could not load background image, using fallback")
        dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill='#f0f0f0'))
        return
    
    # Continue with the original vectorization logic
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


def generate_sd_background(prompt, width=1600, height=600, save_path="./backgrounds", seed_text="", denomination=None):
    """
    Generate background using Stable Diffusion API with the given prompt
    Includes denomination-specific filenames to prevent race conditions
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
            
            # Include denomination in filename to prevent race conditions
            if denomination is not None:
                denom_str = f"d{denomination}_"
            else:
                denom_str = ""
                
            # Include seed text for additional uniqueness if provided
            seed_str = f"_{hashlib.md5(seed_text.encode()).hexdigest()[:4]}" if seed_text else ""
            
            filename = f"bg_{denom_str}{prompt_hash}{seed_str}_{int(time.time())}.png"
            filepath = os.path.join(save_path, filename)
            
            image.save(filepath)
            print(f"[+] Generated background: {filepath}")
            return filepath
        
    except Exception as e:
        print(f"[!] Error generating background: {e}")
        return None


import qrcode
from PIL import Image, ImageDraw
import math
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    print("QR code generation not available. Install qrcode[pil] package.")



def generate_fantasy_banknote(seed_text: str, input_image_path: str, outfile_svg: str,
                               width_mm: float = 160.0, height_mm: float = 60.0,
                               title: str = "灵国国库", subtitle: str = "天圆地方", serial_id: str = "SERIALID", timestamp:str = "TIMESTAMP",
                               denomination: str = "100 卢纳币", specimen: bool = True,
                               fonts = {}):
    timestamp_ms = timestamp or generate_timestamp_ms_precise()
    serial_id = serial_id or generate_serial_id_with_checksum()
    W = mm_to_px(width_mm)
    H = mm_to_px(height_mm)
    dwg = svgwrite.Drawing(outfile_svg, size=(W,H), viewBox=f"0 0 {W} {H}")
    
    denom_value = denom_to_int(denomination)
    denom_exponent = int(round(math.log10(denom_value))) if denom_value > 0 else 0
    seed_text=seed_text
    seed_hash = sha3_512_salted(seed_text, serial_id)
    dwg.add(dwg.rect(insert=(0,0), size=(W,H), fill=denomination_color(denom=denom_value)))
    
    border_info = add_qr_like_border(dwg, seed_text, W, H, serial_id, timestamp_ms)
    
    add_vectorized_background(dwg=dwg, W=W, H=H, seed_text=seed_text, bg_dir="./backgrounds", margin=60, n_segments=1024, 
                              background_prompt=generate_kawaii_mural_from_background(denomination=denom_exponent, filename="background_prompt.txt"), denomination=denomination)
    print("Generated:", path)
    
    add_subtle_frame_and_microgrid(dwg, W, H, border_info, denom_value, timestamp_ms, to_bytes(seed_hash))
    add_decorative_border(dwg, W, H, border_info, denom_value, timestamp_ms)
    im = None
    if input_image_path and os.path.exists(input_image_path):
        im = Image.open(input_image_path).convert("RGB")
    
    center_px_radius = int(min(W,H)*0.32)
    cx, cy = W//2, H//2
    cy = H//2
    d_color = denomination_to_color(denom_exponent)
    small_radius = min(W,H)*0.25
    text_left = f"{str(seed_text)}"
    text_right= f"{str(serial_id)}"
    add_text_seal(dwg, cy=cy, radius=small_radius*0.65, text_left=text_left, text_right=text_right, denom_color=d_color,inner_text="日", include_datetime=True, seed_text=seed_text, serial_id=serial_id, canvas_width=W)
    add_secondary_ring(dwg, cx, cy, radius=center_px_radius*0.88, seed=to_bytes(seed_hash), segments=360, d_color=d_color)
    add_center_seal(dwg, im, cx, cy, center_px_radius*1.2)
    add_center_text(dwg, W, H, title, subtitle, denom_color=denomination_to_color(denom_exponent))
    add_functional_corner_decorations(dwg, W, H, denomination, timestamp_ms, serial_id)
    add_corner_denoms(dwg, W, H, str(denom_value))
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





import svgwrite
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
    ts = datetime.datetime.fromtimestamp(float(timestamp_ms) / 1000.0)
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


def generate_single_banknote(seed_text, input_image_path, single_denom, outfile=None, 
                           specimen=False, serial_id=None, timestamp=None):
    """
    Generate a single banknote with a specific denomination.
    
    Args:
        seed_text: Seed text or name for the note
        input_image_path: Input image path
        single_denom: Specific denomination to generate (e.g., 100)
        outfile: Output SVG file (default: auto-generated)
        specimen: Add SPECIMEN overlay
        serial_id: Serial ID
        timestamp: Timestamp String
    
    Returns:
        Path to the generated SVG file
    """
    # Set default outfile if not provided
    if outfile is None:
        timestamp_str = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        outfile = f"./images/{seed_text}/{single_denom}/{seed_text}_-_{single_denom}_-_{timestamp_str}_FRONT.svg"
    
    # Create output directory
    outfile_dir = os.path.dirname(outfile)
    os.makedirs(outfile_dir, exist_ok=True)
    
    # Load fonts
    fonts_obj = load_fonts("./fonts")
    
    # Generate the single banknote
    denomination_str = f"{single_denom} 卢纳币"
    
    generate_fantasy_banknote(
        seed_text=seed_text,
        input_image_path=input_image_path,
        outfile_svg=outfile,
        specimen=specimen,
        denomination=denomination_str,
        fonts=fonts_obj,
        serial_id=serial_id,
        timestamp=timestamp
    )
    
    print(f"[+] Single bill generated: {outfile}")
    return outfile

def generate_multiple_banknotes(seed_text, input_image_path, copies=1, yen_model=False, 
                              specimen=False, serial_id=None, timestamp=None):
    """
    Generate multiple banknotes with different denominations.
    
    Args:
        seed_text: Seed text or name for the note
        input_image_path: Input image path
        copies: Number of distinct notes to generate
        yen_model: Use 1-100,000,000 denominations
        specimen: Add SPECIMEN overlay
        serial_id: Serial ID
        timestamp: Timestamp String
    
    Returns:
        List of paths to generated SVG files
    """
    # Generate denominations
    if yen_model:
        base_denoms = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
        denominations = base_denoms[:9]  # top 9 denominations
    else:
        denominations = [100 * (i + 1) for i in range(9)]  # default 9 denominations

    fonts_obj = load_fonts("./fonts")
    generated_files = []

    for i in tqdm(range(copies), desc="Generating banknotes"):
        new_seed = seed_text  # no _i prefix in filenames

        for denom in denominations:
            timestamp_str = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
            # Filename format: seed_denomination_datetime.svg
            outfile_svg = f"./images/{new_seed}/{denom}/{new_seed}_-_{denom}_-_{timestamp_str}_FRONT.svg"
            outfile_dir = os.path.dirname(outfile_svg)
            os.makedirs(outfile_dir, exist_ok=True)

            denomination_str = f"{denom} 卢纳币"

            generate_fantasy_banknote(
                seed_text=f"{new_seed}_{i}",  # keep unique seed for generation
                input_image_path=input_image_path,
                outfile_svg=outfile_svg,
                specimen=specimen,
                denomination=denomination_str,
                fonts=fonts_obj,
                serial_id=serial_id,
                timestamp=timestamp_str
            )
            
            generated_files.append(outfile_svg)

    return generated_files

def single_bill_run():
    """
    Command-line wrapper function for single bill generation.
    """
    parser = argparse.ArgumentParser(description="Generate a single fantasy banknote with specific denomination")
    parser.add_argument("seed_text", type=str, help="Seed text or name for the note")
    parser.add_argument("input_image", type=str, help="Input image path")
    parser.add_argument("--single_denom", type=int, required=True, help="Specific denomination to generate (e.g., 100)")
    parser.add_argument("--outfile", type=str, default=None, help="Output SVG file (default: auto-generated)")
    parser.add_argument("--specimen", action="store_true", help="Add SPECIMEN overlay")
    parser.add_argument("--serial_id", type=str, help="Serial ID")
    parser.add_argument("--timestamp", type=str, help="Timestamp String")
    
    args = parser.parse_args()
    
    generate_single_banknote(
        seed_text=args.seed_text,
        input_image_path=args.input_image,
        single_denom=args.single_denom,
        outfile=args.outfile,
        specimen=args.specimen,
        serial_id=args.serial_id,
        timestamp=args.timestamp
    )

def multi_bill_run():
    """
    Command-line wrapper function for multiple bill generation.
    """
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

    generate_multiple_banknotes(
        seed_text=args.seed_text,
        input_image_path=args.input_image,
        copies=args.copies,
        yen_model=args.yen_model,
        specimen=args.specimen,
        serial_id=args.serial_id,
        timestamp=args.timestamp
    )

# Main execution
if __name__ == "__main__":
    import sys
    
    # Check if --single_denom flag is present to use the single bill mode
    if "--single_denom" in sys.argv:
        single_bill_run()
    else:
        multi_bill_run()