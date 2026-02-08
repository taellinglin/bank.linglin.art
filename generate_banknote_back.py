COLORS = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]

def denomination_to_color(denom_exponent: int) -> str:
    """
    Return a visually distinct color for each denomination exponent (0-based).
    Uses a fixed color palette. If denom_exponent is out of range, clamps to valid range.
    Args:
        denom_exponent (int): The exponent/index of the denomination (e.g. 0 for 1, 1 for 10, ...)
    Returns:
        str: Hex color string (e.g. '#FF0000')
    """
    idx = max(0, min(denom_exponent, len(COLORS) - 1))
    return COLORS[idx]

import string
import os
from datetime import datetime
# EisenScript template variable replacement utility
def replace_eisenscript_variables(script: str, context: dict) -> str:
    """
    Replace variables like $denomination in EisenScript templates.
    Context must include 'denomination'.
    """
    if not script:
        return script
    # Jinja2でテンプレート展開
    try:
        from jinja2 import Template
        print("[DEBUG] Jinja2 context before render:", context)
        template = Template(script)
        rendered = template.render(**context)
        # rendered内に未展開の{{denomination}}が残っていれば警告
        if "{{denomination}}" in rendered:
            print("[ERROR] Jinja2 failed to substitute {{denomination}}! Context:", context)
        return rendered
    except Exception as e:
        print(f"[ERROR] Jinja2 template render failed: {e}")
        return script

# --- EisenScript overlay loader for back side ---
def load_eisen_overlay(path: str, context: dict) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        script = f.read()
    return replace_eisenscript_variables(script, context)

import string
import os
import os
import sys
# --- Patch: Prefer .venv site-packages ---
venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_site):
    sys.path.insert(0, venv_site)
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
        response = requests.post("http://127.0.0.1:7777/sdapi/v1/txt2img", json=payload, timeout=120)
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

    # denominationがstrやテンプレート変数の場合は必ずint化、失敗時は1
    import re
    if isinstance(denomination, str):
        m = re.search(r'(\d+)', denomination)
        if m:
            denomination = int(m.group(1))
        else:
            denomination = 1

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
        response = requests.post("http://127.0.0.1:7777/sdapi/v1/txt2img", json=payload, timeout=120)
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
    if denom_str is None:
        raise ValueError("Denomination is None")
    if isinstance(denom_str, (int, float)):
        return int(denom_str)
    match = re.search(r'\d+', str(denom_str))
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
        symbol = datetime.now().strftime("%Y%m%d%H%M%S")

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
def generate_backside_svg(
    outfile: str,
    denomination: int,
    title_text: str,
    phrase_text: str,
    size_px: Tuple[int, int],
    serial_id: str = None,
    timestamp_ms: str = None,
    seed_text: str = "",
    progress_callback=None
):
    # --- Add serials and banknote to DB ---
    # DB・mempool登録処理はgenerate.py等で一元管理するため、ここでは何もしない
    import re
    W, H = size_px
    # Windows用ファイル名サニタイズ
    if outfile:
        outdir, outbase = os.path.split(outfile)
        safe_base = re.sub(r'[<>:"/\\|?*]', '_', outbase)
        outfile = os.path.join(outdir, safe_base)
    # denominationがテンプレート変数の場合は必ず1にフォールバック
    import re
    denom_val = denomination
    if isinstance(denomination, str):
        m = re.search(r'(\d+)', denomination)
        if m:
            denom_val = int(m.group(1))
        else:
            denom_val = 1  # テンプレート変数や空文字列の場合は1
    try:
        denom_exp = int(math.log10(denom_val)) if denom_val > 0 else 0
        denom_exponent = int(round(math.log10(denom_val))) if denom_val > 0 else 0
    except Exception:
        denom_exp = 0
        denom_exponent = 0
    timestamp = timestamp_ms or generate_timestamp_ms_precise()
    serial_id = serial_id or generate_serial_id_combined()
    denom_value = denom_val
    # contextのdenominationは必ずint値をstr化したものを渡す（テンプレート変数が残らないように）
    # denominationは必ず数値文字列化し、テンプレート変数が残っていれば即例外
    context = {
        "denomination": str(int(denom_value)),
        "denomination_label": f"{denom_val} 卢纳币",
        "serial": serial_id,
        "title": title_text,
        "subtitle": phrase_text,
        "username": seed_text,
        "user_id": "",
        # Add more fields as needed
    }
    if "{{" in context["denomination"] or "}}" in context["denomination"]:
        raise ValueError(f"[FATAL] Context['denomination']にテンプレート変数が残っています: {context['denomination']}")
    print("[DEBUG] EisenScript context for back side:", context)
    # Load overlays (prefix, user, suffix)
    # EisenScriptはDB Settingsからのみ取得
    from generate_banknote import get_eisenscript_from_db
    merged_eisen = get_eisenscript_from_db('back', context)

    # --- SVG generation and DB registration only after SVG is written ---
    def on_save_complete(svg_overlay_path):
        """
        SVGファイルの保存が完了したタイミングで呼ばれるコールバック。
        ここでDB登録や後処理、SVGロード処理を行う。
        """
        # バックSVG専用のロード処理をここで呼ぶ（例: load_back_svg）
        if 'load_back_svg' in globals():
            load_back_svg(svg_overlay_path)
        try:
            from app import app
            from models import db, Banknote, SerialNumber, User
            import datetime
            with app.app_context():
                user = User.query.first()
                if not user:
                    print('[!] No user found for banknote registration')
                    return
                existing = Banknote.query.filter_by(serial_number=serial_id).first()
                if existing:
                    print(f'[!] Banknote already exists for serial: {serial_id}')
                    return
                banknote = Banknote(
                    user_id=user.id,
                    serial_number=serial_id,
                    seed_text=seed_text,
                    denomination=str(denomination),
                    side='back',
                    svg_path=outfile,
                    is_public=True,
                    is_verified=False,
                    verification_status='pending',
                    created_at=datetime.datetime.utcnow()
                )
                db.session.add(banknote)
                db.session.commit()
                serial = SerialNumber(
                    serial=serial_id,
                    user_id=user.id,
                    banknote_id=banknote.id,
                    is_active=True,
                    is_mined=False,
                    created_at=datetime.datetime.utcnow()
                )
                db.session.add(serial)
                db.session.commit()
                print(f'[+] Registered Banknote/Serial: {serial_id}')
        except Exception as e:
            print(f'[!] DB/mempool registration error (in callback): {e}')

    # --- LunaMint EisenScript to SVG overlay ---
    svg_saved = False
    if merged_eisen.strip():
        try:
            from lunamint.scripting import render_script_to_svg_html
            from generate import embed_fonts_in_svg_file, resolve_font_dir
            import tempfile
            from pathlib import Path
            import shutil
            out_path = Path(outfile)
            outdir = out_path.parent
            if not outdir.exists():
                outdir.mkdir(parents=True, exist_ok=True)
            render_script_to_svg_html(merged_eisen, out_path)
            embed_fonts_in_svg_file(out_path, resolve_font_dir())
            svg_saved = True
            if out_path.exists():
                print(f"[+] Saved BACK SVG: {outfile}")
                on_save_complete(str(out_path))
            else:
                print(f"[!] BACK SVG was not saved as expected: {outfile}")
        except Exception as e:
            print(f"[!] Failed to apply LunaMint EisenScript overlay or copy SVG: {e}")

    # If not handled above, try to save with svgwrite.Drawing if available
    if not svg_saved:
        try:
            outdir = os.path.dirname(outfile)
            if outdir and not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)
            if 'dwg' in locals() and hasattr(dwg, 'saveas'):
                dwg.saveas(outfile)
                if os.path.exists(outfile):
                    print(f"[+] Saved BACK SVG: {outfile}")
                    on_save_complete(outfile)
                else:
                    print(f"[!] BACK SVG was not saved as expected: {outfile}")
            else:
                print(f"[!] No SVG object found to save for BACK: {outfile}")
        except Exception as e:
            print(f"[!] Exception during BACK SVG save/log: {e}")



from typing import Tuple
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


import glob
# def add_vectorized_background(...) is disabled. All background drawing is now handled by EisenScript only.

# def generate_sd_background(...) is disabled. All background drawing is now handled by EisenScript only.

# def generate_sd_background_from_metadata(...) is disabled. All background drawing is now handled by EisenScript only.
# def add_security_pattern_overlay(...) is disabled. All security pattern overlays are now handled by EisenScript only.

# def add_fallback_security_pattern(...) is disabled. All fallback security patterns are now handled by EisenScript only.
# def fractal_stamp(...) is disabled. All fractal and microprint backgrounds are now handled by EisenScript only.

def run_single_denomination(outdir: str = ".", base_name: str = "banknote", denomination: int = 1, 
                           width_mm: float = 160.0, height_mm: float = 60.0,
                           title_text: str = "灵国国库", phrase_text: str = "灵之意志，天下共识", seed_text: str = "Username", serial_id: str = "SNB-", timestamp: str = None,
                           png: bool = False, dpi: float = 300.0, bg_image: str = None,
                           progress_callback=None):
    # Update global DPI
    global MM_TO_PX
    MM_TO_PX = dpi / 25.4
    
    # Set background image if provided
    if bg_image:
        load_background(bg_image)
    
    W = mm_to_px(width_mm)
    H = mm_to_px(height_mm)
    os.makedirs(outdir, exist_ok=True)
    
    fname = f"{base_name}.svg"
    path = os.path.join(outdir, fname)
    # denominationがテンプレート変数や空文字列の場合は必ず1に変換
    denom_int = 1
    if isinstance(denomination, (int, float)):
        denom_int = int(denomination)
    elif isinstance(denomination, str):
        import re
        m = re.search(r'(\d+)', denomination)
        if m:
            denom_int = int(m.group(1))
    generate_backside_svg(path, denom_int, title_text, phrase_text, (W,H), serial_id, timestamp, seed_text, progress_callback=progress_callback)
    
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
              png: bool = False, dpi: float = 300.0, bg_image: str = None):
    # Update global DPI
    global MM_TO_PX
    MM_TO_PX = dpi / 25.4
    
    # Set background image if provided
    if bg_image:
        load_background(bg_image)
    
    denoms = [10**i for i in range(0,9)]
    W = mm_to_px(width_mm)
    H = mm_to_px(height_mm)
    os.makedirs(outdir, exist_ok=True)
    for d in denoms:
        denom_int = int(d)
        # Include denomination in the filename to avoid overwriting
        fname = f"{base_name}_{denom_int}.svg"  # Add denomination to filename
        path = os.path.join(outdir, fname)
        generate_backside_svg(path, denom_int, title_text, phrase_text, (W,H), serial_id, timestamp, seed_text)
        
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
    parser.add_argument("--dpi", type=float, default=300.0, help="Resolution in DPI (default: 300.0)")
    parser.add_argument("--bg-image", type=str, help="Background image path")
    args = parser.parse_args()

    if args.denomination:
        run_single_denomination(outdir=args.outdir, base_name=args.basename, denomination=args.denomination,
                               width_mm=args.width_mm, height_mm=args.height_mm,
                               title_text=args.title, phrase_text=args.phrase, seed_text=args.seed_text, serial_id=args.serial_id, timestamp=args.timestamp, png=args.png, dpi=args.dpi, bg_image=args.bg_image)
    else:
        run_batch(outdir=args.outdir, base_name=args.basename, width_mm=args.width_mm, height_mm=args.height_mm,
                  title_text=args.title, phrase_text=args.phrase, seed_text=args.seed_text, serial_id=args.serial_id, timestamp=args.timestamp, png=args.png, dpi=args.dpi, bg_image=args.bg_image)