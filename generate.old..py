# -----------------------
# Denomination utility function
# -----------------------
def denom_to_int(denom):
    """Convert denomination to int, handling string/float inputs and removing commas."""
    if denom is None:
        return 1
    if isinstance(denom, int):
        return denom
    try:
        # Remove commas and whitespace, then convert
        return int(float(str(denom).replace(",", "").strip()))
    except Exception:
        return 1
# EisenScript prefix/suffix variable replacement
import string
from datetime import datetime
from jinja2 import Template
from jinja2 import Environment, DebugUndefined
from jinja2.exceptions import TemplateError
from jinja2 import meta
from jinja2 import exceptions as jinja2_exceptions
from lunamint.scripting import render_script_to_svg_html
from models import Banknote, User, Settings, SerialNumber, db
def render_eisenscript_jinja2(script: str, context: dict) -> str:
    """
    Render EisenScript using Jinja2 templating.
    Allows {{ variable }} syntax in EisenScript files.
    Context can be passed from Flask or any Python source.
    """
    if not script:
        return script
    # Provide some default context values if missing
    now = datetime.now()
    denomination = context.get("denomination", 1)
    safe_context = dict(context)
    safe_context.setdefault("username", "")
    safe_context.setdefault("user_id", "")
    safe_context.setdefault("datetime", now.strftime("%Y-%m-%d %H:%M:%S"))
    safe_context.setdefault("date", now.strftime("%Y-%m-%d"))
    safe_context.setdefault("time", now.strftime("%H:%M:%S"))
    safe_context.setdefault("serial", "")
    safe_context.setdefault("serial_id", "")
    safe_context.setdefault("denomination", denomination)
    safe_context.setdefault("title", "")
    safe_context.setdefault("subtitle", "")
    # serial_idがあればserialにも必ず反映
    if safe_context.get("serial_id"):
        safe_context["serial"] = safe_context["serial_id"]
    try:
        import jinja2
        env = jinja2.Environment(undefined=jinja2.DebugUndefined)
        template = env.from_string(script)
        rendered = template.render(**safe_context)
        return rendered
    except Exception as e:
        return script  # fallback: do not break generation
#!/usr/bin/env python3
"""
generate.py - Unified banknote generation script
Can be used as standalone: python generate.py --name NAME --user_id ID
Or imported: from generate import generate_for_user
"""

import os
import random
import subprocess
import time
import glob
import re
import shutil
import requests
import base64
import json
import argparse
import threading
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

import secrets
import hashlib
import xml.etree.ElementTree as ET
import cairosvg
import sys
import tempfile
from pathlib import Path

# IMPORTS WITH BETTER ERROR HANDLING
# =============================================================================

import os
import sys
import traceback

# Add current directory to Python path FIRST
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"[DEBUG] Current directory: {os.getcwd()}")
print(f"[DEBUG] Script directory: {os.path.dirname(os.path.abspath(__file__))}")


def _extract_svg_inner(svg_text: str) -> str:
    start = svg_text.find(">")
    end = svg_text.rfind("</svg>")
    if start == -1 or end == -1:
        return svg_text
    return svg_text[start + 1 : end]


def _extract_svg_size(svg_text: str):
    width_match = re.search(r'width="([0-9.]+)', svg_text)
    height_match = re.search(r'height="([0-9.]+)', svg_text)
    if width_match and height_match:
        return float(width_match.group(1)), float(height_match.group(1))
    viewbox_match = re.search(r'viewBox="[\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+([\d.]+)"', svg_text)
    if viewbox_match:
        return float(viewbox_match.group(1)), float(viewbox_match.group(2))
    return 1600.0, 600.0




# Try to import signatures with multiple fallbacks
HAS_SIGNATURES = False
HAS_FLASK_CONTEXT = False

try:
    # First try to import signatures directly
    import signatures
    print("[+] Successfully imported signatures module")
    
    # Try to get the required functions/classes
    if hasattr(signatures, 'DigitalSignatureManager'):
        DigitalSignatureManager = signatures.DigitalSignatureManager
        HAS_SIGNATURES = True
        print("[+] Got DigitalSignatureManager from signatures")
    
    if hasattr(signatures, 'generate_key_pair'):
        generate_key_pair = signatures.generate_key_pair
        print("[+] Got generate_key_pair from signatures")
    else:
        # Provide a fallback
        def generate_key_pair():
            import hashlib
            import secrets
            priv = secrets.token_hex(32)
            pub = f"04{hashlib.sha256(priv.encode()).hexdigest()[:64]}"
            return priv, pub
        print("[!] Using fallback generate_key_pair")
    

except ImportError as e:
    print(f"[!] Failed to import signatures: {e}")
    # Create fallback implementations
    class DigitalSignatureManager:
        pass


# Import functions from banknote generators
HAS_FRONT_GENERATOR = False
HAS_BACK_GENERATOR = False
generate_front = None
generate_back = None

# Try multiple import approaches for front generator
try:
    from generate_banknote_front import generate_single_banknote as generate_front
    HAS_FRONT_GENERATOR = True
    print("[+] Successfully imported generate_single_banknote from generate_banknote_front.py")
except ImportError as e:
    print(f"[!] First import attempt failed: {e}")
    try:
        # Try adding current directory to path
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from generate_banknote_front import generate_single_banknote as generate_front
        HAS_FRONT_GENERATOR = True
        print("[+] Successfully imported generate_single_banknote (second attempt)")
    except ImportError as e2:
        print(f"[!] Second import attempt failed: {e2}")
        generate_front = None
        HAS_FRONT_GENERATOR = False

# Try multiple import approaches for back generator
try:
    from generate_banknote_back import run_single_denomination as generate_back
    HAS_BACK_GENERATOR = True
    print("[+] Successfully imported run_single_denomination from generate_banknote_back.py")
except ImportError as e:
    print(f"[!] First import attempt failed: {e}")
    try:
        from generate_banknote_back import run_single_denomination as generate_back
        HAS_BACK_GENERATOR = True
        print("[+] Successfully imported run_single_denomination (second attempt)")
    except ImportError as e2:
        print(f"[!] Second import attempt failed: {e2}")
        generate_back = None
        HAS_BACK_GENERATOR = False

# Debug info
print(f"[DEBUG] HAS_FRONT_GENERATOR: {HAS_FRONT_GENERATOR}")
print(f"[DEBUG] HAS_BACK_GENERATOR: {HAS_BACK_GENERATOR}")
print(f"[DEBUG] Current directory: {os.getcwd()}")
print(f"[DEBUG] Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"[DEBUG] Python path: {sys.path}")

# Configuration
# -----------------------
FRONT_SCRIPT = "generate_banknote_front.py"
BACK_SCRIPT = "generate_banknote_back.py"
NAMES_FILE = "master.txt"
OUTPUT_ROOT = "./images"  # single folder per name
PORTRAITS_DIR = "./portraits"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
# SD API URLs with GPU selection support
SD_API_URL = "http://localhost:7777/sdapi/v1/txt2img"
SD_API_URL_GPU0 = os.getenv("SD_API_URL_GPU0", "http://localhost:7777/sdapi/v1/txt2img")
SD_API_URL_GPU1 = os.getenv("SD_API_URL_GPU1", "http://localhost:7777/sdapi/v1/txt2img")

# Check if multi-GPU is available
MULTI_GPU_ENABLED = os.getenv("MULTI_GPU_ENABLED", "false").lower() == "true"
MAX_THREADS = 8  # Increased for 3090!

# Standard denominations
STANDARD_DENOMINATIONS = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

# -----------------------
# Serial generation functions
# -----------------------
def generate_serial_id_with_checksum(timestamp_ms=None):
    """Generate serial ID with built-in checksum for validation (for front)"""
    ts = timestamp_ms or int(datetime.now().timestamp() * 1000000)
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

def generate_serial_id_combined(timestamp_ms=None):
    """Generate a unique, compact serial ID (for back)"""
    ts = timestamp_ms or int(datetime.now().timestamp() * 1000000)
    salt = secrets.token_bytes(4)
    raw = f"{ts}-".encode() + salt
    h = hashlib.sha3_256(raw).digest()
    
    # Use base64 URL-safe encoding
    serial_b64 = base64.urlsafe_b64encode(h[:12]).decode('ascii')
    serial_clean = serial_b64.replace('=', '')[:12]
    
    # Format with prefix and groups for readability
    return f"SN-{serial_clean[:4]}-{serial_clean[4:8]}-{serial_clean[8:12]}"

def generate_timestamp_ms_precise():
    """Generate timestamp with microsecond precision."""
    now = datetime.now()
    return int(now.timestamp() * 1000) + now.microsecond // 1000

# -----------------------
# Digital Signature Functions
# -----------------------
def create_digital_banknote_signature(name, denomination, serial_number, timestamp_ms):
    """Create a digital signature for a banknote"""
    if not HAS_SIGNATURES:
        safe_print("[!] Digital signatures not available, using mock signature")
        return {
            'signature': 'mock_signature_' + hashlib.md5(f"{name}{denomination}{serial_number}".encode()).hexdigest(),
            'public_key': 'mock_public_key',
            'metadata_hash': hashlib.sha256(f"{name}{denomination}{timestamp_ms}".encode()).hexdigest(),
            'is_verified': False
        }
    
    try:
        signature_manager = DigitalSignatureManager()
        
        # Generate key pair for this banknote
        private_key, public_key = generate_key_pair()
        
        # Create bill data
        bill_data = {
            'type': 'banknote',
            'front_serial': f"{serial_number}_FRONT",
            'back_serial': f"{serial_number}_BACK",
            'metadata_hash': hashlib.sha256(f"{name}{denomination}{timestamp_ms}".encode()).hexdigest(),
            'timestamp': timestamp_ms,
            'issued_to': name,
            'denomination': str(denomination)
        }
        
        # Create signed bill
        signed_bill = signature_manager.create_signed_bill(bill_data, private_key)
        
        return {
            'signature': signed_bill.signature,
            'public_key': signed_bill.public_key,
            'private_key': private_key,  # Store for future transactions
            'metadata_hash': bill_data['metadata_hash'],
            'is_verified': True
        }
        
    except Exception as e:
        safe_print(f"[!] Error creating digital signature: {e}")
        # Fallback to simple hash-based signature
        return {
            'signature': hashlib.sha256(f"{name}{denomination}{serial_number}{timestamp_ms}".encode()).hexdigest(),
            'public_key': 'fallback_public_key',
            'metadata_hash': hashlib.sha256(f"{name}{denomination}{timestamp_ms}".encode()).hexdigest(),
            'is_verified': False
        }

def verify_banknote_signature(banknote_data):
    """Verify a banknote's digital signature"""
    if not HAS_SIGNATURES:
        safe_print("[!] Digital signature verification not available")
        return True  # Return True for fallback mode
    
    try:
        signature_manager = DigitalSignatureManager()
        return signature_manager.verify_bill_signature(banknote_data)
    except Exception as e:
        safe_print(f"[!] Error verifying signature: {e}")
        return False

# -----------------------
# Helper functions
# -----------------------
def read_prompt_file(filename, default_prompt=""):
    """Read prompt from file, return default if file doesn't exist"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            safe_print(f"[!] Prompt file {filename} not found, using default")
            return default_prompt
    except Exception as e:
        safe_print(f"[!] Error reading {filename}: {e}")
        return default_prompt

def parse_denomination_from_filename(filename):
    """Extract denomination from filename patterns like '1.svg', '10.svg', etc."""
    basename = os.path.splitext(filename)[0]
    match = re.search(r'(\d+)', basename)
    if match:
        return match.group(1)
    return "1"

def create_proper_filename(name, denom, timestamp, side):
    """Create filename in format: {name}_-_{denom}_-_{timestamp}_{side}.svg"""
    # Windowsで使えない文字を除去
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(name)) if name else "unknown"
    safe_denom = re.sub(r'[<>:"/\\|?*]', '_', str(denom)) if denom else "1"
    safe_side = re.sub(r'[<>:"/\\|?*]', '_', str(side)) if side else "FRONT"
    safe_timestamp = re.sub(r'[<>:"/\\|?*]', '_', str(timestamp)) if timestamp else "0"
    return f"{safe_name}_-_{safe_denom}_-_{safe_timestamp}_{safe_side}.svg"

def create_basename(name, denom, timestamp, side):
    """Create filename in format: {name}_-_{denom}_-_{timestamp}_{side}"""
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(name)) if name else "unknown"
    safe_denom = re.sub(r'[<>:"/\\|?*]', '_', str(denom)) if denom else "1"
    safe_side = re.sub(r'[<>:"/\\|?*]', '_', str(side)) if side else "BACK"
    safe_timestamp = re.sub(r'[<>:"/\\|?*]', '_', str(timestamp)) if timestamp else "0"
    return f"{safe_name}_-_{safe_denom}_-_{safe_timestamp}_{safe_side}"

def safe_print(message):
    """Print message with Unicode fallback handling"""
    try:
        print(message)
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message)

# Move mm_to_px outside so it's globally available
def mm_to_px(mm, dpi=300.0):
    """Convert millimeters to pixels (default DPI=300)."""
    return float(mm) * dpi / 25.4

def generate_png_from_svg(svg_path, png_path, size=(1600, 600)):
    """Generate PNG from SVG file using cairosvg (fast path + cache)."""
    try:
        if os.path.exists(png_path) and os.path.exists(svg_path):
            if os.path.getmtime(png_path) >= os.path.getmtime(svg_path):
                return True

        png_bytes = cairosvg.svg2png(
            url=svg_path,
            output_width=size[0],
            output_height=size[1],
        )
        img = Image.open(BytesIO(png_bytes))
        img.save(png_path, format="PNG", optimize=False, compress_level=1)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to generate PNG from {svg_path}: {e}")
        return False

def generate_pdf_from_svg(svg_path, pdf_path):
    """Generate PDF from SVG file using pure ReportLab"""
    try:
        # Parse SVG manually (simplified approach)
        from reportlab.pdfgen import canvas
        from reportlab.graphics import renderPDF
        from reportlab.graphics.shapes import Drawing
        
        drawing = Drawing(400, 200)  # Adjust size as needed
        
        # Basic SVG parsing - this is simplified
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # You would need to implement proper SVG parsing here
        # This is just a placeholder
        
        c = canvas.Canvas(pdf_path)
        drawing.drawOn(c, 0, 0)
        c.save()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to generate PDF from {svg_path}: {e}")
        return False

# -----------------------
# Portrait generation functions
# -----------------------
def generate_character_portrait(name: str, width: int = 512, height: int = 512, 
                               seed: int = -1, save_path: str = "./portraits", portrait_prompt=None):
    """
    Generate a character portrait based on the name using Stable Diffusion API
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Use provided prompt or read from file (only if not provided)
    if not portrait_prompt:
        portrait_prompt = read_prompt_file(
            "portrait_prompt.txt",
            "portrait of {name}, elegant character, official portrait, banknote portrait, currency art, detailed face, professional, serious expression, high detail, official document style"
        )
    
    negative_prompt = read_prompt_file(
        "negative_prompt.txt",
        "text, words, letters, numbers, blurry, low quality, watermark, signature, ugly, deformed, cartoon, anime, modern, casual"
    )
    
    # Format the prompt with the name
    formatted_prompt = portrait_prompt.format(name=name)
    
    payload = {
        "prompt": formatted_prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": seed if seed != -1 else random.randint(0, 2**32 - 1),
        "steps": 20,  # REDUCED from 30 to 20 - 33% faster!
        "cfg_scale": 7,  # Slightly reduced
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,
        "n_iter": 1,
        "restore_faces": False,  # Disabled for speed
        "tiling": False,
        "enable_hr": False,  # DISABLED hires fix - big speedup!
        "denoising_strength": 0.4,
    }

    try:
        safe_print(f"[+] Generating portrait for: {name}")
        response = requests.post(SD_API_URL, json=payload, timeout=131313)
        response.raise_for_status()
        
        result = response.json()
        images = result.get('images', [])
        
        if images:
            image_data = base64.b64decode(images[0])
            image = Image.open(BytesIO(image_data))
            
            # Clean name for filename
            clean_name = re.sub(r'[^\w\-_]', '_', name)
            filename = f"portrait_{clean_name}.png"
            filepath = os.path.join(save_path, filename)
            
            image.save(filepath)
            safe_print(f"[+] Generated portrait: {filepath}")
            return filepath
        
    except Exception as e:
        safe_print(f"[!] Error generating portrait for {name}: {e}")
        return None

def get_portrait_for_name(name, force_regenerate=False, portrait_prompt=None):
    """
    Get a portrait for the given name - use existing or generate new
    Returns the same portrait path for all denominations
    """
    try:
        clean_name = re.sub(r'[^\w\-_]', '_', name)
    except UnicodeEncodeError:
        try:
            clean_name = re.sub(r'[^\w\-_]', '_', name.encode('ascii', 'ignore').decode('ascii'))
        except:
            clean_name = "unknown"
    
    # Look for existing portrait for this name (without timestamp)
    portrait_patterns = [
        os.path.join(PORTRAITS_DIR, f"portrait_{clean_name}.png"),
        os.path.join(PORTRAITS_DIR, f"portrait_{clean_name}.jpg"),
        os.path.join(PORTRAITS_DIR, f"portrait_{clean_name}.jpeg"),
        os.path.join(PORTRAITS_DIR, f"*{clean_name}*.png"),
        os.path.join(PORTRAITS_DIR, f"*{clean_name}*.jpg"),
        os.path.join(PORTRAITS_DIR, f"*{clean_name}*.jpeg"),
    ]
    
    # Check for existing portrait only if not forcing regeneration
    existing_portrait = None
    if not force_regenerate:
        for pattern in portrait_patterns:
            existing_portraits = glob.glob(pattern)
            if existing_portraits:
                existing_portrait = existing_portraits[0]
                safe_print(f"[+] Using existing portrait: {existing_portrait}")
                return existing_portrait
    
    # No existing portrait found or force_regenerate is True
    if not existing_portrait or force_regenerate:
        safe_print(f"[+] Generating new portrait for {name}...")
        new_portrait = generate_character_portrait(name, portrait_prompt=portrait_prompt)
        if new_portrait and os.path.exists(new_portrait):
            safe_print(f"[+] Successfully generated portrait: {new_portrait}")
            return new_portrait
        else:
            safe_print(f"[!] Failed to generate portrait for {name}")
            return None
    
    return existing_portrait

# -----------------------
# Banknote generation functions
# -----------------------
def generate_front_back_pair(name, denom, img_path, timestamp_ms, denom_folder, user_id=None,
                          width_mm=160.0, height_mm=60.0, title="灵国国库", subtitle="天圆地方", 
                          font_dir="./fonts", bg_dir="./backgrounds", dpi=300.0, bg_image=None, background_prompt=None,
                          eisenscript_front=None, eisenscript_back=None, use_parallel=True):
    """Generate a front+back pair for a single denomination with optional parallel processing"""
    # denom_str, front_svg_path, back_svg_path を関数先頭で定義し、全体で使えるようにする
    # denomがテンプレート変数や空文字列の場合は必ず'1'に置換
    if denom is None or (isinstance(denom, str) and (not denom.strip() or "{{" in denom)):
        denom_str = "1"
    else:
        try:
            denom_str = str(int(denom))
        except Exception:
            denom_str = "1"
    front_serial = generate_serial_id_with_checksum(timestamp_ms)
    back_serial = generate_serial_id_combined(timestamp_ms)

    front_filename = create_proper_filename(name, denom_str, timestamp_ms, "FRONT")
    front_svg_path = os.path.abspath(os.path.join(denom_folder, front_filename))
    # バックサイドのSVGファイル名ロジックをrun_single_denominationと完全一致させる
    back_basename = create_basename(name, denom_str, timestamp_ms, "BACK")
    # run_single_denominationは必ず f"{base_name}.svg" で出力するため、ここも同じにする
    back_svg_path = os.path.abspath(os.path.join(denom_folder, f"{back_basename}.svg"))

    safe_print(f"[+] Created digital signature for serial: {front_serial}")

    def generate_front_task():
        """Generate front in separate thread"""
        try:
            # Accept eisenscript_front as the overlay (prefix+user+suffix), fallback to prefix+suffix if user is None
            # Patch: Replace EisenScript variables with per-bill context
            eisenscript_context = {
                "username": name,
                "user_id": user_id,
                "title": title,
                "subtitle": subtitle,
                "denomination": denom_str if denom_str else "1",  # 空なら"1"をセット
                "serial": front_serial,
                "seed_text": name,
                "qr_url": f"https://bank.linglin.art/verify/{front_serial}",
                "input_image_path": img_path,
                "width_mm": width_mm,
                "height_mm": height_mm,
                
            }
            safe_print(f"[+] Created digital signature for serial: {front_serial}")
            
            # front_svg_pathは関数先頭で一度だけ定義し、以降は上書きしない
            
            # Accept eisenscript_front as the overlay (prefix+suffix+user), fallback to prefix+suffix if user is None
            # Patch: Replace EisenScript variables with per-bill context
            eisenscript_context = {
                "username": name,
                "user_id": user_id,
                "title": title,
                "subtitle": subtitle,
                "denomination": denom_str if denom_str else "1",  # 空なら"1"をセット
                "serial": front_serial,
                "seed_text": name,
                "qr_url": f"https://bank.linglin.art/verify/{front_serial}",
                "input_image_path": img_path,
                "width_mm": width_mm,
                "height_mm": height_mm,
            }
            
            print(f"[DEBUG] EisenScript FRONT context: {eisenscript_context}")
            if not eisenscript_context.get('denomination'):
                print("[ERROR] EisenScript FRONT: denomination is missing or empty! Context:", eisenscript_context)
                eisenscript_context['denomination'] = "1"  # デフォルト値をセット
            
            # Always combine overlays before Jinja2 substitution (front_pre, user_eisen, front_suf)
            # If eisenscript_front is a tuple/list, unpack; else treat as already combined
            if isinstance(eisenscript_front, (tuple, list)) and len(eisenscript_front) == 3:
                front_pre, user_eisen, front_suf = eisenscript_front
                eisenscript_text = merge_eisenscript_with_vars(front_pre, user_eisen, front_suf, eisenscript_context)
            elif isinstance(eisenscript_front, str):
                # fallback: treat as already combined
                eisenscript_text = render_eisenscript_jinja2(eisenscript_front, eisenscript_context)
            else:
                safe_print("[ERROR] eisenscript_front must be a (pre, user, suf) tuple/list or a string.")
                eisenscript_text = ""
                
            eisenscript_file = None
            if eisenscript_text:
                try:
                    os.makedirs(denom_folder, exist_ok=True)
                    eisenscript_file = os.path.join(denom_folder, f"eisenscript_{timestamp_ms}.eisen")
                    with open(eisenscript_file, "w", encoding="utf-8") as f:
                        f.write(eisenscript_text)
                except Exception as script_error:
                    safe_print(f"[!] Failed to write Eisenscript file: {script_error}")
                    eisenscript_file = None


            # SVG保存完了コールバックでのみ後続処理を進める
            svg_saved_event = threading.Event()
            def on_front_svg_saved():
                import time
                max_wait = 1.0  # 最大1秒待つ
                waited = 0.0
                while (not os.path.exists(front_svg_path) or os.path.getsize(front_svg_path) == 0) and waited < max_wait:
                    time.sleep(0.05)
                    waited += 0.05
                if not os.path.exists(front_svg_path) or os.path.getsize(front_svg_path) == 0:
                    safe_print(f"[WARNING] SVG not found or empty after callback: {front_svg_path}")
                safe_print(f"[CALLBACK] FRONT SVG saved: {front_svg_path}")
                svg_saved_event.set()

            # Try using imported function first
            if HAS_FRONT_GENERATOR and generate_front:
                generate_front(
                    seed_text=name,
                    input_image_path=os.path.abspath(img_path),
                    single_denom=denom_str,
                    outfile=front_svg_path,
                    serial_id=front_serial,
                    timestamp=int(timestamp_ms),
                    background_prompt=background_prompt,
                    eisenscript_text=eisenscript_text,
                    progress_callback=lambda *a, **kw: on_front_svg_saved()
                )
                svg_saved_event.wait(timeout=10.0)
                if not os.path.exists(front_svg_path):
                    safe_print(f"[ERROR] Front SVG not ready after generation: {front_svg_path}")
                    return False
            else:
                safe_name = name.replace('&', '_')
                venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
                if os.path.exists(venv_python):
                    python_exe = venv_python
                else:
                    python_exe = sys.executable
                if not os.path.exists(front_svg_path):
                    return False

            # Set GPU 0 environment if multi-GPU is enabled
            gpu_env = os.environ.copy()
            if MULTI_GPU_ENABLED:
                gpu_env['CUDA_VISIBLE_DEVICES'] = '0'
                safe_print(f"[GPU0] Generating front for {denom}卢纳币")

            # Try using imported function first
            if HAS_FRONT_GENERATOR and generate_front:
                generate_front(
                    seed_text=name,
                    input_image_path=os.path.abspath(img_path),
                    single_denom=denom_str,
                    outfile=front_svg_path,
                    serial_id=front_serial,
                    timestamp=int(timestamp_ms),
                    background_prompt=background_prompt,
                    eisenscript_text=eisenscript_text
                )
                # wait_for_svg_ready呼び出しは不要。存在チェックはコールバック/イベントで一元管理。
            else:
                safe_name = name.replace('&', '_')
                # Use .venv python if available, else fallback to sys.executable
                venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
                if os.path.exists(venv_python):
                    python_exe = venv_python
                else:
                    python_exe = sys.executable
                    
                # SVGファイルが存在しない場合は即return（以降の処理は絶対に実行しない）
                if not os.path.exists(front_svg_path):
                    return False
                    
                try:
                    # デバッグ: EisenScriptテキストとSVGパスを出力
                    safe_print("[DEBUG] EisenScript overlay: parsing text below:")
                    safe_print(eisenscript_text)
                    safe_print(f"[DEBUG] SVG path: {svg_path}")
                    
                    with tempfile.TemporaryDirectory(prefix="eisenscript_") as tmp_dir:
                        overlay_svg, _ = render_script_to_svg_html(eisenscript_text, Path(tmp_dir))
                        overlay_text = overlay_svg.read_text(encoding="utf-8")
                        inner = _extract_svg_inner(overlay_text)
                        overlay_w, overlay_h = _extract_svg_size(overlay_text)

                        W = mm_to_px(width_mm)
                        H = mm_to_px(height_mm)
                        scale_x = W / overlay_w if overlay_w else 1.0
                        scale_y = H / overlay_h if overlay_h else 1.0
                        wrapped = f'<g data-eisenscript="1" transform="scale({scale_x},{scale_y})">{inner}</g>'

                        base_text = Path(svg_path).read_text(encoding="utf-8")
                        if "</svg>" in base_text:
                            base_text = base_text.replace("</svg>", wrapped + "</svg>")
                            Path(svg_path).write_text(base_text, encoding="utf-8")
                            
                    safe_print(f"[+] Generated front: {front_svg_path}")
                    return True
                except Exception as e:
                    safe_print(f"[!] Failed to generate front: {e}")
                    return False
        except Exception as e:
            safe_print(f"[!] Failed to generate front: {e}")
    
    def generate_back_task():
        """Generate back in separate thread, with robust error and output diagnostics"""
        try:
            # SVG生成後、ファイルの存在とサイズをポーリングで確認（最大10秒）
            def wait_for_svg_ready(svg_path, timeout=10.0):
                import time
                start = time.time()
                while time.time() - start < timeout:
                    if os.path.exists(svg_path) and os.path.getsize(svg_path) > 0:
                        return True
                    time.sleep(0.1)
                return False

            if os.path.exists(back_svg_path) and os.path.getsize(back_svg_path) > 0:
                safe_print(f"[CACHE] Back SVG exists, skipping generation: {back_svg_path}")
                return True

            # Set GPU 1 environment if multi-GPU is enabled
            gpu_env = os.environ.copy()
            if MULTI_GPU_ENABLED:
                gpu_env['CUDA_VISIBLE_DEVICES'] = '1'
                safe_print(f"[GPU1] Generating back for {denom}卢纳币")

            # Build EisenScript BACK context robustly (like FRONT)
            _denom = str(denom_str) if denom_str not in (None, "", "None") else "1"
            # テンプレート変数が残っていたら必ず'1'に置換
            if "{{" in _denom or not _denom.strip():
                _denom = "1"
            _serial = str(back_serial) if back_serial not in (None, "", "None") else "SN-000000"
            eisenscript_context_back = {
                "username": str(name) if name is not None else "",
                "user_id": str(user_id) if user_id is not None else "",
                "title": str(title) if title is not None else "",
                "subtitle": str(subtitle) if subtitle is not None else "",
                "denomination": _denom,
                "serial": _serial,
                "serial_id": _serial,
                "seed_text": str(name) if name is not None else "",
                "qr_url": f"https://bank.linglin.art/verify/{_serial}",
                "input_image_path": str(img_path) if img_path is not None else "",
                "width_mm": width_mm,
                "height_mm": height_mm,
            }
            # Ensure all context values are strings (except width_mm/height_mm)
            for k in list(eisenscript_context_back.keys()):
                if k not in ("width_mm", "height_mm"):
                    eisenscript_context_back[k] = str(eisenscript_context_back[k]) if eisenscript_context_back[k] is not None else ""
            # Ensure serial is always set from serial_id if present
            if eisenscript_context_back.get("serial_id"):
                eisenscript_context_back["serial"] = eisenscript_context_back["serial_id"]
            print(f"[DEBUG] EisenScript BACK context (full): {json.dumps(eisenscript_context_back, ensure_ascii=False, indent=2)}")
            if not eisenscript_context_back.get('denomination'):
                print("[ERROR] EisenScript BACK: denomination is missing or empty! Context:", eisenscript_context_back)
                eisenscript_context_back['denomination'] = "1"
            if not eisenscript_context_back.get('serial'):
                print("[ERROR] EisenScript BACK: serial is missing or empty! Context:", eisenscript_context_back)
                eisenscript_context_back['serial'] = "SN-000000"

            # --- Use DB-driven EisenScript overlays for back ---
            from generate_banknote import get_eisenscript_from_db
            eisenscript_back_rendered = get_eisenscript_from_db('back', eisenscript_context_back)
            print("[DEBUG] EisenScript BACK rendered text (DB version):")
            print(eisenscript_back_rendered)
            # 展開後に{{denomination}}が残っていたらエラー
            if "{{denomination}}" in eisenscript_back_rendered:
                print("[ERROR] Jinja2 failed to substitute {{denomination}} in BACK! Context:", json.dumps(eisenscript_context_back, ensure_ascii=False, indent=2))

            # Write the rendered EisenScript to a temp file for debugging
            try:
                os.makedirs(denom_folder, exist_ok=True)
                debug_eisen_path = os.path.join(denom_folder, f"debug_back_{timestamp_ms}.eisen")
                with open(debug_eisen_path, "w", encoding="utf-8") as f:
                    f.write(eisenscript_back_rendered)
                print(f"[DEBUG] Wrote debug EisenScript to: {debug_eisen_path}")
            except Exception as script_error:
                safe_print(f"[!] Failed to write debug Eisenscript file: {script_error}")

            # LunaMintのprogress_callbackで完了を検知し、完了時のみ次処理へ
            result = {"ok": False, "error": None}
            import threading
            done_event = threading.Event()
            
            def progress_callback(progress, message=None):
                percent = int(progress * 100) if isinstance(progress, float) else 0
                msg = message or ""
                print(f"[BACK PROGRESS] {percent}% {msg}")
                # 進捗100%または"Completed"で完了判定
                if percent >= 100 or (msg and "complete" in msg.lower()):
                    result["ok"] = True
                    done_event.set()
                    
            if HAS_BACK_GENERATOR and generate_back:
                try:
                    # base_nameは拡張子なし、.svgはrun_single_denomination側で付与される
                    generate_back(
                        outdir=os.path.abspath(denom_folder),
                        base_name=back_basename,
                        denomination=denom_str,
                        seed_text=name,
                        serial_id=back_serial,
                        timestamp=int(timestamp_ms),
                        progress_callback=progress_callback
                    )
                    # 完了まで待機
                    done_event.wait(timeout=120.0)
                    if result["error"]:
                        raise result["error"]
                except Exception as gen_err:
                    safe_print(f"[!] Exception in generate_back: {gen_err}")
                    return False
            else:
                # サブプロセスの場合は従来通り
                safe_name = name.replace('&', '_')
                venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
                if os.path.exists(venv_python):
                    python_exe = venv_python
                else:
                    python_exe = sys.executable
                    
                try:
                    completed = subprocess.run([
                        python_exe, BACK_SCRIPT,
                        '--outdir', os.path.abspath(denom_folder),
                        '--basename', back_basename,
                        '--denomination', denom_str,
                        '--seed-text', safe_name,
                        '--serial-id', back_serial,
                        '--timestamp', str(int(timestamp_ms)),
                        '--width-mm', str(width_mm),
                        '--height-mm', str(height_mm),
                        '--title', title,
                        '--phrase', subtitle,
                        '--dpi', str(dpi)
                    ] + (['--bg-image', os.path.abspath(bg_image)] if bg_image else []),
                        check=False, capture_output=True, timeout=13131313, env=gpu_env)
                        
                    print(f"[DEBUG] Subprocess returncode: {completed.returncode}")
                    print(f"[DEBUG] Subprocess stdout:\n{completed.stdout.decode(errors='replace')}")
                    print(f"[DEBUG] Subprocess stderr:\n{completed.stderr.decode(errors='replace')}")
                except Exception as sub_err:
                    safe_print(f"[!] Exception in back subprocess: {sub_err}")
                    
                # サブプロセスの場合は従来通りwait_for_svg_readyで待機
                if not wait_for_svg_ready(back_svg_path):
                    safe_print(f"[ERROR] Back SVG not ready after generation: {back_svg_path}")
                    try:
                        svg_files = [f for f in os.listdir(denom_folder) if f.lower().endswith('.svg')]
                        safe_print(f"[DEBUG] SVGs in {denom_folder}: {svg_files}")
                    except Exception as list_err:
                        safe_print(f"[!] Could not list SVGs in {denom_folder}: {list_err}")
                    return False
                    
            # SVG内容の先頭だけでもprint
            try:
                with open(back_svg_path, 'r', encoding='utf-8') as f:
                    head = f.read(512)
                    safe_print(f"[DEBUG] Back SVG head: {head[:256]}...")
            except Exception as svg_read_err:
                safe_print(f"[!] Could not read back SVG: {svg_read_err}")
                
            return True
        except Exception as e:
            safe_print(f"[!] Failed to generate back: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    try:
        if use_parallel:
            # Generate front and back in parallel using threading
            safe_print(f"[PARALLEL] Using parallel front/back generation")
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                front_future = executor.submit(generate_front_task)
                back_future = executor.submit(generate_back_task)
                
                # Wait for both to complete
                front_success = front_future.result()
                back_success = back_future.result()
                safe_print(f"[DEBUG] After parallel: front_success={front_success}, back_success={back_success}")
                if not front_success:
                    safe_print(f"[DEBUG] Front SVG generation failed for {name} {denom_str}")
                if not back_success:
                    safe_print(f"[DEBUG] Back SVG generation failed for {name} {denom_str}")
                if not (front_success and back_success):
                    safe_print(f"[FAIL] SVG pair not generated: front={front_success}, back={back_success}")
                    return None
        else:
            # Sequential generation
            safe_print(f"[SEQUENTIAL] Using single GPU sequential generation")
            front_success = generate_front_task()
            back_success = generate_back_task()
            if not (front_success and back_success):
                safe_print(f"[FAIL] SVG pair not generated: front={front_success}, back={back_success}")
                return None
        
        # Generate PNG and PDF files
        front_png_path = front_svg_path.replace(".svg", ".png")
        front_pdf_path = front_svg_path.replace(".svg", ".pdf")
        back_png_path = back_svg_path.replace(".svg", ".png")
        back_pdf_path = back_svg_path.replace(".svg", ".pdf")
        
        # Generate PNGs in parallel
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            front_future = executor.submit(generate_png_from_svg, front_svg_path, front_png_path)
            back_future = executor.submit(generate_png_from_svg, back_svg_path, back_png_path)
            front_ok = front_future.result()
            back_ok = back_future.result()
            if not (front_ok and back_ok):
                raise Exception("Failed to generate front/back PNG")
        
        # Generate PDFs (commented out for now)
        # generate_pdf_from_svg(front_svg_path, front_pdf_path)
        # generate_pdf_from_svg(back_svg_path, back_pdf_path)
        back_pdf_path = ""
        
        # --- Always use save_to_database for DB registration (sequential or parallel) ---
        files = {
            'front_svg': front_svg_path,
            'front_png': front_png_path,
            'front_pdf': front_pdf_path,
            'back_svg': back_svg_path,
            'back_png': back_png_path,
            'back_pdf': back_pdf_path,
            'front_serial': front_serial,
            'back_serial': back_serial,
            'digital_signature': digital_signature_data['signature'],
            'public_key': digital_signature_data['public_key'],
            'private_key': digital_signature_data.get('private_key'),
            'metadata_hash': digital_signature_data['metadata_hash'],
            'is_verified': digital_signature_data['is_verified']
        }
        safe_print(f"[DEBUG] About to call save_to_database for {name} {denom_str}")
        # Mempool (blockchain) registration (if needed)
        try:
            from app import blockchain_daemon_instance
            if blockchain_daemon_instance:
                blockchain_daemon_instance.add_genesis_transaction(
                    serial_number=front_serial,
                    denomination=float(denom_str),
                    issued_to=name
                )
                if back_serial:
                    blockchain_daemon_instance.add_genesis_transaction(
                        serial_number=back_serial,
                        denomination=float(denom_str),
                        issued_to=name
                    )
        except Exception as e:
            safe_print(f"[!] Failed to add banknote to mempool: {e}")

        # DB registration (always via save_to_database)
        from app import app
        with app.app_context():
            save_to_database(name, denom_str, files, user_id)
        return files
        
    except Exception as e:
        safe_print(f"[!] Failed to generate {denom}卢纳币: {e}")
        return None
        import traceback
        traceback.print_exc()
        return None

def save_to_database(name, denom_numeric, files, user_id):
    # (Blockchain/mempool logic removed: only DB registration is performed here)
    """Save the generated banknote pair to database and add to blockchain"""
    safe_print(f"[DEBUG] Entered save_to_database for name={name}, denom={denom_numeric}, user_id={user_id}")
    if not HAS_FLASK_CONTEXT:
        safe_print(f"[!] No Flask context - skipping database save for {name}")
        return False
        safe_print(f"[DEBUG] generate_front_back_pair called for name={name}, denom={denom}, user_id={user_id}")
        
    try:
        # Convert numeric denomination to string and validate
        denom_str = str(denom_numeric)

        # Validate that the denomination is in the allowed set
        allowed_denominations = ["1", "10", "100", "1000", "10000", "100000", 
                               "1000000", "10000000", "100000000"]

        if denom_str not in allowed_denominations:
            safe_print(f"[!] Invalid denomination {denom_str}. Must be one of: {', '.join(allowed_denominations)}")
            return False

        # Prepare transaction data for blockchain
        transaction_data = {
            'type': 'banknote',
            'front_serial': files['front_serial'],
            'back_serial': files['back_serial'],
            'metadata_hash': files.get('metadata_hash', ''),
            'timestamp': int(time.time()),
            'issued_to': name,
            'denomination': denom_str,
            'public_key': files.get('public_key', ''),
            'signature': files.get('digital_signature', '')
        }

        # --- DEBUG: Print transaction data before DB/mempool submission ---
        safe_print("[DEBUG] Pre-validation transaction_data:")
        safe_print(json.dumps(transaction_data, ensure_ascii=False, indent=2))

        # Save front banknote with digital signature data
        front_banknote = Banknote(
            user_id=user_id,
            serial_number=files['front_serial'],
            seed_text=name,
            denomination=denom_str,
            side="front",
            svg_path=files['front_svg'],
            png_path=files['front_png'],
            pdf_path=files['front_pdf'],
            is_public=True,
            transaction_data=json.dumps(transaction_data),
            digital_signature=files.get('digital_signature'),
            public_key=files.get('public_key'),
            metadata_hash=files.get('metadata_hash')
        )
        # --- DEBUG: Print front_banknote fields ---
        safe_print("[DEBUG] Pre-validation front_banknote:")
        safe_print(json.dumps({
            'user_id': front_banknote.user_id,
            'serial_number': front_banknote.serial_number,
            'seed_text': front_banknote.seed_text,
            'denomination': front_banknote.denomination,
            'side': front_banknote.side,
            'svg_path': front_banknote.svg_path,
            'png_path': front_banknote.png_path,
            'pdf_path': front_banknote.pdf_path,
            'is_public': front_banknote.is_public,
            'transaction_data': front_banknote.transaction_data,
            'digital_signature': front_banknote.digital_signature,
            'public_key': front_banknote.public_key,
            'metadata_hash': front_banknote.metadata_hash
        }, ensure_ascii=False, indent=2))

        db.session.add(front_banknote)
        db.session.flush()

        front_serial_record = SerialNumber(
            serial=files['front_serial'],
            user_id=user_id,
            banknote_id=front_banknote.id,
            is_active=True
        )
        # --- DEBUG: Print front_serial_record fields ---
        safe_print("[DEBUG] Pre-validation front_serial_record:")
        safe_print(json.dumps({
            'serial': front_serial_record.serial,
            'user_id': front_serial_record.user_id,
            'banknote_id': front_serial_record.banknote_id,
            'is_active': front_serial_record.is_active
        }, ensure_ascii=False, indent=2))

        db.session.add(front_serial_record)

        back_banknote = Banknote(
            user_id=user_id,
            serial_number=files['back_serial'],
            seed_text=name,
            denomination=denom_str,
            side="back",
            svg_path=files['back_svg'],
            png_path=files['back_png'],
            pdf_path=files['back_pdf'],
            is_public=True,
            transaction_data=json.dumps(transaction_data),
            digital_signature=files.get('digital_signature'),
            public_key=files.get('public_key'),
            metadata_hash=files.get('metadata_hash')
        )
        # --- DEBUG: Print back_banknote fields ---
        safe_print("[DEBUG] Pre-validation back_banknote:")
        safe_print(json.dumps({
            'user_id': back_banknote.user_id,
            'serial_number': back_banknote.serial_number,
            'seed_text': back_banknote.seed_text,
            'denomination': back_banknote.denomination,
            'side': back_banknote.side,
            'svg_path': back_banknote.svg_path,
            'png_path': back_banknote.png_path,
            'pdf_path': back_banknote.pdf_path,
            'is_public': back_banknote.is_public,
            'transaction_data': back_banknote.transaction_data,
            'digital_signature': back_banknote.digital_signature,
            'public_key': back_banknote.public_key,
            'metadata_hash': back_banknote.metadata_hash
        }, ensure_ascii=False, indent=2))

        db.session.add(back_banknote)
        db.session.flush()

        # (Mempool/blockchain submission removed by request)
        # --- 追加: 生成ペアができたかを明示的にprint ---
        if os.path.exists(files['front_svg']) and os.path.exists(files['back_svg']):
            safe_print(f"[SUCCESS] SVG pair generated: {files['front_svg']} & {files['back_svg']}")
        else:
            safe_print(f"[FAIL] SVG pair missing: {files['front_svg']} or {files['back_svg']}")
        return True

    except Exception as e:
        db.session.rollback()
        safe_print(f"[!] Failed to save to database: {e}")
        # ...existing code...
        traceback.print_exc()
        return False

def process_denomination(args_tuple):
    """Helper function for parallel denomination processing"""
    name, denom, img_path, timestamp_ms, denom_folder, user_id, width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, background_prompt, eisenscript_front, eisenscript_back = args_tuple
    result = generate_front_back_pair(name, denom, img_path, timestamp_ms, denom_folder, user_id, width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, background_prompt, eisenscript_front, eisenscript_back)
    if result:
        result['denomination'] = denom  # Add denomination to result
    return result

def process_name(name, user_id, force_regenerate=False, specific_denom=None, single_denom=False, images=None,
               width_mm=160.0, height_mm=60.0, title="灵国国库", subtitle="天圆地方", 
               font_dir="./fonts", bg_dir="./backgrounds", dpi=300.0, bg_image=None, portrait_prompt=None, background_prompt=None,
               eisenscript_front=None, eisenscript_back=None):
    """Process a single name with all its denominations in parallel"""
    try:
        safe_print(f"[+] Processing: {name}")
        safe_print("=" * 50)
    except UnicodeEncodeError:
        safe_name = name.encode('ascii', 'replace').decode('ascii')
        safe_print(f"\n[+] Processing: {safe_name}")
        safe_print("=" * 50)

    # Get or generate ONE portrait for this name
    img_path = get_portrait_for_name(name, force_regenerate, portrait_prompt)
    
    # If portrait generation/retrieval failed, try multiple times to generate a new one
    retry_count = 0
    max_retries = 3
    
    while (not img_path or not os.path.exists(img_path)) and retry_count < max_retries:
        retry_count += 1
        safe_print(f"[!] Portrait not found for {name}, attempt {retry_count}/{max_retries} to generate...")
        img_path = generate_character_portrait(name, portrait_prompt=portrait_prompt)
        
        if img_path and os.path.exists(img_path):
            safe_print(f"[+] Successfully generated portrait: {img_path}")
            
            # Update user's avatar in database if this portrait was just created
            try:
                from app import app, db
                from models import User
                with app.app_context():
                    user = User.query.get(user_id)
                    if user:
                        # Store just the filename for the avatar
                        portrait_filename = os.path.basename(img_path)
                        # Check if user has an avatar field (some schemas may not have this)
                        if hasattr(user, 'avatar'):
                            user.avatar = portrait_filename
                            db.session.commit()
                            safe_print(f"[+] Updated user avatar to: {portrait_filename}")
                        else:
                            safe_print(f"[+] User model doesn't have avatar field, skipping avatar update")
            except Exception as avatar_error:
                safe_print(f"[!] Warning: Could not update user avatar: {avatar_error}")
            
            break
    
    # If still no portrait after retries, fail the generation
    if not img_path or not os.path.exists(img_path):
        safe_print(f"[!] Failed to generate portrait for {name} after {max_retries} attempts")
        safe_print(f"[!] Cannot proceed without a portrait. Please check SD API connection.")
        return 0

    safe_print(f"[+] Using portrait for all bills: {img_path}")

    # CLEAN THE NAME - Remove trailing/leading whitespace and invalid characters
    clean_name = name.strip()
    # Replace any remaining problematic characters
    clean_name = re.sub(r'[<>:"/\\|?*]', '_', clean_name)
    # Remove multiple consecutive spaces
    clean_name = re.sub(r'\s+', ' ', clean_name)
    
    name_folder = os.path.join(OUTPUT_ROOT, clean_name)
    os.makedirs(name_folder, exist_ok=True)

    # Determine which denominations to generate
    if specific_denom:
        if single_denom:
            denominations = [specific_denom]
            safe_print(f"[+] Generating only denomination: {specific_denom}")
        else:
            denominations = [d for d in STANDARD_DENOMINATIONS if d == specific_denom]
            safe_print(f"[+] Generating denomination: {specific_denom}")
    else:
        denominations = STANDARD_DENOMINATIONS
        safe_print(f"[+] Generating all standard denominations: {denominations}")

    # Prepare arguments for parallel processing
    args_list = []
    for denom in denominations:
        denom_str = str(denom)
        denom_numeric = int(denom)
        denom_folder = os.path.join(name_folder, denom_str)
        os.makedirs(denom_folder, exist_ok=True)
        timestamp_ms = generate_timestamp_ms_precise()

        # --- Patch: Per-denomination EisenScript variable replacement ---
        # Prepare context for this bill
        eisenscript_context = {
            "username": name,
            "user_id": user_id,
            "title": title,
            "subtitle": subtitle,
            "denomination": denom_str,
            # serial will be filled after serial is generated, but for overlays before serial, leave blank or generate here if needed
        }
        # Replace variables in overlays for this denomination
        eisenscript_front_d = render_eisenscript_jinja2(eisenscript_front, eisenscript_context) if eisenscript_front else None
        eisenscript_back_d = render_eisenscript_jinja2(eisenscript_back, eisenscript_context) if eisenscript_back else None

        args_list.append((name, denom_numeric, img_path, timestamp_ms, denom_folder, user_id, width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, background_prompt, eisenscript_front_d, eisenscript_back_d))

    # Use sequential processing to avoid subprocess issues
    svg_pairs_created = 0
    results = []
    
    safe_print("[+] Using sequential processing for stability")

    for args in args_list:
        try:
            result = process_denomination(args)
            results.append(result)
            if result:
                denom_str = str(result['denomination'])
                safe_print(f"[DEBUG] Attempting to save to database: name={name}, denom={denom_str}, user_id={user_id}, serials: front={result.get('front_serial')}, back={result.get('back_serial')}")
                db_success = save_to_database(name, denom_str, result, user_id)
                if db_success:
                    safe_print("[SUCCESS] Saved Bill to Database.")
                    svg_pairs_created += 1
                else:
                    safe_print(f"[FAIL] Failed to save to database for denomination {denom_str} (see error above)")
        except Exception as single_error:
            import traceback
            safe_print(f"[!] Sequential processing failed for denomination: {single_error}")
            traceback.print_exc()
            results.append(None)

    safe_print(f"[+] Completed {name}: {svg_pairs_created} SVG pairs created")
    
    # Send email notification if email service is available
    if svg_pairs_created > 0:
        try:
            from app import app, db
            from models import User
            with app.app_context():
                user = User.query.get(user_id)
                if user and user.email_verified:
                    from email_service import send_banknote_generation_notification
                    
                    # Collect serial numbers from results
                    serial_numbers = []
                    denoms_generated = []
                    
                    for r in results:
                        if r:
                            denoms_generated.append(r.get('denomination'))
                            # Try to get serial number from result
                            if 'serial_number' in r:
                                serial_numbers.append(r['serial_number'])
                    
                    # Format denomination info
                    if len(set(denoms_generated)) == 1:
                        denom_str = f"{denoms_generated[0]}"
                    elif len(denoms_generated) <= 3:
                        denom_str = ", ".join(str(d) for d in sorted(set(denoms_generated)))
                    else:
                        denom_str = f"Multiple ({len(set(denoms_generated))} denominations)"
                    
                    send_banknote_generation_notification(
                        user.email,
                        user.username,
                        denomination=denom_str,
                        count=svg_pairs_created,
                        serial_numbers=serial_numbers[:5] if serial_numbers else None
                    )
                    safe_print(f"[+] ✉️  Email notification sent to {user.email}")
                elif user and not user.email_verified:
                    safe_print(f"[!] Email not verified for {user.username}, skipping notification")
        except Exception as email_error:
            safe_print(f"[!] Failed to send email notification: {email_error}")
    
    return svg_pairs_created

# -----------------------
# Main API function
# -----------------------
def _merge_eisenscript(prefix: str, user_script: str, suffix: str) -> str:
    parts = []
    for part in (prefix, user_script, suffix):
        if part:
            parts.append(part.strip())
    return "\n\n".join(parts)

# --- Patch: EisenScript variable support in overlays ---
def merge_eisenscript_with_vars(prefix, user_script, suffix, context):
    """
    Merge and replace variables in EisenScript overlays.
    """
    merged = _merge_eisenscript(prefix, user_script, suffix)
    return render_eisenscript_jinja2(merged, context)


def generate_for_user(username, user_id, force_regenerate=False, specific_denom=None, single_denom=False, max_threads=1,
                   width_mm=None, height_mm=None, title=None, subtitle=None, 
                   font_dir=None, bg_dir=None, dpi=None, bg_image=None, background_prompt=None, portrait_prompt=None,
                   custom_eisenscript=None):
    """
    Generate banknotes for a specific user
    
    Args:
        username (str): The name to generate banknotes for
        user_id (int): The user ID for database association
        force_regenerate (bool): Whether to force regeneration of portraits
        specific_denom (int): Specific denomination to generate (None for all)
        single_denom (bool): If True, generate only the specific denomination
        max_threads (int): Maximum number of parallel threads
        width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, portrait_prompt, background_prompt: Override settings from database
    """
    # Get settings from database
    settings = Settings.query.first()
    if settings:
        # Use database settings as defaults, but allow overrides
        width_mm = width_mm if width_mm is not None else settings.bill_width_mm
        height_mm = height_mm if height_mm is not None else settings.bill_height_mm
        title = title if title is not None else settings.bill_title
        subtitle = subtitle if subtitle is not None else settings.bill_subtitle
        font_dir = font_dir if font_dir is not None else settings.font_dir
        bg_dir = bg_dir if bg_dir is not None else settings.bg_dir
        dpi = dpi if dpi is not None else settings.bill_dpi
        # Get background_prompt from settings first (check for non-empty), then file as fallback
        if background_prompt is None:
            background_prompt = (settings.background_prompt if settings.background_prompt and settings.background_prompt.strip() 
                               else read_prompt_file("background_prompt.txt", "A beautiful fantasy landscape with mountains and rivers, mystical atmosphere"))
        # Get portrait_prompt from settings first (check for non-empty), then file as fallback  
        if portrait_prompt is None:
            portrait_prompt = (settings.portrait_prompt if settings.portrait_prompt and settings.portrait_prompt.strip()
                             else read_prompt_file("portrait_prompt.txt", "A professional portrait of a person, high quality, studio lighting, detailed face"))
    else:
        # Fallback defaults if no settings in database
        width_mm = width_mm or 160.0
        height_mm = height_mm or 60.0
        title = title or "灵国国库"
        subtitle = subtitle or "天圆地方"
        font_dir = font_dir or "./fonts"
        bg_dir = bg_dir or "./backgrounds"
        dpi = dpi or 300.0
        # Try to read prompts from file if no prompt provided
        if not background_prompt:
            background_prompt = read_prompt_file("background_prompt.txt", "A beautiful fantasy landscape with mountains and rivers, mystical atmosphere")
        if not portrait_prompt:
            portrait_prompt = read_prompt_file("portrait_prompt.txt", "A professional portrait of a person, high quality, studio lighting, detailed face")

    # Load existing portraits
    images = []
    if os.path.exists(PORTRAITS_DIR):
        for ext in IMAGE_EXTS:
            pattern = os.path.join(PORTRAITS_DIR, f"*{ext}")
            images.extend(glob.glob(pattern))
    
    # Ensure prompts are not None before passing
    if not portrait_prompt:
        portrait_prompt = "A professional portrait of a person, high quality, studio lighting, detailed face"
    if not background_prompt:
        background_prompt = "A beautiful fantasy landscape with mountains and rivers, mystical atmosphere"

    eisenscript_prefix_front = settings.eisenscript_prefix_front if settings else ""
    eisenscript_suffix_front = settings.eisenscript_suffix_front if settings else ""
    eisenscript_prefix_back = settings.eisenscript_prefix_back if settings else ""
    eisenscript_suffix_back = settings.eisenscript_suffix_back if settings else ""

    # Prepare context for variable replacement
    eisenscript_context = {
        "username": username,
        "user_id": user_id,
        "title": title,
        "subtitle": subtitle,
        # serial/denomination will be filled per-denomination if needed
    }
    # For global overlays, $serial/$denomination will be blank, but per-denomination code can extend this
    combined_front = merge_eisenscript_with_vars(eisenscript_prefix_front, custom_eisenscript, eisenscript_suffix_front, eisenscript_context)
    combined_back = merge_eisenscript_with_vars(eisenscript_prefix_back, custom_eisenscript, eisenscript_suffix_back, eisenscript_context)

    # Process the name with all denominations
    return process_name(username, user_id, force_regenerate, specific_denom, single_denom, images,
                       width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, portrait_prompt, background_prompt,
                       eisenscript_front=combined_front, eisenscript_back=combined_back)

# -----------------------
# Command line argument parsing
# -----------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate banknotes for a user")
    parser.add_argument("--name", required=True, help="Name for banknote generation")
    parser.add_argument("--user-id", type=int, required=True, help="User ID for database association")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regeneration of portraits")
    parser.add_argument("--denom", type=int, help="Specific denomination to generate")
    parser.add_argument("--single-denom", action="store_true", help="Generate only the specific denomination")
    parser.add_argument("--threads", type=int, default=1, help="Maximum number of threads")
    # New customization arguments
    parser.add_argument("--width-mm", type=float, default=160.0, help="Width in mm (default: 160.0)")
    parser.add_argument("--height-mm", type=float, default=60.0, help="Height in mm (default: 60.0)")
    parser.add_argument("--title", type=str, default="灵国国库", help="Title text (default: 灵国国库)")
    parser.add_argument("--subtitle", type=str, default="天圆地方", help="Subtitle text (default: 天圆地方)")
    parser.add_argument("--font-dir", type=str, default="./fonts", help="Directory containing font files (default: ./fonts)")
    parser.add_argument("--bg-dir", type=str, default="./backgrounds", help="Directory containing background images (default: ./backgrounds)")
    parser.add_argument("--dpi", type=float, default=300.0, help="Resolution in DPI (default: 300.0)")
    parser.add_argument("--bg-image", type=str, help="Background image path for back")
    parser.add_argument("--background-prompt", type=str, help="Background generation prompt")
    parser.add_argument("--eisenscript", type=str, help="Inline EisenScript overlay")
    parser.add_argument("--eisenscript-file", type=str, help="Path to EisenScript file")
    return parser.parse_args()

    def main():
        args = parse_arguments()
        
        if not args.name or not args.user_id:
            print("Error: --name and --user_id are required")
            return 1
        
        custom_eisenscript = args.eisenscript
        if not custom_eisenscript and args.eisenscript_file and os.path.exists(args.eisenscript_file):
            try:
                with open(args.eisenscript_file, "r", encoding="utf-8") as f:
                    custom_eisenscript = f.read()
            except Exception as script_error:
                safe_print(f"[!] Failed to read Eisenscript file: {script_error}")

        # Use the API function
        result = generate_for_user(
            username=args.name,
            user_id=args.user_id,
            force_regenerate=args.force_regenerate,
            specific_denom=args.denom,
            single_denom=args.single_denom,
            max_threads=args.threads,
            width_mm=args.width_mm,
            height_mm=args.height_mm,
            title=args.title,
            subtitle=args.subtitle,
            font_dir=args.font_dir,
            bg_dir=args.bg_dir,
            dpi=args.dpi,
            bg_image=args.bg_image,
            background_prompt=args.background_prompt,
            custom_eisenscript=custom_eisenscript
        )
        
        safe_print(f"\n[+] Banknote generation finished! Created {result} SVG pairs!")
        return 0

    if __name__ == "__main__":
        sys.exit(main())