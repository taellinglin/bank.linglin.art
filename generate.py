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
import datetime
import secrets
import hashlib
import xml.etree.ElementTree as ET
import cairosvg
import sys
# Add the current directory to Python path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# IMPORTS WITH BETTER ERROR HANDLING
# =============================================================================

import os
import sys
import traceback

# Add current directory to Python path FIRST
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"[DEBUG] Current directory: {os.getcwd()}")
print(f"[DEBUG] Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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
        def __init__(self):
            pass
        def create_signed_bill(self, bill_data, private_key):
            import hashlib
            import json
            return type('obj', (object,), {
                'signature': hashlib.sha256(json.dumps(bill_data).encode()).hexdigest(),
                'public_key': 'fallback_key'
            })
        def verify_bill_signature(self, bill_data):
            return True
    
    def generate_key_pair():
        import hashlib
        import secrets
        priv = secrets.token_hex(32)
        pub = f"04{hashlib.sha256(priv.encode()).hexdigest()[:64]}"
        return priv, pub
    
    print("[!] Using fallback signatures")
    HAS_SIGNATURES = True  # Still mark as available for fallback

# Try to import Flask models
try:
    from models import Banknote, SerialNumber, User, Settings, db
    HAS_FLASK_CONTEXT = True
    print("[+] Successfully imported database models")
except ImportError as e:
    print(f"[!] Failed to import Flask models: {e}")
    HAS_FLASK_CONTEXT = False
    # Create dummy classes for fallback
    class Banknote:
        pass
    class SerialNumber:
        pass
    class User:
        pass
    class db:
        class session:
            @staticmethod
            def add(obj): pass
            @staticmethod
            def commit(): pass
            @staticmethod
            def rollback(): pass
            @staticmethod
            def flush(): pass
            @staticmethod
            def query(cls): 
                class Query:
                    def get(self, id): return None
                    def filter_by(self, **kwargs): return self
                    def first(self): return None
                return Query()
    
    print("[!] Using fallback database classes")

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
SD_API_URL = "http://localhost:3333/sdapi/v1/txt2img"
SD_API_URL_GPU0 = os.getenv("SD_API_URL_GPU0", "http://localhost:3333/sdapi/v1/txt2img")
SD_API_URL_GPU1 = os.getenv("SD_API_URL_GPU1", "http://localhost:3333/sdapi/v1/txt2img")

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
    ts = timestamp_ms or int(datetime.datetime.now().timestamp() * 1000000)
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
    ts = timestamp_ms or int(datetime.datetime.now().timestamp() * 1000000)
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
    now = datetime.datetime.now()
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
    return f"{name}_-_{denom}_-_{timestamp}_{side}.svg"

def create_basename(name, denom, timestamp, side):
    """Create filename in format: {name}_-_{denom}_-_{timestamp}_{side}"""
    return f"{name}_-_{denom}_-_{timestamp}_{side}"

def safe_print(message):
    """Print message with Unicode fallback handling"""
    try:
        print(message)
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message)

def generate_png_from_svg(svg_path, png_path, size=(1600, 600)):
    """Generate PNG from SVG file using cairosvg"""
    try:
        # Convert SVG to PNG
        cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=size[0], output_height=size[1])
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
                          use_parallel=True):
    """Generate a front+back pair for a single denomination with optional parallel processing"""
    front_serial = generate_serial_id_with_checksum(timestamp_ms)
    back_serial = generate_serial_id_combined(timestamp_ms)
    
    denom_str = str(denom)  # Ensure denomination is string
    safe_print(f"[+] Generating {denom}卢纳币 bill (timestamp: {timestamp_ms})")
    
    # Generate digital signature for the banknote
    digital_signature_data = create_digital_banknote_signature(
        name=name,
        denomination=denom,
        serial_number=front_serial,
        timestamp_ms=timestamp_ms
    )
    
    safe_print(f"[+] Created digital signature for serial: {front_serial}")
    
    # Prepare filenames
    front_filename = create_proper_filename(name, denom_str, timestamp_ms, "FRONT")
    front_svg_path = os.path.join(denom_folder, front_filename)
    back_basename = create_basename(name, denom_str, timestamp_ms, "BACK")
    back_svg_path = os.path.join(denom_folder, f"{back_basename}.svg")
    
    def generate_front_task():
        """Generate front in separate thread"""
        try:
            # Set GPU 0 environment if multi-GPU is enabled
            gpu_env = os.environ.copy()
            if MULTI_GPU_ENABLED:
                gpu_env['CUDA_VISIBLE_DEVICES'] = '0'
                safe_print(f"[GPU0] Generating front for {denom}卢纳币")
            
            # Try using imported function first
            if HAS_FRONT_GENERATOR and generate_front:
                generate_front(
                    seed_text=name,
                    input_image_path=img_path,
                    single_denom=denom_str,
                    outfile=front_svg_path,
                    serial_id=front_serial,
                    timestamp=int(timestamp_ms),
                    background_prompt=background_prompt
                )
            else:
                safe_name = name.replace('&', '_')
                subprocess.run([
                    'python', FRONT_SCRIPT,
                    safe_name,
                    img_path,
                    '--outfile', front_svg_path,
                    '--single_denom', denom_str,
                    '--serial_id', front_serial,
                    '--timestamp', str(int(timestamp_ms)),
                    '--width-mm', str(width_mm),
                    '--height-mm', str(height_mm),
                    '--title', title,
                    '--subtitle', subtitle,
                    '--font-dir', font_dir,
                    '--bg-dir', bg_dir,
                    '--dpi', str(dpi),
                    '--background-prompt', background_prompt or ''
                ], check=True, timeout=13131313, env=gpu_env)
            
            safe_print(f"[+] Generated front: {front_svg_path}")
            return True
        except Exception as e:
            safe_print(f"[!] Failed to generate front: {e}")
            return False
    
    def generate_back_task():
        """Generate back in separate thread"""
        try:
            # Set GPU 1 environment if multi-GPU is enabled
            gpu_env = os.environ.copy()
            if MULTI_GPU_ENABLED:
                gpu_env['CUDA_VISIBLE_DEVICES'] = '1'
                safe_print(f"[GPU1] Generating back for {denom}卢纳币")
            
            if HAS_BACK_GENERATOR and generate_back:
                generate_back(
                    outdir=denom_folder,
                    base_name=back_basename,
                    denomination=denom_str,
                    seed_text=name,
                    serial_id=back_serial,
                    timestamp=int(timestamp_ms)
                )
            else:
                safe_name = name.replace('&', '_')
                subprocess.run([
                    'python', BACK_SCRIPT,
                    '--outdir', denom_folder,
                    '--basename', back_basename,
                    '--denomination', denom_str,
                    '--seed_text', safe_name,
                    '--serial_id', back_serial,
                    '--timestamp', str(int(timestamp_ms)),
                    '--width-mm', str(width_mm),
                    '--height-mm', str(height_mm),
                    '--title', title,
                    '--phrase', subtitle,
                    '--dpi', str(dpi)
                ] + (['--bg-image', bg_image] if bg_image else []), check=True, timeout=13131313, env=gpu_env)
            
            safe_print(f"[+] Generated back: {back_svg_path}")
            return True
        except Exception as e:
            safe_print(f"[!] Failed to generate back: {e}")
            return False
    
    try:
        if use_parallel and MULTI_GPU_ENABLED:
            # Generate front and back in parallel using threading
            safe_print(f"[PARALLEL] Using multi-GPU parallel generation")
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                front_future = executor.submit(generate_front_task)
                back_future = executor.submit(generate_back_task)
                
                # Wait for both to complete
                front_success = front_future.result()
                back_success = back_future.result()
                
                if not (front_success and back_success):
                    raise Exception("Failed to generate front or back")
        else:
            # Sequential generation
            safe_print(f"[SEQUENTIAL] Using single GPU sequential generation")
            if not generate_front_task():
                raise Exception("Failed to generate front")
            if not generate_back_task():
                raise Exception("Failed to generate back")
        
        # Generate PNG and PDF files
        front_png_path = front_svg_path.replace(".svg", ".png")
        front_pdf_path = front_svg_path.replace(".svg", ".pdf")
        back_png_path = back_svg_path.replace(".svg", ".png")
        back_pdf_path = back_svg_path.replace(".svg", ".pdf")
        
        # Generate PNGs
        generate_png_from_svg(front_svg_path, front_png_path)
        generate_png_from_svg(back_svg_path, back_png_path)
        
        # Generate PDFs (commented out for now)
        # generate_pdf_from_svg(front_svg_path, front_pdf_path)
        # generate_pdf_from_svg(back_svg_path, back_pdf_path)
        back_pdf_path = ""
        
        return {
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
        
    except Exception as e:
        safe_print(f"[!] Failed to generate {denom}卢纳币: {e}")
        return None
        import traceback
        traceback.print_exc()
        return None

def save_to_database(name, denom_numeric, files, user_id):
    """Save the generated banknote pair to database and add to blockchain"""
    if not HAS_FLASK_CONTEXT:
        safe_print(f"[!] No Flask context - skipping database save for {name}")
        return False
        
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
        db.session.add(front_banknote)
        db.session.flush()
        
        front_serial_record = SerialNumber(
            serial=files['front_serial'],
            user_id=user_id,
            banknote_id=front_banknote.id,
            is_active=True
        )
        db.session.add(front_serial_record)
        
        # Save back banknote
        back_banknote = Banknote(
            user_id=user_id,
            serial_number=files['back_serial'],
            seed_text=name,
            denomination=denom_str,
            side="back",
            svg_path=files['back_svg'],
            png_path=files['back_png'],
            pdf_path=files['back_pdf'],
            is_public=True
        )
        db.session.add(back_banknote)
        db.session.flush()
        
        back_serial_record = SerialNumber(
            serial=files['back_serial'],
            user_id=user_id,
            banknote_id=back_banknote.id,
            is_active=True
        )
        db.session.add(back_serial_record)
        
        # Update user balance
        user = User.query.get(user_id)
        if user:
            denom_value = float(denom_str)
            user.balance += denom_value
            user.last_generation = datetime.datetime.utcnow()
        
        db.session.commit()
        safe_print(f"[+] Added banknote pair to DB for {denom_str} 卢纳币")
        safe_print(f"[+] Digital signature: {files.get('digital_signature', 'N/A')[:20]}...")

        # Add genesis transactions to blockchain AFTER DB commit succeeds
        try:
            from app import blockchain_daemon_instance

            safe_print(f"[DEBUG] Blockchain daemon instance: {blockchain_daemon_instance}")
            safe_print(f"[DEBUG] Blockchain daemon type: {type(blockchain_daemon_instance)}")

            if blockchain_daemon_instance:
                safe_print(f"[DEBUG] Blockchain daemon attributes: {[attr for attr in dir(blockchain_daemon_instance) if not attr.startswith('_')]}")

                if hasattr(blockchain_daemon_instance, 'mempool'):
                    safe_print(f"[DEBUG] Mempool size: {len(blockchain_daemon_instance.mempool)}")
                else:
                    safe_print(f"[DEBUG] No mempool attribute found")

                if hasattr(blockchain_daemon_instance, 'is_running'):
                    safe_print(f"[DEBUG] Daemon running: {blockchain_daemon_instance.is_running}")

                safe_print(f"[DEBUG] Adding genesis transaction for serial: {files['front_serial']}")
                genesis_success = blockchain_daemon_instance.add_genesis_transaction(
                    serial_number=files['front_serial'],
                    denomination=float(denom_str),
                    issued_to=name
                )

                back_genesis_success = False
                if files.get('back_serial'):
                    safe_print(f"[DEBUG] Adding genesis transaction for serial: {files['back_serial']}")
                    back_genesis_success = blockchain_daemon_instance.add_genesis_transaction(
                        serial_number=files['back_serial'],
                        denomination=float(denom_str),
                        issued_to=name
                    )

                safe_print(f"[DEBUG] Genesis transaction result (front): {genesis_success}")
                safe_print(f"[DEBUG] Genesis transaction result (back): {back_genesis_success}")

                if genesis_success:
                    safe_print(f"[+] ✓ Genesis transaction added to mempool for serial: {files['front_serial']}")
                else:
                    safe_print(f"[!] Failed to add genesis transaction for serial: {files['front_serial']}")

                if files.get('back_serial'):
                    if back_genesis_success:
                        safe_print(f"[+] ✓ Genesis transaction added to mempool for serial: {files['back_serial']}")
                    else:
                        safe_print(f"[!] Failed to add genesis transaction for serial: {files['back_serial']}")

                if hasattr(blockchain_daemon_instance, 'mempool'):
                    safe_print(f"[DEBUG] Mempool size after add: {len(blockchain_daemon_instance.mempool)}")
                    genesis_txs = [tx for tx in blockchain_daemon_instance.mempool if tx.get('type') in ['genesis', 'GTX_Genesis']]
                    safe_print(f"[DEBUG] Genesis transactions in mempool: {len(genesis_txs)}")
            else:
                safe_print(f"[!] Blockchain daemon instance is None - not initialized")

        except ImportError as e:
            safe_print(f"[!] Could not import blockchain_daemon_instance: {e}")
        except Exception as e:
            safe_print(f"[!] Error with blockchain integration: {e}")
            import traceback
            traceback.print_exc()
        return True
        
    except Exception as e:
        db.session.rollback()
        safe_print(f"[!] Failed to save to database: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_denomination(args_tuple):
    """Helper function for parallel denomination processing"""
    name, denom, img_path, timestamp_ms, denom_folder, user_id, width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, background_prompt = args_tuple
    result = generate_front_back_pair(name, denom, img_path, timestamp_ms, denom_folder, user_id, width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, background_prompt)
    if result:
        result['denomination'] = denom  # Add denomination to result
    return result

def process_name(name, user_id, force_regenerate=False, specific_denom=None, single_denom=False, images=None,
               width_mm=160.0, height_mm=60.0, title="灵国国库", subtitle="天圆地方", 
               font_dir="./fonts", bg_dir="./backgrounds", dpi=300.0, bg_image=None, portrait_prompt=None, background_prompt=None):
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
        args_list.append((name, denom_numeric, img_path, timestamp_ms, denom_folder, user_id, width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, background_prompt))

    # Use sequential processing to avoid subprocess issues
    svg_pairs_created = 0
    results = []
    
    safe_print("[+] Using sequential processing for stability")
    for args in args_list:
        try:
            result = process_denomination(args)
            results.append(result)
        except Exception as single_error:
            safe_print(f"[!] Sequential processing failed for denomination: {single_error}")
            results.append(None)
    
    for result in results:
        if result:
            denom_str = str(result['denomination'])
            if save_to_database(name, denom_str, result, user_id):
                safe_print("Saved Bill to Database.")
                svg_pairs_created += 1

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
def generate_for_user(username, user_id, force_regenerate=False, specific_denom=None, single_denom=False, max_threads=1,
                   width_mm=None, height_mm=None, title=None, subtitle=None, 
                   font_dir=None, bg_dir=None, dpi=None, bg_image=None, background_prompt=None, portrait_prompt=None):
    """
    Generate banknotes for a specific user
    
    Args:
        username (str): The name to generate banknotes for
        user_id (int): The user ID for database association
        force_regenerate (bool): Whether to force regeneration of portraits
        specific_denom (int): Specific denomination to generate (None for all)
        single_denom (bool): If True, generate only the specific denomination
        max_threads (int): Maximum number of parallel threads
        width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, portrait_prompt: Override settings from database
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

    # Process the name with all denominations
    return process_name(username, user_id, force_regenerate, specific_denom, single_denom, images,
                       width_mm, height_mm, title, subtitle, font_dir, bg_dir, dpi, bg_image, portrait_prompt, background_prompt)

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
    return parser.parse_args()

    def main():
        args = parse_arguments()
        
        if not args.name or not args.user_id:
            print("Error: --name and --user_id are required")
            return 1
        
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
            background_prompt=args.background_prompt
        )
        
        safe_print(f"\n[+] Banknote generation finished! Created {result} SVG pairs!")
        return 0

    if __name__ == "__main__":
        sys.exit(main())