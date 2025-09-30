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

# Try to import database models and app (will fail if not in Flask context)
try:
    from models import Banknote, SerialNumber, User, db
    from signatures import DigitalSignatureManager, generate_key_pair
    HAS_FLASK_CONTEXT = True
    HAS_SIGNATURES = True
except ImportError as e:
    HAS_FLASK_CONTEXT = False
    HAS_SIGNATURES = False
    print(f"[!] Running without Flask context - database operations disabled: {e}")

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
SD_API_URL = "http://localhost:3014/sdapi/v1/txt2img"
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
                               seed: int = -1, save_path: str = "./portraits"):
    """
    Generate a character portrait based on the name using Stable Diffusion API
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Read prompts from files
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

def get_portrait_for_name(name, force_regenerate=False):
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
    
    if not force_regenerate:
        for pattern in portrait_patterns:
            existing_portraits = glob.glob(pattern)
            if existing_portraits:
                safe_print(f"[+] Using existing portrait: {existing_portraits[0]}")
                return existing_portraits[0]
    
    # Generate new portrait with consistent filename
    return generate_character_portrait(name)

# -----------------------
# Banknote generation functions
# -----------------------
def generate_front_back_pair(name, denom, img_path, timestamp_ms, denom_folder, user_id=None):
    """Generate a front+back pair for a single denomination"""
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
    
    # Generate front
    front_filename = create_proper_filename(name, denom_str, timestamp_ms, "FRONT")
    front_svg_path = os.path.join(denom_folder, front_filename)
    
    try:
        # Try using imported function first
        if HAS_FRONT_GENERATOR and generate_front:
            generate_front(
                seed_text=name,
                input_image_path=img_path,
                single_denom=denom_str,  # Already string
                outfile=front_svg_path,
                serial_id=front_serial,
                timestamp=int(timestamp_ms)
            )
        else:
            # Fallback to subprocess - ensure all arguments are strings
            subprocess.run([
                'python', FRONT_SCRIPT,
                name,
                img_path,
                '--outfile', front_svg_path,
                '--single_denom', denom_str,  # Already string
                '--serial_id', front_serial,
                '--timestamp', str(int(timestamp_ms))  # Convert to string
            ], check=True, timeout=13131313)
        
        safe_print(f"[+] Generated front: {front_svg_path}")
        
        # Generate back
        back_basename = create_basename(name, denom_str, timestamp_ms, "BACK")
        back_svg_path = os.path.join(denom_folder, f"{back_basename}.svg")
        
        if HAS_BACK_GENERATOR and generate_back:
            generate_back(
                outdir=denom_folder,
                base_name=back_basename,
                denomination=denom_str,  # Pass as string
                seed_text=name,
                serial_id=back_serial,
                timestamp=int(timestamp_ms)
            )
        else:
            # Fallback to subprocess - ensure all arguments are strings
            subprocess.run([
                'python', BACK_SCRIPT,
                '--outdir', denom_folder,
                '--basename', back_basename,
                '--denomination', denom_str,  # Already string
                '--seed_text', name,
                '--serial_id', back_serial,
                '--timestamp', str(int(timestamp_ms))  # Convert to string
            ], check=True, timeout=13131313)
        
        safe_print(f"[+] Generated back: {back_svg_path}")
        
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
            
            # Update this section in save_to_database function:
            # In your save_to_database function, replace the blockchain section with:

            # Add genesis transaction to blockchain daemon - WITH DEBUGGING
            try:
                from app import blockchain_daemon_instance
                
                safe_print(f"[DEBUG] Blockchain daemon instance: {blockchain_daemon_instance}")
                safe_print(f"[DEBUG] Blockchain daemon type: {type(blockchain_daemon_instance)}")
                
                if blockchain_daemon_instance:
                    safe_print(f"[DEBUG] Blockchain daemon attributes: {[attr for attr in dir(blockchain_daemon_instance) if not attr.startswith('_')]}")
                    
                    # Check if mempool exists
                    if hasattr(blockchain_daemon_instance, 'mempool'):
                        safe_print(f"[DEBUG] Mempool size: {len(blockchain_daemon_instance.mempool)}")
                    else:
                        safe_print(f"[DEBUG] No mempool attribute found")
                    
                    # Check if the daemon is running
                    if hasattr(blockchain_daemon_instance, 'is_running'):
                        safe_print(f"[DEBUG] Daemon running: {blockchain_daemon_instance.is_running}")
                    
                    # Create genesis transaction for the front serial
                    safe_print(f"[DEBUG] Adding genesis transaction for serial: {files['front_serial']}")
                    genesis_success = blockchain_daemon_instance.add_genesis_transaction(
                        serial_number=files['front_serial'],
                        denomination=denom_value,
                        issued_to=name
                    )
                    
                    safe_print(f"[DEBUG] Genesis transaction result: {genesis_success}")
                    
                    if genesis_success:
                        safe_print(f"[+] ✓ Genesis transaction added to mempool for serial: {files['front_serial']}")
                        # Verify it was added
                        if hasattr(blockchain_daemon_instance, 'mempool'):
                            safe_print(f"[DEBUG] Mempool size after add: {len(blockchain_daemon_instance.mempool)}")
                            
                            # Show what's in the mempool
                            genesis_txs = [tx for tx in blockchain_daemon_instance.mempool if tx.get('type') in ['genesis', 'GTX_Genesis']]
                            safe_print(f"[DEBUG] Genesis transactions in mempool: {len(genesis_txs)}")
                    else:
                        safe_print(f"[!] Failed to add genesis transaction for serial: {files['front_serial']}")
                else:
                    safe_print(f"[!] Blockchain daemon instance is None - not initialized")
                    
            except ImportError as e:
                safe_print(f"[!] Could not import blockchain_daemon_instance: {e}")
            except Exception as e:
                safe_print(f"[!] Error with blockchain integration: {e}")
                import traceback
                traceback.print_exc()
                        
            user.last_generation = datetime.datetime.utcnow()
        
        db.session.commit()
        safe_print(f"[+] Added banknote pair to DB for {denom_str} 卢纳币")
        safe_print(f"[+] Digital signature: {files.get('digital_signature', 'N/A')[:20]}...")
        return True
        
    except Exception as e:
        db.session.rollback()
        safe_print(f"[!] Failed to save to database: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_denomination(args_tuple):
    """Helper function for parallel denomination processing"""
    name, denom, img_path, timestamp_ms, denom_folder, user_id = args_tuple
    result = generate_front_back_pair(name, denom, img_path, timestamp_ms, denom_folder, user_id)
    if result:
        result['denomination'] = denom  # Add denomination to result
    return result

def process_name(name, user_id, force_regenerate=False, specific_denom=None, single_denom=False, images=None):
    """Process a single name with all its denominations in parallel"""
    try:
        safe_print(f"[+] Processing: {name}")
        safe_print("=" * 50)
    except UnicodeEncodeError:
        safe_name = name.encode('ascii', 'replace').decode('ascii')
        safe_print(f"\n[+] Processing: {safe_name}")
        safe_print("=" * 50)

    # Get or generate ONE portrait for this name
    img_path = get_portrait_for_name(name, force_regenerate)
    if not img_path:
        safe_print(f"[!] Failed to get portrait for {name}, using random existing one")
        if images and isinstance(images, list) and len(images) > 0:
            valid_images = [img for img in images if isinstance(img, str) and os.path.exists(img)]
            if valid_images:
                img_path = random.choice(valid_images)
                safe_print(f"[+] Using random portrait: {img_path}")
            else:
                safe_print(f"[!] No valid portraits available for {name}, skipping")
                return 0
        else:
            safe_print(f"[!] No portraits available for {name}, skipping")
            return 0

    safe_print(f"[+] Using portrait for all bills: {img_path}")

    name_folder = os.path.join(OUTPUT_ROOT, name)
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
        args_list.append((name, denom_numeric, img_path, timestamp_ms, denom_folder, user_id))

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
    return svg_pairs_created

# -----------------------
# Main API function
# -----------------------
def generate_for_user(username, user_id, force_regenerate=False, specific_denom=None, single_denom=False, max_threads=1):
    """
    Generate banknotes for a specific user
    
    Args:
        username (str): The name to generate banknotes for
        user_id (int): The user ID for database association
        force_regenerate (bool): Whether to force regeneration of portraits
        specific_denom (int): Specific denomination to generate (None for all)
        single_denom (bool): If True, generate only the specific denomination
        max_threads (int): Maximum number of parallel threads
    
    Returns:
        int: Number of SVG pairs created (1 for single denomination, more for multiple)
    """
    # Load existing portraits
    images = []
    if os.path.exists(PORTRAITS_DIR):
        for ext in IMAGE_EXTS:
            pattern = os.path.join(PORTRAITS_DIR, f"*{ext}")
            images.extend(glob.glob(pattern))
        images = [img for img in images if os.path.isfile(img)]
        safe_print(f"[+] Found {len(images)} existing portraits")

    # For single denomination, we only want to create 1 pair
    if single_denom and specific_denom:
        safe_print(f"[SINGLE DENOM] Generating only denomination {specific_denom}")
        result = process_name(username, user_id, force_regenerate, specific_denom, True, images)
        return result
    else:
        # For multiple denominations, process normally
        return process_name(username, user_id, force_regenerate, specific_denom, single_denom, images)

# -----------------------
# Standalone execution
# -----------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate banknotes for names")
    parser.add_argument("--name", type=str, help="Generate notes for a specific name only")
    parser.add_argument("--user_id", type=int, help="User ID")
    parser.add_argument("--denom", type=int, help="Generate notes for a specific denomination only")
    parser.add_argument("--force-regenerate", action="store_true", 
                       help="Force regeneration of portraits even if they exist")
    parser.add_argument("--threads", type=int, default=MAX_THREADS,
                       help=f"Number of parallel threads (default: {MAX_THREADS})")
    parser.add_argument("--single-denom", action="store_true",
                       help="Generate only one denomination (use with --denom)")
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
        max_threads=args.threads
    )
    
    safe_print(f"\n[+] Banknote generation finished! Created {result} SVG pairs!")
    return 0

if __name__ == "__main__":
    sys.exit(main())