# utils.py
import os
import unicodedata
import subprocess
import threading
from datetime import datetime, timedelta
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from urllib.request import pathname2url
import glob
from PIL import Image
import cairosvg
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import pyotp
import qrcode
import io
import base64
from pyzbar.pyzbar import decode as qr_decode
from flask import flash, redirect, url_for
from sqlalchemy import desc
from models import db, User, GenerationTask, Banknote, SerialNumber
import bleach
from bleach.sanitizer import ALLOWED_TAGS, ALLOWED_ATTRIBUTES
from sqlalchemy.exc import IntegrityError
import sys
import cv2
import numpy as np
from qreader import QReader
from urllib.parse import urlparse, parse_qs
import time
import queue
import concurrent.futures
from typing import Dict, List, Optional, Tuple

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration
IMAGES_ROOT = "./images"
GENERATION_LOCK = threading.Lock()
GENERATION_THREADS = {}

# Initialize the advanced QR reader
qreader = QReader()

# =============================================================================
# TASK QUEUE SYSTEM (NEW)
# =============================================================================

import threading
import time
import queue
from typing import Dict, List, Optional, Tuple


import subprocess
import threading
import time
from typing import Dict, List, Optional
import os
import json
from flask import current_app

# In utils.py, replace the GenerationQueue with a threaded version
class GenerationQueue:
    """Queue that runs generation in separate threads instead of processes"""
    def __init__(self, max_workers=1):
        self.active_tasks: Dict[int, int] = {}  # user_id -> task_id
        self.lock = threading.Lock()
    
    def add_task(self, user_id: int, username: str) -> Optional[int]:
        """Add a generation task - returns immediately"""
        try:
            from app import app
            with app.app_context():
                # Create task record
                task = GenerationTask(
                    user_id=user_id, 
                    status='queued', 
                    message="Starting generation process..."
                )
                db.session.add(task)
                db.session.commit()
                task_id = task.id
                
                # Launch generation in separate THREAD (not process)
                self._launch_generation_thread(user_id, username, task_id)
                
                print(f"[QUEUE] Launched generation thread for user {user_id}, task {task_id}")
                return task_id
                
        except Exception as e:
            print(f"[QUEUE ERROR] Failed to start generation: {e}")
            return None
    
    def _launch_generation_thread(self, user_id: int, username: str, task_id: int):
        """Launch generation in a thread with proper app context"""
        def run_generation():
            try:
                from app import app
                with app.app_context():
                    from generate import generate_for_user
                    from models import GenerationTask, db
                    
                    # Update task status
                    task = GenerationTask.query.get(task_id)
                    if task:
                        task.status = 'processing'
                        task.message = "Generation in progress..."
                        db.session.commit()
                    
                    denominations = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
                    results = []
                    total_pairs = 0
                    
                    # Process denominations
                    for i, denom in enumerate(denominations):
                        try:
                            if task:
                                task.message = f"Generating denomination {denom} ({i+1}/{len(denominations)})..."
                                db.session.commit()
                            
                            pairs_created = generate_for_user(
                                username=username,
                                user_id=user_id,
                                force_regenerate=False,
                                specific_denom=denom,
                                single_denom=True,
                                max_threads=1
                            )
                            
                            results.append({
                                'denom': denom,
                                'success': True,
                                'pairs_created': pairs_created
                            })
                            total_pairs += pairs_created
                            
                            print(f"[GENERATION] Denomination {denom}: {pairs_created} pairs created")
                            
                        except Exception as e:
                            error_msg = f"Error generating {denom}: {str(e)}"
                            print(f"[GENERATION ERROR] {error_msg}")
                            results.append({
                                'denom': denom,
                                'success': False,
                                'error': error_msg
                            })
                    
                    # Calculate final status
                    successful = [r for r in results if r['success']]
                    failed = [r for r in results if not r['success']]
                    
                    if len(successful) == len(denominations):
                        status = 'completed'
                        message = f"All {len(denominations)} denominations generated! {total_pairs} pairs created."
                    elif successful:
                        status = 'partial'
                        message = f"Partial: {len(successful)}/{len(denominations)} denominations. {total_pairs} pairs created."
                    else:
                        status = 'failed'
                        message = "All denominations failed to generate."
                    
                    # Update task
                    if task:
                        task.status = status
                        task.message = message
                        task.completed_at = datetime.utcnow()
                        db.session.commit()
                    
                    # Clean up
                    with self.lock:
                        if user_id in self.active_tasks and self.active_tasks[user_id] == task_id:
                            del self.active_tasks[user_id]
                    
                    print(f"[THREAD] Generation completed for task {task_id}: {status}")
                    
            except Exception as e:
                print(f"[THREAD ERROR] {e}")
                import traceback
                traceback.print_exc()
                
                # Mark as failed
                try:
                    with self.lock:
                        if user_id in self.active_tasks:
                            del self.active_tasks[user_id]
                    
                    from app import app
                    with app.app_context():
                        from models import GenerationTask, db
                        task = GenerationTask.query.get(task_id)
                        if task:
                            task.status = 'failed'
                            task.message = f"Thread error: {str(e)}"
                            task.completed_at = datetime.utcnow()
                            db.session.commit()
                except:
                    pass
        
        # Start the thread
        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()
        
        with self.lock:
            self.active_tasks[user_id] = task_id
    
    def _mark_task_failed(self, task_id: int, error_msg: str):
        """Mark a task as failed"""
        try:
            from app import app
            with app.app_context():
                task = GenerationTask.query.get(task_id)
                if task:
                    task.status = 'failed'
                    task.message = f"Process error: {error_msg}"
                    task.completed_at = datetime.utcnow()
                    db.session.commit()
        except Exception as e:
            print(f"[TASK UPDATE ERROR] {e}")
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        with self.lock:
            return {
                'queue_size': 0,  # We don't track queue size with this approach
                'active_tasks': list(self.active_tasks.keys()),
                'is_running': True
            }
    
    def cleanup_completed_task(self, user_id: int, task_id: int):
        """Clean up a completed task"""
        with self.lock:
            if user_id in self.active_tasks and self.active_tasks[user_id] == task_id:
                del self.active_tasks[user_id]
                print(f"[CLEANUP] Removed completed task {task_id} for user {user_id}")


# Define generation_queue FIRST
generation_queue = GenerationQueue(max_workers=1)

# THEN define the functions that use it
def run_generation_task(user_id: int, username: str) -> Optional[int]:
    """Non-blocking generation task - returns immediately"""
    return generation_queue.add_task(user_id, username)

def get_generation_queue_status() -> Dict:
    """Get current generation queue status"""
    return generation_queue.get_queue_status()

def mark_generation_complete(user_id: int, task_id: int, status: str, message: str):
    """Mark a generation task as complete (called by worker process)"""
    generation_queue.cleanup_completed_task(user_id, task_id)
    
    try:
        from app import app
        with app.app_context():
            task = GenerationTask.query.get(task_id)
            if task:
                task.status = status
                task.message = message
                task.completed_at = datetime.utcnow()
                db.session.commit()
                print(f"[COMPLETE] Task {task_id} marked as {status}")
    except Exception as e:
        print(f"[COMPLETION ERROR] {e}")

def initialize_queue():
    """Initialize the generation queue"""
    try:
        # The new GenerationQueue doesn't need explicit starting
        # It starts processes on demand when add_task() is called
        print("[QUEUE] Generation queue initialized successfully")
    except Exception as e:
        print(f"[QUEUE INIT ERROR] {e}")

# Auto-initialize
initialize_queue()
# =============================================================================
# CLEANED UTILITY FUNCTIONS (ONLY USED ONES)
# =============================================================================

def extract_serial_from_text(text: str) -> Optional[str]:
    """Extract complete 3-part serial number from text"""
    if not text:
        return None
    
    # Direct serial match
    if text.startswith('SN-'):
        parts = text.split('-')
        if len(parts) >= 4:
            return f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[3]}"
        elif len(parts) == 3:
            return text
    
    # URL pattern match
    if 'bank.linglin.art' in text or 'verify' in text:
        url_match = re.search(r'/verify/(SN-[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)', text)
        if url_match:
            return url_match.group(1)
    
    # General pattern match
    sn_match = re.search(r'(SN-[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)', text)
    if sn_match:
        return sn_match.group(1)
    
    return None

def extract_serials_from_png(png_path: str) -> Optional[List[str]]:
    """Extract QR codes from PNG file"""
    try:
        image = cv2.imread(png_path)
        if image is None:
            return None
        
        serials = []
        decoded_texts = qreader.detect_and_decode(image=image)
        
        for text in decoded_texts:
            if text:
                serial = extract_serial_from_text(text)
                if serial and serial not in serials:
                    serials.append(serial)
        
        return serials if serials else None
        
    except Exception as e:
        print(f"[QR ERROR] {png_path}: {e}")
        return None

def extract_info_from_png_path(png_path: str) -> Tuple[str, str, str]:
    """Extract username, denomination, and side from PNG file path"""
    try:
        path_parts = png_path.split(os.sep)
        
        if len(path_parts) >= 4:
            username = path_parts[-3]
            denomination = path_parts[-2]
            filename = path_parts[-1]
            
            if 'back' in filename.lower():
                side = 'back'
            else:
                side = 'front'
            
            return username, denomination, side
        
        # Fallback extraction
        filename = os.path.basename(png_path)
        side = 'back' if 'back' in filename.lower() else 'front'
        numbers = re.findall(r'\d+', filename)
        denomination = numbers[0] if numbers else '1'
        
        username = re.split(r'\d+|front|back', filename, flags=re.IGNORECASE)[0]
        username = username.replace('_', ' ').replace('-', ' ').strip(' _-').title()
        
        return username or "Unknown", denomination, side
        
    except Exception as e:
        print(f"[PATH EXTRACT ERROR] {png_path}: {e}")
        return "Unknown", "1", "front"

import re
from markupsafe import Markup
import html

def sanitize_bio(raw_bio):
    """Sanitize bio with limited HTML support and BBCode"""
    # List of allowed HTML tags and their allowed attributes
    allowed_tags = {
        'b': [],
        'strong': [],
        'i': [],
        'em': [],
        'u': [],
        's': [],
        'strike': [],
        'br': [],
        'p': [],
        'div': ['style'],
        'span': ['style', 'class'],
        'a': ['href', 'title', 'rel'],
        'img': ['src', 'alt', 'style', 'width', 'height'],
        'blockquote': [],
        'pre': [],
        'code': [],
        'ul': [],
        'ol': [],
        'li': [],
        'hr': []
    }
    
    # Allowed CSS properties for style attributes
    allowed_style_props = [
        'color', 'background-color', 'font-size', 'font-weight', 
        'text-align', 'text-decoration', 'font-style', 'margin',
        'padding', 'border', 'width', 'height', 'max-width'
    ]
    
    def clean_style(style):
        """Clean style attribute to only allow safe properties"""
        if not style:
            return ''
        
        clean_rules = []
        for rule in style.split(';'):
            if ':' in rule:
                prop, value = rule.split(':', 1)
                prop = prop.strip().lower()
                value = value.strip()
                
                # Only allow specific CSS properties
                if prop in allowed_style_props:
                    # Basic validation for color values
                    if prop in ['color', 'background-color']:
                        if (value.startswith('#') and len(value) in [4, 7] and 
                            all(c in '0123456789abcdefABCDEF' for c in value[1:])) or \
                           value in ['red', 'blue', 'green', 'yellow', 'orange', 
                                   'purple', 'pink', 'black', 'white', 'gray', 
                                   'lightgray', 'darkgray']:
                            clean_rules.append(f"{prop}:{value}")
                    elif prop == 'font-size':
                        if value.endswith(('px', 'em', 'rem', '%')):
                            clean_rules.append(f"{prop}:{value}")
                    else:
                        clean_rules.append(f"{prop}:{value}")
        
        return ';'.join(clean_rules)
    
    def clean_attributes(tag, attributes):
        """Clean tag attributes"""
        clean_attrs = {}
        
        for attr, value in attributes.items():
            attr = attr.lower()
            
            # Clean style attributes
            if attr == 'style':
                clean_value = clean_style(value)
                if clean_value:
                    clean_attrs[attr] = clean_value
            
            # Clean href/src attributes
            elif attr in ['href', 'src']:
                # Basic URL validation - allow http, https, and relative URLs
                if value.startswith(('/#', '/', '#', 'http://', 'https://')):
                    # Escape the URL but preserve the structure
                    clean_attrs[attr] = html.escape(value)
            
            # Clean class attributes - only allow specific classes
            elif attr == 'class':
                allowed_classes = ['rainbow-text', 'flash-text']
                classes = [c.strip() for c in value.split() if c.strip() in allowed_classes]
                if classes:
                    clean_attrs[attr] = ' '.join(classes)
            
            # Allow other attributes if they're in the allowed list for this tag
            elif attr in allowed_tags.get(tag, []):
                clean_attrs[attr] = html.escape(value)
        
        return clean_attrs
    
    # First, convert BBCode to HTML
    bio_with_bbcode = convert_bbcode_to_html(raw_bio)
    
    # Then parse and clean the HTML
    safe_bio = clean_html(bio_with_bbcode, allowed_tags, clean_attributes)
    
    return Markup(safe_bio)

def convert_bbcode_to_html(text):
    """Convert BBCode to HTML"""
    # Convert newlines to <br> first
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n\s*\n', '</p><p>', text)  # Double newlines become paragraphs
    text = text.replace('\n', '<br>')  # Single newlines become line breaks
    
    # BBCode to HTML conversion
    bbcode_patterns = [
        (r'\[b\](.*?)\[/b\]', r'<strong>\1</strong>'),
        (r'\[i\](.*?)\[/i\]', r'<em>\1</em>'),
        (r'\[u\](.*?)\[/u\]', r'<u>\1</u>'),
        (r'\[s\](.*?)\[/s\]', r'<strike>\1</strike>'),
        (r'\[url=(.*?)\](.*?)\[/url\]', r'<a href="\1" rel="nofollow">\2</a>'),
        (r'\[url\](.*?)\[/url\]', r'<a href="\1" rel="nofollow">\1</a>'),
        (r'\[img\](.*?)\[/img\]', r'<img src="\1" style="max-width: 100%; height: auto;" alt="Image">'),
        (r'\[quote\](.*?)\[/quote\]', r'<blockquote>\1</blockquote>'),
        (r'\[code\](.*?)\[/code\]', r'<pre><code>\1</code></pre>'),
        (r'\[rainbow\](.*?)\[/rainbow\]', r'<span class="rainbow-text">\1</span>'),
        (r'\[flash\](.*?)\[/flash\]', r'<span class="flash-text">\1</span>'),
        (r'\[center\](.*?)\[/center\]', r'<div style="text-align: center;">\1</div>'),
        (r'\[size=(\d+)\](.*?)\[/size\]', r'<span style="font-size: \1px;">\2</span>'),
        (r'\[color=(.*?)\](.*?)\[/color\]', r'<span style="color: \1;">\2</span>'),
        (r'\[ul\](.*?)\[/ul\]', r'<ul>\1</ul>'),
        (r'\[ol\](.*?)\[/ol\]', r'<ol>\1</ol>'),
        (r'\[li\](.*?)\[/li\]', r'<li>\1</li>'),
        (r'\[hr\]', r'<hr>')
    ]
    
    for pattern, replacement in bbcode_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.DOTALL)
    
    # Wrap in paragraph if no block-level tags
    if not re.search(r'<(p|div|blockquote|pre|ul|ol|li|hr)', text, re.IGNORECASE):
        text = f'<p>{text}</p>'
    
    return text

def clean_html(html_content, allowed_tags, clean_attributes_func):
    """Basic HTML cleaner (you might want to use a proper HTML sanitizer like bleach in production)"""
    # This is a simplified cleaner - consider using bleach library for production
    from html.parser import HTMLParser
    
    class HTMLCleaner(HTMLParser):
        def __init__(self, allowed_tags, clean_attributes_func):
            super().__init__()
            self.allowed_tags = allowed_tags
            self.clean_attributes_func = clean_attributes_func
            self.result = []
            self.open_tags = []
            
        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            if tag in self.allowed_tags:
                attrs_dict = dict(attrs)
                clean_attrs = self.clean_attributes_func(tag, attrs_dict)
                
                attr_str = ' '.join([f'{k}="{v}"' for k, v in clean_attrs.items()])
                if attr_str:
                    self.result.append(f'<{tag} {attr_str}>')
                else:
                    self.result.append(f'<{tag}>')
                
                self.open_tags.append(tag)
            elif tag in ['br', 'hr', 'img']:  # Self-closing tags
                attrs_dict = dict(attrs)
                clean_attrs = self.clean_attributes_func(tag, attrs_dict)
                attr_str = ' '.join([f'{k}="{v}"' for k, v in clean_attrs.items()])
                if attr_str:
                    self.result.append(f'<{tag} {attr_str}>')
                else:
                    self.result.append(f'<{tag}>')
        
        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in self.allowed_tags and tag in self.open_tags:
                self.result.append(f'</{tag}>')
                if tag in self.open_tags:
                    self.open_tags.remove(tag)
        
        def handle_data(self, data):
            self.result.append(html.escape(data))
        
        def get_clean_html(self):
            # Close any unclosed tags
            for tag in reversed(self.open_tags):
                self.result.append(f'</{tag}>')
            return ''.join(self.result)
    
    cleaner = HTMLCleaner(allowed_tags, clean_attributes_func)
    cleaner.feed(html_content)
    return cleaner.get_clean_html()
# Add CSS styles for rainbow and flash effects to your template


def get_user_by_username(username: str) -> Optional[User]:
    """Get user object by username"""
    return User.query.filter_by(username=username).first()

def get_user_avatar(username: str) -> Optional[str]:
    """Get avatar image path for a user"""
    clean_username = re.sub(r'[^\w\-_]', '_', username)
    
    for ext in ['.png', '.jpg', '.jpeg', '.svg']:
        portrait_path = f"portraits/portrait_{clean_username}{ext}"
        if os.path.exists(portrait_path):
            return portrait_path
        
        simple_path = f"portraits/{clean_username}{ext}"
        if os.path.exists(simple_path):
            return simple_path
    
    return None

def get_initials(username: str) -> str:
    """Convert username to initials format"""
    if not username:
        return "?"
    
    name_parts = re.split(r'[\s_\-\.]+', username)
    initials = [part[0].upper() for part in name_parts if part]
    
    if not initials:
        return "?"
    
    return '.'.join(initials) + '.'

def get_current_user():
    """Get current user from session"""
    from flask import session
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return user
        session.pop('user_id', None)
    return None

def generate_qr_code(uri: str) -> str:
    """Generate QR code from URI and return as base64"""
    img = qrcode.make(uri)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_thumbnail(svg_path: str, png_path: str, size: Tuple[int, int] = (600, 300)) -> bool:
    """Generate PNG thumbnail from SVG file"""
    try:
        svg_url = f"file:{pathname2url(os.path.abspath(svg_path))}"
        cairosvg.svg2png(url=svg_url, write_to=png_path, output_width=size[0], output_height=size[1])
        return True
    except Exception as e:
        print(f"[THUMBNAIL ERROR] {svg_path}: {e}")
        return False

def generate_pdf(svg_path: str, pdf_path: str) -> bool:
    """Convert SVG to PDF"""
    try:
        png_buffer = BytesIO()
        cairosvg.svg2png(url=svg_path, write_to=png_buffer)
        png_buffer.seek(0)
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        img = ImageReader(png_buffer)
        c.drawImage(img, 50, 50, width=500, height=250, preserveAspectRatio=True)
        c.save()
        return True
    except Exception as e:
        print(f"[PDF ERROR] {svg_path}: {e}")
        return False

def add_banknote(user_id: int, username: str, denom: str, side: str, serial_number: str, svg_path: str) -> int:
    """Add a banknote to the database with associated files"""
    print(f"[ADD BANKNOTE] {serial_number} ({side}) for {username}")
    
    # Check if banknote already exists
    existing_banknote = Banknote.query.filter_by(
        serial_number=serial_number, 
        side=side
    ).first()
    
    if existing_banknote:
        print(f"[SKIP] Banknote exists: {serial_number}")
        return 0
    
    # Check if serial number exists
    existing_serial = SerialNumber.query.filter_by(serial=serial_number).first()
    if existing_serial:
        print(f"[SKIP] Serial exists: {serial_number}")
        return 0
    
    # Generate PNG and PDF files
    png_path, pdf_path = None, None
    if svg_path and os.path.exists(svg_path):
        png_path = svg_path.replace(".svg", ".png")
        pdf_path = svg_path.replace(".svg", ".pdf")
        
        generate_thumbnail(svg_path, png_path, size=(3200, 1200))
        #generate_pdf(svg_path, pdf_path)
    
    try:
        # Create banknote record
        banknote = Banknote(
            user_id=user_id,
            serial_number=serial_number,
            seed_text=username,
            denomination=denom,
            side=side,
            svg_path=svg_path,
            png_path=png_path,
            pdf_path=pdf_path,
            qr_data=serial_number,
            is_public=True
        )
        db.session.add(banknote)
        db.session.flush()
        
        # Create serial number record
        serial = SerialNumber(
            serial=serial_number,
            user_id=user_id,
            banknote_id=banknote.id,
            is_active=True
        )
        db.session.add(serial)
        db.session.commit()
        
        print(f"[SUCCESS] Added banknote: {serial_number}")
        return 1
        
    except IntegrityError:
        db.session.rollback()
        print(f"[INTEGRITY ERROR] {serial_number}")
        return 0
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] {serial_number}: {e}")
        return 0

def has_banknotes(user_id: int) -> bool:
    """Check if a user has any banknotes"""
    return Banknote.query.filter_by(user_id=user_id).first() is not None

def validate_serial_id(serial_id: str) -> Dict:
    """Validate serial number format"""
    if not serial_id.startswith("SN-"):
        return {"valid": False, "reason": "Missing prefix 'SN-'"}
    
    parts = serial_id.split('-')[1:]
    parts = [p for p in parts if p]
    
    if not parts:
        return {"valid": False, "reason": "No valid groups after SN- prefix"}
    
    if len(parts) == 2:
        body, checksum = parts
        if re.match(r'^[A-Za-z0-9_-]+$', body) and re.match(r'^[A-Za-z0-9_-]+$', checksum):
            return {
                "valid": True,
                "type": "with_checksum",
                "serial_body": body,
                "checksum": checksum
            }
    
    if all(re.match(r'^[A-Za-z0-9_-]+$', p) for p in parts):
        return {
            "valid": True,
            "type": "combined",
            "groups": parts,
            "checksum": None
        }
    
    return {"valid": False, "reason": "Invalid format"}
def get_user_avatar_or_default(username):
    """
    Get avatar URL or return a default with initials
    """
    avatar_path = get_user_avatar(username)
    
    if avatar_path:
        relative_path = os.path.relpath(avatar_path, '.')
        return url_for('serve_static', filename=relative_path)
    
    # Return a data URL with initials as fallback
    import base64
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    # Create a simple avatar with initials
    initials = get_initials(username) if username else "?"
    bg_color = f"hsl({random.randint(0, 360)}, 70%, 60%)"
    
    # Create image
    img = Image.new('RGB', (100, 100), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Draw initials
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((100 - text_width) // 2, (100 - text_height) // 2)
    
    draw.text(position, initials, fill="white", font=font)
    
    # Convert to data URL
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"
def get_user_avatar_url(username):
    """
    Get the URL for a user's avatar using the portraits route
    """
    avatar_path = get_user_avatar(username)
    
    if avatar_path:
        # Extract just the filename (e.g., "portrait_Ling_Lin.png")
        filename = os.path.basename(avatar_path)
        return url_for('serve_portrait', filename=filename)
    
    return None
def get_formatted_initials(username):
    """
    Convert username to formatted initials:
    - "Ling Lin" -> "L.L."
    - "linglin" -> "L" 
    - "John Doe Smith" -> "J.D.S."
    - "mary" -> "M"
    """
    if not username:
        return "?"
    
    # Split by spaces, underscores, hyphens, and dots
    name_parts = re.split(r'[\s_\-\.]+', username.strip())
    
    # Filter out empty parts and get first letter of each part
    initials = [part[0].upper() for part in name_parts if part]
    
    if not initials:
        return "?"
    
    # If only one initial, return just that letter
    if len(initials) == 1:
        return initials[0]
    
    # If multiple initials, format as "L.L." style
    return '.'.join(initials) + '.'
def regenerate_all_pngs():
    """Regenerate all PNG thumbnails from SVG files"""
    from app import app
    with app.app_context():
        svg_files = glob.glob('./images/**/*.svg', recursive=True)
        print(f"Regenerating {len(svg_files)} PNG files")
        
        for svg_path in svg_files:
            png_path = svg_path.replace('.svg', '.png')
            
            if not os.path.exists(png_path):
                print(f"Generating: {png_path}")
                if generate_thumbnail(svg_path, png_path):
                    print(f"✓ Created {png_path}")
                else:
                    print(f"✗ Failed {png_path}")

# =============================================================================
# NEW GENERATION TASK FUNCTION (NON-BLOCKING)
# =============================================================================