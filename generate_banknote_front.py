# -----------------------
# to_bytes utility function
# -----------------------
def to_bytes(data, encoding="utf-8"):
    """Convert string or bytes-like object to bytes."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode(encoding)
    raise TypeError(f"Cannot convert type {type(data)} to bytes")
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
import os
import sys
# --- Patch: Prefer .venv site-packages ---
venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_site):
    sys.path.insert(0, venv_site)
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
from pathlib import Path
import tempfile
import binascii
from PIL import Image, ImageOps
import numpy as np
from sklearn.cluster import KMeans
import requests
import os
import time
import tqdm
import re
import hashlib
import random
import datetime
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



def generate_single_banknote(seed_text, input_image_path, single_denom, outfile=None, 
                           specimen=False, serial_id=None, timestamp=None,
                           width_mm=160.0, height_mm=60.0, title="灵国国库", subtitle="天圆地方",
                           font_dir="./fonts", bg_dir="./backgrounds", dpi=300.0, background_prompt="",
                           progress_callback=None, eisenscript_text=""):
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
        width_mm: Width in mm
        height_mm: Height in mm
        title: Title text
        subtitle: Subtitle text
        font_dir: Directory containing font files
        bg_dir: Directory containing background images
        dpi: Resolution in DPI
    
    Returns:
        Path to the generated SVG file
    """
    # Update global DPI
    global MM_TO_PX
    MM_TO_PX = dpi / 25.4
    
    # Set default outfile if not provided
    if outfile is None:
        timestamp_str = timestamp or datetime.time.strftime("%Y%m%d-%H%M%S")
        outfile = f"./images/{seed_text}/{single_denom}/{seed_text}_-_{single_denom}_-_{timestamp_str}_FRONT.svg"
    
    # Create output directory
    outfile_dir = os.path.dirname(outfile)
    os.makedirs(outfile_dir, exist_ok=True)
    
    # Load fonts
    fonts_obj = load_fonts(font_dir)
    

    # EisenScript/Jinja2テンプレート変数として渡す（必ず数値+単位にする）
    denomination_str = f"{single_denom} 卢纳币"

    generate_fantasy_banknote(
        seed_text=seed_text,
        input_image_path=input_image_path,
        outfile_svg=outfile,
        width_mm=width_mm,
        height_mm=height_mm,
        title=title,
        subtitle=subtitle,
        specimen=specimen,
        denomination=denomination_str,
        fonts=fonts_obj,
        serial_id=serial_id,
        timestamp=timestamp,
        bg_dir=bg_dir,
        background_prompt=background_prompt,
        progress_callback=progress_callback,
        eisenscript_text=eisenscript_text
    )
    
    print(f"[+] Single bill generated: {outfile}")
    return outfile

def generate_multiple_banknotes(seed_text, input_image_path, copies=1, yen_model=False, 
                              specimen=False, serial_id=None, timestamp=None,
                              width_mm=160.0, height_mm=60.0, title="灵国国库", subtitle="天圆地方",
                              font_dir="./fonts", bg_dir="./backgrounds", dpi=300.0, background_prompt="",
                              eisenscript_text=""):
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
        width_mm: Width in mm
        height_mm: Height in mm
        title: Title text
        subtitle: Subtitle text
        font_dir: Directory containing font files
        bg_dir: Directory containing background images
        dpi: Resolution in DPI
    
    Returns:
        List of paths to generated SVG files
    """
    # Update global DPI
    global MM_TO_PX
    MM_TO_PX = dpi / 25.4
    
    # Generate denominations
    if yen_model:
        base_denoms = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
        denominations = base_denoms[:9]  # top 9 denominations
    else:
        denominations = [100 * (i + 1) for i in range(9)]  # default 9 denominations

    fonts_obj = load_fonts(font_dir)
    generated_files = []

    for i in tqdm(range(copies), desc="Generating banknotes"):
        new_seed = seed_text  # no _i prefix in filenames

        for denom in denominations:
            timestamp_str = timestamp or datetime.time().strftime("%Y%m%d%H%M%S")
            # Filename format: seed_denomination_datetime.svg
            outfile_svg = f"./images/{new_seed}/{denom}/{new_seed}_-_{denom}_-_{timestamp_str}_FRONT.svg"
            outfile_dir = os.path.dirname(outfile_svg)
            os.makedirs(outfile_dir, exist_ok=True)

            denomination_str = f"{denom} 卢纳币"

            generate_fantasy_banknote(
                seed_text=f"{new_seed}_{i}",  # keep unique seed for generation
                input_image_path=input_image_path,
                outfile_svg=outfile_svg,
                width_mm=width_mm,
                height_mm=height_mm,
                title=title,
                subtitle=subtitle,
                specimen=specimen,
                denomination=denomination_str,
                fonts=fonts_obj,
                serial_id=serial_id,
                timestamp=timestamp_str,
                bg_dir=bg_dir,
                background_prompt=background_prompt,
                eisenscript_text=eisenscript_text
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
    parser.add_argument("--width-mm", type=float, default=160.0, help="Width in mm (default: 160.0)")
    parser.add_argument("--height-mm", type=float, default=60.0, help="Height in mm (default: 60.0)")
    parser.add_argument("--title", type=str, default="灵国国库", help="Title text (default: 灵国国库)")
    parser.add_argument("--subtitle", type=str, default="天圆地方", help="Subtitle text (default: 天圆地方)")
    parser.add_argument("--font-dir", type=str, default="./fonts", help="Directory containing font files (default: ./fonts)")
    parser.add_argument("--bg-dir", type=str, default="./backgrounds", help="Directory containing background images (default: ./backgrounds)")
    parser.add_argument("--dpi", type=float, default=300.0, help="Resolution in DPI (default: 300.0)")
    parser.add_argument("--background-prompt", type=str, help="Background generation prompt")
    parser.add_argument("--eisenscript", type=str, help="Inline EisenScript overlay")
    parser.add_argument("--eisenscript-file", type=str, help="Path to EisenScript file")
    
    args = parser.parse_args()
    
    eisenscript_text = args.eisenscript or ""
    if not eisenscript_text and args.eisenscript_file and os.path.exists(args.eisenscript_file):
        try:
            with open(args.eisenscript_file, "r", encoding="utf-8") as f:
                eisenscript_text = f.read()
        except Exception as script_error:
            print(f"[!] Failed to read Eisenscript file: {script_error}")

    generate_single_banknote(
        seed_text=args.seed_text,
        input_image_path=args.input_image,
        single_denom=args.single_denom,
        outfile=args.outfile,
        specimen=args.specimen,
        serial_id=args.serial_id,
        timestamp=args.timestamp,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
        title=args.title,
        subtitle=args.subtitle,
        font_dir=args.font_dir,
        bg_dir=args.bg_dir,
        dpi=args.dpi,
        background_prompt=args.background_prompt,
        eisenscript_text=eisenscript_text
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
    parser.add_argument("--width-mm", type=float, default=160.0, help="Width in mm (default: 160.0)")
    parser.add_argument("--height-mm", type=float, default=60.0, help="Height in mm (default: 60.0)")
    parser.add_argument("--title", type=str, default="灵国国库", help="Title text (default: 灵国国库)")
    parser.add_argument("--subtitle", type=str, default="天圆地方", help="Subtitle text (default: 天圆地方)")
    parser.add_argument("--font-dir", type=str, default="./fonts", help="Directory containing font files (default: ./fonts)")
    parser.add_argument("--bg-dir", type=str, default="./backgrounds", help="Directory containing background images (default: ./backgrounds)")
    parser.add_argument("--dpi", type=float, default=300.0, help="Resolution in DPI (default: 300.0)")
    parser.add_argument("--background-prompt", type=str, help="Background generation prompt")
    parser.add_argument("--eisenscript", type=str, help="Inline EisenScript overlay")
    parser.add_argument("--eisenscript-file", type=str, help="Path to EisenScript file")
    
    args = parser.parse_args()

    eisenscript_text = args.eisenscript or ""
    if not eisenscript_text and args.eisenscript_file and os.path.exists(args.eisenscript_file):
        try:
            with open(args.eisenscript_file, "r", encoding="utf-8") as f:
                eisenscript_text = f.read()
        except Exception as script_error:
            print(f"[!] Failed to read Eisenscript file: {script_error}")

    generate_multiple_banknotes(
        seed_text=args.seed_text,
        input_image_path=args.input_image,
        copies=args.copies,
        yen_model=args.yen_model,
        specimen=args.specimen,
        serial_id=args.serial_id,
        timestamp=args.timestamp,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
        title=args.title,
        subtitle=args.subtitle,
        font_dir=args.font_dir,
        bg_dir=args.bg_dir,
        dpi=args.dpi,
        background_prompt=args.background_prompt,
        eisenscript_text=eisenscript_text
    )

# Main execution
if __name__ == "__main__":
    import sys
    
    # Check if --single_denom flag is present to use the single bill mode
    if "--single_denom" in sys.argv:
        single_bill_run()
    else:
        multi_bill_run()