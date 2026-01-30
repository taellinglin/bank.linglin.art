# --- GTX_Genesis combined submission helper ---
import json
import hashlib
import time
import threading
def submit_gtx_genesis(front_serial, back_serial, denomination, svg_path_front, svg_path_back, issued_to, extra_fields=None):
    from app import blockchain_daemon_instance
    try:
        denomination = float(denomination)
    except Exception:
        denomination = 0.0
    tx = {
        "type": "GTX_Genesis",
        "front_serial": front_serial,
        "back_serial": back_serial,
        "denomination": denomination,
        "svg_path_front": svg_path_front,
        "svg_path_back": svg_path_back,
        "timestamp": int(time.time()),
        "issued_to": issued_to,
        "metadata_hash": None,
    }
    if extra_fields:
        tx.update(extra_fields)
    # Compute hash (as required by validation)
    tx_json = json.dumps(tx, sort_keys=True, separators=(',', ':'))
    tx["hash"] = hashlib.sha256(tx_json.encode("utf-8")).hexdigest()
    result = blockchain_daemon_instance.add_transaction(tx)
    if result:
        print("[+] GTX_Genesis submitted to mempool via lunalib")
    else:
        print("[!] Failed to submit GTX_Genesis via lunalib")
    return result
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



if __name__ == "__main__":
    import argparse
    import os
    import json
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

    # --- State tracking file ---
    STATE_FILE = f"{args.seed_text}_generation_state.json"

    def load_state():
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {"last_successful": {"copy": -1, "denom_index": -1}}
        return {"last_successful": {"copy": -1, "denom_index": -1}}

    def save_state(copy_index, denom_index):
        with open(STATE_FILE, 'w') as f:
            json.dump({"last_successful": {"copy": copy_index, "denom_index": denom_index}}, f)

    state = load_state()
    last_copy = state['last_successful']['copy']
    last_denom_idx = state['last_successful']['denom_index']

    start_copy = last_copy if last_copy != -1 else 0
    if last_denom_idx == len(denominations) - 1:
        start_copy += 1 # Start next copy
        

    for i in tqdm(range(start_copy, args.copies), desc="Generating banknotes", initial=start_copy, total=args.copies):
        new_seed = args.seed_text

        start_denom_idx = 0
        if i == last_copy and last_denom_idx != -1:
            start_denom_idx = last_denom_idx + 1

        for denom_idx, denom in enumerate(tqdm(denominations[start_denom_idx:], desc=f"Copy {i+1}/{args.copies}", leave=False)):
            actual_denom_idx = start_denom_idx + denom_idx
            import re
            import time
            base, ext = os.path.splitext(args.outfile)
            # Windows-safe filename: remove forbidden chars
            safe_seed = re.sub(r'[^\w\-]', '_', str(new_seed))
            safe_denom = re.sub(r'[^\w\-]', '_', str(denom))
            outfile_svg_front = f"{safe_seed}_{i}_{safe_denom}_front{ext}"
            outfile_svg_back = f"{safe_seed}_{i}_{safe_denom}_back{ext}"

            # Skip if already exists (both sides)
            if os.path.exists(outfile_svg_front) and os.path.exists(outfile_svg_back):
                print(f"Skipping existing files: {outfile_svg_front}, {outfile_svg_back}")
                save_state(i, actual_denom_idx)
                continue

            denomination_str = f"{denom} 卢纳币"

            # --- Coordination events ---
            front_done = threading.Event()
            back_done = threading.Event()

            def on_front_complete():
                front_done.set()

            def on_back_complete():
                back_done.set()

            # --- Generate front SVG (with callback) ---
            def front_thread():
                generate_fantasy_banknote(
                    seed_text=f"{new_seed}_{i}",
                    input_image_path=args.input_image,
                    outfile_svg=outfile_svg_front,
                    specimen=args.specimen,
                    denomination=str(int(denom)),  # 数値部分のみ
                    denomination_label=denomination_str,  # ラベル用途
                    fonts=fonts
                )
                on_front_complete()

            # --- Generate back SVG (with callback) ---
            def back_thread():
                from generate_banknote_back import generate_backside_svg
                generate_backside_svg(
                    outfile=outfile_svg_back,
                    denomination=int(denom),  # 必ずintで渡す
                    title_text="灵国国库",
                    phrase_text="灵之意志，天下共识",
                    size_px=(mm_to_px(160.0), mm_to_px(60.0)),
                    serial_id=None,
                    timestamp_ms=None,
                    seed_text=f"{new_seed}_{i}",
                    progress_callback=None
                )
                on_back_complete()

            t_front = threading.Thread(target=front_thread)
            t_back = threading.Thread(target=back_thread)
            t_front.start()
            t_back.start()

            # Wait for both to finish
            front_done.wait()
            back_done.wait()

            # --- Retrieve serials from DB ---
            from app import app
            with app.app_context():
                from models import Banknote, User
                user = User.query.first()
                front_bn = Banknote.query.filter_by(svg_path=outfile_svg_front, side='front').first()
                back_bn = Banknote.query.filter_by(svg_path=outfile_svg_back, side='back').first()
                if not front_bn or not back_bn:
                    print(f"[!] Could not find both front and back banknote records for: {outfile_svg_front}, {outfile_svg_back}")
                    continue
                front_serial = front_bn.serial_number
                back_serial = back_bn.serial_number
                issued_to = user.username if user else "unknown"

            # --- Submit GTX_Genesis transaction ---
            submit_gtx_genesis(
                front_serial=front_serial,
                back_serial=back_serial,
                denomination=denom,
                svg_path_front=outfile_svg_front,
                svg_path_back=outfile_svg_back,
                issued_to=issued_to
            )

            save_state(i, actual_denom_idx)

# --- EisenScriptをDBから取得してオーバーレイ適用 ---
# 仮定: get_eisenscript_from_db(note_type) でEisenScript文字列を取得できる
# note_type: 'front' または 'back' を判定
def get_eisenscript_from_db(note_type, context):
    # Settingsモデルからpre/suffix両方を取得し、Jinja2 contextでレンダリング
    try:
        from app import app
        with app.app_context():
            from models import Settings
            settings = Settings.query.first()
            if not settings:
                print("[LUNAMINT ERROR] No Settings found in DB.")
                return ''
            from jinja2 import Environment, StrictUndefined
            if note_type == 'back':
                pre = settings.eisenscript_prefix_back or ''
                suf = settings.eisenscript_suffix_back or ''
            else:
                pre = settings.eisenscript_prefix_front or ''
                suf = settings.eisenscript_suffix_front or ''
            print("[DEBUG] EisenScript context for DB render:", context)
            print("[DEBUG] EisenScript pre template:\n", pre)
            print("[DEBUG] EisenScript suf template:\n", suf)
            env = Environment(undefined=StrictUndefined)
            pre_rendered = env.from_string(pre).render(**context)
            suf_rendered = env.from_string(suf).render(**context)
            full_rendered = pre_rendered + '\n' + suf_rendered
            print("[DEBUG] EisenScript rendered text:\n", full_rendered)
            if "{{denomination}}" in full_rendered or "{{ denomination }}" in full_rendered:
                raise RuntimeError(f"[FATAL] Jinja2 failed to substitute {{denomination}}! Context: {context}\nRendered: {full_rendered}")
            return full_rendered
    except Exception as e:
        print(f"[LUNAMINT ERROR] Failed to fetch/render EisenScript pre/suffix from Settings: {e}")
        raise
    
    if '_BACK' in outfile_svg or 'back' in outfile_svg.lower():
        note_type = 'back'
    else:
        note_type = 'front'

    # EisenScript contextをここで構築し、必ず渡す
    # 例: context = {...}（既にどこかで作られている場合はそれを渡す）
    # ここでは仮にeisenscript_contextという変数名で存在していると仮定

    eisenscript_context = locals().get('eisenscript_context') or locals().get('context') or {}
    print("[DEBUG] EisenScript context before DB render:", eisenscript_context)
    eisenscript_content = get_eisenscript_from_db(note_type, eisenscript_context)

    # outfile_svgがファイルパスであることを保証
    import os
    if not outfile_svg or outfile_svg.strip() in ('.', './') or os.path.isdir(outfile_svg):
        print(f"[LUNAMINT ERROR] Invalid SVG file path for overlay: {outfile_svg}")
    else:
        try:
            from lunamint import render_script_to_svg_html
            svg_result = render_script_to_svg_html(
                eisenscript_content,
                outfile_svg,
                on_progress=None,
                on_complete=None,
                on_error=None
            )
            # 結果を上書き保存（APIがファイルを直接書き換えない場合のみ）
            if svg_result and isinstance(svg_result, str):
                with open(outfile_svg, 'w', encoding='utf-8') as f:
                    f.write(svg_result)
            print(f"[LUNAMINT] Overlay applied from DB: {note_type} to {outfile_svg}")
        except Exception as e:
            print(f"[LUNAMINT ERROR] Overlay from DB failed: {e}")

    # SVGファイルが生成されるまで最大60秒待機
    max_wait = 60.0
    waited = 0.0
    while not os.path.exists(outfile_svg) and waited < max_wait:
        time.sleep(0.2)
        waited += 0.2
    if not os.path.exists(outfile_svg):
        print(f"[ERROR] SVG not generated after {max_wait} seconds: {outfile_svg}")
    else:
        print(f"[OK] SVG generated: {outfile_svg}")

    # 成功時のみステート保存
    save_state(i, actual_denom_idx)

