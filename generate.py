# --- Proxy for app.py/utils.py compatibility ---
def generate_for_user(*args, **kwargs):
    """
    Proxy for legacy code: generates a banknote pair for a user.
    Accepts flexible positional/keyword arguments for compatibility.
    """
    # Positional fallback
    name = args[0] if len(args) > 0 else None
    denomination = args[1] if len(args) > 1 else None

    # Keyword aliases
    name = kwargs.get("name", name)
    name = kwargs.get("username", name)
    denomination = kwargs.get("denomination", denomination)
    denomination = kwargs.get("denom", denomination)
    denomination = kwargs.get("specific_denom", denomination)

    output_dir = kwargs.get("output_dir") or kwargs.get("outdir")
    width_mm = kwargs.get("width_mm", 160.0)
    height_mm = kwargs.get("height_mm", 60.0)
    extra_context = kwargs.get("extra_context")
    progress_callback = kwargs.get("progress_callback")

    if name is None or denomination is None:
        raise TypeError("generate_for_user requires name and denomination")

    denomination = normalize_denomination(denomination)

    # Merge any unknown kwargs into extra_context
    ctx = dict(extra_context) if extra_context else {}
    ctx.update(kwargs)

    return generate_banknote_pair(
        name,
        denomination,
        output_dir,
        width_mm,
        height_mm,
        ctx,
        progress_callback=progress_callback
    )
#!/usr/bin/env python3
"""
generate.py - Minimal, robust banknote generator
- Jinja2 context substitution for EisenScript (front/back)
- EisenScript rendering (LunaMint)
- Parallel generation: front (GPU0), back (GPU1)
- PNG generation, mempool, Banknote DB, SerialNumber DB registration
"""
import os
import sys
import time
import math
import json
import threading
import shutil
from pathlib import Path
from io import BytesIO
from datetime import datetime
from jinja2 import Environment, DebugUndefined, meta
from lunamint.scripting import render_script_to_svg_html
from models import Banknote, SerialNumber, db, User, Settings
from PIL import Image
import cairosvg

# --- Utility ---
def mm_to_px(mm, dpi=300.0):
    return float(mm) * dpi / 25.4

def denomination_color(denom: int) -> str:
    """Match front denomination color palette (ROYGBIV, light tint)."""
    try:
        denom = max(1, min(100_000_000, int(denom)))
    except Exception:
        denom = 1
    exp = math.log10(denom) / math.log10(100_000_000)
    roygbiv = [
        (255, 0, 0),
        (255, 165, 0),
        (255, 255, 0),
        (0, 128, 0),
        (0, 0, 255),
        (75, 0, 130),
        (143, 0, 255),
    ]
    idx = int(exp * (len(roygbiv) - 1))
    frac = exp * (len(roygbiv) - 1) - idx
    c1 = roygbiv[idx]
    c2 = roygbiv[min(idx + 1, len(roygbiv) - 1)]
    r = int(c1[0] + (c2[0] - c1[0]) * frac)
    g = int(c1[1] + (c2[1] - c1[1]) * frac)
    b = int(c1[2] + (c2[2] - c1[2]) * frac)
    r = int(0.7 * 255 + 0.3 * r)
    g = int(0.7 * 255 + 0.3 * g)
    b = int(0.7 * 255 + 0.3 * b)
    return f"#{r:02X}{g:02X}{b:02X}"

def normalize_denomination(value, default="1"):
    """Normalize denomination to a numeric string for EisenScript."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    text = str(value).strip()
    if not text:
        return default
    # Extract numeric part
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits if digits else default

def denomination_to_words(value) -> str:
    """Convert a denomination to English words (e.g., 1000 -> One Thousand)."""
    try:
        num = int(str(value).strip())
    except Exception:
        return "Unknown"

    if num == 0:
        return "Zero"

    units = [
        "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
        "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
        "Seventeen", "Eighteen", "Nineteen",
    ]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    scales = [(10**9, "Billion"), (10**6, "Million"), (10**3, "Thousand"), (10**2, "Hundred")]

    def two_digit(n):
        if n < 20:
            return units[n]
        return tens[n // 10] + (" " + units[n % 10] if n % 10 else "")

    words = []
    remainder = num

    for scale_value, scale_name in scales:
        if remainder >= scale_value:
            chunk = remainder // scale_value
            remainder = remainder % scale_value
            if scale_value == 100:
                words.append(units[chunk] + " " + scale_name)
            else:
                words.append(two_digit(chunk) + " " + scale_name)

    if remainder:
        words.append(two_digit(remainder))

    return " ".join(w for w in words if w).strip()

def denomination_to_compact_lkc(value) -> str:
    """Convert a denomination to compact LKC format (1kLKC, 1mLKC, 1gLKC)."""
    try:
        num = int(str(value).strip())
    except Exception:
        return "Unknown"

    if num >= 1_000_000_000:
        return f"{num // 1_000_000_000}gLKC"
    if num >= 1_000_000:
        return f"{num // 1_000_000}mLKC"
    if num >= 1_000:
        return f"{num // 1_000}kLKC"
    return f"{num}LKC"

def denomination_to_chinese(value) -> str:
    """Convert a denomination to Chinese numerals (up to 100,000,000)."""
    try:
        num = int(str(value).strip())
    except Exception:
        return "未知"

    if num == 0:
        return "零"

    digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    units = ["", "十", "百", "千"]
    big_units = ["", "万", "亿"]

    def four_to_chinese(n: int) -> str:
        result = []
        zero = False
        for i in range(4):
            d = n % 10
            if d == 0:
                if result and not zero:
                    result.append(digits[0])
                    zero = True
            else:
                result.append(units[i])
                result.append(digits[d])
                zero = False
            n //= 10
        return "".join(reversed(result)).rstrip(digits[0])

    parts = []
    unit_index = 0
    while num > 0 and unit_index < len(big_units):
        chunk = num % 10000
        if chunk:
            chunk_str = four_to_chinese(chunk)
            parts.append(chunk_str + big_units[unit_index])
        else:
            parts.append("")
        num //= 10000
        unit_index += 1

    result = "".join(reversed([p for p in parts if p]))
    result = result.replace("一十", "十")
    result = result.replace("零零", "零").strip("零")
    return result or "零"

def denomination_to_chinese_lkc(value) -> str:
    """Convert a denomination to Chinese numerals with LKC unit (e.g., 一LKC, 一kLKC, 一mLKC, 一gLKC)."""
    try:
        num = int(str(value).strip())
    except Exception:
        return "未知"

    if num >= 1_000_000_000:
        return f"{denomination_to_chinese(num // 1_000_000_000)}gLKC"
    if num >= 1_000_000:
        return f"{denomination_to_chinese(num // 1_000_000)}mLKC"
    if num >= 1_000:
        return f"{denomination_to_chinese(num // 1_000)}kLKC"
    return f"{denomination_to_chinese(num)}LKC"

def denomination_to_exponent(value) -> int:
    """Map denomination 1..100000000 to exponent 1..9."""
    try:
        num = int(str(value).strip())
    except Exception:
        return 1
    if num <= 0:
        return 1
    exponent = int(math.log10(num)) + 1
    return max(1, min(9, exponent))

def sanitize_username_for_filename(name: str) -> str:
    if not name:
        return "unknown"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))
    return safe or "unknown"
def _load_eisen_file(path: str) -> str:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ""
def get_portrait_path(username: str) -> str:
    safe_name = sanitize_username_for_filename(username)
    return os.path.join("portraits", f"portrait_{safe_name}.png")

def get_user_denom_output_dir(username: str, denomination: str) -> str:
    safe_name = sanitize_username_for_filename(username)
    safe_denom = sanitize_username_for_filename(str(denomination))
    return os.path.join("images", safe_name, safe_denom)

def render_eisenscript_jinja2(script: str, context: dict) -> str:
    safe_context = dict(context) if context else {}
    safe_context.setdefault("title", "")
    safe_context.setdefault("subtitle", "")
    safe_context.setdefault("denomination", "1")
    safe_context.setdefault("serial", "")
    env = Environment(undefined=DebugUndefined)
    try:
        parsed = env.parse(script or "")
        vars_used = sorted(meta.find_undeclared_variables(parsed))
        print(f"[JINJA2 DEBUG] Vars used: {vars_used}")
        print(f"[JINJA2 DEBUG] Context keys: {sorted(safe_context.keys())}")
        if "title" in vars_used or "subtitle" in vars_used:
            print(f"[JINJA2 DEBUG] title={safe_context.get('title')}, subtitle={safe_context.get('subtitle')}")
    except Exception as e:
        print(f"[JINJA2 DEBUG] Failed to parse template vars: {e}")
    template = env.from_string(script)
    return template.render(**safe_context)

def load_eisenscript_parts(side: str):
    """
    Load EisenScript pre, user, suf for 'front' or 'back' from Settings model.
    Returns (pre, user, suf) tuple.
    """
    from app import app
    with app.app_context():
        settings = Settings.query.first()
        if not settings:
            return ('', '', '')
        if side == 'front':
            pre = getattr(settings, 'eisenscript_prefix_front', '') or ''
            user = getattr(settings, 'eisenscript_user_front', '') or ''
            suf = getattr(settings, 'eisenscript_suffix_front', '') or ''
        else:
            pre = getattr(settings, 'eisenscript_prefix_back', '') or ''
            user = getattr(settings, 'eisenscript_user_back', '') or ''
            suf = getattr(settings, 'eisenscript_suffix_back', '') or ''
        return (pre, user, suf)

def merge_eisenscript_with_vars(pre, user, suf, context):
    """
    Jinja2展開したpre, user, sufを結合して1つのEisenScriptにする
    """
    return '\n'.join([
        render_eisenscript_jinja2(pre, context),
        render_eisenscript_jinja2(user, context),
        render_eisenscript_jinja2(suf, context)
    ])

def inject_back_denom_background(script: str, color: str, width_px: int = 1600, height_px: int = 600) -> str:
    """Inject a solid background rect into back EisenScript."""
    if not script:
        return script
    if "# denom_bg_rect" in script:
        return script
    rect_line = f"rect 0 0 {width_px} {height_px} {color}"
    lines = script.splitlines()
    insert_at = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith("size "):
            insert_at = idx + 1
            break
    lines.insert(insert_at, "# denom_bg_rect")
    lines.insert(insert_at + 1, rect_line)
    return "\n".join(lines)

def generate_png_from_svg(svg_path, png_path, size=(1600, 600)):
    try:
        svg_uri = Path(svg_path).resolve().as_uri()
        for attempt in range(3):
            try:
                png_bytes = cairosvg.svg2png(url=svg_uri, output_width=size[0], output_height=size[1])
                img = Image.open(BytesIO(png_bytes))
                img.save(png_path, format="PNG", optimize=False, compress_level=1)
                return True
            except Exception as inner:
                if attempt < 2:
                    time.sleep(0.1)
                    continue
                raise inner
    except Exception as e:
        print(f"[ERROR] Failed to generate PNG from {svg_path}: {e}")
        return False

def resolve_svg_file(svg_path: Path) -> Path:
    """Resolve to an actual SVG file if a directory is provided."""
    svg_path = Path(svg_path)
    if svg_path.exists() and svg_path.is_dir():
        eisen = svg_path / "eisen.svg"
        if eisen.exists():
            return eisen
        candidates = sorted(svg_path.glob("*.svg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return svg_path

def write_eisen_file(eisen_path: Path, script: str) -> None:
    try:
        eisen_path = Path(eisen_path)
        eisen_path.parent.mkdir(parents=True, exist_ok=True)
        eisen_path.write_text(script or "", encoding="utf-8")
    except Exception as e:
        print(f"[WARNING] Failed to write EisenScript file: {e}")

def normalize_svg_output(result_path, target_path: Path) -> Path:
    """Ensure the SVG ends up at target_path, even if renderer returns a directory."""
    target_path = Path(target_path)
    if result_path:
        result_path = Path(result_path)
        if result_path.exists() and result_path.is_dir():
            candidate = result_path / "eisen.svg"
            if candidate.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if candidate.resolve() != target_path.resolve():
                    shutil.move(str(candidate), str(target_path))
                return target_path
        if result_path.exists() and result_path.suffix.lower() == ".svg":
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if result_path.resolve() != target_path.resolve():
                shutil.move(str(result_path), str(target_path))
            return target_path
    return target_path

def apply_svg_background_color(svg_path: Path, color: str) -> None:
    """Insert a background rect as the first SVG element if not present."""
    try:
        svg_path = Path(svg_path)
        if not svg_path.exists():
            return
        text = svg_path.read_text(encoding="utf-8")
        if 'data-denom-bg="1"' in text:
            return
        insert_idx = text.find("<svg")
        if insert_idx == -1:
            return
        start_tag_end = text.find(">", insert_idx)
        if start_tag_end == -1:
            return
        rect = f"<rect data-denom-bg=\"1\" width=\"100%\" height=\"100%\" fill=\"{color}\"/>"
        text = text[:start_tag_end+1] + rect + text[start_tag_end+1:]
        svg_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"[WARNING] Failed to apply background color: {e}")

def wait_for_svg_ready(svg_path, timeout=6.0, interval=0.05):
    """Wait until SVG exists and is non-empty."""
    start = time.time()
    svg_path = resolve_svg_file(Path(svg_path))
    while time.time() - start < timeout:
        try:
            if svg_path.exists() and svg_path.stat().st_size > 0:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False

def create_serial_id():
    return f"SN-{os.urandom(4).hex()}-{int(time.time()*1000)}"

def create_filename(name, denom, timestamp, side):
    safe = lambda s: str(s).replace('/', '_').replace('\\', '_')
    return f"{safe(name)}_-_{safe(denom)}_-_{timestamp}_{side}.svg"

def save_to_database(user, serial, denom, side, svg_path, png_path):
    from app import app
    with app.app_context():
        try:
            banknote = Banknote(
                user_id=user.id,
                serial_number=serial,
                seed_text=user.username,
                denomination=str(denom),
                side=side,
                svg_path=svg_path,
                png_path=png_path,
                is_public=True,
                created_at=datetime.utcnow()
            )
            db.session.add(banknote)
            db.session.commit()
            serial_rec = SerialNumber(
                serial=serial,
                user_id=user.id,
                banknote_id=banknote.id,
                is_active=True,
                created_at=datetime.utcnow()
            )
            db.session.add(serial_rec)
            db.session.commit()
            print(f"[+] Registered {side} banknote/serial: {serial}")
            return True
        except Exception as e:
            db.session.rollback()
            print(f"[ERROR] Failed to register {side} banknote/serial: {e}")
            return False

# --- Main generation logic ---

def generate_banknote_pair(name, denom, output_dir, width_mm=160.0, height_mm=60.0, extra_context=None, progress_callback=None):
    """
    Generate a front/back SVG+PNG pair for a given name/denom, using EisenScript parts from DB.
    Can be called from app.py or CLI.
    """
    timestamp = int(time.time()*1000)
    from app import app
    with app.app_context():
        user = User.query.first()
        if not user:
            print('[!] No user found')
            return
        settings = Settings.query.first()
    serial_front = create_serial_id()
    serial_back = create_serial_id()
    denom = normalize_denomination(denom)
    if not output_dir:
        output_dir = get_user_denom_output_dir(name, denom)
    denom_color = denomination_color(denom)
    denom_exponent = denomination_to_exponent(denom)
    denom_words = denomination_to_words(denom)
    denom_compact = denomination_to_compact_lkc(denom)
    denom_words_cn = denomination_to_chinese(denom)
    denom_compact_cn = denomination_to_chinese_lkc(denom)
    title = (settings.bill_title if settings else "") or ""
    subtitle = (settings.bill_subtitle if settings else "") or ""
    context_base = {
        "username": name,
        "denomination": denom,
        "denom_exponent": denom_exponent,
        "dendom_exp": denom_exponent,
        "pow_level": denom_exponent,
        "denomination_words": denom_words,
        "denomination_compact": denom_compact,
        "denomination_words_cn": denom_words_cn,
        "denomination_compact_cn": denom_compact_cn,
        "serial": serial_front,
        "title": title,
        "subtitle": subtitle,
        "denomination_color": denom_color,
        "denom_color": denom_color,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "timestamp": timestamp
    }
    if extra_context:
        context_base.update(extra_context)
    context_front = dict(context_base)
    context_back = dict(context_base)
    context_back["serial"] = serial_back

    # EisenScript parts from DB
    front_pre, front_user, front_suf = load_eisenscript_parts('front')
    back_pre, back_user, back_suf = load_eisenscript_parts('back')

    # Apply per-user custom EisenScript (from profile)
    custom_eisenscript = ""
    if extra_context and isinstance(extra_context, dict):
        custom_eisenscript = (extra_context.get("custom_eisenscript") or "").strip()
    if custom_eisenscript:
        front_user = f"{front_user}\n{custom_eisenscript}" if front_user else custom_eisenscript
        back_user = f"{back_user}\n{custom_eisenscript}" if back_user else custom_eisenscript
    eisenscript_front = merge_eisenscript_with_vars(front_pre, front_user, front_suf, context_front)
    eisenscript_back = merge_eisenscript_with_vars(back_pre, back_user, back_suf, context_back)
    eisenscript_back = inject_back_denom_background(eisenscript_back, denom_color)

    front_filename = create_filename(name, denom, timestamp, "FRONT")
    back_filename = create_filename(name, denom, timestamp, "BACK")
    front_stem = Path(front_filename).stem
    back_stem = Path(back_filename).stem
    front_dir = Path(output_dir) / front_stem
    back_dir = Path(output_dir) / back_stem
    svg_front = front_dir / f"{front_stem}.svg"
    svg_back = back_dir / f"{back_stem}.svg"
    eisen_front = front_dir / f"{front_stem}.eisen"
    eisen_back = back_dir / f"{back_stem}.eisen"
    png_front = front_dir / f"{front_stem}.png"
    png_back = back_dir / f"{back_stem}.png"
    os.makedirs(front_dir, exist_ok=True)
    os.makedirs(back_dir, exist_ok=True)
    write_eisen_file(eisen_front, eisenscript_front)
    write_eisen_file(eisen_back, eisenscript_back)

    status = {
        "front_png": False,
        "back_png": False,
        "front_db": False,
        "back_db": False,
    }

    def on_front_svg_saved(svg_path):
        svg_file = resolve_svg_file(Path(svg_path))
        if wait_for_svg_ready(svg_file):
            status["front_png"] = generate_png_from_svg(svg_file, png_front)
        else:
            print(f"[WARNING] SVG not ready for PNG (front): {svg_file}")

    def on_back_svg_saved(svg_path):
        svg_file = resolve_svg_file(Path(svg_path))
        if wait_for_svg_ready(svg_file):
            status["back_png"] = generate_png_from_svg(svg_file, png_back)
        else:
            print(f"[WARNING] SVG not ready for PNG (back): {svg_file}")

    def render_front():
        svg_result = render_script_to_svg_html(eisenscript_front, front_dir)
        if isinstance(svg_result, tuple):
            svg_result = svg_result[0]
        svg_path = Path(svg_result) if svg_result else svg_front
        svg_file = resolve_svg_file(normalize_svg_output(svg_path, svg_front))
        print(f"[+] Saved FRONT SVG: {svg_file}")
        on_front_svg_saved(svg_file)
        status["front_db"] = save_to_database(user, serial_front, denom, "front", str(svg_front), str(png_front))

    def render_back():
        svg_result = render_script_to_svg_html(eisenscript_back, back_dir)
        if isinstance(svg_result, tuple):
            svg_result = svg_result[0]
        svg_path = Path(svg_result) if svg_result else svg_back
        svg_file = resolve_svg_file(normalize_svg_output(svg_path, svg_back))
        print(f"[+] Saved BACK SVG: {svg_file}")
        apply_svg_background_color(svg_file, denom_color)
        on_back_svg_saved(svg_file)
        status["back_db"] = save_to_database(user, serial_back, denom, "back", str(svg_back), str(png_back))

    t_front = threading.Thread(target=render_front)
    t_back = threading.Thread(target=render_back)
    t_front.start()
    t_back.start()
    t_front.join()
    t_back.join()
    print("[+] Banknote pair generated.")
    success = status["front_png"] and status["back_png"] and status["front_db"] and status["back_db"]
    if success:
        try:
            from app import blockchain_daemon_instance
            if blockchain_daemon_instance:
                blockchain_daemon_instance.add_genesis_transaction(
                    serial_number=serial_front,
                    denomination=float(denom),
                    issued_to=name,
                )
                blockchain_daemon_instance.add_genesis_transaction(
                    serial_number=serial_back,
                    denomination=float(denom),
                    issued_to=name,
                )
                print("[+] Added GTX_Genesis transactions to mempool")
                try:
                    from app import app
                    with app.app_context():
                        user_ref = User.query.get(user.id)
                        if user_ref:
                            user_ref.balance = float(user_ref.balance or 0) + float(denom)
                            db.session.commit()
                            print(f"[+] Updated user balance: {user_ref.balance}")
                except Exception as bal_err:
                    db.session.rollback()
                    print(f"[WARNING] Failed to update user balance: {bal_err}")
        except Exception as e:
            print(f"[WARNING] Failed to add to mempool: {e}")
    if callable(progress_callback):
        try:
            if success:
                progress_callback("complete")
        except Exception:
            pass

    return 1 if success else 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate banknote SVG/PNG pair with EisenScript and Jinja2.")
    parser.add_argument('--name', type=str, default="", required=True)
    parser.add_argument('--denomination', type=str, default="", required=True)
    parser.add_argument('--output_dir', type=str, default='./images')
    parser.add_argument('--width_mm', type=float, default=160.0)
    parser.add_argument('--height_mm', type=float, default=60.0)
    args = parser.parse_args()
    generate_banknote_pair(
        args.name,
        args.denomination,
        args.output_dir,
        args.width_mm,
        args.height_mm
    )
