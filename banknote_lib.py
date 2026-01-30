"""
banknote_lib.py - Standalone banknote SVG/PNG generation helpers.

This module intentionally avoids Flask/DB dependencies and can be used as a
lightweight library for generating banknote assets.
"""
from __future__ import annotations

import os
import subprocess
import datetime
import secrets
import hashlib
from io import BytesIO
from typing import Optional, Dict

import cairosvg
from PIL import Image

# Try to import generator functions directly (faster); fallback to subprocess.
try:
    from generate_banknote_front import generate_single_banknote as _generate_front
except Exception:
    _generate_front = None

try:
    from generate_banknote_back import run_single_denomination as _generate_back
except Exception:
    _generate_back = None

FRONT_SCRIPT = "generate_banknote_front.py"
BACK_SCRIPT = "generate_banknote_back.py"


def generate_timestamp_ms_precise() -> int:
    """Generate timestamp with microsecond precision (ms)."""
    now = datetime.datetime.now()
    return int(now.timestamp() * 1000) + now.microsecond // 1000


def generate_serial_id_with_checksum(timestamp_ms: Optional[int] = None) -> str:
    """Generate serial ID with checksum (front)."""
    ts = timestamp_ms or generate_timestamp_ms_precise()
    salt = secrets.token_bytes(3)
    raw = f"{ts}-".encode() + salt
    h = hashlib.sha3_256(raw).digest()

    serial_bytes = h[:10]
    serial_b64 = (
        __import__("base64").urlsafe_b64encode(serial_bytes)
        .decode("ascii")
        .replace("=", "")[:14]
    )

    checksum_bytes = h[-2:]
    checksum_b64 = (
        __import__("base64").urlsafe_b64encode(checksum_bytes)
        .decode("ascii")
        .replace("=", "")[:3]
    )

    return f"SN-{serial_b64}-{checksum_b64}"


def generate_serial_id_combined(timestamp_ms: Optional[int] = None) -> str:
    """Generate compact serial ID (back)."""
    ts = timestamp_ms or generate_timestamp_ms_precise()
    salt = secrets.token_bytes(4)
    raw = f"{ts}-".encode() + salt
    h = hashlib.sha3_256(raw).digest()

    serial_b64 = __import__("base64").urlsafe_b64encode(h[:12]).decode("ascii")
    serial_clean = serial_b64.replace("=", "")[:12]
    return f"SN-{serial_clean[:4]}-{serial_clean[4:8]}-{serial_clean[8:12]}"


def create_proper_filename(name: str, denom: str, timestamp_ms: int, side: str) -> str:
    """Create filename: {name}_-_{denom}_-_{timestamp}_{side}.svg"""
    return f"{name}_-_{denom}_-_{timestamp_ms}_{side}.svg"


def create_basename(name: str, denom: str, timestamp_ms: int, side: str) -> str:
    """Create basename: {name}_-_{denom}_-_{timestamp}_{side}"""
    return f"{name}_-_{denom}_-_{timestamp_ms}_{side}"


def generate_png_from_svg(svg_path: str, png_path: str, size=(1600, 600)) -> bool:
    """Generate PNG from SVG file using cairosvg (with simple cache)."""
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


def generate_banknote_pair_svgs_pngs(
    name: str,
    denom: int,
    portrait_path: str,
    output_dir: str,
    timestamp_ms: Optional[int] = None,
    front_serial: Optional[str] = None,
    back_serial: Optional[str] = None,
    width_mm: float = 160.0,
    height_mm: float = 60.0,
    title: str = "灵国国库",
    subtitle: str = "天圆地方",
    font_dir: str = "./fonts",
    bg_dir: str = "./backgrounds",
    dpi: float = 300.0,
    bg_image: Optional[str] = None,
    background_prompt: Optional[str] = None,
    eisenscript_prefix_front: Optional[str] = None,
    eisenscript_suffix_front: Optional[str] = None,
    eisenscript_prefix_back: Optional[str] = None,
    eisenscript_suffix_back: Optional[str] = None,
    eisenscript_user_front: Optional[str] = None,
    eisenscript_user_back: Optional[str] = None,
    on_progress: Optional[callable] = None,
    on_complete: Optional[callable] = None,
    on_error: Optional[callable] = None,
) -> Dict[str, str]:
    """Generate front+back SVG and PNG files for a denomination using LunaMint EisenScript overlays.

    Returns a dict with paths and serials. Does not touch DB/Flask.
    """
    if not portrait_path or not os.path.exists(portrait_path):
        raise FileNotFoundError("portrait_path is required and must exist")

    from jinja2 import Template
    from lunamint.scripting import render_script_to_svg_html
    import tempfile
    from pathlib import Path
    import time

    os.makedirs(output_dir, exist_ok=True)
    ts = timestamp_ms or generate_timestamp_ms_precise()
    # Always generate front and back serials independently to avoid confusion
    front_serial = front_serial or generate_serial_id_with_checksum(ts)
    # If back_serial is not provided, generate a new one even if ts is the same
    back_serial = back_serial or generate_serial_id_combined(ts + 1)

    denom_str = str(int(denom))
    front_filename = create_proper_filename(name, denom_str, ts, "FRONT")
    back_basename = create_basename(name, denom_str, ts, "BACK")

    front_svg_path = os.path.join(output_dir, front_filename)
    back_svg_path = os.path.join(output_dir, f"{back_basename}.svg")

    # EisenScript context
    eisenscript_context_front = {
        "username": name,
        "title": title,
        "subtitle": subtitle,
        "denomination": denom_str,
        "serial": front_serial,
        "serial_id": front_serial,
        "seed_text": name,
        "qr_url": f"https://bank.linglin.art/verify/{front_serial}",
        "input_image_path": portrait_path,
        "width_mm": width_mm,
        "height_mm": height_mm,
    }
    eisenscript_context_back = dict(eisenscript_context_front)
    eisenscript_context_back["serial"] = back_serial
    eisenscript_context_back["serial_id"] = back_serial
    eisenscript_context_back["denomination"] = denom_str
    eisenscript_context_back["qr_url"] = f"https://bank.linglin.art/verify/{back_serial}"

    def merge_eisenscript(prefix, user, suffix, context):
        parts = [p.strip() for p in (prefix, user, suffix) if p and p.strip()]
        merged = "\n\n".join(parts)
        print("[DEBUG] Merged EisenScript template:")
        print(merged)
        try:
            rendered = Template(merged).render(**context)
            print("[DEBUG] Rendered EisenScript result:")
            print(rendered)
            return rendered
        except Exception as e:
            print(f"[ERROR] Jinja2 rendering failed: {e}")
            return merged

    # Load EisenScript templates if not provided
    def load_eisen(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                # Remove BOM if present
                if text.startswith("\ufeff"):
                    text = text[1:]
                # Remove all non-ASCII invisible/control characters
                import re
                text = re.sub(r'[\u200B-\u200F\uFEFF\u202A-\u202E]', '', text)
                return text
        except Exception:
            return ""

    prefix_front = eisenscript_prefix_front or load_eisen("eisen/front_pre.eisen")
    suffix_front = eisenscript_suffix_front or load_eisen("eisen/front_suf.eisen")
    prefix_back = eisenscript_prefix_back or load_eisen("eisen/back_pre.eisen")
    suffix_back = eisenscript_suffix_back or load_eisen("eisen/back_suf.eisen")

    # 合成
    print("[DEBUG] EisenScript BACK context:", eisenscript_context_back)
    eisenscript_front = merge_eisenscript(prefix_front, eisenscript_user_front, suffix_front, eisenscript_context_front)
    eisenscript_back = merge_eisenscript(prefix_back, eisenscript_user_back, suffix_back, eisenscript_context_back)

    import threading
    from tqdm import tqdm

    def run_svg_with_tqdm(gen_func, label, *args, **kwargs):
        done_event = threading.Event()
        result = {"ok": False, "error": None}
        pbar = tqdm(total=100, desc=label, leave=True, ncols=80)
        last_progress = [0.0]
        def _progress(progress, message):
            percent = int(progress * 100)
            if percent > last_progress[0]:
                pbar.n = percent
                pbar.set_postfix_str(message)
                pbar.refresh()
                last_progress[0] = percent
        def _complete(*_):
            pbar.n = 100
            pbar.set_postfix_str("Completed")
            pbar.refresh()
            pbar.close()
            print(f"[COMPLETED] {label} SVG generation complete.")
            result["ok"] = True
            done_event.set()
        def _error(exc):
            pbar.close()
            print(f"[ERROR] {label} SVG generation failed: {exc}")
            result["error"] = exc
            done_event.set()
        kwargs["progress_callback"] = lambda progress, msg=None: _progress(progress if isinstance(progress, float) else 0.0, msg or str(progress))
        kwargs["on_complete"] = _complete
        kwargs["on_error"] = _error
        gen_func(*args, **kwargs)
        done_event.wait(timeout=120.0)
        if result["error"]:
            raise result["error"]
        return result["ok"]

    # Front SVG (main)
    from generate_banknote_front import generate_single_banknote
    ok_front = run_svg_with_tqdm(
        generate_single_banknote,
        "FrontSVG",
        name,
        portrait_path,
        denom_str,
        outfile=front_svg_path,
        specimen=False,
        serial_id=front_serial,
        timestamp=ts,
        width_mm=width_mm,
        height_mm=height_mm,
        title=title,
        subtitle=subtitle,
        font_dir=font_dir,
        bg_dir=bg_dir,
        dpi=dpi,
        background_prompt=background_prompt,
        eisenscript_text=eisenscript_front
    )
    if not ok_front:
        raise RuntimeError("Failed to generate front SVG with LunaMint")

    # Back SVG (main)
    from generate_banknote_back import generate_backside_svg
    ok_back = run_svg_with_tqdm(
        generate_backside_svg,
        "BackSVG",
        back_svg_path,
        denom,
        title,
        subtitle,
        (int(width_mm * dpi / 25.4), int(height_mm * dpi / 25.4)),
        serial_id=back_serial,
        timestamp_ms=ts,
        seed_text=name,
    )
    if not ok_back:
        raise RuntimeError("Failed to generate back SVG with LunaMint")

    # EisenScript overlay (front)
    def run_eisen_svg_with_tqdm(eisenscript_text, svg_path, width_mm, height_mm, label):
        if not eisenscript_text:
            return False
        done_event = threading.Event()
        result = {"ok": False, "error": None}
        svg_data = {"svg": None}
        pbar = tqdm(total=100, desc=label, leave=True, ncols=80)
        last_progress = [0.0]
        def _progress(progress, message):
            percent = int(progress * 100)
            if percent > last_progress[0]:
                pbar.n = percent
                pbar.set_postfix_str(message)
                pbar.refresh()
                last_progress[0] = percent
        def _complete(svg, html):
            pbar.n = 100
            pbar.set_postfix_str("Completed")
            pbar.refresh()
            pbar.close()
            print(f"[COMPLETED] {label} SVG generation complete: {svg_path}")
            # SVGサイズ調整
            import re
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
            overlay_text = svg
            inner = _extract_svg_inner(overlay_text)
            overlay_w, overlay_h = _extract_svg_size(overlay_text)
            W = width_mm * dpi / 25.4
            H = height_mm * dpi / 25.4
            scale_x = W / overlay_w if overlay_w else 1.0
            scale_y = H / overlay_h if overlay_h else 1.0
            wrapped = f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">' + \
                      f'<g data-eisenscript="1" transform="scale({scale_x},{scale_y})">{inner}</g></svg>'
            Path(svg_path).write_text(wrapped, encoding="utf-8")
            svg_data["svg"] = wrapped
            result["ok"] = True
            done_event.set()
        def _error(exc):
            pbar.close()
            print(f"[ERROR] {label} SVG generation failed: {exc}")
            result["error"] = exc
            done_event.set()
        from lunamint.scripting import render_script_to_svg_html
        render_script_to_svg_html(
            eisenscript_text,
            on_progress=_progress,
            on_complete=_complete,
            on_error=_error
        )
        # Wait for completion (only check SVG after completed callback)
        done_event.wait(timeout=60.0)
        if result["error"]:
            raise result["error"]
        # Only check SVG file after completed
        if not (os.path.exists(svg_path) and os.path.getsize(svg_path) > 0):
            print(f"[ERROR] {label} SVG file not found or empty after completion: {svg_path}")
            return False
        return result["ok"]

    ok_front_overlay = run_eisen_svg_with_tqdm(
        eisenscript_front, front_svg_path, width_mm, height_mm, label="FrontOverlay"
    )
    ok_back_overlay = run_eisen_svg_with_tqdm(
        eisenscript_back, back_svg_path, width_mm, height_mm, label="BackOverlay"
    )
    if not (ok_front_overlay and ok_back_overlay):
        raise RuntimeError("Failed to generate front/back SVG with LunaMint EisenScript overlay")

    front_png_path = front_svg_path.replace(".svg", ".png")
    back_png_path = back_svg_path.replace(".svg", ".png")
    generate_png_from_svg(front_svg_path, front_png_path)
    generate_png_from_svg(back_svg_path, back_png_path)

    return {
        "front_svg": front_svg_path,
        "front_png": front_png_path,
        "back_svg": back_svg_path,
        "back_png": back_png_path,
        "front_serial": front_serial,
        "back_serial": back_serial,
        "timestamp_ms": ts,
    }
