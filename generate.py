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

    is_coin = bool(
        kwargs.get("is_coin")
        or (isinstance(extra_context, dict) and extra_context.get("is_coin"))
    )

    if name is None or denomination is None:
        raise TypeError("generate_for_user requires name and denomination")

    if not is_coin:
        denomination = normalize_denomination(denomination)

    # Merge any unknown kwargs into extra_context
    ctx = dict(extra_context) if extra_context else {}
    ctx.update(kwargs)
    if is_coin:
        ctx["is_coin"] = True

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
import hashlib
import sys
import time
import math
import json
import threading
import shutil
import random
import base64
import re
import requests
from pathlib import Path
from io import BytesIO
from datetime import datetime
from jinja2 import Environment, DebugUndefined, meta
from lunamint.scripting import render_script_to_svg_html
from models import Banknote, SerialNumber, db, User, Settings
from PIL import Image
import cairosvg
try:
    from fontTools.ttLib import TTFont
except Exception:
    TTFont = None

GENERIC_FONT_FAMILIES = {
    "serif",
    "sans-serif",
    "monospace",
    "cursive",
    "fantasy",
    "system-ui",
}


def resolve_font_dir(settings=None) -> str:
    if settings and getattr(settings, "font_dir", None):
        return str(settings.font_dir)
    env_font_dir = os.environ.get("EISENSCRIPT_FONT_DIR") or os.environ.get("FONT_DIR")
    if env_font_dir:
        return env_font_dir
    return "./fonts"


def _split_font_families(raw_value: str) -> list[str]:
    families = []
    for part in raw_value.split(","):
        cleaned = part.strip().strip("\"").strip("'").strip()
        if cleaned:
            families.append(cleaned)
    return families


def _extract_svg_font_families(svg_content: str) -> set[str]:
    families = set()
    for match in re.findall(r"font-family\s*:\s*([^;\"'}]+)", svg_content, flags=re.IGNORECASE):
        families.update(_split_font_families(match))
    for match in re.findall(r"font-family=['\"]([^'\"]+)['\"]", svg_content, flags=re.IGNORECASE):
        families.update(_split_font_families(match))
    return {family for family in families if family and family.lower() not in GENERIC_FONT_FAMILIES}


def _font_family_from_file(font_path: Path) -> str | None:
    if TTFont is None:
        return font_path.stem
    try:
        font = TTFont(str(font_path), lazy=True)
        name_table = font["name"].names
        preferred = None
        fallback = None
        for record in name_table:
            if record.nameID == 16 and not preferred:
                preferred = record.toUnicode()
            if record.nameID == 1 and not fallback:
                fallback = record.toUnicode()
        font.close()
        return preferred or fallback or font_path.stem
    except Exception:
        return font_path.stem


def _font_format_from_suffix(suffix: str) -> tuple[str, str]:
    suffix = suffix.lower()
    if suffix == ".woff2":
        return ("font/woff2", "woff2")
    if suffix == ".woff":
        return ("font/woff", "woff")
    if suffix == ".otf":
        return ("font/otf", "opentype")
    return ("font/ttf", "truetype")


def _load_font_family_map(font_dir: str) -> dict[str, tuple[Path, str, str]]:
    font_dir_path = Path(font_dir)
    if not font_dir_path.exists():
        return {}
    font_map = {}
    for font_path in font_dir_path.rglob("*"):
        if not font_path.is_file():
            continue
        if font_path.suffix.lower() not in (".ttf", ".otf", ".woff", ".woff2"):
            continue
        family = _font_family_from_file(font_path)
        if not family:
            continue
        mime, fmt = _font_format_from_suffix(font_path.suffix)
        font_map[family] = (font_path, mime, fmt)
        font_map[family.lower()] = (font_path, mime, fmt)
    return font_map


def embed_fonts_in_svg_content(svg_content: str, font_dir: str) -> str:
    if not svg_content or "data-embedded-fonts=\"1\"" in svg_content:
        return svg_content
    families = _extract_svg_font_families(svg_content)
    if not families:
        return svg_content
    font_map = _load_font_family_map(font_dir)
    if not font_map:
        return svg_content

    css_rules = []
    for family in sorted(families):
        entry = font_map.get(family) or font_map.get(family.lower())
        if not entry:
            continue
        font_path, mime, fmt = entry
        try:
            font_bytes = font_path.read_bytes()
        except Exception:
            continue
        encoded = base64.b64encode(font_bytes).decode("ascii")
        css_rules.append(
            "@font-face {"
            f"font-family: '{family}';"
            f"src: url(data:{mime};base64,{encoded}) format('{fmt}');"
            "font-weight: normal;"
            "font-style: normal;"
            "}"
        )
    if not css_rules:
        return svg_content

    insert_idx = svg_content.find("<svg")
    if insert_idx == -1:
        return svg_content
    start_tag_end = svg_content.find(">", insert_idx)
    if start_tag_end == -1:
        return svg_content
    style_block = "<defs><style data-embedded-fonts=\"1\"><![CDATA[" + "".join(css_rules) + "]]></style></defs>"
    return svg_content[: start_tag_end + 1] + style_block + svg_content[start_tag_end + 1 :]


def embed_fonts_in_svg_file(svg_path: Path, font_dir: str) -> None:
    try:
        svg_path = Path(svg_path)
        if not svg_path.exists():
            return
        svg_content = svg_path.read_text(encoding="utf-8")
        updated = embed_fonts_in_svg_content(svg_content, font_dir)
        if updated != svg_content:
            svg_path.write_text(updated, encoding="utf-8")
    except Exception as exc:
        print(f"[WARNING] Failed to embed fonts into SVG: {exc}")

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

def format_coin_amount(value) -> str:
    try:
        amount = float(str(value).strip())
    except Exception:
        amount = 0.0
    text = f"{amount:.3f}".rstrip("0").rstrip(".")
    return text or "0"

def _get_render_endpoints() -> list[str]:
    endpoints_text = os.getenv("LUNAMINT_RENDER_ENDPOINTS", "").strip()
    if not endpoints_text:
        try:
            from app import app
            with app.app_context():
                settings = Settings.query.first()
        except Exception:
            settings = None
        if not settings:
            return []
        if not (
            getattr(settings, "lunamint_use_custom_server", False)
            or getattr(settings, "lunamint_server_url", "")
            or getattr(settings, "lunamint_server_urls", "")
        ):
            return []
        endpoints_text = settings.lunamint_server_urls or settings.lunamint_server_url or ""
    endpoints = [endpoint.strip().rstrip("/") for endpoint in endpoints_text.split(",") if endpoint.strip()]
    return list(dict.fromkeys(endpoints))

def _render_eisenscript_via_remote(script_text: str) -> str | None:
    endpoints = _get_render_endpoints()
    if not endpoints:
        return None
    for endpoint in endpoints:
        try:
            payload = {"script": script_text, "html": False}
            if endpoint.endswith("/mint/compile"):
                payload = {"script": script_text, "name": "eisenscript"}
            response = requests.post(endpoint, json=payload, timeout=20)
            if not response.ok:
                continue
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                payload = response.json()
                if payload.get("ok") is False:
                    continue
                svg_content = payload.get("svg") or payload.get("html")
                if svg_content:
                    return svg_content
            else:
                text = response.text
                if text and "<svg" in text:
                    return text
        except Exception as exc:
            print(f"[RENDER REMOTE] Failed at {endpoint}: {exc}")
            continue
    return None

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

    def three_digit(n):
        if n < 100:
            return two_digit(n)
        hundreds = n // 100
        remainder = n % 100
        if remainder:
            return units[hundreds] + " Hundred " + two_digit(remainder)
        return units[hundreds] + " Hundred"

    words = []
    remainder = num

    for scale_value, scale_name in scales:
        if remainder >= scale_value:
            chunk = remainder // scale_value
            remainder = remainder % scale_value
            if scale_value == 100:
                words.append(units[chunk] + " " + scale_name)
            else:
                words.append(three_digit(chunk) + " " + scale_name)

    if remainder:
        words.append(three_digit(remainder))

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

def read_prompt_file(filepath: str, default: str = "") -> str:
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass
    return default

def generate_character_portrait(name: str, portrait_prompt: str = None, save_path: str = "./portraits") -> str:
    """Generate a portrait via Stable Diffusion API; returns file path or empty string."""
    try:
        os.makedirs(save_path, exist_ok=True)
        safe_name = sanitize_username_for_filename(name)
        output_path = os.path.join(save_path, f"portrait_{safe_name}.png")

        prompt = portrait_prompt or read_prompt_file(
            "portrait_prompt.txt",
            "A professional portrait of a person, high quality, detailed face, neutral background"
        )
        if "{name}" in prompt:
            prompt = prompt.format(name=name)
        negative_prompt = read_prompt_file(
            "negative_prompt.txt",
            "text, words, letters, numbers, blurry, low quality, watermark, signature"
        )

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": 512,
            "height": 512,
            "seed": random.randint(0, 2**32 - 1),
            "steps": 25,
            "cfg_scale": 7.5,
            "sampler_name": "DPM++ 2M Karras",
            "batch_size": 1,
            "n_iter": 1,
            "restore_faces": True,
            "tiling": False,
        }

        api_url = os.getenv("SD_API_URL", "http://127.0.0.1:7777/sdapi/v1/txt2img")
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        images = result.get("images", [])
        if not images:
            return ""
        image_data = base64.b64decode(images[0])
        image = Image.open(BytesIO(image_data))
        image.save(output_path)
        print(f"[+] Generated portrait: {output_path}")
        return output_path
    except Exception as e:
        print(f"[WARNING] Portrait generation failed: {e}")
        return ""

def ensure_portrait_exists(name: str, settings: Settings = None, extra_context: dict = None) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    portrait_path = get_portrait_path(name)
    portrait_abs_path = os.path.join(base_dir, portrait_path)
    portraits_dir = os.path.join(base_dir, "portraits")
    if portrait_path and os.path.isfile(portrait_abs_path) and os.path.getsize(portrait_abs_path) > 0:
        return portrait_path

    portrait_prompt = None
    if settings and getattr(settings, "portrait_prompt", None):
        portrait_prompt = settings.portrait_prompt
    if extra_context and isinstance(extra_context, dict):
        portrait_prompt = extra_context.get("portrait_prompt") or portrait_prompt

    for attempt in range(1, 4):
        print(f"[!] Portrait not found for {name}, generating (attempt {attempt}/3)...")
        generated = generate_character_portrait(
            name,
            portrait_prompt=portrait_prompt,
            save_path=portraits_dir,
        )
        generated_path = generated or portrait_abs_path
        if generated_path and os.path.isfile(generated_path) and os.path.getsize(generated_path) > 0:
            return generated or portrait_path
    return ""

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
    env.filters["human_color"] = human_color
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


def human_color(value):
    """Format color strings into a human-readable form."""
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    lower = text.lower()

    hex_match = re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})", lower)
    if hex_match:
        hex_value = hex_match.group(1)
        if len(hex_value) == 3:
            r, g, b = [int(c * 2, 16) for c in hex_value]
            return f"RGB({r}, {g}, {b})"
        if len(hex_value) == 6:
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            return f"RGB({r}, {g}, {b})"
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
        a = int(hex_value[6:8], 16) / 255
        return f"RGBA({r}, {g}, {b}, {a:.2f})"

    rgb_match = re.fullmatch(
        r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)",
        lower,
    )
    if rgb_match:
        r, g, b = (int(float(rgb_match.group(i))) for i in range(1, 4))
        a_raw = rgb_match.group(4)
        if a_raw is None:
            return f"RGB({r}, {g}, {b})"
        try:
            a = float(a_raw)
        except ValueError:
            a = 1.0
        return f"RGBA({r}, {g}, {b}, {a:.2f})"

    return text

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
        elif side == 'back':
            pre = getattr(settings, 'eisenscript_prefix_back', '') or ''
            user = getattr(settings, 'eisenscript_user_back', '') or ''
            suf = getattr(settings, 'eisenscript_suffix_back', '') or ''
        elif side == 'coin_front':
            pre = getattr(settings, 'eisenscript_prefix_coin_front', '') or ''
            user = getattr(settings, 'eisenscript_user_coin_front', '') or ''
            suf = getattr(settings, 'eisenscript_suffix_coin_front', '') or ''
        elif side == 'coin_back':
            pre = getattr(settings, 'eisenscript_prefix_coin_back', '') or ''
            user = getattr(settings, 'eisenscript_user_coin_back', '') or ''
            suf = getattr(settings, 'eisenscript_suffix_coin_back', '') or ''
        else:
            pre = ''
            user = ''
            suf = ''
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
    """Inject a denomination background using EisenScript `background` command."""
    if not script:
        return script
    lines = script.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("background ") or stripped == "# denom_bg":
            return script
    insert_at = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith("size "):
            insert_at = idx + 1
            break
    lines.insert(insert_at, "# denom_bg")
    lines.insert(insert_at + 1, f"background {color}")
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
        user = None
        user_id = None
        if extra_context and isinstance(extra_context, dict):
            user_id = extra_context.get("user_id") or extra_context.get("userId")
        if user_id:
            user = User.query.get(user_id)
        if not user and name:
            user = User.query.filter_by(username=name).first()
        if not user:
            user = User.query.first()
        if not user:
            print("[!] No user found")
            return
        settings = Settings.query.first()
    font_dir = resolve_font_dir(settings)
    ensure_portrait_exists(name, settings, extra_context)
    serial_front = create_serial_id()
    serial_back = create_serial_id()
    is_coin = bool(extra_context and isinstance(extra_context, dict) and extra_context.get("is_coin"))
    if not is_coin:
        denom = normalize_denomination(denom)
    if not output_dir:
        output_dir = get_user_denom_output_dir(name, denom)
    if is_coin:
        denom_text = str(denom).strip()
        denom_color = denomination_color(1)
        denom_exponent = 0
        denom_words = denom_text
        denom_compact = denom_text
        denom_words_cn = denom_text
        denom_compact_cn = denom_text
        coin_amount_text = format_coin_amount(denom_text)
        render_size = (512, 512)
    else:
        denom_color = denomination_color(denom)
        denom_exponent = denomination_to_exponent(denom)
        denom_words = denomination_to_words(denom)
        denom_compact = denomination_to_compact_lkc(denom)
        denom_words_cn = denomination_to_chinese(denom)
        denom_compact_cn = denomination_to_chinese_lkc(denom)
        coin_amount_text = ""
        render_size = (1600, 600)
    title = (settings.bill_title if settings else "") or ""
    subtitle = (settings.bill_subtitle if settings else "") or ""
    context_base = {
        "username": name,
        "denomination": denom,
        "denom_exponent": denom_exponent,
        "denom_exp": denom_exponent,
        "pow_level": denom_exponent,
        "denomination_words": denom_words,
        "denomination_compact": denom_compact,
        "denomination_words_cn": denom_words_cn,
        "denomination_compact_cn": denom_compact_cn,
        "serial": serial_front,
        "title": title,
        "subtitle": subtitle,
        "portrait_path": get_portrait_path(name),
        "input_image_path": get_portrait_path(name),
        "denomination_color": denom_color,
        "denom_color": denom_color,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "timestamp": timestamp
    }
    if is_coin:
        context_base["coin_amount"] = denom
        context_base["coin_amount_text"] = coin_amount_text
    if extra_context:
        context_base.update(extra_context)
    context_front = dict(context_base)
    context_back = dict(context_base)
    context_back["serial"] = serial_back

    # EisenScript parts from DB
    if is_coin:
        front_pre, front_user, front_suf = load_eisenscript_parts('coin_front')
        back_pre, back_user, back_suf = load_eisenscript_parts('coin_back')
        if not (front_pre or front_user or front_suf):
            front_pre, front_user, front_suf = load_eisenscript_parts('front')
        if not (back_pre or back_user or back_suf):
            back_pre, back_user, back_suf = load_eisenscript_parts('back')
    else:
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
    if not is_coin:
        eisenscript_back = inject_back_denom_background(
            eisenscript_back,
            denom_color,
            width_px=render_size[0],
            height_px=render_size[1],
        )

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
            status["front_png"] = generate_png_from_svg(svg_file, png_front, size=render_size)
        else:
            print(f"[WARNING] SVG not ready for PNG (front): {svg_file}")

    def on_back_svg_saved(svg_path):
        svg_file = resolve_svg_file(Path(svg_path))
        if wait_for_svg_ready(svg_file):
            status["back_png"] = generate_png_from_svg(svg_file, png_back, size=render_size)
        else:
            print(f"[WARNING] SVG not ready for PNG (back): {svg_file}")

    def render_front():
        if progress_callback:
            progress_callback("Rendering front EisenScript...")
        svg_content = _render_eisenscript_via_remote(eisenscript_front)
        if svg_content:
            svg_front.write_text(embed_fonts_in_svg_content(svg_content, font_dir), encoding="utf-8")
            svg_file = resolve_svg_file(svg_front)
        else:
            if progress_callback:
                progress_callback("Rendering front locally...")
            svg_result = render_script_to_svg_html(eisenscript_front, front_dir)
            if isinstance(svg_result, tuple):
                svg_result = svg_result[0]
            svg_path = Path(svg_result) if svg_result else svg_front
            svg_file = resolve_svg_file(normalize_svg_output(svg_path, svg_front))
        embed_fonts_in_svg_file(svg_file, font_dir)
        print(f"[+] Saved FRONT SVG: {svg_file}")
        if progress_callback:
            progress_callback("Front SVG ready.")
        on_front_svg_saved(svg_file)
        status["front_db"] = save_to_database(user, serial_front, denom, "front", str(svg_front), str(png_front))

    def render_back():
        if progress_callback:
            progress_callback("Rendering back EisenScript...")
        svg_content = _render_eisenscript_via_remote(eisenscript_back)
        if svg_content:
            svg_back.write_text(embed_fonts_in_svg_content(svg_content, font_dir), encoding="utf-8")
            svg_file = resolve_svg_file(svg_back)
        else:
            if progress_callback:
                progress_callback("Rendering back locally...")
            svg_result = render_script_to_svg_html(eisenscript_back, back_dir)
            if isinstance(svg_result, tuple):
                svg_result = svg_result[0]
            svg_path = Path(svg_result) if svg_result else svg_back
            svg_file = resolve_svg_file(normalize_svg_output(svg_path, svg_back))
        embed_fonts_in_svg_file(svg_file, font_dir)
        print(f"[+] Saved BACK SVG: {svg_file}")
        if progress_callback:
            progress_callback("Back SVG ready.")
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
                if is_coin:
                    blockchain_daemon_instance.add_genesis_transaction(
                        serial_number=serial_front,
                        denomination=float(denom),
                        issued_to=name,
                        bill_type="coin",
                        is_coin=True,
                    )
                    blockchain_daemon_instance.add_genesis_transaction(
                        serial_number=serial_back,
                        denomination=float(denom),
                        issued_to=name,
                        bill_type="coin",
                        is_coin=True,
                    )
                else:
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

    if success:
        return 1
    if is_coin and (status["front_png"] or status["back_png"] or status["front_db"] or status["back_db"]):
        return 1
    return 0

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
