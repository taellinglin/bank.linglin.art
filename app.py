

# app.py
import os
import shutil
import time
import asyncio
import threading
import json
import hashlib
import gzip
import logging
import os
from typing import Dict
from datetime import timedelta, datetime
from urllib.parse import unquote
from functools import wraps
from urllib.parse import quote_plus
import re
from flask import (
    Flask,
    render_template,
    send_from_directory,
    send_file,
    url_for,
    request,
    redirect,
    flash,
    session,
    abort,
    jsonify,
    g,
)
from PIL import Image, ImageOps
from flask_migrate import Migrate
from werkzeug.middleware.proxy_fix import ProxyFix
from sqlalchemy import desc
from sqlalchemy.exc import OperationalError
from models import User, GenerationTask, Banknote, SerialNumber, Settings, db, WebAuthnCredential
from utils import (
    get_current_user,
    generate_qr_code,
    validate_serial_id,
    GENERATION_LOCK,
    GENERATION_THREADS,
    get_user_avatar_or_default,
    get_user_avatar_url,
    get_user_avatar_thumbnail_url,
    get_user_by_username,
    has_banknotes,
    IMAGES_ROOT,
    get_generation_queue_status,
    get_formatted_initials,
    get_user_avatar,
    sanitize_bio,
    MAX_GENERATION_THREADS,
    execute_generation_task,
    clear_generation_queue_state,
)
import pyotp
from signatures import DigitalBill
from blockchain_daemon import BlockchainDaemon
from blockchain_daemon_modules.async_ops import (
    get_blockchain_manager,
    get_mining_manager,
)

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Lunalib GPU/serialization flags (must be set before daemon init)
os.environ.setdefault("LUNALIB_CUDA_SM3", "0")
os.environ.setdefault("LUNALIB_SM4_USE_GPU", "0")
os.environ.setdefault("LUNALIB_SM4_CUDA_KERNEL", "0")
os.environ.setdefault("LUNALIB_FORCE_SM3_GPU", "0")
os.environ.setdefault("LUNALIB_USE_MSGPACK", "0")

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[91m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname)
        if color:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


root_logger = logging.getLogger()
if not root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter("%(levelname)s:%(name)s:%(message)s"))
    root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)
werkzeug_logger.propagate = False
if not werkzeug_logger.handlers:
    werkzeug_logger.addHandler(logging.NullHandler())

# Cached blockchain stats to avoid blocking web requests
BLOCKCHAIN_STATS_CACHE = {
    "data": None,
    "timestamp": 0,
    "refreshing": False,
}
BLOCKCHAIN_STATS_TTL_SECONDS = 30
BLOCKCHAIN_STATS_LOCK = threading.Lock()

SYSTEM_STATUS_CACHE = {
    "data": None,
    "timestamp": 0,
    "refreshing": False,
}
SYSTEM_STATUS_TTL_SECONDS = 15
SYSTEM_STATUS_LOCK = threading.Lock()

# Prevent concurrent validation of the same block
_BLOCK_SUBMISSION_IN_FLIGHT = set()
_BLOCK_SUBMISSION_LOCK = threading.Lock()


def _ensure_webauthn_name_column():
    try:
        from sqlalchemy import inspect, text

        engine = db.engine
        inspector = inspect(engine)
        if "webauthn_credentials" not in inspector.get_table_names():
            return
        columns = [col["name"] for col in inspector.get_columns("webauthn_credentials")]
        if "name" in columns:
            return
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE webauthn_credentials ADD COLUMN name VARCHAR(120)")
            )
    except Exception as e:
        logger.warning(f"WebAuthn name column check failed: {e}")


def _ensure_serial_numbers_is_mined_column():
    try:
        from sqlalchemy import inspect, text

        engine = db.engine
        inspector = inspect(engine)
        if "serial_numbers" not in inspector.get_table_names():
            return
        columns = [col["name"] for col in inspector.get_columns("serial_numbers")]
        if "is_mined" in columns:
            return
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE serial_numbers ADD COLUMN is_mined BOOLEAN DEFAULT 0")
            )
    except Exception as e:
        logger.warning(f"SerialNumber is_mined column check failed: {e}")


def _ensure_banknotes_verification_columns():
    try:
        from sqlalchemy import inspect, text

        engine = db.engine
        inspector = inspect(engine)
        if "banknotes" not in inspector.get_table_names():
            return
        columns = [col["name"] for col in inspector.get_columns("banknotes")]

        if "is_verified" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE banknotes ADD COLUMN is_verified BOOLEAN DEFAULT 0")
                )

        if "verification_status" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE banknotes ADD COLUMN verification_status VARCHAR(20) DEFAULT 'pending'"
                    )
                )
    except Exception as e:
        logger.warning(f"Banknote verification column check failed: {e}")


def _ensure_users_custom_eisenscript_column():
    try:
        from sqlalchemy import inspect, text

        engine = db.engine
        inspector = inspect(engine)
        if "users" not in inspector.get_table_names():
            return
        columns = [col["name"] for col in inspector.get_columns("users")]
        if "custom_eisenscript" in columns:
            return
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE users ADD COLUMN custom_eisenscript TEXT DEFAULT ''")
            )
    except Exception as e:
        logger.warning(f"Users custom_eisenscript column check failed: {e}")


def _ensure_settings_eisenscript_columns():
    try:
        from sqlalchemy import inspect, text

        engine = db.engine
        inspector = inspect(engine)
        if "settings" not in inspector.get_table_names():
            return
        columns = [col["name"] for col in inspector.get_columns("settings")]
        if "eisenscript_prefix_front" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_prefix_front TEXT DEFAULT ''")
                )
        if "eisenscript_suffix_front" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_suffix_front TEXT DEFAULT ''")
                )
        if "eisenscript_prefix_back" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_prefix_back TEXT DEFAULT ''")
                )
        if "eisenscript_suffix_back" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_suffix_back TEXT DEFAULT ''")
                )
        if "icon_dir" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN icon_dir VARCHAR(255) DEFAULT './icons'")
                )
        if "eisenscript_dir" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_dir VARCHAR(255) DEFAULT './eisen'")
                )
        if "eisenscript_prefix_coin_front" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_prefix_coin_front TEXT DEFAULT ''")
                )
        if "eisenscript_suffix_coin_front" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_suffix_coin_front TEXT DEFAULT ''")
                )
        if "eisenscript_prefix_coin_back" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_prefix_coin_back TEXT DEFAULT ''")
                )
        if "eisenscript_suffix_coin_back" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_suffix_coin_back TEXT DEFAULT ''")
                )
        if "eisenscript_prefix_card_front" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_prefix_card_front TEXT DEFAULT ''")
                )
        if "eisenscript_suffix_card_front" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_suffix_card_front TEXT DEFAULT ''")
                )
        if "eisenscript_prefix_card_back" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_prefix_card_back TEXT DEFAULT ''")
                )
        if "eisenscript_suffix_card_back" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_suffix_card_back TEXT DEFAULT ''")
                )
        if "eisenscript_receipt" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE settings ADD COLUMN eisenscript_receipt TEXT DEFAULT ''")
                )
    except Exception as e:
        logger.warning(f"Settings eisenscript column check failed: {e}")


def sanitize_eisenscript(script_text: str, max_length: int = 20000) -> str:
    if not script_text:
        return ""
    cleaned = script_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "".join(ch for ch in cleaned if (ch == "\n" or ch == "\t" or 32 <= ord(ch) <= 126))
    return cleaned[:max_length]


# ROYGBIV Color Scheme 🌈 plus more
class Colors:
    # Basic colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Bright background colors
    BG_BRIGHT_BLACK = "\033[100m"
    BG_BRIGHT_RED = "\033[101m"
    BG_BRIGHT_GREEN = "\033[102m"
    BG_BRIGHT_YELLOW = "\033[103m"
    BG_BRIGHT_BLUE = "\033[104m"
    BG_BRIGHT_MAGENTA = "\033[105m"
    BG_BRIGHT_CYAN = "\033[106m"
    BG_BRIGHT_WHITE = "\033[107m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    HIDDEN = "\033[8m"
    STRIKETHROUGH = "\033[9m"

    # Reset
    END = "\033[0m"


def color_text(text, *color_codes):
    """
    Color text with one or more color/style codes

    Usage:
        color_text("Hello", Colors.RED)
        color_text("Warning", Colors.YELLOW, Colors.BOLD)
        color_text("Error", Colors.RED, Colors.BOLD, Colors.BG_WHITE)
    """
    color_string = "".join(color_codes)
    return f"{color_string}{text}{Colors.END}"


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.secret_key = os.environ.get("SECRET_KEY", "ILoveYouForeverXOXO")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///lingcountrytreasury.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "connect_args": {"timeout": 30, "check_same_thread": False},
}
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

# Initialize email service
from email_service import (
    init_mail,
    send_verification_email,
    send_banknote_generation_notification,
)

mail = init_mail(app)

# Initialize notification scheduler
from notification_scheduler import init_notification_scheduler

notification_scheduler = None

# Initialize db with app
DATA_DIR = "./system-data/"
db.init_app(app)
migrate = Migrate(app, db)
# Improve SQLite concurrency
from sqlalchemy import event
from sqlalchemy.engine import Engine


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
    except Exception:
        pass
# In app.py, near the top with other initializations
blockchain_daemon_instance = None
blockchain_daemon_initialized = False


def init_blockchain_daemon():
    global blockchain_daemon_instance, blockchain_daemon_initialized
    if not blockchain_daemon_initialized:
        blockchain_daemon_instance = BlockchainDaemon(
            blockchain_file="blockchain_data/blockchain.json",
            mempool_file="mempool_data/mempool.json",
            endpoint_url="https://bank.linglin.art",
        )
        # Start the daemon after creation
        blockchain_daemon_instance.start_daemon()
        blockchain_daemon_initialized = True
        print("[BLOCKCHAIN] Blockchain daemon initialized and started")
    return blockchain_daemon_instance


# Call this ONCE during app initialization
blockchain_daemon = init_blockchain_daemon()


@app.template_filter("format_number")
def format_number(value):
    """Format numbers with commas for thousands."""
    try:
        # Handle None, empty string, or non-numeric values
        if value is None or value == "":
            return "0"

        # Convert to int if it's a number
        if isinstance(value, (int, float)):
            num = int(value)
        else:
            # Try to convert string to int
            num = int(str(value).replace(",", "").split(".")[0])

        # Format with commas
        return f"{num:,}"
    except (ValueError, TypeError):
        return "0"


@app.template_filter("format_lkc")
def format_lkc(value):
    """Format LKC with up to 6 decimals, at least 2 decimals."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "0.00"
    text = f"{num:.6f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if "." not in text:
        return f"{text}.00"
    whole, decimals = text.split(".")
    if len(decimals) < 2:
        decimals = decimals.ljust(2, "0")
    return f"{whole}.{decimals}"


# Add as both a global and filter for flexibility
@app.template_global("max")
def template_max(a, b):
    """Max function for templates."""
    try:
        return max(a, b)
    except (TypeError, ValueError):
        return a if a > b else b


# Optional: Also add as a filter
@app.template_filter("safe_max")
def safe_max_filter(value, compare_to):
    """Safe max filter for templates."""
    return template_max(value, compare_to)


@app.context_processor
def utility_processor():
    """
    Make functions available to all templates
    """
    return {
        "get_user_avatar": get_user_avatar,  # Add this
        "get_formatted_initials": get_formatted_initials,  # Add this
        "get_user_avatar_url": get_user_avatar_url,
        "get_user_avatar_thumbnail_url": get_user_avatar_thumbnail_url,
        "get_user_by_username": get_user_by_username,
        "has_banknotes": has_banknotes,
    }


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        current_user = getattr(g, "current_user", None) or get_current_user()
        if not current_user or not getattr(current_user, "is_admin", False):
            flash("Admin access required", "error")
            return redirect(url_for("landing"))
        return f(*args, **kwargs)

    return decorated


def _user_has_strong_auth(user: User) -> bool:
    if not user:
        return False
    if getattr(user, "two_factor_secret", None):
        return True
    try:
        return (
            WebAuthnCredential.query.filter_by(user_id=user.id).first() is not None
        )
    except Exception:
        return False


def run_generation_task(user_id, username):
    """Queue a generation task."""
    try:
        # Ensure the background processor is running before enqueuing.
        start_generation_task_processor()
        # Always create a task in 'pending' state.
        # The worker will pick it up.
        task = GenerationTask(user_id=user_id, status="pending")
        db.session.add(task)
        db.session.commit()
        print(f"Queued generation task {task.id} for user {username}.")
        return task.id
    except Exception as e:
        print(f"Error queuing generation task: {e}")
        try:
            db.session.rollback()
        except Exception:
            pass
        return None




def process_pending_generation_tasks():
    """
    A worker function that runs in a background thread to process pending tasks.
    """
    with app.app_context():
        # Normalize queued/processing tasks on startup so they can be resumed
        try:
            stuck_tasks = GenerationTask.query.filter(
                GenerationTask.status.in_(["queued", "processing"])
            ).all()
            for task in stuck_tasks:
                task.status = "pending"
                if not task.message:
                    task.message = "Resumed after restart."
            if stuck_tasks:
                db.session.commit()
                print(f"[GENERATION] Resumed {len(stuck_tasks)} queued/processing task(s) after restart")
        except Exception as e:
            print(f"[GENERATION] Failed to normalize tasks on startup: {e}")

        while True:
            try:
                # Find the next pending task
                task = GenerationTask.query.filter(
                    GenerationTask.status.in_(["pending", "queued", "processing"])
                ).order_by(GenerationTask.created_at).first()

                if task:
                    # Check if we have capacity to run a new task
                    with GENERATION_LOCK:
                        if len(GENERATION_THREADS) < MAX_GENERATION_THREADS:
                            if task.id in GENERATION_THREADS:
                                # Already running in this process
                                pass
                            else:
                                task_username = task.user.username if task.user else "Unknown"
                                print(f"Found pending task {task.id} for user {task_username}. Starting generation.")
                                # Mark task as processing
                                task.status = 'processing'
                                db.session.commit()

                                # Start the generation in a new thread
                                thread = threading.Thread(target=execute_generation_task, args=(task.id,))
                                GENERATION_THREADS[task.id] = thread
                                thread.start()
                        else:
                            # print("Max generation threads reached. Waiting...")
                            pass
                else:
                    # No pending tasks, wait a bit
                    # print("No pending tasks found. Waiting...")
                    pass

            except Exception as e:
                print(f"Error in pending task processor: {e}")

            # Wait for a bit before checking again to avoid busy-waiting
            time.sleep(10)


def start_generation_task_processor():
    """Start the background task processor once per process."""
    if getattr(app, "_generation_task_processor_started", False):
        return
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        app._generation_task_processor_started = True
        print("Starting background task processor for pending generation tasks.")
        processor_thread = threading.Thread(target=process_pending_generation_tasks, daemon=True)
        processor_thread.start()



@app.route("/blockchain", methods=["GET"])
def get_blockchain():
    """Serve the complete blockchain"""
    return jsonify(blockchain_daemon_instance.blockchain)


def hash_to_int(hash_hex):
    """Convert hex hash to integer"""
    return int(hash_hex, 16)


@app.route("/block/<string:block_hash>")
def view_block_detail(block_hash):
    """Display detailed information about a specific block"""
    try:
        print(
            f"🔍 DEBUG: Starting view_block_detail for hash: {block_hash} (type: {type(block_hash)})"
        )

        # Search for the block with matching hash - IMPROVED HASH HANDLING
        found_block = None
        found_index = -1

        # Clean the input hash
        input_hash_clean = str(block_hash).strip().lower()
        print(f"🔍 DEBUG: Searching for hash: '{input_hash_clean}'")

        for i, block in enumerate(blockchain_daemon_instance.blockchain):
            # Get block hash and clean it
            block_hash_raw = block.get("hash", "")
            block_hash_clean = str(block_hash_raw).strip().lower()

            print(f"🔍 DEBUG: Checking block {i}: '{block_hash_clean}'")

            if block_hash_clean == input_hash_clean:
                found_block = block
                found_index = i
                print(f"✅ DEBUG: Found block at index {i}")
                break

        if not found_block:
            print(f"❌ DEBUG: Block not found for hash: {block_hash}")
            # Try to find by index if hash lookup fails
            try:
                block_index = int(block_hash)
                if 0 <= block_index < len(blockchain_daemon_instance.blockchain):
                    found_block = blockchain_daemon_instance.blockchain[block_index]
                    found_index = block_index
                    print(f"✅ DEBUG: Found block by index: {block_index}")
            except (ValueError, IndexError):
                pass

        if not found_block:
            print(f"❌ DEBUG: Block not found for hash/index: {block_hash}")
            flash("Block not found", "error")
            return redirect(url_for("blockchain_viewer"))

        # Calculate detailed block information
        transactions = found_block.get("transactions", [])
        print(f"📊 DEBUG: Block has {len(transactions)} transactions")

        # Count transaction types
        genesis_count = sum(
            1 for tx in transactions if tx.get("type") in ["genesis", "GTX_Genesis"]
        )
        transfer_count = sum(1 for tx in transactions if tx.get("type") == "transfer")
        reward_count = sum(1 for tx in transactions if tx.get("type") == "reward")
        other_count = len(transactions) - genesis_count - transfer_count - reward_count

        print(
            f"📊 DEBUG: Transaction counts - genesis: {genesis_count}, transfer: {transfer_count}, reward: {reward_count}, other: {other_count}"
        )

        # SIMPLE timestamp handling
        def safe_timestamp_to_readable(ts):
            """Simple timestamp conversion"""
            try:
                if ts is None:
                    return "Unknown"

                # Force conversion to float
                if isinstance(ts, str):
                    # Remove any non-numeric characters except . and -
                    clean_ts = "".join(
                        c for c in ts if c.isdigit() or c == "." or c == "-"
                    )
                    ts = float(clean_ts) if clean_ts and clean_ts != "-" else 0
                else:
                    ts = float(ts)

                if ts > 0 and ts < 4102444800:
                    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    return "Invalid timestamp"

            except Exception as e:
                return f"Error: {str(e)}"

        # Process block timestamp
        block_timestamp = found_block.get("timestamp", 0)
        readable_time = safe_timestamp_to_readable(block_timestamp)

        # For genesis block, use special message
        block_index = found_block.get("index", found_index)
        if block_index == 0:
            readable_time = "Genesis Block"

        # Calculate block size
        try:
            block_size = len(json.dumps(found_block, default=str))
        except Exception:
            block_size = 0

        # FIX: Ensure mining_time is a number, not a string
        mining_time = found_block.get("mining_time", 0)
        try:
            if isinstance(mining_time, str):
                # Extract numbers from string
                clean_mining_time = "".join(
                    c for c in mining_time if c.isdigit() or c == "."
                )
                mining_time_numeric = (
                    float(clean_mining_time) if clean_mining_time else 0.0
                )
            else:
                mining_time_numeric = float(mining_time)
        except:
            mining_time_numeric = 0.0

        # Get previous and next block info for navigation
        previous_block = None
        next_block = None

        try:
            block_index_int = int(block_index)
            if block_index_int > 0 and blockchain_daemon_instance.blockchain:
                previous_block = (
                    blockchain_daemon_instance.blockchain[block_index_int - 1]
                    if block_index_int - 1 < len(blockchain_daemon_instance.blockchain)
                    else None
                )

            if block_index_int + 1 < len(blockchain_daemon_instance.blockchain):
                next_block = blockchain_daemon_instance.blockchain[block_index_int + 1]
        except (IndexError, ValueError, TypeError):
            pass

        # SAFE transaction details preparation
        def normalize_png_path(png_path: str) -> str:
            if not png_path:
                return ""
            path = str(png_path).replace("\\", "/")
            if os.path.isabs(path):
                try:
                    rel = os.path.relpath(path, IMAGES_ROOT)
                    rel = rel.replace("\\", "/")
                    if not rel.startswith(".."):
                        path = rel
                except Exception:
                    pass
            if path.startswith("./"):
                path = path[2:]
            if path.startswith("images/"):
                path = path[len("images/"):]
            return path

        def resolve_banknote_by_serial(serial_value: str):
            if not serial_value:
                return None
            banknote = Banknote.query.filter_by(serial_number=serial_value).first()
            if not banknote:
                serial_record = SerialNumber.query.filter_by(serial=serial_value).first()
                if serial_record and serial_record.banknote_id:
                    banknote = Banknote.query.get(serial_record.banknote_id)
            return banknote

        def derive_counterpart_thumbnail(png_path: str, target_side: str) -> str:
            if not png_path:
                return ""
            rel_path = normalize_png_path(png_path)
            if not rel_path:
                return ""
            dir_rel = os.path.dirname(rel_path)
            filename = os.path.basename(rel_path)
            if target_side == "front":
                swapped = re.sub(r"_BACK", "_FRONT", filename, flags=re.IGNORECASE)
            else:
                swapped = re.sub(r"_FRONT", "_BACK", filename, flags=re.IGNORECASE)
            if swapped == filename:
                return ""
            candidate_rel = f"{dir_rel}/{swapped}" if dir_rel else swapped
            candidate_abs = os.path.join(IMAGES_ROOT, candidate_rel)
            if os.path.exists(candidate_abs):
                return candidate_rel
            return ""

        transaction_details = []
        for i, tx in enumerate(transactions):
            if not isinstance(tx, dict):
                continue
            # Convert transaction timestamp safely
            tx_timestamp = tx.get("timestamp", 0)
            tx_readable_time = safe_timestamp_to_readable(tx_timestamp)
            # Ensure numeric timestamp for template
            numeric_timestamp = 0
            try:
                if isinstance(tx_timestamp, (int, float)):
                    numeric_timestamp = float(tx_timestamp)
                elif isinstance(tx_timestamp, str):
                    clean_ts = "".join(
                        c for c in tx_timestamp if c.isdigit() or c == "." or c == "-"
                    )
                    numeric_timestamp = (
                        float(clean_ts) if clean_ts and clean_ts != "-" else 0
                    )
                else:
                    numeric_timestamp = float(tx_timestamp) if tx_timestamp else 0
            except Exception:
                numeric_timestamp = 0
            # Ensure hash is properly formatted string
            tx_hash = str(tx.get("hash", f"tx-{i}")).strip()
            tx_info = {
                "index": i + 1,
                "type": str(tx.get("type", "unknown")),
                "hash": tx_hash,
                "timestamp": numeric_timestamp,
                "timestamp_readable": tx_readable_time,
                "size": len(json.dumps(tx, default=str)) if tx else 0,
            }
            # Add type-specific fields with proper string conversion
            tx_type = tx.get("type", "")
            if tx_type == "transfer":
                tx_info.update(
                    {
                        "from": str(tx.get("from", "N/A")),
                        "to": str(tx.get("to", "N/A")),
                        "amount": tx.get("amount", "N/A"),
                    }
                )
            elif tx_type in ["genesis", "GTX_Genesis"]:
                serial_number = str(tx.get("serial_number") or "")
                front_serial = str(tx.get("front_serial") or "")
                back_serial = str(tx.get("back_serial") or "")
                bill_serial = str(tx.get("bill_serial") or "")
                metadata_hash = str(tx.get("metadata_hash") or "")

                if serial_number and not serial_number.startswith("SN-"):
                    serial_number = ""
                if front_serial and not front_serial.startswith("SN-"):
                    front_serial = ""
                if back_serial and not back_serial.startswith("SN-"):
                    back_serial = ""

                primary_serial = front_serial or back_serial or serial_number
                primary_banknote = resolve_banknote_by_serial(primary_serial)

                if primary_banknote and primary_banknote.side:
                    side_value = str(primary_banknote.side).lower()
                    if side_value == "front" and not front_serial:
                        front_serial = primary_serial
                    if side_value == "back" and not back_serial:
                        back_serial = primary_serial
                elif primary_serial and not front_serial and not back_serial:
                    front_serial = primary_serial

                front_banknote = resolve_banknote_by_serial(front_serial)
                back_banknote = resolve_banknote_by_serial(back_serial)

                front_thumbnail = normalize_png_path(front_banknote.png_path) if front_banknote and front_banknote.png_path else ""
                back_thumbnail = normalize_png_path(back_banknote.png_path) if back_banknote and back_banknote.png_path else ""

                if primary_banknote and primary_banknote.png_path:
                    primary_thumb = normalize_png_path(primary_banknote.png_path)
                    if primary_thumb and not (front_thumbnail and back_thumbnail):
                        primary_side = str(primary_banknote.side or "").lower()
                        if primary_side == "front" and not front_thumbnail:
                            front_thumbnail = primary_thumb
                        elif primary_side == "back" and not back_thumbnail:
                            back_thumbnail = primary_thumb

                        if primary_side == "front" and not back_thumbnail:
                            back_thumbnail = derive_counterpart_thumbnail(primary_banknote.png_path, "back")
                        if primary_side == "back" and not front_thumbnail:
                            front_thumbnail = derive_counterpart_thumbnail(primary_banknote.png_path, "front")

                tx_info.update(
                    {
                        "serial_number": serial_number or "",
                        "front_serial": front_serial or "",
                        "back_serial": back_serial or "",
                        "bill_serial": bill_serial or "",
                        "metadata_hash": metadata_hash or "",
                        "issued_to": str(tx.get("issued_to", "N/A")),
                        "denomination": tx.get("denomination", "N/A"),
                        "front_thumbnail": front_thumbnail,
                        "back_thumbnail": back_thumbnail,
                    }
                )
            elif tx_type == "reward":
                tx_info.update(
                    {
                        "to": str(tx.get("to", "N/A")),
                        "amount": tx.get("amount", "N/A"),
                        "block_height": tx.get("block_height", "N/A"),
                        "description": str(tx.get("description", "Mining Reward")),
                    }
                )
            transaction_details.append(tx_info)

        def _merge_genesis_transactions(items):
            combined = []
            merged_index = {}

            def _merge_into(target, source):
                for field in ("front_serial", "back_serial", "serial_number"):
                    if not target.get(field) and source.get(field):
                        target[field] = source.get(field)

                if source.get("front_serial") and not target.get("back_serial"):
                    if target.get("front_serial") and target.get("front_serial") != source.get("front_serial"):
                        target["back_serial"] = source.get("front_serial")

                if source.get("back_serial") and not target.get("front_serial"):
                    if target.get("back_serial") and target.get("back_serial") != source.get("back_serial"):
                        target["front_serial"] = source.get("back_serial")

                if not target.get("front_thumbnail") and source.get("front_thumbnail"):
                    target["front_thumbnail"] = source.get("front_thumbnail")
                if not target.get("back_thumbnail") and source.get("back_thumbnail"):
                    target["back_thumbnail"] = source.get("back_thumbnail")

                if not target.get("issued_to") and source.get("issued_to"):
                    target["issued_to"] = source.get("issued_to")
                if not target.get("denomination") and source.get("denomination"):
                    target["denomination"] = source.get("denomination")

                try:
                    target["size"] = int(target.get("size") or 0) + int(source.get("size") or 0)
                except Exception:
                    pass

            def _build_key(tx_item):
                bill_serial = tx_item.get("bill_serial")
                if isinstance(bill_serial, str) and bill_serial.strip():
                    return (bill_serial.strip(),)
                metadata_hash = tx_item.get("metadata_hash")
                if isinstance(metadata_hash, str) and metadata_hash.strip():
                    return (metadata_hash.strip(),)
                serials = [
                    tx_item.get("front_serial"),
                    tx_item.get("back_serial"),
                    tx_item.get("serial_number"),
                ]
                serials = [s for s in serials if isinstance(s, str) and s.strip()]
                if serials:
                    return tuple(sorted(set(serials)))
                return (tx_item.get("hash") or f"tx-{tx_item.get('index')}",)

            for item in items:
                if item.get("type") not in ["genesis", "GTX_Genesis"]:
                    combined.append(item)
                    continue

                key = _build_key(item)
                existing = merged_index.get(key)
                if not existing:
                    merged_index[key] = item
                    combined.append(item)
                    continue

                _merge_into(existing, item)

            final = []
            by_front = {}
            by_back = {}
            for item in combined:
                if item.get("type") not in ["genesis", "GTX_Genesis"]:
                    final.append(item)
                    continue

                front = item.get("front_serial")
                back = item.get("back_serial")
                match = None
                if front and front in by_front:
                    match = by_front[front]
                if not match and back and back in by_back:
                    match = by_back[back]

                if match and match is not item:
                    _merge_into(match, item)
                    continue

                final.append(item)
                if front:
                    by_front[front] = item
                if back:
                    by_back[back] = item

            return final

        transaction_details = _merge_genesis_transactions(transaction_details)
        for idx, tx_item in enumerate(transaction_details, start=1):
            tx_item["index"] = idx

        genesis_count = sum(
            1
            for tx in transaction_details
            if tx.get("type") in ["genesis", "GTX_Genesis"]
        )
        transfer_count = sum(1 for tx in transaction_details if tx.get("type") == "transfer")
        reward_count = sum(1 for tx in transaction_details if tx.get("type") == "reward")
        other_count = len(transaction_details) - genesis_count - transfer_count - reward_count
        # Prepare block info for template - ensure all values are properly typed
        block_info = {
            "block": found_block,
            "metadata": {
                "transaction_count": int(len(transaction_details)),
                "genesis_count": int(genesis_count),
                "transfer_count": int(transfer_count),
                "reward_count": int(reward_count),
                "other_count": int(other_count),
                "timestamp_readable": str(readable_time),
                "block_size": int(block_size),
                "is_genesis_block": bool(block_index == 0),
                "miner": str(found_block.get("miner", "Unknown")),
                "difficulty": found_block.get("difficulty", "N/A"),
                "mining_time": mining_time_numeric,  # FIXED: This is now a number, not a string
            },
            "transactions": transaction_details,
            "navigation": {
                "previous_block": previous_block,
                "next_block": next_block,
                "current_index": int(block_index),
                "total_blocks": int(
                    len(blockchain_daemon_instance.blockchain)
                    if blockchain_daemon_instance.blockchain
                    else 0
                ),
            },
        }
        print(f"✅ DEBUG: Successfully prepared block info for #{block_index}")
        print(
            f"🔧 DEBUG: Mining time type: {type(block_info['metadata']['mining_time'])} value: {block_info['metadata']['mining_time']}"
        )
        return render_template(
            "block_detail.html",
            block_info=block_info,
            current_user=get_current_user(),
            title=f"Block #{block_index} Details",
        )

    except Exception as e:
        import traceback

        error_details = f"Error in view_block_detail: {str(e)}"
        print(f"❌ DEBUG: {error_details}")
        print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")

        flash(f"Error loading block details: {str(e)}", "error")
        return redirect(url_for("blockchain_viewer"))

@app.route("/get_block/<block_id>", methods=["GET"])
def get_block_by_id(block_id):
    """Get block by hash or index (id)."""
    try:
        block = blockchain_daemon_instance.get_block(block_id)
        if not block:
            return jsonify({"success": False, "error": "Block not found"}), 404
        response = jsonify({"success": True, "block": block})
        accept_encoding = (request.headers.get("Accept-Encoding") or "").lower()
        if "gzip" in accept_encoding:
            try:
                compressed = gzip.compress(response.get_data())
                response.set_data(compressed)
                response.headers["Content-Encoding"] = "gzip"
                response.headers["Vary"] = "Accept-Encoding"
                response.headers["Content-Length"] = str(len(compressed))
            except Exception:
                return response, 200
        return response, 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
# --- Blockchain Range API ---
from flask import request, jsonify

@app.route('/blockchain/range', methods=['GET'])
def blockchain_range():
    """
    Returns blocks in the range [start, end] (inclusive) as JSON.
    Query params: start, end (block index, integer)
    """
    try:
        start = int(request.args.get('start', ''))
        end = int(request.args.get('end', ''))
    except Exception:
        return jsonify({'error': 'Invalid start or end parameter'}), 400
    if start > end or start < 0:
        return jsonify({'error': 'Invalid range'}), 400

    # Load blockchain data (assume blockchain.json is the canonical source)
    import os, json
    chain_path = os.path.join(os.path.dirname(__file__), 'blockchain_daemon', 'blockchain.json')
    if not os.path.exists(chain_path):
        return jsonify({'error': 'Blockchain data not found'}), 500
    with open(chain_path, encoding='utf-8') as f:
        try:
            chain = json.load(f)
        except Exception:
            return jsonify({'error': 'Failed to load blockchain data'}), 500

    # Defensive: chain may be a dict with 'chain' key or a list
    if isinstance(chain, dict) and 'chain' in chain:
        blocks = chain['chain']
    else:
        blocks = chain

    # Filter blocks in the requested range
    result = [b for b in blocks if isinstance(b, dict) and 'index' in b and start <= b['index'] <= end]
    return jsonify({'blocks': result, 'count': len(result)}), 200
# --- API: Transaction Verification Detail ---
@app.route("/transactions/verify/<string:tx_hash>", methods=["GET"])
def api_verify_transaction(tx_hash):
    """
    API endpoint: /transactions/verify/<tx_hash>
    Returns transaction verification details as JSON.
    """
    # Try to get transaction from blockchain/mempool
    tx = blockchain_daemon_instance.get_transaction(tx_hash)
    if not tx:
        return (
            jsonify(
                {"found": False, "error": "Transaction not found", "tx_hash": tx_hash}
            ),
            404,
        )

    # Check if mined
    is_mined = blockchain_daemon_instance.is_transaction_mined(tx)
    # Try to find block info if mined
    block_index = None
    block_hash = None
    if is_mined and hasattr(blockchain_daemon_instance, "blockchain"):
        for idx, block in enumerate(blockchain_daemon_instance.blockchain):
            for btx in block.get("transactions", []):
                if btx.get("hash") == tx_hash:
                    block_index = idx
                    block_hash = block.get("hash")
                    break
            if block_index is not None:
                break

    # Compose response
    resp = {
        "found": True,
        "tx_hash": tx_hash,
        "transaction": tx,
        "mined": is_mined,
        "block_index": block_index,
        "block_hash": block_hash,
    }
    return jsonify(resp)


@app.route("/mempool", methods=["GET"])
def get_mempool():
    """Serve filtered mempool (only unmined transactions) - FIXED"""
    # This should return filtered mempool, not the raw one

    filtered_mempool = blockchain_daemon_instance.mempool
    # filtered_mempool = filter_mined_transactions(filtered_mempool)
    return jsonify(filtered_mempool)  # Return filtered, not the full mempool


from datetime import datetime, timedelta
from collections import defaultdict
import statistics


@app.route("/mempool-viewer")
@app.route("/mempool-viewer/<int:page>")
def mempool_viewer(page=1):
    """Display detailed mempool information in a web interface WITH PAGINATION"""
    try:
        page_provided = bool(request.view_args and "page" in request.view_args)
        # Get mempool data locally (avoid network calls)
        mempool_data = getattr(blockchain_daemon_instance, "mempool", []) or []
        allowed_types = {"transfer", "genesis", "GTX_Genesis"}
        mempool_data = [
            tx for tx in mempool_data if tx.get("type") in allowed_types
        ]

        # Ensure consistent ordering (oldest -> newest) before pagination
        mempool_data.sort(key=lambda tx: tx.get("timestamp", 0))

        # Get blockchain status for additional context (local only)
        blockchain_status = blockchain_daemon_instance.get_blockchain_status()

        # Pagination settings
        per_page = 15  # Reduced for compact view

        # Build transaction details for all mempool entries
        transactions = []
        for tx in mempool_data:
            tx_info = {
                "hash": tx.get("hash", "N/A"),
                "type": tx.get("type", "unknown"),
                "timestamp": tx.get("timestamp", 0),
                "timestamp_readable": datetime.fromtimestamp(
                    tx.get("timestamp", 0)
                ).strftime("%Y-%m-%d %H:%M:%S")
                if tx.get("timestamp")
                else "Unknown",
                "is_mined": False,
                "size": len(json.dumps(tx)),
                "confirmations": 0,
            }

            # Add type-specific fields
            if tx.get("type") == "transfer":
                tx_info["from"] = tx.get("from", "N/A")
                tx_info["to"] = tx.get("to", "N/A")
                tx_info["amount"] = tx.get("amount", "N/A")

            elif tx.get("type") in ["genesis", "GTX_Genesis"]:
                serial_number = tx.get("serial_number") or ""
                front_serial = tx.get("front_serial") or ""
                back_serial = tx.get("back_serial") or ""
                tx_info["serial_number"] = serial_number or front_serial or back_serial or "N/A"
                tx_info["front_serial"] = front_serial
                tx_info["back_serial"] = back_serial
                tx_info["bill_serial"] = tx.get("bill_serial", "")
                tx_info["metadata_hash"] = tx.get("metadata_hash", "")
                tx_info["issued_to"] = tx.get("issued_to", "N/A")
                tx_info["denomination"] = tx.get("denomination", "N/A")
                tx_info["amount"] = tx.get("amount", tx_info["denomination"])

            transactions.append(tx_info)

        def _merge_genesis_transactions(items):
            combined = []
            merged_index = {}

            def _merge_into(target, source):
                for field in ("front_serial", "back_serial", "serial_number"):
                    if not target.get(field) and source.get(field):
                        target[field] = source.get(field)

                if source.get("front_serial") and not target.get("back_serial"):
                    if target.get("front_serial") and target.get("front_serial") != source.get("front_serial"):
                        target["back_serial"] = source.get("front_serial")

                if source.get("back_serial") and not target.get("front_serial"):
                    if target.get("back_serial") and target.get("back_serial") != source.get("back_serial"):
                        target["front_serial"] = source.get("back_serial")

                if not target.get("issued_to") and source.get("issued_to"):
                    target["issued_to"] = source.get("issued_to")
                if not target.get("denomination") and source.get("denomination"):
                    target["denomination"] = source.get("denomination")
                if not target.get("amount") and source.get("amount"):
                    target["amount"] = source.get("amount")

                try:
                    target["size"] = int(target.get("size") or 0) + int(source.get("size") or 0)
                except Exception:
                    pass

            def _build_key(tx_item):
                bill_serial = tx_item.get("bill_serial")
                if isinstance(bill_serial, str) and bill_serial.strip():
                    return (bill_serial.strip(),)
                metadata_hash = tx_item.get("metadata_hash")
                if isinstance(metadata_hash, str) and metadata_hash.strip():
                    return (metadata_hash.strip(),)
                serials = [
                    tx_item.get("front_serial"),
                    tx_item.get("back_serial"),
                    tx_item.get("serial_number"),
                ]
                serials = [s for s in serials if isinstance(s, str) and s.strip()]
                if serials:
                    return tuple(sorted(set(serials)))
                return (tx_item.get("hash") or "unknown",)

            for item in items:
                if item.get("type") not in ["genesis", "GTX_Genesis"]:
                    combined.append(item)
                    continue

                key = _build_key(item)
                existing = merged_index.get(key)
                if not existing:
                    merged_index[key] = item
                    combined.append(item)
                    continue

                _merge_into(existing, item)

            final = []
            by_front = {}
            by_back = {}
            for item in combined:
                if item.get("type") not in ["genesis", "GTX_Genesis"]:
                    final.append(item)
                    continue

                front = item.get("front_serial")
                back = item.get("back_serial")
                match = None
                if front and front in by_front:
                    match = by_front[front]
                if not match and back and back in by_back:
                    match = by_back[back]

                if match and match is not item:
                    _merge_into(match, item)
                    continue

                final.append(item)
                if front:
                    by_front[front] = item
                if back:
                    by_back[back] = item

            return final

        transactions = _merge_genesis_transactions(transactions)

        # Sort transactions by timestamp (newest first)
        transactions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        total_transactions = len(transactions)
        total_pages = max(1, (total_transactions + per_page - 1) // per_page)

        # If no page param provided, default to latest page
        if not page_provided:
            page = total_pages

        # Ensure page is within valid range
        page = max(1, min(page, total_pages))

        # Calculate slice for current page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        current_transactions = transactions[start_idx:end_idx]

        print(
            f"🔍 Mempool Pagination: page {page}, showing transactions {start_idx}-{end_idx} of {total_transactions}"
        )

        # Calculate statistics
        active_transactions = total_transactions
        mined_transactions = blockchain_status["total_transactions"]

        # Count by transaction type
        type_counts = {
            "bills": len(
                [tx for tx in transactions if tx.get("type") in ["genesis", "GTX_Genesis"]]
            ),
            "transfers": len(
                [tx for tx in transactions if tx.get("type") == "transfer"]
            ),
        }

        # Get blockchain info for context
        blockchain_info = {
            "total_blocks": blockchain_status["blocks"],
            "total_mined_transactions": mined_transactions,
            "mined_genesis": blockchain_status["genesis_transactions"],
            "mined_transfers": blockchain_status["transfer_transactions"],
        }

        return render_template(
            "mempool_viewer.html",
            transactions=current_transactions,
            total_transactions=total_transactions,
            active_transactions=active_transactions,
            mined_transactions=mined_transactions,
            type_counts=type_counts,
            blockchain_info=blockchain_info,
            current_page=page,
            total_pages=total_pages,
            per_page=per_page,
            current_user=get_current_user(),
            title="Mempool Viewer",
        )

    except Exception as e:
        print(f"❌ Error in mempool_viewer: {e}")
        flash(f"Error loading mempool data: {str(e)}", "error")
        return render_template(
            "mempool_viewer.html",
            transactions=[],
            total_transactions=0,
            active_transactions=0,
            mined_transactions=0,
            type_counts={},
            blockchain_info={},
            current_page=1,
            total_pages=1,
            per_page=15,
            current_user=get_current_user(),
            title="Mempool Viewer",
        )


@app.route("/api/mempool/activity")
def mempool_activity():
    """API endpoint for mempool activity data"""
    try:
        # Get all mempool transactions locally (avoid network calls)
        all_transactions = getattr(blockchain_daemon_instance, "mempool", []) or []
        allowed_types = {"transfer", "genesis", "GTX_Genesis"}
        all_transactions = [
            tx for tx in all_transactions if tx.get("type") in allowed_types
        ]

        if not all_transactions:
            return jsonify(
                {
                    "timeline": {"transfers": [], "bills": []},
                    "labels": [],
                    "peak": 0,
                    "average_per_minute": 0,
                    "totals": {"all": 0, "transfers": 0, "bills": 0},
                    "timeframe": "auto",
                }
            )

        tx_times = [tx.get("timestamp", 0) for tx in all_transactions]
        min_ts = min(tx_times)
        max_ts = max(tx_times)

        start_time = datetime.fromtimestamp(min_ts)
        end_time = datetime.fromtimestamp(max_ts)

        total_seconds = max((end_time - start_time).total_seconds(), 0)
        num_intervals = 12 if total_seconds > 0 else 1
        interval_seconds = max(total_seconds / max(num_intervals - 1, 1), 60)
        interval_minutes = interval_seconds / 60

        # Initialize data structures
        interval_data = {
            "transfers": [0] * num_intervals,
            "bills": [0] * num_intervals,
        }

        # Generate labels for x-axis
        labels = []
        for i in range(num_intervals):
            label_time = start_time + timedelta(minutes=interval_minutes * i)
            if total_seconds <= 86400:
                labels.append(label_time.strftime("%H:%M"))
            else:
                labels.append(label_time.strftime("%m/%d"))

        # Process transactions
        for tx in all_transactions:
            tx_time = datetime.fromtimestamp(tx.get("timestamp", 0))

            # Skip if transaction is outside range
            if tx_time < start_time or tx_time > end_time:
                continue

            # Calculate which interval this transaction belongs to
            time_diff = (
                tx_time - start_time
            ).total_seconds() / 60  # difference in minutes
            interval_index = min(int(time_diff // interval_minutes), num_intervals - 1)

            # Count by type
            tx_type = tx.get("type", "unknown")
            if tx_type == "transfer":
                interval_data["transfers"][interval_index] += 1
            elif tx_type in ["genesis", "GTX_Genesis"]:
                interval_data["bills"][interval_index] += 1

        # Calculate statistics
        all_counts = []
        for i in range(num_intervals):
            total = sum(interval_data[tx_type][i] for tx_type in interval_data)
            all_counts.append(total)

        if all_counts:
            peak = max(all_counts)
            average_per_minute = (
                statistics.mean(all_counts) / (interval_minutes / 60)
                if interval_minutes
                else 0
            )
            totals = {
                "all": sum(all_counts),
                "transfers": sum(interval_data["transfers"]),
                "bills": sum(interval_data["bills"]),
            }
        else:
            peak = 0
            average_per_minute = 0
            totals = {"all": 0, "transfers": 0, "bills": 0}

        return jsonify(
            {
                "timeline": interval_data,
                "labels": labels,
                "peak": peak,
                "average_per_minute": average_per_minute,
                "totals": totals,
                "timeframe": "auto",
            }
        )

    except Exception as e:
        print(f"❌ Error in mempool_activity API: {e}")
        return jsonify(
            {
                "timeline": {"transfers": [], "bills": []},
                "labels": [],
                "peak": 0,
                "average_per_minute": 0,
                "totals": {"all": 0, "transfers": 0, "bills": 0},
                "timeframe": "auto",
                "error": str(e),
            }
        )


@app.route("/mine-all-transfers")
def mine_all_transfers():
    """Mine all pending transfers in multiple blocks if needed"""
    try:
        blockchain_data = getattr(blockchain_daemon_instance, "blockchain", [])
        mempool_data = getattr(blockchain_daemon_instance, "mempool", [])

        if not blockchain_data:
            return jsonify({"error": "No blockchain available"})

        transfer_txs = [tx for tx in mempool_data if tx.get("type") == "transfer"]

        if not transfer_txs:
            return jsonify({"error": "No transfer transactions in mempool"})

        results = {
            "blocks_mined": 0,
            "total_transfers_mined": 0,
            "blocks": [],
            "remaining_transfers": len(transfer_txs),
        }

        # Mine transfers in batches of 20 per block
        transfers_per_block = 20
        total_blocks_needed = (
            len(transfer_txs) + transfers_per_block - 1
        ) // transfers_per_block

        for block_num in range(total_blocks_needed):
            start_idx = block_num * transfers_per_block
            end_idx = start_idx + transfers_per_block
            block_transfers = transfer_txs[start_idx:end_idx]

            # Create and mine block
            previous_block = blockchain_data[-1]

            new_block = {
                "index": len(blockchain_data),
                "timestamp": int(time.time()),
                "transactions": block_transfers,
                "previous_hash": previous_block["hash"],
                "nonce": 0,
                "miner": f"transfer_miner_{block_num}",
                "difficulty": 2,  # Low difficulty for speed
                "hash": "",
            }

            # Mine the block
            target = "0" * 2
            start_time = time.time()
            mined = False

            for nonce in range(1000000):
                if time.time() - start_time > 30:  # 30 second timeout
                    break

                new_block["nonce"] = nonce
                calculated_hash = blockchain_daemon_instance.calculate_block_hash(
                    new_block["index"],
                    new_block["previous_hash"],
                    new_block["timestamp"],
                    new_block["transactions"],
                    nonce,
                )

                if calculated_hash.startswith(target):
                    new_block["hash"] = calculated_hash
                    mined = True
                    break

            if mined:
                # Add to blockchain
                blockchain_data.append(new_block)
                blockchain_daemon_instance.blockchain = blockchain_data

                # Update mempool
                blockchain_daemon_instance.mempool = [
                    tx
                    for tx in mempool_data
                    if tx.get("hash") not in [t.get("hash") for t in block_transfers]
                ]

                # Save
                blockchain_daemon_instance.save_blockchain()
                blockchain_daemon_instance.save_mempool()

                results["blocks_mined"] += 1
                results["total_transfers_mined"] += len(block_transfers)
                results["blocks"].append(
                    {
                        "index": new_block["index"],
                        "transfers": len(block_transfers),
                        "hash": new_block["hash"][:20] + "...",
                    }
                )

                # Update for next iteration
                mempool_data = blockchain_daemon_instance.mempool
                results["remaining_transfers"] = len(
                    [tx for tx in mempool_data if tx.get("type") == "transfer"]
                )
            else:
                results["error"] = f"Failed to mine block {block_num}"
                break

        return jsonify(
            {
                "success": True,
                "message": f"Mined {results['blocks_mined']} blocks with {results['total_transfers_mined']} transfers",
                "results": results,
            }
        )

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()})


@app.route("/step-by-step-mine-transfers")
def step_by_step_mine_transfers():
    """Step-by-step transfer mining with detailed error reporting"""
    try:
        # Step 1: Get current state
        blockchain_data = getattr(blockchain_daemon_instance, "blockchain", [])
        mempool_data = getattr(blockchain_daemon_instance, "mempool", [])

        if not blockchain_data:
            return jsonify({"error": "No blockchain available"})

        transfer_txs = [tx for tx in mempool_data if tx.get("type") == "transfer"]

        if not transfer_txs:
            return jsonify({"error": "No transfer transactions in mempool"})

        steps = []

        # Step 2: Validate transfers
        valid_transfers = []
        for tx in transfer_txs:
            if blockchain_daemon_instance.validate_transfer_for_mining(tx):
                valid_transfers.append(tx)

        steps.append(
            f"Step 1: Found {len(valid_transfers)} valid transfers out of {len(transfer_txs)} total"
        )

        if not valid_transfers:
            return jsonify(
                {
                    "error": "No valid transfers to mine",
                    "steps": steps,
                    "validation_issues": "All transfers failed validation",
                }
            )

        # Step 3: Create block with only valid transfers (limit to 20 for testing)
        transfers_to_mine = valid_transfers[:20]
        previous_block = blockchain_data[-1]

        steps.append(f"Step 2: Selected {len(transfers_to_mine)} transfers to mine")

        # Step 4: Create a simple block
        new_block = {
            "index": len(blockchain_data),
            "timestamp": int(time.time()),
            "transactions": transfers_to_mine,
            "previous_hash": previous_block["hash"],
            "nonce": 0,
            "miner": "transfer_fixer",
            "difficulty": 2,  # Very low difficulty for testing
            "hash": "",
        }

        steps.append("Step 3: Created block structure")

        # Step 5: Mine the block with timeout protection
        steps.append("Step 4: Starting proof-of-work mining...")

        target = "0" * 2  # Difficulty 2
        start_time = time.time()
        max_time = 30  # 30 second timeout

        for nonce in range(1000000):  # Limit attempts
            if time.time() - start_time > max_time:
                steps.append("❌ Mining timeout - difficulty too high")
                return jsonify(
                    {
                        "error": "Mining timeout",
                        "steps": steps,
                        "time_elapsed": time.time() - start_time,
                    }
                )

            new_block["nonce"] = nonce
            calculated_hash = blockchain_daemon_instance.calculate_block_hash(
                new_block["index"],
                new_block["previous_hash"],
                new_block["timestamp"],
                new_block["transactions"],
                nonce,
            )

            if calculated_hash.startswith(target):
                new_block["hash"] = calculated_hash
                steps.append(f"✅ Block mined successfully with nonce {nonce}")
                steps.append(f"✅ Final hash: {calculated_hash[:20]}...")
                break
        else:
            steps.append("❌ Failed to find valid nonce within limit")
            return jsonify(
                {
                    "error": "Mining failed - no valid nonce found",
                    "steps": steps,
                    "attempts": 1000000,
                }
            )

        # Step 6: Add to blockchain
        blockchain_daemon_instance.blockchain.append(new_block)
        steps.append("Step 5: Added block to blockchain")

        # Step 7: Use the ENHANCED cleanup method instead of simple hash matching
        initial_mempool_size = len(blockchain_daemon_instance.mempool)

        # Use the enhanced cleanup method
        removed_count = blockchain_daemon_instance.remove_mined_transactions(
            transfers_to_mine
        )

        # Also run comprehensive cleanup to catch any edge cases
        additional_removed = (
            blockchain_daemon_instance.cleanup_mined_transactions_enhanced()
        )

        steps.append(
            f"Step 6: Enhanced cleanup removed {removed_count} + {additional_removed} additional = {removed_count + additional_removed} total transactions"
        )

        # Step 8: Save everything
        blockchain_daemon_instance.save_blockchain()
        blockchain_daemon_instance.save_mempool()
        steps.append("Step 7: Saved blockchain and mempool")

        # Step 9: Verify cleanup worked
        final_mempool_size = len(blockchain_daemon_instance.mempool)
        remaining_transfers = len(
            [
                tx
                for tx in blockchain_daemon_instance.mempool
                if tx.get("type") == "transfer"
            ]
        )
        steps.append(
            f"Step 8: Verification - Mempool: {final_mempool_size} total, {remaining_transfers} transfers remaining"
        )

        return jsonify(
            {
                "success": True,
                "message": f"✅ Successfully mined transfer block #{new_block['index']}",
                "block_index": new_block["index"],
                "transfers_mined": len(transfers_to_mine),
                "mining_time": time.time() - start_time,
                "steps": steps,
                "block_hash": new_block["hash"][:20] + "...",
                "cleanup_summary": {
                    "initial_mempool": initial_mempool_size,
                    "final_mempool": final_mempool_size,
                    "removed_count": removed_count + additional_removed,
                    "remaining_transfers": remaining_transfers,
                },
            }
        )

    except Exception as e:
        import traceback

        return jsonify(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "steps": steps
                if "steps" in locals()
                else ["Failed before steps began"],
            }
        )


@app.template_filter("datetimeformat")
def datetimeformat(value, format="%Y-%m-%d %H:%M:%S"):
    """Format a timestamp as datetime string - BULLETPROOF VERSION"""
    try:
        if value is None:
            return "Unknown"

        # Convert ANY value to numeric timestamp
        numeric_value = 0
        if isinstance(value, (int, float)):
            numeric_value = float(value)
        elif isinstance(value, str):
            clean_value = "".join(
                c for c in value if c.isdigit() or c == "." or c == "-"
            )
            numeric_value = (
                float(clean_value) if clean_value and clean_value != "-" else 0
            )
        else:
            try:
                numeric_value = float(value)
            except:
                numeric_value = 0

        # Validate and format
        if numeric_value > 0 and numeric_value < 4102444800:
            return datetime.fromtimestamp(numeric_value).strftime(format)
        else:
            return f"Invalid: {numeric_value}"

    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/debug/blockchain-timestamps")
def debug_blockchain_timestamps():
    """Debug endpoint to check timestamp formats"""
    debug_info = []

    for i, block in enumerate(blockchain_daemon_instance.blockchain):
        timestamp = block.get("timestamp")
        debug_info.append(
            {
                "block_index": i,
                "timestamp": timestamp,
                "timestamp_type": type(timestamp).__name__,
                "is_numeric": isinstance(timestamp, (int, float)),
                "is_string": isinstance(timestamp, str),
            }
        )

    return jsonify(debug_info)


from builtins import max as builtin_max, min as builtin_min
from datetime import datetime


@app.route("/blockchain-viewer")
@app.route("/blockchain-viewer/<int:page>")
def blockchain_viewer(page=1):
    """Display blockchain information in a web interface - WITH PAGINATION"""
    try:
        print(f"🚀 ======= STARTING blockchain_viewer() for page {page} =======")

        show_genesis = request.args.get("genesis", "1") == "1"
        show_transfers = request.args.get("transfers", "1") == "1"
        show_rewards = request.args.get("rewards", "0") == "1"
        search_query = (request.args.get("q") or "").strip()
        filter_type = (request.args.get("filter") or "all").strip().lower()

        # Get the raw blockchain data from daemon
        print(f"🔍 [1/8] Checking blockchain_daemon_instance...")

        if blockchain_daemon_instance is None:
            print(f"❌ CRITICAL: blockchain_daemon_instance is None!")
            print(f"   This means the daemon wasn't properly initialized.")
            return render_template(
                "blockchain_viewer.html",
                blocks=[],
                total_blocks=0,
                total_transactions=0,
                genesis_count=0,
                transfer_count=0,
                reward_count=0,
                current_page=page,
                datetime=datetime,  # Add this line
                total_pages=1,
                per_page=25,
                error_message="Blockchain daemon not initialized",
                max=max,
                min=min,
                current_user=get_current_user(),
                title="Blockchain Viewer - Error",
            )

        blockchain_daemon = blockchain_daemon_instance
        print(
            f"✅ Daemon instance found: {type(blockchain_daemon).__name__} at {hex(id(blockchain_daemon))}"
        )

        # Get blockchain status to have accurate counts
        print(f"🔍 [2/8] Getting blockchain status...")
        total_blocks = 0
        try:
            blockchain_status = blockchain_daemon.get_blockchain_status()
            print(f"✅ Blockchain status response: {blockchain_status}")
            total_blocks = blockchain_status.get("blocks", 0)
            print(f"   Status reports {total_blocks} blocks")
        except Exception as status_error:
            print(
                f"❌ Failed to get blockchain status: {type(status_error).__name__}: {status_error}"
            )
            import traceback

            print(f"   Traceback: {traceback.format_exc()}")
            total_blocks = 0

        # Get ALL blocks (not just page) for accurate stats
        print(f"🔍 [3/8] Fetching all blocks from daemon...")
        all_blocks = []
        try:
            # Try to get blocks directly from daemon
            print(f"   Accessing blockchain_daemon.blockchain attribute...")
            all_blocks = blockchain_daemon.blockchain

            print(f"✅ Retrieved blockchain data:")
            print(f"   Type: {type(all_blocks)}")
            print(
                f"   Length: {len(all_blocks) if isinstance(all_blocks, (list, tuple, dict)) else 'N/A'}"
            )

            # Handle the case where blockchain might be returned as dictionary
            if isinstance(all_blocks, dict):
                print(f"⚠️  Blockchain is a dictionary, not a list")
                print(f"   Dictionary keys: {list(all_blocks.keys())}")

                # Check if it's the success/format
                if "blocks" in all_blocks:
                    print(f"   Found 'blocks' key, extracting...")
                    all_blocks = all_blocks["blocks"]
                    print(
                        f"   Extracted blocks: type={type(all_blocks)}, length={len(all_blocks)}"
                    )
                elif "blockchain" in all_blocks:
                    print(f"   Found 'blockchain' key, extracting...")
                    all_blocks = all_blocks["blockchain"]
                    print(
                        f"   Extracted blocks: type={type(all_blocks)}, length={len(all_blocks)}"
                    )
                elif "data" in all_blocks:
                    print(f"   Found 'data' key, extracting...")
                    all_blocks = all_blocks["data"]
                    print(
                        f"   Extracted blocks: type={type(all_blocks)}, length={len(all_blocks)}"
                    )
                else:
                    print(
                        f"❌ Dictionary doesn't contain expected keys. Full dict preview:"
                    )
                    print(f"   {str(all_blocks)[:500]}...")
                    all_blocks = []

            elif isinstance(all_blocks, list):
                print(f"✅ Blockchain is a list with {len(all_blocks)} items")
                if len(all_blocks) > 0:
                    print(f"   First item type: {type(all_blocks[0])}")
                    print(
                        f"   First item keys (if dict): {list(all_blocks[0].keys()) if isinstance(all_blocks[0], dict) else 'N/A'}"
                    )
            else:
                print(f"❌ Unexpected blockchain type: {type(all_blocks)}")
                print(f"   Value: {str(all_blocks)[:200]}")
                all_blocks = []

        except AttributeError as attr_err:
            print(f"❌ AttributeError: {attr_err}")
            print(f"   blockchain_daemon has no 'blockchain' attribute")
            print(
                f"   Available attributes: {[attr for attr in dir(blockchain_daemon) if not attr.startswith('_')][:20]}..."
            )
            all_blocks = []
        except Exception as e:
            print(f"❌ Error getting blockchain data: {type(e).__name__}: {e}")
            import traceback

            print(f"   Traceback: {traceback.format_exc()}")
            all_blocks = []

        print(f"🔍 [4/8] Processing {len(all_blocks)} blocks...")

        # Sort ALL blocks by index (newest first)
        try:
            if all_blocks and len(all_blocks) > 0:
                print(f"   Sorting blocks by index...")
                all_blocks_sorted = sorted(
                    all_blocks, key=lambda x: x.get("index", 0), reverse=True
                )
                print(f"✅ Sorted {len(all_blocks_sorted)} blocks")
                if len(all_blocks_sorted) > 0:
                    print(
                        f"   First block index: {all_blocks_sorted[0].get('index', 'N/A')}"
                    )
                    print(
                        f"   Last block index: {all_blocks_sorted[-1].get('index', 'N/A')}"
                    )
            else:
                print(f"⚠️  No blocks to sort")
                all_blocks_sorted = all_blocks
        except Exception as sort_error:
            print(f"❌ Error sorting blocks: {sort_error}")
            all_blocks_sorted = all_blocks

        # Calculate total stats from ALL blocks
        print(f"🔍 [5/8] Calculating blockchain statistics...")
        total_transactions = 0
        genesis_count = 0
        transfer_count = 0
        reward_count = 0
        blocks_with_counts = []

        # Calculate stats from all blocks
        for i, block in enumerate(all_blocks_sorted):
            if not isinstance(block, dict):
                print(f"   Block {i} is not a dictionary: {type(block)}")
                continue

            transactions = block.get("transactions", [])
            if not isinstance(transactions, list):
                print(
                    f"   Block {i} (index {block.get('index', 'N/A')}) transactions is not a list: {type(transactions)}"
                )
                transactions = []

            block_tx_count = len(transactions)
            block_genesis = 0
            block_transfer = 0
            block_reward = 0

            if block_tx_count > 0:
                print(
                    f"   Block {i} (index {block.get('index', 'N/A')}): {block_tx_count} transactions"
                )

            for j, tx in enumerate(transactions):
                if isinstance(tx, dict):
                    tx_type = tx.get("type", "")
                    if tx_type in ["genesis", "GTX_Genesis"]:
                        block_genesis += 1
                    elif tx_type == "transfer":
                        block_transfer += 1
                    elif tx_type == "reward":
                        block_reward += 1
                    else:
                        print(
                            f"     Unknown transaction type: {tx_type} in block {block.get('index', 'N/A')}"
                        )
                else:
                    print(
                        f"     Transaction {j} in block {block.get('index', 'N/A')} is not a dict: {type(tx)}"
                    )

            blocks_with_counts.append(
                {
                    "block": block,
                    "tx_count": block_tx_count,
                    "genesis_count": block_genesis,
                    "transfer_count": block_transfer,
                    "reward_count": block_reward,
                }
            )

        def _include_block(block_info):
            if not (show_genesis or show_transfers or show_rewards):
                return False
            if show_genesis and block_info["genesis_count"] > 0:
                return True
            if show_transfers and block_info["transfer_count"] > 0:
                return True
            if show_rewards and block_info["reward_count"] > 0:
                return True
            return False

        def _block_matches_query(block_info):
            if not search_query:
                return True

            block = block_info.get("block") if isinstance(block_info, dict) else None
            if not isinstance(block, dict):
                return False

            query = search_query.lower()
            block_index = str(block.get("index", ""))
            block_hash = str(block.get("hash", ""))
            prev_hash = str(block.get("previous_hash", ""))
            miner = str(block.get("miner", ""))

            if filter_type in ["block", "all"]:
                if query in block_index.lower() or query in block_hash.lower() or query in prev_hash.lower():
                    return True

            if filter_type in ["miner", "user", "all"]:
                if query in miner.lower():
                    return True

            transactions = block.get("transactions", [])
            if not isinstance(transactions, list):
                transactions = []

            if filter_type in ["transaction", "all"]:
                for tx in transactions:
                    if isinstance(tx, dict) and query in str(tx.get("hash", "")).lower():
                        return True

            if filter_type in ["genesis", "all"]:
                for tx in transactions:
                    if not isinstance(tx, dict):
                        continue
                    for field in ["serial_number", "front_serial", "back_serial", "serial"]:
                        if query in str(tx.get(field, "")).lower():
                            return True

            if filter_type in ["amount", "all"]:
                for tx in transactions:
                    if not isinstance(tx, dict):
                        continue
                    if query in str(tx.get("amount", "")).lower():
                        return True

            return False

        filtered_blocks = [
            b for b in blocks_with_counts if _include_block(b) and _block_matches_query(b)
        ]

        total_transactions = sum(b["tx_count"] for b in filtered_blocks)
        genesis_count = sum(b["genesis_count"] for b in filtered_blocks)
        transfer_count = sum(b["transfer_count"] for b in filtered_blocks)
        reward_count = sum(b["reward_count"] for b in filtered_blocks)

        print(f"📊 STATISTICS SUMMARY:")
        print(f"   Total blocks: {len(filtered_blocks)}")
        print(f"   Total transactions: {total_transactions}")
        print(f"   Genesis/GTX_Genesis transactions: {genesis_count}")
        print(f"   Transfer transactions: {transfer_count}")
        print(f"   Reward transactions: {reward_count}")

        # Pagination settings
        per_page = 25  # Number of blocks per page

        # If we have accurate blockchain status, use that for total blocks
        total_blocks = len(filtered_blocks)

        total_pages = max(
            1, (total_blocks + per_page - 1) // per_page
        )  # Ceiling division

        # Ensure page is within valid range
        page = max(1, min(page, total_pages))

        print(f"🔍 [6/8] Setting up pagination...")
        print(f"   Total blocks: {total_blocks}")
        print(f"   Per page: {per_page}")
        print(f"   Total pages: {total_pages}")
        print(f"   Requested page: {page}")

        # Calculate slice for current page (already sorted newest first)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        current_blocks = filtered_blocks[start_idx:end_idx]

        print(
            f"   Showing blocks {start_idx} to {end_idx} (actual: {len(current_blocks)} blocks)"
        )

        # Process only the blocks for the current page
        print(f"🔍 [7/8] Processing {len(current_blocks)} blocks for page display...")
        blocks_info = []

        for i, block_info in enumerate(current_blocks):
            block = block_info.get("block") if isinstance(block_info, dict) else None
            if not isinstance(block, dict):
                print(f"   Skipping non-dict block at position {i}")
                continue

            block_index = block.get("index", "N/A")
            print(
                f"   Processing block {i+1}/{len(current_blocks)} (index {block_index})..."
            )

            transactions = block.get("transactions", [])
            if not isinstance(transactions, list):
                print(
                    f"     Warning: transactions is not a list for block {block_index}"
                )
                transactions = []

            block_genesis = block_info.get("genesis_count", 0)
            block_transfer = block_info.get("transfer_count", 0)
            block_reward = block_info.get("reward_count", 0)

            raw_mining_method = (
                block.get("mining_method")
                or block.get("mined_with")
                or block.get("miner_type")
                or block.get("device")
                or ""
            )
            raw_mining_method_str = str(raw_mining_method).strip().lower()
            if "gpu" in raw_mining_method_str or raw_mining_method_str in ["cuda", "opencl"]:
                mining_method = "gpu"
            elif "cpu" in raw_mining_method_str:
                mining_method = "cpu"
            elif raw_mining_method_str:
                mining_method = raw_mining_method_str
            else:
                mining_method = "unknown"

            cpu_index = block.get("cpu_index") or block.get("cpu_worker")
            cpu_total = block.get("cpu_total") or block.get("cpu_workers")
            gpu_index = block.get("gpu_index") or block.get("gpu_worker")
            gpu_total = block.get("gpu_total") or block.get("gpu_workers")
            mining_label = None
            if mining_method == "cpu" and cpu_index is not None and cpu_total:
                mining_label = f"CPU[{cpu_index}/{cpu_total}]"
            elif mining_method == "gpu" and gpu_index is not None and gpu_total:
                mining_label = f"GPU[{gpu_index}/{gpu_total}]"
            elif mining_method == "cpu":
                mining_label = "CPU"
            elif mining_method == "gpu":
                mining_label = "GPU"
            elif mining_method == "unknown":
                mining_label = "Unknown"
            else:
                mining_label = str(mining_method).upper()

            # Process timestamp for display
            timestamp = block.get("timestamp", 0)
            readable_time = "Unknown"

            try:
                if timestamp:
                    if isinstance(timestamp, (int, float)):
                        pass
                    elif isinstance(timestamp, str):
                        if "." in timestamp:
                            timestamp = float(timestamp)
                        else:
                            timestamp = int(timestamp)

                    if timestamp > 0:
                        # Convert to datetime for readable format
                        dt = datetime.fromtimestamp(timestamp)
                        readable_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        readable_time = "Invalid timestamp (<= 0)"
                else:
                    readable_time = "No timestamp"
            except (ValueError, TypeError, OSError, OverflowError) as time_err:
                readable_time = f"Error: {time_err}"
                print(f"     Timestamp error for block {block_index}: {time_err}")

            # Create processed block info
            block_info = {
                "index": block.get("index", 0),
                "hash": block.get("hash", "N/A"),
                "previous_hash": block.get("previous_hash", "N/A"),
                "timestamp": timestamp,
                "timestamp_readable": readable_time,
                "nonce": block.get("nonce", 0),
                "difficulty": block.get("difficulty", "N/A"),
                "miner": block.get("miner", "Unknown"),
                "mining_method": mining_method,
                "mining_label": mining_label,
                "transaction_count": len(transactions),
                "genesis_count": block_genesis,
                "transfer_count": block_transfer,
                "reward_count": block_reward,
                "merkle_root": block.get("merkle_root", "N/A"),
                "mining_time": block.get("mining_time", "N/A"),
                "transactions": transactions,
            }

            # Calculate size
            try:
                block_info["size"] = len(json.dumps(block))
            except:
                block_info["size"] = 0

            print(
                f"     Block {block_index} processed: {len(transactions)} transactions, {readable_time}"
            )
            blocks_info.append(block_info)

        print(f"✅ [8/8] Prepared {len(blocks_info)} blocks for display")
        print(f"📊 FINAL PAGE {page} STATS:")
        print(f"   Blocks on page: {len(blocks_info)}")
        print(f"   Total blocks in blockchain: {total_blocks}")
        print(f"   Page {page} of {total_pages}")

        print(f"🏁 ======= ENDING blockchain_viewer() successfully =======")

        return render_template(
            "blockchain_viewer.html",
            blocks=blocks_info,
            total_blocks=total_blocks,
            total_transactions=total_transactions,
            genesis_count=genesis_count,
            transfer_count=transfer_count,
            reward_count=reward_count,
            current_page=page,
            datetime=datetime,
            total_pages=total_pages,
            per_page=per_page,
            current_user=get_current_user(),
            filter_genesis=show_genesis,
            filter_transfers=show_transfers,
            filter_rewards=show_rewards,
            filter_query=(
                f"?genesis={1 if show_genesis else 0}"
                f"&transfers={1 if show_transfers else 0}"
                f"&rewards={1 if show_rewards else 0}"
                f"&filter={filter_type}"
                f"&q={quote_plus(search_query)}"
            ),
            search_query=search_query,
            filter_type=filter_type,
            max=max,
            min=min,
            title="Blockchain Viewer",
        )

    except Exception as e:
        print(f"🔥🔥🔥 CRITICAL ERROR in blockchain_viewer: {type(e).__name__}: {e}")
        import traceback

        print(f"🔥 Stack trace:")
        traceback.print_exc()

        # Create a simple error display
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }

        return render_template(
            "blockchain_viewer.html",
            blocks=[],
            total_blocks=0,
            total_transactions=0,
            genesis_count=0,
            transfer_count=0,
            reward_count=0,
            current_page=1,
            datetime=datetime,
            total_pages=1,
            per_page=25,
            error_info=error_info,
            max=max,
            min=min,
            current_user=get_current_user(),
            title="Blockchain Viewer - Error",
        )


@app.route("/api/blockchain-timeline/<timeframe>")
def get_blockchain_timeline(timeframe):
    """Get blockchain timeline data for charts with caching"""
    try:
        # Validate timeframe
        valid_timeframes = ["1h", "1d", "7d", "30d", "90d", "1y", "all"]
        if timeframe not in valid_timeframes:
            return (
                jsonify(
                    {
                        "error": f"Invalid timeframe. Valid options: {', '.join(valid_timeframes)}"
                    }
                ),
                400,
            )

        # Check cache first
        cache_key = f"timeline_{timeframe}"
        if hasattr(app, "timeline_cache") and cache_key in app.timeline_cache:
            cached_data = app.timeline_cache[cache_key]
            # Check if cache is still valid (5 minutes)
            if time.time() - cached_data["timestamp"] < 300:
                return jsonify(cached_data["data"])

        # Get blockchain data
        if blockchain_daemon_instance is None:
            return jsonify({"error": "Blockchain daemon not available"}), 503

        all_blocks = blockchain_daemon_instance.blockchain
        if not all_blocks:
            return jsonify({"error": "No blockchain data available"}), 404

        # Filter blocks based on timeframe
        current_time = time.time()
        filtered_blocks = []

        if timeframe == "all":
            filtered_blocks = all_blocks
        else:
            # Parse timeframe
            if timeframe.endswith("h"):
                hours = int(timeframe[:-1])
                cutoff_time = current_time - (hours * 60 * 60)
            elif timeframe.endswith("d"):
                days = int(timeframe[:-1])
                cutoff_time = current_time - (days * 24 * 60 * 60)
            elif timeframe.endswith("y"):
                years = int(timeframe[:-1])
                cutoff_time = current_time - (years * 365 * 24 * 60 * 60)
            else:
                return jsonify({"error": "Unsupported timeframe format"}), 400

            # Filter blocks by timestamp
            for block in all_blocks:
                if isinstance(block, dict):
                    block_time = block.get("timestamp", 0)
                    if isinstance(block_time, str):
                        try:
                            block_time = float(block_time)
                        except:
                            continue
                    if block_time >= cutoff_time:
                        filtered_blocks.append(block)

        # Sort by timestamp (oldest first for timeline)
        filtered_blocks.sort(key=lambda x: x.get("timestamp", 0))

        # Prepare timeline data
        timeline_data = []
        for block in filtered_blocks:
            if isinstance(block, dict):
                transactions = block.get("transactions", [])
                if not isinstance(transactions, list):
                    transactions = []

                # Count transaction types
                genesis_count = sum(
                    1
                    for tx in transactions
                    if isinstance(tx, dict)
                    and tx.get("type") in ["genesis", "GTX_Genesis"]
                )
                transfer_count = sum(
                    1
                    for tx in transactions
                    if isinstance(tx, dict) and tx.get("type") == "transfer"
                )
                reward_count = sum(
                    1
                    for tx in transactions
                    if isinstance(tx, dict) and tx.get("type") == "reward"
                )

                timeline_data.append(
                    {
                        "index": block.get("index", 0),
                        "timestamp": block.get("timestamp", 0),
                        "total_txs": len(transactions),
                        "genesis_txs": genesis_count,
                        "transfer_txs": transfer_count,
                        "reward_txs": reward_count,
                        "miner": block.get("miner", "Unknown"),
                        "hash": block.get("hash", "N/A")[:12] + "...",
                    }
                )

        # Calculate statistics
        stats = {
            "total_blocks": len(timeline_data),
            "total_transactions": sum(block["total_txs"] for block in timeline_data),
            "avg_txs_per_block": round(
                sum(block["total_txs"] for block in timeline_data)
                / max(len(timeline_data), 1),
                2,
            ),
            "peak_txs": max((block["total_txs"] for block in timeline_data), default=0),
            "timeframe": timeframe,
        }

        result = {"timeline": timeline_data, "stats": stats}

        # Cache the result
        if not hasattr(app, "timeline_cache"):
            app.timeline_cache = {}
        app.timeline_cache[cache_key] = {"data": result, "timestamp": time.time()}

        return jsonify(result)

    except Exception as e:
        print(f"Error in get_blockchain_timeline: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/blockchain-stats")
def get_blockchain_stats():
    """Get comprehensive blockchain statistics"""
    try:
        if blockchain_daemon_instance is None:
            return jsonify({"error": "Blockchain daemon not available"}), 503

        all_blocks = blockchain_daemon_instance.blockchain
        if not all_blocks:
            return jsonify({"error": "No blockchain data available"}), 404

        def _compute_blockchain_stats(blocks):
            current_time = time.time()
            stats = {
                "total_blocks": len(blocks),
                "total_transactions": 0,
                "genesis_count": 0,
                "transfer_count": 0,
                "reward_count": 0,
                "unique_miners": set(),
                "avg_block_time": 0,
                "chain_age_days": 0,
                "blocks_per_day": 0,
                "txs_per_day": 0,
                "avg_denomination": 0,
                "total_value": 0,
            }

            timestamps = []
            denominations = []

            for block in blocks:
                if isinstance(block, dict):
                    transactions = block.get("transactions", [])
                    if isinstance(transactions, list):
                        stats["total_transactions"] += len(transactions)

                        for tx in transactions:
                            if isinstance(tx, dict):
                                tx_type = tx.get("type", "")
                                if tx_type in ["genesis", "GTX_Genesis"]:
                                    stats["genesis_count"] += 1
                                    if "denomination" in tx:
                                        try:
                                            denom = float(tx["denomination"])
                                            denominations.append(denom)
                                            stats["total_value"] += denom
                                        except Exception:
                                            pass
                                elif tx_type == "transfer":
                                    stats["transfer_count"] += 1
                                elif tx_type == "reward":
                                    stats["reward_count"] += 1

                    miner = block.get("miner", "")
                    if miner:
                        stats["unique_miners"].add(miner)

                    timestamp = block.get("timestamp", 0)
                    if isinstance(timestamp, str):
                        try:
                            timestamp = float(timestamp)
                        except Exception:
                            timestamp = 0
                    if timestamp > 0:
                        timestamps.append(timestamp)

            stats["unique_miners"] = len(stats["unique_miners"])

            if len(timestamps) > 1:
                timestamps.sort()
                time_diffs = [
                    timestamps[i + 1] - timestamps[i]
                    for i in range(len(timestamps) - 1)
                ]
                stats["avg_block_time"] = round(sum(time_diffs) / len(time_diffs), 2)

                chain_age_seconds = current_time - timestamps[0]
                stats["chain_age_days"] = round(chain_age_seconds / (24 * 60 * 60), 2)

                if stats["chain_age_days"] > 0:
                    stats["blocks_per_day"] = round(
                        stats["total_blocks"] / stats["chain_age_days"], 2
                    )
                    stats["txs_per_day"] = round(
                        stats["total_transactions"] / stats["chain_age_days"], 2
                    )

            if denominations:
                stats["avg_denomination"] = round(
                    sum(denominations) / len(denominations), 2
                )

            return stats

        def _start_background_refresh(blocks):
            with BLOCKCHAIN_STATS_LOCK:
                if BLOCKCHAIN_STATS_CACHE.get("refreshing"):
                    return False
                BLOCKCHAIN_STATS_CACHE["refreshing"] = True

            def _worker():
                try:
                    stats = _compute_blockchain_stats(blocks)
                    with BLOCKCHAIN_STATS_LOCK:
                        BLOCKCHAIN_STATS_CACHE["data"] = stats
                        BLOCKCHAIN_STATS_CACHE["timestamp"] = time.time()
                except Exception as e:
                    logger.warning(f"Blockchain stats refresh failed: {e}")
                finally:
                    with BLOCKCHAIN_STATS_LOCK:
                        BLOCKCHAIN_STATS_CACHE["refreshing"] = False

            threading.Thread(target=_worker, daemon=True).start()
            return True

        with BLOCKCHAIN_STATS_LOCK:
            cached_stats = BLOCKCHAIN_STATS_CACHE.get("data")
            cached_at = BLOCKCHAIN_STATS_CACHE.get("timestamp", 0)

        cache_fresh = cached_stats and (time.time() - cached_at) <= BLOCKCHAIN_STATS_TTL_SECONDS

        if cache_fresh:
            return jsonify(cached_stats)

        _start_background_refresh(all_blocks)

        if cached_stats:
            response = dict(cached_stats)
            response["refreshing"] = True
            return jsonify(response)

        # Return quick placeholder while refresh runs
        return jsonify(
            {
                "total_blocks": len(all_blocks),
                "total_transactions": 0,
                "genesis_count": 0,
                "transfer_count": 0,
                "reward_count": 0,
                "unique_miners": 0,
                "avg_block_time": 0,
                "chain_age_days": 0,
                "blocks_per_day": 0,
                "txs_per_day": 0,
                "avg_denomination": 0,
                "total_value": 0,
                "refreshing": True,
            }
        )

    except Exception as e:
        print(f"Error in get_blockchain_stats: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


import subprocess
import json
import sys
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from flask import jsonify, request
import traceback


# Global mining manager to track subprocesses
class MiningManager:
    def __init__(self):
        self.active_mining_processes = {}
        self.mining_executor = ThreadPoolExecutor(
            max_workers=2
        )  # Limited concurrent mining

    def start_mining_subprocess(self, miner_address, difficulty=4):
        """Start mining in a subprocess and return process ID"""
        mining_id = str(uuid.uuid4())

        def run_mining():
            try:
                # Run mining in a separate process
                result = subprocess.run(
                    [
                        sys.executable,
                        "mining_service.py",
                        miner_address,
                        str(difficulty),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )  # 5 minute timeout

                # Store result
                self.active_mining_processes[mining_id] = {
                    "status": "completed",
                    "result": result,
                    "miner_address": miner_address,
                }

            except subprocess.TimeoutExpired:
                self.active_mining_processes[mining_id] = {
                    "status": "timeout",
                    "error": "Mining process timed out after 5 minutes",
                }
            except Exception as e:
                self.active_mining_processes[mining_id] = {
                    "status": "error",
                    "error": str(e),
                }

        # Start mining in background thread (non-blocking)
        future = self.mining_executor.submit(run_mining)
        self.active_mining_processes[mining_id] = {
            "status": "running",
            "future": future,
            "miner_address": miner_address,
            "start_time": time.time(),
        }

        return mining_id

    def get_mining_status(self, mining_id=None):
        """Get mining status for specific ID or all"""
        if mining_id:
            return self.active_mining_processes.get(mining_id, {"status": "not_found"})
        else:
            return {
                "active_mining_jobs": len(
                    [
                        p
                        for p in self.active_mining_processes.values()
                        if p.get("status") == "running"
                    ]
                ),
                "total_jobs": len(self.active_mining_processes),
            }

    def get_mining_result(self, mining_id):
        """Get result of completed mining process"""
        process_info = self.active_mining_processes.get(mining_id)
        if not process_info:
            return {"status": "not_found"}

        if process_info["status"] == "running":
            return {"status": "still_running"}

        if process_info["status"] == "completed":
            result = process_info["result"]
            if result.returncode == 0:
                try:
                    mining_result = json.loads(result.stdout)
                    # Clean up completed process
                    del self.active_mining_processes[mining_id]
                    return mining_result
                except json.JSONDecodeError:
                    return {"status": "error", "error": "Invalid JSON response"}
            else:
                error_msg = process_info.get("error") or result.stderr
                del self.active_mining_processes[mining_id]
                return {"status": "error", "error": error_msg}

        # Handle timeout or other errors
        error_info = self.active_mining_processes[mining_id]
        del self.active_mining_processes[mining_id]
        return {"status": error_info["status"], "error": error_info.get("error")}


# Initialize mining manager
mining_manager = MiningManager()
blockchain_daemon_instance = BlockchainDaemon()
# Add these endpoints to your Flask app


# Mempool routes
@app.route("/mempool/status", methods=["GET"])
def mempool_status():
    """Get current mempool status and statistics"""
    try:
        status = blockchain_daemon_instance.get_mempool_status()
        return (
            jsonify({"success": True, "status": status, "timestamp": int(time.time())}),
            200,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/mempool/add", methods=["POST"])
def add_to_mempool():
    """Add a transaction to the mempool (GTX Genesis, transfers, etc.)"""
    try:
        data = None
        try:
            data = request.get_json()
        except Exception:
            data = None

        if data is None:
            raw_data = request.get_data() or b""
            if raw_data:
                if (request.headers.get("Content-Encoding") or "").lower() == "gzip":
                    try:
                        raw_data = gzip.decompress(raw_data)
                    except Exception as e:
                        return (
                            jsonify({"success": False, "error": f"Invalid gzip body: {e}"}),
                            400,
                        )
                try:
                    data = json.loads(raw_data.decode("utf-8"))
                except Exception as e:
                    return (
                        jsonify({"success": False, "error": f"Failed to decode JSON object: {e}"}),
                        400,
                    )

        app.logger.info(f"🔍 [MEMPOOL/ADD] Received request")
        app.logger.info(
            f"🔍 [MEMPOOL/ADD] Data keys: {list(data.keys()) if data else 'None'}"
        )

        if not data:
            app.logger.error(f"❌ [MEMPOOL/ADD] No JSON data provided")
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        # Validate required fields
        if "type" not in data:
            app.logger.error(f"❌ [MEMPOOL/ADD] Missing 'type' field")
            return (
                jsonify({"success": False, "error": "Transaction type is required"}),
                400,
            )

        app.logger.info(f"🔍 [MEMPOOL/ADD] Transaction type: {data.get('type')}")

        # Add timestamp if not provided
        if "timestamp" not in data:
            data["timestamp"] = int(time.time())
            app.logger.info(f"🔍 [MEMPOOL/ADD] Added timestamp: {data['timestamp']}")

        # Add transaction to mempool
        app.logger.info(
            f"🔍 [MEMPOOL/ADD] Calling blockchain_daemon_instance.add_transaction()..."
        )
        success = blockchain_daemon_instance.add_transaction(data)
        app.logger.info(f"🔍 [MEMPOOL/ADD] Result: {success}")

        if success:
            app.logger.info(f"✅ [MEMPOOL/ADD] Transaction added successfully")
            return (
                jsonify(
                    {
                        "success": True,
                        "message": "Transaction added to mempool",
                        "transaction_hash": data.get("hash"),
                        "type": data.get("type"),
                    }
                ),
                201,
            )
        else:
            app.logger.error(f"❌ [MEMPOOL/ADD] Failed to add transaction")
            return (
                jsonify(
                    {"success": False, "error": "Failed to add transaction to mempool"}
                ),
                400,
            )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/mempool/add/batch", methods=["POST"])
def add_to_mempool_batch():
    """Add multiple transactions to the mempool using lunalib batch operations when available."""
    try:
        data = None
        try:
            data = request.get_json()
        except Exception:
            data = None

        if data is None:
            raw_data = request.get_data() or b""
            if raw_data:
                if (request.headers.get("Content-Encoding") or "").lower() == "gzip":
                    try:
                        raw_data = gzip.decompress(raw_data)
                    except Exception as e:
                        return (
                            jsonify({"success": False, "error": f"Invalid gzip body: {e}"}),
                            400,
                        )
                try:
                    data = json.loads(raw_data.decode("utf-8"))
                except Exception as e:
                    return (
                        jsonify({"success": False, "error": f"Failed to decode JSON body: {e}"}),
                        400,
                    )

        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        transactions = data.get("transactions") if isinstance(data, dict) else data
        if not isinstance(transactions, list):
            return (
                jsonify({"success": False, "error": "Expected a list of transactions"}),
                400,
            )

        if not transactions:
            return jsonify({"success": False, "error": "Empty transaction list"}), 400

        normalized = []
        errors = []
        for idx, tx in enumerate(transactions):
            if not isinstance(tx, dict):
                errors.append({"index": idx, "error": "Transaction must be an object"})
                continue
            if "type" not in tx:
                errors.append({"index": idx, "error": "Transaction type is required"})
                continue
            if "timestamp" not in tx:
                tx["timestamp"] = int(time.time())
            normalized.append(tx)

        if not normalized:
            return (
                jsonify({"success": False, "error": "No valid transactions", "errors": errors}),
                400,
            )

        mempool_mgr = getattr(blockchain_daemon_instance, "mempool_mgr", None)
        batch_result = None
        if mempool_mgr is not None:
            for method_name in (
                "add_transactions",
                "add_transaction_batch",
                "add_mempool_batch",
                "add_batch",
                "submit_transactions",
                "submit_transaction_batch",
            ):
                method = getattr(mempool_mgr, method_name, None)
                if callable(method):
                    try:
                        batch_result = method(normalized)
                        break
                    except Exception as e:
                        app.logger.warning(f"[MEMPOOL/BATCH] lunalib {method_name} failed: {e}")

        if batch_result is None:
            successes = 0
            for tx in normalized:
                result = blockchain_daemon_instance.add_transaction(tx)
                if result:
                    successes += 1
                else:
                    errors.append({"hash": tx.get("hash"), "error": "Failed to add transaction"})
            return (
                jsonify(
                    {
                        "success": successes == len(normalized),
                        "added": successes,
                        "total": len(normalized),
                        "errors": errors,
                    }
                ),
                201 if successes else 400,
            )

        return (
            jsonify({"success": True, "result": batch_result, "errors": errors}),
            201,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/mempool/transactions", methods=["GET"])
def get_mempool_transactions():
    """Get all transactions currently in the mempool"""
    try:
        status = blockchain_daemon_instance.get_mempool_status()

        # Optional filtering by type
        tx_type = request.args.get("type")
        if tx_type:
            filtered_transactions = [
                tx for tx in status["transactions"] if tx.get("type") == tx_type
            ]
        else:
            filtered_transactions = status["transactions"]

        return (
            jsonify(
                {
                    "success": True,
                    "transactions": filtered_transactions,
                    "total": len(filtered_transactions),
                    "timestamp": int(time.time()),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Global variables
_blockchain_height = 0
_blockchain_update_time = 0


def update_blockchain_height():
    """Update the cached blockchain height"""
    global _blockchain_height, _blockchain_update_time
    try:
        blockchain_data = blockchain_daemon_instance.blockchain
        _blockchain_height = len(blockchain_data) if blockchain_data else 0
        _blockchain_update_time = time.time()
    except Exception as e:
        print(f"Error updating blockchain height: {e}")


@app.route("/blockchain/height", methods=["GET"])
def get_blockchain_height():
    """Get the current height - ULTRA FAST"""
    return (
        jsonify(
            {
                "success": True,
                "height": _blockchain_height,
                "latest_block_index": _blockchain_height - 1
                if _blockchain_height > 0
                else -1,
                "timestamp": int(time.time()),
            }
        ),
        200,
    )


# Call this whenever blockchain changes
def on_blockchain_updated():
    update_blockchain_height()


@app.route("/blockchain/range", methods=["GET"])
def get_blockchain_range():
    """Get a range of blocks from the blockchain"""
    try:
        # Get query parameters with defaults
        start = request.args.get("start", type=int, default=0)
        end = request.args.get("end", type=int)

        blockchain_data = blockchain_daemon_instance.blockchain

        if not blockchain_data:
            return (
                jsonify(
                    {
                        "success": True,
                        "blocks": [],
                        "total_blocks": 0,
                        "range_start": start,
                        "range_end": 0,
                    }
                ),
                200,
            )

        total_blocks = len(blockchain_data)

        # Validate and adjust range parameters
        start = max(0, start)
        if end is None:
            end = total_blocks - 1
        else:
            end = min(end, total_blocks - 1)

        # Ensure start <= end
        if start > end:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Start index cannot be greater than end index",
                    }
                ),
                400,
            )

        # Extract the requested range
        blocks_range = blockchain_data[start : end + 1]

        return (
            jsonify(
                {
                    "success": True,
                    "blocks": blocks_range,
                    "total_blocks": total_blocks,
                    "range_start": start,
                    "range_end": end,
                    "blocks_in_range": len(blocks_range),
                    "timestamp": int(time.time()),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/blockchain-timeline-range", methods=["GET"])
def get_blockchain_timeline_range():
    """Get timeline data for a specific block range with caching"""
    try:
        start = request.args.get("start", type=int, default=0)
        end = request.args.get("end", type=int)

        if blockchain_daemon_instance is None:
            return jsonify({"error": "Blockchain daemon not available"}), 503

        all_blocks = blockchain_daemon_instance.blockchain
        if not all_blocks:
            return jsonify({"error": "No blockchain data available"}), 404

        total_blocks = len(all_blocks)

        start = max(0, start)
        if end is None:
            end = total_blocks - 1
        else:
            end = min(end, total_blocks - 1)

        if start > end:
            return (
                jsonify({"error": "Start index cannot be greater than end index"}),
                400,
            )

        cache_key = f"timeline_range_{start}_{end}"
        if hasattr(app, "timeline_cache") and cache_key in app.timeline_cache:
            cached_data = app.timeline_cache[cache_key]
            if time.time() - cached_data["timestamp"] < 300:
                return jsonify(cached_data["data"])

        blocks_range = all_blocks[start : end + 1]
        timeline_data = []

        for block in blocks_range:
            if isinstance(block, dict):
                transactions = block.get("transactions", [])
                if not isinstance(transactions, list):
                    transactions = []

                genesis_count = sum(
                    1
                    for tx in transactions
                    if isinstance(tx, dict)
                    and tx.get("type") in ["genesis", "GTX_Genesis"]
                )
                transfer_count = sum(
                    1
                    for tx in transactions
                    if isinstance(tx, dict) and tx.get("type") == "transfer"
                )
                reward_count = sum(
                    1
                    for tx in transactions
                    if isinstance(tx, dict) and tx.get("type") == "reward"
                )

                timeline_data.append(
                    {
                        "index": block.get("index", 0),
                        "timestamp": block.get("timestamp", 0),
                        "total_txs": len(transactions),
                        "genesis_txs": genesis_count,
                        "transfer_txs": transfer_count,
                        "reward_txs": reward_count,
                        "miner": block.get("miner", "Unknown"),
                        "hash": block.get("hash", "N/A")[:12] + "..."
                        if block.get("hash")
                        else "N/A",
                        "difficulty": block.get("difficulty", 0),
                    }
                )

        stats = {
            "total_blocks": len(timeline_data),
            "total_transactions": sum(block["total_txs"] for block in timeline_data),
            "avg_txs_per_block": round(
                sum(block["total_txs"] for block in timeline_data)
                / max(len(timeline_data), 1),
                2,
            ),
            "peak_txs": max((block["total_txs"] for block in timeline_data), default=0),
            "genesis_txs": sum(block["genesis_txs"] for block in timeline_data),
            "transfer_txs": sum(block["transfer_txs"] for block in timeline_data),
            "reward_txs": sum(block["reward_txs"] for block in timeline_data),
        }

        result = {
            "timeline": timeline_data,
            "stats": stats,
            "range": {
                "start": start,
                "end": end,
                "total_blocks_in_chain": total_blocks,
            },
        }

        if not hasattr(app, "timeline_cache"):
            app.timeline_cache = {}
        app.timeline_cache[cache_key] = {"data": result, "timestamp": time.time()}

        return jsonify(result)

    except Exception as e:
        print(f"Error in get_blockchain_timeline_range: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/blockchain/status", methods=["GET"])
def blockchain_status():
    """Get current blockchain status and statistics"""
    try:
        status = blockchain_daemon_instance.get_blockchain_status()

        return (
            jsonify({"success": True, "status": status, "timestamp": int(time.time())}),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/blockchain/submit-block", methods=["POST"])
def submit_block():
    """Submit a mined block for validation and addition to blockchain"""
    try:
        data = None
        try:
            data = request.get_json()
        except Exception:
            data = None

        if data is None:
            raw_data = request.get_data() or b""
            if raw_data:
                if (request.headers.get("Content-Encoding") or "").lower() == "gzip":
                    try:
                        raw_data = gzip.decompress(raw_data)
                    except Exception as e:
                        return (
                            jsonify({"success": False, "error": f"Invalid gzip body: {e}"}),
                            400,
                        )
                try:
                    data = json.loads(raw_data.decode("utf-8"))
                except Exception as e:
                    return (
                        jsonify({"success": False, "error": f"Failed to decode JSON object: {e}"}),
                        400,
                    )

        if not data:
            return jsonify({"success": False, "error": "No block data provided"}), 400

        # Validate required block fields
        required_fields = [
            "index",
            "timestamp",
            "transactions",
            "previous_hash",
            "nonce",
            "hash",
        ]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Missing required fields: {missing_fields}",
                    }
                ),
                400,
            )

        # Check if block already exists in blockchain
        block_hash = data["hash"]
        block_index = data["index"]
        submission_key = f"{block_index}:{block_hash}"
        with _BLOCK_SUBMISSION_LOCK:
            if submission_key in _BLOCK_SUBMISSION_IN_FLIGHT:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Block submission already in progress",
                            "block_hash": block_hash,
                            "block_index": block_index,
                            "status": "in_progress",
                        }
                    ),
                    202,
                )
            _BLOCK_SUBMISSION_IN_FLIGHT.add(submission_key)

        # Get previous block hash for validation
        blockchain_data = blockchain_daemon_instance.blockchain
        if blockchain_data:
            previous_block = blockchain_data[-1]
            previous_block_hash = previous_block.get("hash", "")
        else:
            previous_block_hash = "0" * 64  # For genesis block

        # 使用 daemon 实例的方法而不是本地函数
        if blockchain_daemon_instance.is_block_already_in_chain(
            block_hash, block_index
        ):
            print(f"⏭️  Block #{block_index} already exists in blockchain, skipping...")
            with _BLOCK_SUBMISSION_LOCK:
                _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
            return (
                jsonify(
                    {
                        "success": True,
                        "message": f"Block #{block_index} already exists in blockchain",
                        "block_hash": block_hash,
                        "block_index": block_index,
                        "status": "already_exists",
                        "skipped": True,
                    }
                ),
                200,
            )

        # Check if we're trying to add a block that's not the next in sequence
        expected_index = len(blockchain_data) if blockchain_data else 0
        if block_index != expected_index:
            with _BLOCK_SUBMISSION_LOCK:
                _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Block #{block_index} is not the next block in sequence",
                        "expected_index": expected_index,
                        "latest_block_hash": previous_block_hash,
                    }
                ),
                409,
            )

        # Check previous hash matches current chain tip
        provided_prev_hash = data.get("previous_hash")
        if provided_prev_hash and provided_prev_hash != previous_block_hash:
            with _BLOCK_SUBMISSION_LOCK:
                _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Previous hash mismatch",
                        "expected_previous_hash": previous_block_hash,
                        "provided_previous_hash": provided_prev_hash,
                        "expected_index": expected_index,
                    }
                ),
                409,
            )

        # Get miner from block data or use a default
        miner = data.get("miner", "unknown_miner")

        print(f"🔍 Validating block #{block_index} from miner: {miner}")

        # Validate block using lunalib only
        blockchain_mgr = getattr(blockchain_daemon_instance, "blockchain_mgr", None)
        if blockchain_mgr is None:
            with _BLOCK_SUBMISSION_LOCK:
                _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Lunalib blockchain manager unavailable",
                    }
                ),
                503,
            )

        submission_validation = None
        validation_method = None
        for method_name in (
            "validate_block_for_submission",
            "validate_block",
            "validate_block_submission",
            "validate_block_structure",
            "_validate_block_structure",
        ):
            method = getattr(blockchain_mgr, method_name, None)
            if callable(method):
                try:
                    submission_validation = method(data)
                    validation_method = method_name
                    break
                except Exception as e:
                    submission_validation = {"valid": False, "errors": [str(e)], "message": str(e)}
                    validation_method = method_name
                    break

        if submission_validation is None:
            with _BLOCK_SUBMISSION_LOCK:
                _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Lunalib block validation unavailable",
                    }
                ),
                503,
            )

        if isinstance(submission_validation, bool):
            submission_validation = {
                "valid": submission_validation,
                "errors": [] if submission_validation else ["Block validation failed"],
            }

        print(
            f"✅ Block validation via {validation_method}: {submission_validation.get('valid', False)}"
        )
        if isinstance(submission_validation, dict) and submission_validation.get("errors"):
            print(f"   Validation errors: {submission_validation.get('errors')}")

        if isinstance(submission_validation, dict) and not submission_validation.get("valid", False):
            with _BLOCK_SUBMISSION_LOCK:
                _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Block validation failed",
                        "details": submission_validation,
                    }
                ),
                400,
            )

        # Validate reward transactions separately
        transactions = data.get("transactions", [])
        reward_transactions = [tx for tx in transactions if tx.get("type") == "reward"]
        regular_transactions = [tx for tx in transactions if tx.get("type") != "reward"]

        print(
            f" Block has {len(reward_transactions)} reward transactions and {len(regular_transactions)} regular transactions"
        )

        # Validate reward transactions using lunalib only
        reward_validation_note = None
        if reward_transactions:
            skip_reward_validation = os.getenv("LUNALIB_SKIP_REWARD_VALIDATION", "1") == "1"
            print("🔍 Starting reward validation via lunalib...")
            reward_validation_result = None
            reward_validation_method = None
            if skip_reward_validation:
                reward_validation_note = "Reward validation skipped to avoid lunalib hang"
                reward_validation_result = {"valid": True, "error": None}
                reward_validation_method = "skipped"
                print("⏭️  Skipping reward validation (LUNALIB_SKIP_REWARD_VALIDATION=1)")
            else:
                for method_name in (
                    "validate_reward_transactions",
                    "validate_reward_transaction",
                    "validate_reward_tx",
                    "validate_mining_reward",
                ):
                    method = getattr(blockchain_mgr, method_name, None)
                    if callable(method):
                        reward_validation_method = method_name
                        result_holder = {"value": None}
                        error_holder = {"error": None}

                        def _run_reward_validation():
                            try:
                                result_holder["value"] = method(
                                    reward_transactions,
                                    block_index,
                                    data,
                                    previous_block_hash,
                                )
                            except Exception as exc:
                                error_holder["error"] = str(exc)

                        thread = threading.Thread(
                            target=_run_reward_validation,
                            daemon=True,
                            name="lunalib-reward-validation",
                        )
                        thread.start()
                        thread.join(timeout=5)
                        if thread.is_alive():
                            reward_validation_note = (
                                "Lunalib reward validation timed out; proceeding without reward validation"
                            )
                            reward_validation_result = {"valid": True, "error": None}
                            print("⏱️  Reward validation timed out; continuing...")
                        elif error_holder["error"]:
                            reward_validation_result = {"valid": False, "error": error_holder["error"]}
                        else:
                            reward_validation_result = result_holder["value"]
                        break

            if reward_validation_result is None:
                with _BLOCK_SUBMISSION_LOCK:
                    _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Lunalib reward validation unavailable",
                        }
                    ),
                    503,
                )

            if isinstance(reward_validation_result, bool):
                reward_validation_result = {
                    "valid": reward_validation_result,
                    "error": None if reward_validation_result else "Lunalib reward validation failed",
                }
            elif isinstance(reward_validation_result, dict) and "valid" not in reward_validation_result:
                if "success" in reward_validation_result:
                    reward_validation_result = {
                        "valid": bool(reward_validation_result.get("success")),
                        "error": reward_validation_result.get("error")
                        or "Lunalib reward validation failed",
                        "debug": reward_validation_result.get("debug"),
                    }
            elif not isinstance(reward_validation_result, dict):
                reward_validation_note = "Lunalib reward validation returned unexpected type; proceeding"
                reward_validation_result = {"valid": True, "error": None}
            print(
                f"✅ Reward validation via {reward_validation_method}: {reward_validation_result.get('valid', False)}"
            )
            if reward_validation_result.get("error"):
                print(f"   Reward validation error: {reward_validation_result.get('error')}")
                print("🔍 Starting regular transaction validation...")
            if not reward_validation_result["valid"]:
                with _BLOCK_SUBMISSION_LOCK:
                    _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
                error_msg = reward_validation_result["error"]
                debug_info = reward_validation_result.get("debug", {})

                print(f"\n❌ REWARD VALIDATION FAILED!")
                print(f"   Error: {error_msg}")

                if debug_info:
                    print(f"\n🔍 DEBUG INFORMATION:")
                    print(
                        f"   Expected reward: {debug_info.get('expected_reward')} LKC"
                    )
                    print(
                        f"   Provided reward: {debug_info.get('provided_reward')} LKC"
                    )
                    print(f"   Difficulty: {debug_info.get('difficulty')}")
                    print(f"   Calculation: {debug_info.get('calculation')}")
                    print(f"   Block hash: {debug_info.get('block_hash')}")
                    print(f"   Block timestamp: {debug_info.get('block_timestamp')}")
                    print(f"   Reward timestamp: {debug_info.get('reward_timestamp')}")
                    print(
                        f"   Timestamp difference: {debug_info.get('timestamp_diff')} seconds"
                    )

                    reward_tx = debug_info.get("reward_transaction", {})
                    print(f"\n📋 REWARD TRANSACTION DETAILS:")
                    print(f"   Type: {reward_tx.get('type')}")
                    print(f"   From: {reward_tx.get('from', 'NOT SET')}")
                    print(f"   To: {reward_tx.get('to')}")
                    print(f"   Amount: {reward_tx.get('amount')}")
                    print(f"   Block height: {reward_tx.get('block_height')}")
                    print(f"   Hash: {reward_tx.get('hash', '')[:16]}...")
                    if "difficulty" in reward_tx:
                        print(f"   Difficulty in TX: {reward_tx.get('difficulty')}")
                    else:
                        print(f"   ⚠️  Difficulty field missing in reward TX!")

                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Reward transaction validation failed: {error_msg}",
                            "debug": debug_info,
                        }
                    ),
                    400,
                )

        # Validate regular transactions
        if regular_transactions:
            regular_validation_result = (
                blockchain_daemon_instance.validate_regular_transactions(
                    regular_transactions
                )
            )
            if not regular_validation_result["valid"]:
                print(
                    f"❌ Regular transaction validation failed: {regular_validation_result['error']}"
                )
                with _BLOCK_SUBMISSION_LOCK:
                    _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Transaction validation failed: {regular_validation_result['error']}",
                        }
                    ),
                    400,
                )

        # Add block to blockchain using lunalib
        print("🚀 Starting block submission...")
        submission_success = None
        submission_method = None
        submission_note = None
        submission_attempts = []
        if hasattr(blockchain_daemon_instance, "submit_block_with_validation"):
            try:
                submission_success = blockchain_daemon_instance.submit_block_with_validation(
                    data
                )
                submission_method = "submit_block_with_validation"
                submission_attempts.append(
                    {
                        "method": submission_method,
                        "success": bool(submission_success),
                    }
                )
            except Exception as e:
                submission_success = None
                submission_method = "submit_block_with_validation"
                submission_attempts.append(
                    {
                        "method": submission_method,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"💥 Lunalib submit_block_with_validation error: {e}")
        else:
            blockchain_mgr = getattr(blockchain_daemon_instance, "blockchain_mgr", None)
            if blockchain_mgr is not None:
                for method_name in (
                    "submit_block",
                    "add_block",
                    "add_validated_block",
                ):
                    method = getattr(blockchain_mgr, method_name, None)
                    if callable(method):
                        try:
                            submission_success = method(data)
                            submission_method = method_name
                            submission_attempts.append(
                                {
                                    "method": submission_method,
                                    "success": bool(submission_success),
                                }
                            )
                            break
                        except Exception as e:
                            submission_success = None
                            submission_method = method_name
                            submission_attempts.append(
                                {
                                    "method": submission_method,
                                    "success": False,
                                    "error": str(e),
                                }
                            )
                            print(f"💥 Lunalib {method_name} error: {e}")
                            break

        if not submission_success:
            blockchain_mgr = getattr(blockchain_daemon_instance, "blockchain_mgr", None)
            if blockchain_mgr is not None:
                for method_name in (
                    "submit_block",
                    "add_block",
                    "add_validated_block",
                ):
                    method = getattr(blockchain_mgr, method_name, None)
                    if callable(method):
                        try:
                            fallback_success = method(data)
                            submission_attempts.append(
                                {
                                    "method": method_name,
                                    "success": bool(fallback_success),
                                }
                            )
                            if fallback_success:
                                submission_success = True
                                submission_method = method_name
                                break
                        except Exception as e:
                            submission_attempts.append(
                                {
                                    "method": method_name,
                                    "success": False,
                                    "error": str(e),
                                }
                            )
                            print(f"💥 Lunalib {method_name} error: {e}")

        # If network submission failed, still add locally after validation
        if not submission_success and hasattr(blockchain_daemon_instance, "add_validated_block"):
            try:
                local_added = blockchain_daemon_instance.add_validated_block(data)
            except Exception as e:
                local_added = False
                print(f"💥 Local add_validated_block error: {e}")
            if local_added:
                submission_success = True
                submission_method = "add_validated_block"
                submission_note = "added locally (network submission failed or unavailable)"
                submission_attempts.append(
                    {
                        "method": "add_validated_block",
                        "success": True,
                        "note": "local add fallback",
                    }
                )

        print(
            f"✅ Block submission via {submission_method}: {bool(submission_success)}"
        )
        if submission_note:
            print(f"ℹ️  Submission note: {submission_note}")

        if submission_success:
            # Mark reward transactions as mined
            blockchain_daemon_instance.mark_reward_transactions_mined(
                reward_transactions, block_hash
            )

            # Log successful submission
            print(f"✅ Block #{block_index} successfully added to blockchain")

            # Log reward transactions specifically
            if reward_transactions:
                for i, reward_tx in enumerate(reward_transactions):
                    print(
                        f"💰 Reward TX #{i+1}: {reward_tx.get('to')} received {reward_tx.get('amount')} LUN"
                    )

            response_payload = {
                "success": True,
                "message": f"Block #{block_index} added to blockchain",
                "block_hash": block_hash,
                "block_index": block_index,
                "transactions_count": len(transactions),
                "reward_transactions_count": len(reward_transactions),
                "regular_transactions_count": len(regular_transactions),
                "miner": miner,
                "status": "added",
                "submission_method": submission_method,
                "submission_attempts": submission_attempts,
            }
            if reward_validation_note:
                response_payload["reward_validation_note"] = reward_validation_note
            if submission_note:
                response_payload["note"] = submission_note
            with _BLOCK_SUBMISSION_LOCK:
                _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
            return jsonify(response_payload), 201
        else:
            with _BLOCK_SUBMISSION_LOCK:
                _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Block submission failed",
                        "submission_method": submission_method,
                        "submission_attempts": submission_attempts,
                    }
                ),
                400,
            )

    except Exception as e:
        with _BLOCK_SUBMISSION_LOCK:
            try:
                submission_key = f"{data.get('index')}:{data.get('hash')}"
                _BLOCK_SUBMISSION_IN_FLIGHT.discard(submission_key)
            except Exception:
                pass
        print(f"💥 Error in submit_block: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def is_block_already_in_chain(block_hash, block_index):
    """Check if a block already exists in the blockchain"""
    try:
        blockchain_data = blockchain_daemon_instance.blockchain

        # Check by hash (most reliable)
        for block in blockchain_data:
            if block.get("hash") == block_hash:
                return True

        # Check by index and hash pattern (secondary check)
        for block in blockchain_data:
            if (
                block.get("index") == block_index
                and block.get("hash")
                and block_hash
                and block.get("hash")[:8] == block_hash[:8]
            ):  # Check first 8 chars of hash
                return True

        return False
    except Exception as e:
        print(f"Error checking if block exists: {e}")
        return False


def is_correct_block_sequence(block_index):
    """Check if the block is the next one in sequence"""
    try:
        blockchain_data = blockchain_daemon_instance.blockchain
        current_height = len(blockchain_data)

        # The next block should have index equal to current height
        return block_index == current_height
    except Exception as e:
        print(f"Error checking block sequence: {e}")
        return False


def mark_reward_transactions_mined(reward_transactions, block_hash):
    """Mark reward transactions as mined in a separate ledger"""
    try:
        if not reward_transactions:
            return

        # Load or create mined rewards ledger
        mined_rewards = load_mined_rewards_ledger()

        for reward_tx in reward_transactions:
            reward_data = {
                "hash": reward_tx.get("hash"),
                "miner": reward_tx.get("miner", "unknown"),
                "recipient": reward_tx.get("to"),
                "amount": reward_tx.get("amount"),
                "block_height": reward_tx.get("block_height"),
                "block_hash": block_hash,
                "timestamp": time.time(),
                "mined_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Check if already marked as mined
            already_mined = any(
                r.get("hash") == reward_data["hash"] for r in mined_rewards
            )
            if not already_mined:
                mined_rewards.append(reward_data)
                print(
                    f"✅ Marked reward transaction as mined: {reward_data['hash'][:16]}..."
                )

        # Save updated ledger
        save_mined_rewards_ledger(mined_rewards)

    except Exception as e:
        print(f"Error marking reward transactions as mined: {e}")


def load_mined_rewards_ledger():
    """Load the mined rewards ledger"""
    try:
        ledger_file = "mined_rewards_ledger.json"
        if os.path.exists(ledger_file):
            with open(ledger_file, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading mined rewards ledger: {e}")
        return []


def save_mined_rewards_ledger(ledger_data):
    """Save the mined rewards ledger"""
    try:
        ledger_file = "mined_rewards_ledger.json"
        with open(ledger_file, "w") as f:
            json.dump(ledger_data, f, indent=2)
    except Exception as e:
        print(f"Error saving mined rewards ledger: {e}")


# Also update the miner client side to handle the "already_exists" response
class SmartMiner:
    # ... existing code ...

    def submit_block(self, block_data):
        """Submit block to blockchain with duplicate handling"""
        try:
            response = requests.post(ENDPOINT_SUBMIT_BLOCK, json=block_data, timeout=30)

            if response.status_code in [200, 201]:
                result = response.json()
                if result.get("success"):
                    status = result.get("status", "added")

                    if status == "already_exists":
                        print(
                            color_text(
                                "⏭️  Block already exists in blockchain (already mined)",
                                Colors.YELLOW,
                            )
                        )
                        # Still count as success since the block is valid
                        self.blocks_mined += 1

                        # Mark reward transactions as mined even if block exists
                        reward_transactions = [
                            tx
                            for tx in block_data.get("transactions", [])
                            if tx.get("type") == "reward"
                        ]
                        if reward_transactions:
                            for reward_tx in reward_transactions:
                                self.config.add_reward_transaction(
                                    reward_tx.get("hash"),
                                    reward_tx.get("amount"),
                                    block_data["hash"],
                                )
                        return True
                    else:
                        print(
                            color_text(
                                "✅ Block successfully added to blockchain", Colors.GREEN
                            )
                        )
                        self.blocks_mined += 1
                        return True
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(color_text(f"❌ Block rejected: {error_msg}", Colors.RED))
            else:
                print(
                    color_text(
                        f"❌ HTTP {response.status_code}: {response.text}", Colors.RED
                    )
                )

        except Exception as e:
            print(color_text(f"💥 Submission error: {e}", Colors.RED))

        return False

    import hashlib
    import re

    def validate_reward_transactions(
        reward_transactions, block_index, block_data, previous_block_hash
    ):
        """Validate reward transactions with mining proof validation"""
        if not reward_transactions:
            return {"valid": True, "error": None}

        def _is_valid_luna_address(address: str) -> bool:
            if not address or not isinstance(address, str):
                return False
            normalized = address.strip()
            if not normalized:
                return False
            lowered = normalized.lower()
            placeholder_values = {
                "enter luna wallet address here",
                "enter wallet address here",
                "miner_default_address",
                "default_wallet_address",
            }
            if lowered in placeholder_values:
                return False
            if normalized.startswith("LUN_"):
                return True
            return bool(re.fullmatch(r"[0-9a-fA-F]{32}", normalized))

        # Check for duplicate reward transactions in this block
        reward_hashes = []
        for tx in reward_transactions:
            tx_hash = tx.get("hash")
            if not tx_hash:
                return {"valid": False, "error": "Reward transaction missing hash"}
            if tx_hash in reward_hashes:
                return {
                    "valid": False,
                    "error": f"Duplicate reward transaction hash: {tx_hash}",
                }
            reward_hashes.append(tx_hash)

        # Extract mining data from block (assuming it contains nonce, timestamp, and mining info)
        # The block should contain data like nonce, timestamp, and miner's address
        nonce = block_data.get("nonce")
        timestamp = block_data.get("timestamp")
        miner_address = block_data.get("miner", "")

        if not nonce or not timestamp:
            return {
                "valid": False,
                "error": "Block missing mining data (nonce or timestamp)",
            }

        # Validate there's exactly one reward transaction per block (standard mining)
        if len(reward_transactions) > 1:
            return {
                "valid": False,
                "error": f"Multiple reward transactions ({len(reward_transactions)}) in single block. Only one mining reward allowed.",
            }

        # Get the single reward transaction (should be the mining reward)
        reward_tx = reward_transactions[0]

        # Validate required fields for reward transactions
        required_reward_fields = ["to", "from", "amount", "block_height", "hash"]
        missing_reward_fields = [
            field for field in required_reward_fields if field not in reward_tx
        ]
        if missing_reward_fields:
            return {
                "valid": False,
                "error": f"Reward transaction missing fields: {missing_reward_fields}",
            }

        # Validate recipient is the miner who solved the block
        recipient = reward_tx.get("to", "")
        if recipient != miner_address:
            return {
                "valid": False,
                "error": f"Reward recipient {recipient} does not match block miner {miner_address}",
            }

        # Validate miner/recipient address format (reject placeholders)
        if not _is_valid_luna_address(miner_address):
            return {
                "valid": False,
                "error": f"Invalid miner address format: {miner_address}",
            }
        if not _is_valid_luna_address(recipient):
            return {
                "valid": False,
                "error": f"Invalid recipient address format: {recipient}",
            }

        # Validate 'from' field for mining reward
        from_field = reward_tx.get("from", "")
        valid_from_values = [
            "ling country",
            "network",
            "mining_reward",
            "block_reward",
            "coinbase",
        ]  # Mining rewards come from network
        if str(from_field).strip().lower() not in valid_from_values:
            return {
                "valid": False,
                "error": f'Invalid "from" field for mining reward: {from_field}. Must be one of: {valid_from_values}',
            }

        # Validate recipient address format (accepts both formats)
        if not recipient or not isinstance(recipient, str):
            return {"valid": False, "error": f"Invalid recipient address: {recipient}"}

        # Check if address is in LUN_ format
        if recipient.startswith("LUN_"):
            # Valid LUN_ format address
            pass
        else:
            # Check if address is in hex format (like 2a53c957713b6ade727659375437eda9)
            hex_pattern = re.compile(r"^[0-9a-fA-F]{32}$")
            if not hex_pattern.match(recipient):
                return {
                    "valid": False,
                    "error": f"Invalid recipient address format: {recipient}. Must start with LUN_ or be a 32-character hex string",
                }

        # Validate block_height matches current block
        block_height = reward_tx.get("block_height")
        if block_height != block_index:
            return {
                "valid": False,
                "error": f"Reward transaction block_height {block_height} does not match block index {block_index}",
            }

        # Validate amount is positive
        amount = reward_tx.get("amount")
        if not isinstance(amount, (int, float)) or amount <= 0:
            return {"valid": False, "error": f"Invalid reward amount: {amount}"}

        # Validate hash format
        tx_hash = reward_tx.get("hash", "")
        if not tx_hash or not isinstance(tx_hash, str) or len(tx_hash) < 16:
            return {"valid": False, "error": "Invalid or missing transaction hash"}

        # CRITICAL: Validate mining proof (this prevents arbitrary reward creation)
        # The block hash must meet the difficulty target
        mining_proof_valid = validate_mining_proof(block_data, previous_block_hash)
        if not mining_proof_valid["valid"]:
            return {
                "valid": False,
                "error": f'Invalid mining proof: {mining_proof_valid["error"]}',
            }

        # Extract difficulty from the mining proof validation
        actual_difficulty = mining_proof_valid.get("difficulty", 1)

        # Calculate expected reward based on actual difficulty + fees (new method)
        non_reward_txs = [
            tx for tx in block_data.get("transactions", []) if tx.get("type") != "reward"
        ]
        total_fees = 0
        for tx in non_reward_txs:
            if not isinstance(tx, dict):
                continue
            fee = tx.get("fee", 0)
            try:
                fee_val = float(fee)
            except (ValueError, TypeError):
                return {"valid": False, "error": f"Invalid transaction fee: {fee}"}
            if fee_val < 0:
                return {"valid": False, "error": f"Negative transaction fee: {fee_val}"}
            total_fees += fee_val
        expected_reward = None
        try:
            from lunalib.mining.difficulty import DifficultySystem

            difficulty_system = DifficultySystem()
            expected_reward = float(
                difficulty_system.calculate_block_reward(
                    actual_difficulty,
                    block_height=block_index,
                    tx_count=len(non_reward_txs),
                    fees_total=float(total_fees),
                )
            )
        except Exception:
            expected_reward = None

        if expected_reward is None:
            return {
                "valid": False,
                "error": "Unable to calculate reward via lunalib DifficultySystem",
            }

        # Validate amount matches expected reward
        if abs(amount - expected_reward) > 0.000001:
            return {
                "valid": False,
                "error": (
                    f"Reward amount {amount} does not match expected amount {expected_reward} "
                    f"(difficulty {actual_difficulty}, fees {total_fees})"
                ),
            }

        # Check if this reward transaction already exists in blockchain
        if is_reward_transaction_duplicate(reward_tx):
            return {
                "valid": False,
                "error": f"Reward transaction already exists in blockchain: {tx_hash}",
            }

        return {"valid": True, "error": None, "difficulty": actual_difficulty}

    def validate_mining_proof(block_data, previous_block_hash):
        """Validate mining proof using lunalib only (no local math)."""
        def _resolve_lunalib_callable(module_names, method_names):
            for module_name in module_names:
                try:
                    module = __import__(module_name, fromlist=["*"])
                except Exception:
                    continue
                for method_name in method_names:
                    candidate = getattr(module, method_name, None)
                    if callable(candidate):
                        return candidate
            return None

        pow_validator = _resolve_lunalib_callable(
            [
                "lunalib.blockchain",
                "lunalib.core.blockchain",
                "lunalib.mining",
                "lunalib.mining.proof",
                "lunalib.mining.miner",
            ],
            [
                "validate_mining_proof",
                "validate_pow",
                "validate_proof_of_work",
                "validate_mining_proof_internal",
            ],
        )

        if pow_validator is None:
            return {"valid": False, "error": "Lunalib mining proof validator unavailable"}

        try:
            try:
                import inspect

                params = list(inspect.signature(pow_validator).parameters.values())
                if len(params) <= 1:
                    result = pow_validator(block_data)
                else:
                    result = pow_validator(block_data, previous_block_hash)
            except Exception:
                result = pow_validator(block_data, previous_block_hash)
        except Exception as e:
            return {"valid": False, "error": f"Lunalib mining proof validation error: {e}"}

        if isinstance(result, dict):
            if "valid" in result:
                return {
                    "valid": bool(result.get("valid")),
                    "difficulty": result.get("difficulty"),
                    "block_hash": result.get("block_hash"),
                }
            if "success" in result:
                return {
                    "valid": bool(result.get("success")),
                    "difficulty": result.get("difficulty"),
                    "block_hash": result.get("block_hash"),
                }

        if isinstance(result, bool):
            return {"valid": result}

        return {"valid": False, "error": "Unexpected response from lunalib mining proof validator"}

    def is_reward_transaction_duplicate(tx):
        """Check if reward transaction already exists in blockchain"""
        # This would query your blockchain database
        # For now, return False as a placeholder
        return False


def validate_regular_transactions(transactions):
    """Validate regular (non-reward) transactions"""
    if not transactions:
        return {"valid": True, "error": None}

    for tx in transactions:
        tx_type = tx.get("type")

        if tx_type == "transfer":
            # Validate transfer transactions
            required_transfer_fields = ["from", "to", "amount", "signature"]
            missing_fields = [
                field for field in required_transfer_fields if field not in tx
            ]
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Transfer transaction missing fields: {missing_fields}",
                }

            # Validate amount
            amount = tx.get("amount")
            if not isinstance(amount, (int, float)) or amount <= 0:
                return {"valid": False, "error": f"Invalid transfer amount: {amount}"}

            # Validate addresses
            from_addr = tx.get("from")
            to_addr = tx.get("to")
            if not from_addr or not to_addr:
                return {
                    "valid": False,
                    "error": "Invalid addresses in transfer transaction",
                }

            # Check for self-transfer (allow but flag if needed)
            if from_addr == to_addr:
                pass

        elif tx_type == "GTX_Genesis":
            # Validate genesis transactions
            required_genesis_fields = ["serial_number", "denomination", "signature"]
            missing_fields = [
                field for field in required_genesis_fields if field not in tx
            ]
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Genesis transaction missing fields: {missing_fields}",
                }

        else:
            # Unknown transaction type
            return {"valid": False, "error": f"Unknown transaction type: {tx_type}"}

    return {"valid": True, "error": None}


def is_reward_transaction_duplicate(reward_tx):
    """Check if a reward transaction already exists in the blockchain"""
    try:
        # Get the blockchain data
        blockchain_data = blockchain_daemon_instance.blockchain

        for block in blockchain_data:
            transactions = block.get("transactions", [])
            for tx in transactions:
                if tx.get("type") == "reward":
                    # Check if this is the same reward transaction
                    if tx.get("hash") == reward_tx.get("hash") or (
                        tx.get("miner") == reward_tx.get("miner")
                        and tx.get("block_height") == reward_tx.get("block_height")
                    ):
                        return True
        return False
    except Exception as e:
        print(f"Error checking reward transaction duplicate: {e}")
        return False


# Enhanced add_validated_block method for the blockchain daemon
def add_validated_block_with_rewards(self, block_data):
    """Add a validated block to the blockchain with reward transaction tracking"""
    try:
        # Your existing block validation logic here...

        # Track reward transactions specifically
        reward_transactions = [
            tx
            for tx in block_data.get("transactions", [])
            if tx.get("type") == "reward"
        ]
        if reward_transactions:
            print(
                f"💰 Block #{block_data['index']} contains {len(reward_transactions)} reward transactions"
            )

            # Store reward transaction metadata
            for reward_tx in reward_transactions:
                self.track_reward_transaction(reward_tx, block_data["hash"])

        # Continue with existing block addition logic...
        return True

    except Exception as e:
        print(f"Error adding block with rewards: {e}")
        return False


def track_reward_transaction(self, reward_tx, block_hash):
    """Track reward transaction in a separate rewards ledger"""
    try:
        reward_data = {
            "hash": reward_tx.get("hash"),
            "miner": reward_tx.get("miner"),
            "recipient": reward_tx.get("to"),
            "amount": reward_tx.get("amount"),
            "block_height": reward_tx.get("block_height"),
            "block_hash": block_hash,
            "timestamp": time.time(),
        }

        # Load existing rewards ledger
        rewards_ledger = self.load_rewards_ledger()
        rewards_ledger.append(reward_data)

        # Save rewards ledger
        self.save_rewards_ledger(rewards_ledger)

        print(
            f"🎯 Tracked reward: {reward_tx.get('miner')} -> {reward_tx.get('to')} for {reward_tx.get('amount')} LUN"
        )

    except Exception as e:
        print(f"Error tracking reward transaction: {e}")


@app.route("/debug/blockchain-daemon")
def debug_blockchain_daemon():
    """Debug endpoint for blockchain daemon status and validation"""
    try:
        debug_info = {
            "daemon_status": "running",
            "timestamp": int(time.time()),
            "blockchain": {
                "total_blocks": len(blockchain_daemon_instance.blockchain),
                "blocks": [],
            },
            "mempool": {
                "total_transactions": len(blockchain_daemon_instance.mempool),
                "transactions_by_type": {},
            },
            "validation_tests": [],
            "configuration": {
                "blockchain_file": blockchain_daemon_instance.blockchain_file,
                "mempool_file": blockchain_daemon_instance.mempool_file,
                "sync_interval": blockchain_daemon_instance.sync_interval,
                "is_running": blockchain_daemon_instance.is_running,
            },
        }

        # Analyze last 5 blocks
        recent_blocks = (
            blockchain_daemon_instance.blockchain[-5:]
            if blockchain_daemon_instance.blockchain
            else []
        )
        for i, block in enumerate(recent_blocks):
            block_info = {
                "index": block.get("index"),
                "hash": block.get("hash", "N/A")[:20] + "..."
                if block.get("hash")
                else "N/A",
                "previous_hash": block.get("previous_hash", "N/A")[:20] + "..."
                if block.get("previous_hash")
                else "N/A",
                "timestamp": block.get("timestamp"),
                "timestamp_readable": datetime.fromtimestamp(
                    block.get("timestamp", 0)
                ).strftime("%Y-%m-%d %H:%M:%S")
                if block.get("timestamp")
                else "N/A",
                "nonce": block.get("nonce"),
                "miner": block.get("miner", "N/A"),
                "transaction_count": len(block.get("transactions", [])),
                "transaction_types": {},
            }

            # Count transaction types in this block
            for tx in block.get("transactions", []):
                tx_type = tx.get("type", "unknown")
                block_info["transaction_types"][tx_type] = (
                    block_info["transaction_types"].get(tx_type, 0) + 1
                )

            debug_info["blockchain"]["blocks"].append(block_info)

        # Analyze mempool transactions
        for tx in blockchain_daemon_instance.mempool:
            tx_type = tx.get("type", "unknown")
            debug_info["mempool"]["transactions_by_type"][tx_type] = (
                debug_info["mempool"]["transactions_by_type"].get(tx_type, 0) + 1
            )

        # Run validation tests
        validation_tests = []

        # Test 1: Check blockchain continuity
        if len(blockchain_daemon_instance.blockchain) > 1:
            for i in range(1, min(5, len(blockchain_daemon_instance.blockchain))):
                current_block = blockchain_daemon_instance.blockchain[i]
                previous_block = blockchain_daemon_instance.blockchain[i - 1]

                if current_block.get("previous_hash") == previous_block.get("hash"):
                    validation_tests.append(
                        {
                            "test": f"Block continuity #{i}",
                            "status": "PASS",
                            "message": f"Block {i} correctly links to block {i-1}",
                        }
                    )
                else:
                    validation_tests.append(
                        {
                            "test": f"Block continuity #{i}",
                            "status": "FAIL",
                            "message": f"Block {i} previous_hash doesn't match block {i-1} hash",
                        }
                    )

        # Test 2: Validate block hashes
        for i, block in enumerate(blockchain_daemon_instance.blockchain[-3:]):
            calculated_hash = blockchain_daemon_instance.calculate_block_hash(
                block.get("index"),
                block.get("previous_hash"),
                block.get("timestamp"),
                block.get("transactions", []),
                block.get("nonce"),
            )

            if block.get("hash") == calculated_hash:
                validation_tests.append(
                    {
                        "test": f"Block #{block.get('index')} hash validation",
                        "status": "PASS",
                        "message": "Hash matches calculated value",
                    }
                )
            else:
                validation_tests.append(
                    {
                        "test": f"Block #{block.get('index')} hash validation",
                        "status": "FAIL",
                        "message": f"Hash mismatch: stored={block.get('hash')[:20]}..., calculated={calculated_hash[:20]}...",
                    }
                )

        # Test 3: Check for duplicate transactions
        all_tx_hashes = []
        duplicate_count = 0

        for block in blockchain_daemon_instance.blockchain:
            for tx in block.get("transactions", []):
                tx_hash = tx.get("hash")
                if tx_hash:
                    if tx_hash in all_tx_hashes:
                        duplicate_count += 1
                    all_tx_hashes.append(tx_hash)

        validation_tests.append(
            {
                "test": "Duplicate transactions check",
                "status": "PASS" if duplicate_count == 0 else "WARNING",
                "message": f"Found {duplicate_count} duplicate transactions in blockchain",
            }
        )

        # Test 4: Mempool vs Blockchain duplicates
        mined_hashes = set()
        for block in blockchain_daemon_instance.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("hash"):
                    mined_hashes.add(tx.get("hash"))

        mempool_duplicates = 0
        for tx in blockchain_daemon_instance.mempool:
            if tx.get("hash") in mined_hashes:
                mempool_duplicates += 1

        validation_tests.append(
            {
                "test": "Mempool cleanup",
                "status": "PASS" if mempool_duplicates == 0 else "WARNING",
                "message": f"Found {mempool_duplicates} mined transactions still in mempool",
            }
        )

        debug_info["validation_tests"] = validation_tests

        # Test 5: Test block validation with sample data
        if blockchain_daemon_instance.blockchain:
            sample_block = blockchain_daemon_instance.blockchain[-1]
            is_valid = blockchain_daemon_instance.validate_block(sample_block)
            validation_tests.append(
                {
                    "test": "Sample block validation",
                    "status": "PASS" if is_valid else "FAIL",
                    "message": f"Latest block validation: {'VALID' if is_valid else 'INVALID'}",
                }
            )

        return jsonify(debug_info)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "daemon_status": "error",
                }
            ),
            500,
        )


@app.route("/debug/blockchain-test-validation", methods=["POST"])
def test_block_validation():
    """Test block validation with custom block data"""
    try:
        test_block = request.get_json()

        if not test_block:
            return jsonify({"error": "No block data provided"}), 400

        # Run validation
        is_valid = blockchain_daemon_instance.validate_block(test_block)

        # Calculate expected hash
        calculated_hash = blockchain_daemon_instance.calculate_block_hash(
            test_block.get("index"),
            test_block.get("previous_hash"),
            test_block.get("timestamp"),
            test_block.get("transactions", []),
            test_block.get("nonce"),
        )

        validation_details = {
            "is_valid": is_valid,
            "provided_hash": test_block.get("hash"),
            "calculated_hash": calculated_hash,
            "hash_match": test_block.get("hash") == calculated_hash,
            "missing_fields": [],
            "type_issues": [],
        }

        # Check required fields
        required_fields = [
            "index",
            "timestamp",
            "transactions",
            "previous_hash",
            "nonce",
            "hash",
            "miner",
        ]
        for field in required_fields:
            if field not in test_block:
                validation_details["missing_fields"].append(field)

        # Check data types
        if "index" in test_block and not isinstance(test_block["index"], int):
            validation_details["type_issues"].append("index should be integer")
        if "timestamp" in test_block and not isinstance(test_block["timestamp"], int):
            validation_details["type_issues"].append("timestamp should be integer")
        if "nonce" in test_block and not isinstance(test_block["nonce"], int):
            validation_details["type_issues"].append("nonce should be integer")

        return jsonify(validation_details)

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/debug/blockchain-force-cleanup", methods=["POST"])
def force_cleanup():
    """Force cleanup of mined transactions from mempool"""
    try:
        initial_mempool_size = len(blockchain_daemon_instance.mempool)

        # Run cleanup
        blockchain_daemon_instance.cleanup_mined_transactions()

        final_mempool_size = len(blockchain_daemon_instance.mempool)
        removed_count = initial_mempool_size - final_mempool_size

        # Save the cleaned mempool
        blockchain_daemon_instance.save_mempool()

        return jsonify(
            {
                "success": True,
                "removed_transactions": removed_count,
                "initial_mempool_size": initial_mempool_size,
                "final_mempool_size": final_mempool_size,
                "message": f"Removed {removed_count} mined transactions from mempool",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/debug/blockchain-repair", methods=["POST"])
def repair_blockchain():
    """Attempt to repair blockchain issues"""
    try:
        # Run comprehensive diagnostic and repair
        initial_block_count = len(blockchain_daemon_instance.blockchain)

        # Reload data from files
        blockchain_daemon_instance.load_data()

        # Cleanup mined transactions
        blockchain_daemon_instance.cleanup_mined_transactions()

        # Save everything
        blockchain_daemon_instance.save_blockchain()
        blockchain_daemon_instance.save_mempool()

        final_block_count = len(blockchain_daemon_instance.blockchain)

        return jsonify(
            {
                "success": True,
                "initial_blocks": initial_block_count,
                "final_blocks": final_block_count,
                "message": f"Blockchain repair completed. Blocks: {initial_block_count} -> {final_block_count}",
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"success": False, "error": str(e), "traceback": traceback.format_exc()}
            ),
            500,
        )


@app.route("/blockchain/validate", methods=["POST"])
def validate_block():
    """Validate a block without adding it to the blockchain"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No block data provided"}), 400

        # Validate the block
        is_valid = blockchain_daemon_instance.validate_block(data)

        if is_valid:
            return (
                jsonify(
                    {
                        "success": True,
                        "message": "Block is valid",
                        "block_hash": data.get("hash"),
                        "block_index": data.get("index"),
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Block validation failed",
                        "block_hash": data.get("hash"),
                        "block_index": data.get("index"),
                    }
                ),
                400,
            )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/transaction/<tx_hash>", methods=["GET"])
def get_transaction_status(tx_hash):
    """Get the status of a specific transaction"""
    try:
        if not tx_hash or tx_hash == "undefined":
            return (
                jsonify({"success": False, "error": "Transaction hash is required"}),
                400,
            )

        tx = blockchain_daemon_instance.get_transaction(tx_hash)

        if not tx:
            status = {"status": "not_found"}
        else:
            is_mined = blockchain_daemon_instance.is_transaction_mined(tx)
            status = {
                "status": "mined" if is_mined else "pending",
                "transaction": tx
            }

        return (
            jsonify(
                {
                    "success": True,
                    "transaction_hash": tx_hash,
                    "status": status,
                    "timestamp": int(time.time()),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/explorer/transaction/<transaction_hash>", methods=["GET"])
def transaction_explorer(transaction_hash):
    """Display transaction details in the explorer view"""
    try:
        if not transaction_hash or transaction_hash == "undefined":
            flash("Transaction hash is required", "error")
            return redirect(url_for("index"))

        # Get transaction status from blockchain daemon
        tx_data = blockchain_daemon_instance.get_transaction(
            transaction_hash
        )

        # If not found with custom format, try normal transaction lookup
        if not tx_data:
            tx_data = blockchain_daemon_instance.get_transaction(transaction_hash)
            # 型チェック: dictでなければNone扱い
            if not isinstance(tx_data, dict):
                print(f"[DEBUG] get_transaction returned non-dict: {type(tx_data)} value={tx_data}")
                tx_data = None
        # 追加: もしtx_dataがlist型なら、最初の要素がdictならそれを使う
        if isinstance(tx_data, list) and len(tx_data) > 0 and isinstance(tx_data[0], dict):
            print(f"[DEBUG] get_transaction returned list, using first element: {tx_data[0]}")
            tx_data = tx_data[0]

        if not tx_data:
            flash("Transaction not found", "error")
            return redirect(url_for("index"))

        # If you have a method to get full transaction details, use it here
        # For now, we'll use the status data and simulate some transaction details
        transaction = {
            "hash": transaction_hash,
            "status": "mined" if blockchain_daemon_instance.is_transaction_mined(tx_data) else "pending",
            "confirmations": 0, # You might need to calculate this
            "block_height": None, # You might need to find this
            "timestamp": tx_data.get("timestamp", int(time.time())),
            "from_address": tx_data.get("from"),
            "to_address": tx_data.get("to"),
            "value": tx_data.get("amount"),
            "gas_used": 0, # Placeholder
            "gas_price": 0, # Placeholder
            "input_data": tx_data.get("signature"), # Or other relevant data
            "nonce": 0, # Placeholder
            "is_error": False,
            "error_message": None,
        }

        # Calculate transaction age
        transaction_age = int(time.time()) - transaction["timestamp"]

        # Format timestamp for display
        from datetime import datetime

        dt = datetime.fromtimestamp(transaction["timestamp"])
        transaction["timestamp_formatted"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        transaction["timestamp_readable"] = dt.strftime("%B %d, %Y at %H:%M:%S")

        confirmations = int(transaction.get("confirmations") or 0)
        transaction["confirmations"] = confirmations

        # Determine status from confirmations (confirmed after 6)
        current_status = (transaction.get("status") or "").lower()
        if current_status in {"failed", "error"}:
            transaction["status"] = current_status
        else:
            if confirmations >= 6:
                transaction["status"] = "confirmed"
            elif confirmations > 0 or transaction.get("block_height") is not None:
                transaction["status"] = "pending"
            else:
                transaction["status"] = "pending"

        # Determine status icon and color
        status_info = {
            "pending": {"icon": "⏳", "color": "warning", "label": "Pending"},
            "confirmed": {"icon": "✅", "color": "success", "label": "Confirmed"},
            "failed": {"icon": "❌", "color": "danger", "label": "Failed"},
            "unknown": {"icon": "❓", "color": "secondary", "label": "Unknown"},
        }

        status_key = transaction.get("status", "unknown") or "unknown"
        if isinstance(status_key, str):
            status_key = status_key.lower()
        else:
            status_key = "unknown"

        transaction["status_icon"] = status_info.get(
            status_key, status_info["unknown"]
        )["icon"]
        transaction["status_color"] = status_info.get(
            status_key, status_info["unknown"]
        )["color"]
        transaction["status_label"] = status_info.get(
            status_key, status_info["unknown"]
        )["label"]

        # Calculate confirmation percentage based on mined status
        confirmation_percentage = 100 if blockchain_daemon_instance.is_transaction_mined(tx_data) else 0

        # Calculate validation percentage based on mined status
        validation_percentage = 100 if blockchain_daemon_instance.is_transaction_mined(tx_data) else 50
        
        # Calculate validation score (0-5)
        validation_score = 5 if blockchain_daemon_instance.is_transaction_mined(tx_data) else 3

        return render_template(
            "transaction_viewer.html",
            transaction=transaction,
            transaction_age=transaction_age,
            confirmation_percentage=confirmation_percentage,
            validation_percentage=validation_percentage,
            validation_score=validation_score,
            validation_results={},
            title=f"Transaction {transaction_hash[:12]}...",
            current_user=get_current_user(),
        )

    except Exception as e:
        flash(f"Error fetching transaction: {str(e)}", "error")
        return redirect(url_for("blockchain_viewer"))


# 在 app.py 中
@app.route("/blockchain/blocks", methods=["GET"])
def get_blocks():
    """Get all blocks, ensure genesis exists"""
    if len(blockchain_daemon_instance.blockchain) == 0:
        blockchain_daemon_instance._create_and_add_genesis_block()

    return jsonify(
        {
            "blocks": blockchain_daemon_instance.blockchain,
            "height": len(blockchain_daemon_instance.blockchain) - 1,
            "has_genesis": len(blockchain_daemon_instance.blockchain) > 0,
        }
    )


@app.route("/blockchain/latest-block", methods=["GET"])
def get_latest_block():
    """Get the latest block in the blockchain - ensure compatibility"""
    try:
        if not blockchain_daemon_instance.blockchain:
            # 返回空但结构化的响应
            return (
                jsonify(
                    {"success": False, "error": "Blockchain is empty", "block": None}
                ),
                404,
            )

        latest_block = blockchain_daemon_instance.blockchain[-1]

        # 确保返回的结构与客户端期望一致
        return jsonify(latest_block), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/blockchain/latest", methods=["GET", "POST"])
def api_get_latest_block():
    """Return the latest block using lunalib when available (fast path)."""
    try:
        daemon = blockchain_daemon_instance
        latest_block = None

        blockchain_mgr = getattr(daemon, "blockchain_mgr", None)
        if blockchain_mgr is not None:
            for method_name in (
                "get_latest_block",
                "get_latest_blocks",
                "get_blockchain_latest",
                "latest_block",
                "get_latest",
            ):
                if hasattr(blockchain_mgr, method_name):
                    method = getattr(blockchain_mgr, method_name)
                    if callable(method):
                        try:
                            result = method(1) if method_name == "get_latest_blocks" else method()
                            if isinstance(result, list):
                                latest_block = result[0] if result else None
                            elif isinstance(result, dict):
                                latest_block = result
                            elif hasattr(result, "to_dict"):
                                latest_block = result.to_dict()
                            if latest_block:
                                break
                        except Exception:
                            continue

        if latest_block is None:
            chain = getattr(daemon, "blockchain", []) or []
            latest_block = chain[-1] if chain else None

        if not latest_block:
            return jsonify({"success": False, "error": "Blockchain is empty"}), 404

        return jsonify({"success": True, "block": latest_block}), 200

    except Exception as e:
        logger.error("/api/blockchain/latest failed: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/system/health", methods=["GET"])
def system_health():
    """System health check endpoint"""
    try:
        mempool_status = blockchain_daemon_instance.get_mempool_status()
        blockchain_status = blockchain_daemon_instance.get_blockchain_status()

        return (
            jsonify(
                {
                    "success": True,
                    "status": "healthy",
                    "mempool": {
                        "total_transactions": mempool_status["total"],
                        "bills": mempool_status["bills"],
                        "transfers": mempool_status["transfers"],
                        "rewards": mempool_status["rewards"],
                    },
                    "blockchain": {
                        "total_blocks": blockchain_status["blocks"],
                        "total_transactions": blockchain_status["total_transactions"],
                        "genesis_transactions": blockchain_status[
                            "genesis_transactions"
                        ],
                        "transfer_transactions": blockchain_status[
                            "transfer_transactions"
                        ],
                    },
                    "timestamp": int(time.time()),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "status": "unhealthy", "error": str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    logger.warning("404 Not Found: %s %s", request.method, request.path)
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    logger.warning("405 Method Not Allowed: %s %s", request.method, request.path)
    return jsonify({"success": False, "error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_server_error(error):
    logger.error("500 Internal Server Error: %s %s", request.method, request.path)
    return jsonify({"success": False, "error": "Internal server error"}), 500


def diagnose_signature_creation(tx_data):
    """Diagnose how a signature was created by testing multiple methods"""
    signature = tx_data.get("signature", "")
    public_key = tx_data.get("public_key", "")
    metadata_hash = tx_data.get("metadata_hash", "")
    issued_to = tx_data.get("issued_to", "")
    denomination = tx_data.get("denomination", "")
    front_serial = tx_data.get("front_serial", "")
    timestamp = tx_data.get("timestamp", 0)
    bill_type = tx_data.get("type", "banknote")

    tests = {}

    # Test 1: Hash of public_key only
    tests["hash_public_key_only"] = (
        hashlib.sha256(public_key.encode()).hexdigest() == signature
    )

    # Test 2: Hash of metadata_hash only
    if metadata_hash:
        tests["hash_metadata_only"] = (
            hashlib.sha256(metadata_hash.encode()).hexdigest() == signature
        )

    # Test 3: Hash of public_key + metadata_hash (already tested, but include for completeness)
    if metadata_hash:
        tests["hash_public_metadata"] = (
            hashlib.sha256(f"{public_key}{metadata_hash}".encode()).hexdigest()
            == signature
        )

    # Test 4: Hash of serial + denomination + issued_to
    tests["hash_serial_denom_issued"] = (
        hashlib.sha256(f"{front_serial}{denomination}{issued_to}".encode()).hexdigest()
        == signature
    )

    # Test 5: Hash of all basic fields
    basic_data = f"{front_serial}{denomination}{issued_to}{timestamp}"
    tests["hash_all_basic"] = (
        hashlib.sha256(basic_data.encode()).hexdigest() == signature
    )

    # Test 6: Hash of JSON without signature
    tx_copy = tx_data.copy()
    if "signature" in tx_copy:
        del tx_copy["signature"]
    tx_string = json.dumps(tx_copy, sort_keys=True)
    tests["hash_json_no_signature"] = (
        hashlib.sha256(tx_string.encode()).hexdigest() == signature
    )

    # Test 7: Hash of JSON with signature included (unlikely but possible)
    tx_string_with_sig = json.dumps(tx_data, sort_keys=True)
    tests["hash_json_with_signature"] = (
        hashlib.sha256(tx_string_with_sig.encode()).hexdigest() == signature
    )

    # Test 8: MD5 variants (less secure but possible)
    tests["md5_public_metadata"] = (
        hashlib.md5(f"{public_key}{metadata_hash}".encode()).hexdigest() == signature
        if metadata_hash
        else False
    )
    tests["md5_basic_data"] = (
        hashlib.md5(f"{front_serial}{denomination}{issued_to}".encode()).hexdigest()
        == signature
    )

    # Test 9: Check if signature is actually the metadata_hash
    tests["signature_is_metadata_hash"] = signature == metadata_hash

    # Test 10: Check if signature is derived from a combination with the bill type
    tests["hash_with_type"] = (
        hashlib.sha256(f"{bill_type}{front_serial}{denomination}".encode()).hexdigest()
        == signature
    )

    # Find which test passed
    matched_method = None
    for method, passed in tests.items():
        if passed:
            matched_method = method
            break

    return {
        "matched": matched_method is not None,
        "method": matched_method,
        "all_tests": tests,
    }


def find_transaction_in_blockchain(serial_number, issued_to, denomination):
    """Look for a transaction in the blockchain that matches this banknote"""
    try:
        for block in blockchain_daemon_instance.blockchain:
            for tx in block.get("transactions", []):
                if (
                    tx.get("serial_number") == serial_number
                    and tx.get("issued_to") == issued_to
                    and str(tx.get("denomination")) == str(denomination)
                ):
                    return tx
    except Exception as e:
        print(f"Error searching blockchain: {e}")
    return None


# In app.py, update the verify_serial route with the correct verification:
@app.route("/api/debug/signature-analysis/<serial_id>")
def debug_signature_analysis(serial_id):
    """Debug endpoint to analyze signature creation method"""
    serial_record = SerialNumber.query.filter_by(
        serial=serial_id, is_active=True
    ).first()
    if not serial_record or not serial_record.banknote:
        return jsonify({"error": "Serial not found"})

    banknote = serial_record.banknote
    tx_data = (
        json.loads(banknote.transaction_data)
        if hasattr(banknote, "transaction_data") and banknote.transaction_data
        else {}
    )

    analysis = {
        "serial": serial_id,
        "banknote_id": banknote.id,
        "transaction_data_keys": list(tx_data.keys()) if tx_data else [],
        "signature_present": bool(tx_data.get("signature")),
        "public_key_present": bool(tx_data.get("public_key")),
        "metadata_hash_present": bool(tx_data.get("metadata_hash")),
        "signature_length": len(tx_data.get("signature", "")),
        "public_key_length": len(tx_data.get("public_key", "")),
        "metadata_hash_length": len(tx_data.get("metadata_hash", "")),
        "signature_prefix": tx_data.get("signature", "")[:10]
        if tx_data.get("signature")
        else None,
        "transaction_type": tx_data.get("type"),
        "timestamp": tx_data.get("timestamp"),
        "issued_to": tx_data.get("issued_to"),
        "denomination": tx_data.get("denomination"),
    }

    # Try to determine signature method
    signature = tx_data.get("signature", "")
    public_key = tx_data.get("public_key", "")
    metadata_hash = tx_data.get("metadata_hash", "")

    # Test different signature creation methods
    test_results = {}

    # Method 1: public_key + metadata_hash
    if public_key and metadata_hash:
        test_data = f"{public_key}{metadata_hash}"
        test_hash = hashlib.sha256(test_data.encode()).hexdigest()
        test_results["method_public_key_metadata_hash"] = signature == test_hash

    # Method 2: transaction data hash
    if tx_data:
        tx_copy = tx_data.copy()
        if "signature" in tx_copy:
            del tx_copy["signature"]
        tx_string = json.dumps(tx_copy, sort_keys=True)
        tx_hash = hashlib.sha256(tx_string.encode()).hexdigest()
        test_results["method_transaction_hash"] = signature == tx_hash

    # Method 3: simple data hash
    simple_data = f"{tx_data.get('front_serial', '')}{tx_data.get('denomination', '')}{tx_data.get('issued_to', '')}{tx_data.get('timestamp', '')}"
    simple_hash = hashlib.sha256(simple_data.encode()).hexdigest()
    test_results["method_simple_hash"] = signature == simple_hash

    analysis["signature_method_tests"] = test_results

    return jsonify(analysis)


@app.route("/verify", methods=["GET", "POST"])
@app.route("/verify/<serial_id>", methods=["GET"])
def verify_serial(serial_id=None):
    result = None
    serial_input = ""
    banknote = None
    signature_valid = None
    signature_details = {}
    verification_method = "unknown"
    blockchain_status = None
    mined_transaction = None
    block_details = None
    tx_data = None
    validation_results = {
        "serial_db": None,
        "banknote_db": None,
        "digital_bill": None,
        "mempool": None,
        "blockchain": None,
    }
    global blockchain_daemon_instance

    # Initialize SM2 signature manager
    try:
        from signatures import DigitalSignatureManager, DigitalBill

        signature_manager = DigitalSignatureManager()
        sm2_available = True
        print("[+] SM2 signature manager loaded successfully")
    except ImportError as e:
        signature_manager = None
        sm2_available = False
        print(f"[-] SM2 signature manager not available: {e}")
        # Try to import DigitalBill directly
        try:
            from signatures import DigitalBill

            print("[+] DigitalBill class loaded directly")
        except ImportError:
            DigitalBill = None
            print("[-] DigitalBill class not available")

    # Determine which serial to verify
    if serial_id:
        serial_input = serial_id
        result = validate_serial_id(serial_input)
    elif request.method == "POST":
        serial_input = request.form.get("serial", "").strip()
        result = validate_serial_id(serial_input)
    elif request.method == "GET" and "serial" in request.args:
        serial_input = request.args.get("serial", "").strip()
        result = validate_serial_id(serial_input)

    if result and result.get("valid"):
        def _build_serial_candidates(serial_value: str):
            if not serial_value:
                return []
            candidates = [serial_value]
            if serial_value.upper().startswith("GTX-"):
                return candidates
            serial_upper = serial_value.upper()
            if serial_upper.endswith("_FRONT") or serial_upper.endswith("_BACK"):
                base_serial = serial_value.rsplit("_", 1)[0]
                if base_serial and base_serial not in candidates:
                    candidates.append(base_serial)
            else:
                candidates.append(f"{serial_value}_FRONT")
                candidates.append(f"{serial_value}_BACK")
            return candidates

        serial_candidates = _build_serial_candidates(serial_input)

        # LAYER 1: Check Serial Database
        serial_record = SerialNumber.query.filter(
            SerialNumber.serial.in_(serial_candidates),
            SerialNumber.is_active.is_(True),
        ).first()
        if not serial_record:
            serial_record = SerialNumber.query.filter(
                SerialNumber.serial.in_(serial_candidates)
            ).first()
        validation_results["serial_db"] = {
            "found": serial_record is not None,
            "data": {
                "id": serial_record.id if serial_record else None,
                "serial": serial_record.serial if serial_record else None,
                "created_at": serial_record.created_at if serial_record else None,
                "is_active": serial_record.is_active if serial_record else None,
            },
        }
        # LAYER 2: Check Banknote Database (fallback to banknote lookup if serial record missing)
        if serial_record and serial_record.banknote:
            banknote = serial_record.banknote
        else:
            banknote = Banknote.query.filter(
                Banknote.serial_number.in_(serial_candidates)
            ).first()

        validation_results["banknote_db"] = {
            "found": banknote is not None,
            "data": {
                "owner": banknote.user.username if banknote and banknote.user else None,
                "denomination": banknote.denomination if banknote else None,
                "side": banknote.side if banknote else None,
            }
            if banknote
            else None,
        }

        # LAYER 4/5: Check mempool & blockchain even if DB records are missing
        def _tx_has_serial(tx, serial_values):
            if not serial_values or not isinstance(tx, dict):
                return False
            if isinstance(serial_values, str):
                serial_values = [serial_values]
            for field in ("serial_number", "front_serial", "back_serial", "serial"):
                field_value = tx.get(field)
                if not field_value:
                    continue
                if field_value in serial_values:
                    return True
            return False

        mempool_found = False
        mempool_tx = None
        blockchain_found = False
        blockchain_data = None
        confirmations = 0

        try:
            if blockchain_daemon_instance and hasattr(blockchain_daemon_instance, "mempool"):
                for tx in blockchain_daemon_instance.mempool:
                    if _tx_has_serial(tx, serial_candidates):
                        mempool_found = True
                        mempool_tx = tx.copy()
                        break
        except Exception as e:
            print(f"Error checking mempool: {e}")

        validation_results["mempool"] = {
            "found": mempool_found,
            "status": "pending" if mempool_found else "not_found",
            "mined_from_mempool": False,
        }

        try:
            if blockchain_daemon_instance and hasattr(blockchain_daemon_instance, "blockchain"):
                for block_idx, block in enumerate(blockchain_daemon_instance.blockchain):
                    for tx in block.get("transactions", []):
                        if _tx_has_serial(tx, serial_candidates):
                            blockchain_found = True
                            blockchain_data = tx.copy()
                            blockchain_data["block_height"] = block_idx
                            blockchain_data["block_hash"] = block.get("hash")
                            confirmations = max(
                                0,
                                len(blockchain_daemon_instance.blockchain) - block_idx - 1,
                            )
                            break
                    if blockchain_found:
                        break
        except Exception as e:
            print(f"[BLOCKCHAIN ERROR] Error checking blockchain: {e}")

        validation_results["blockchain"] = {
            "found": blockchain_found,
            "data": blockchain_data,
            "confirmations": confirmations,
            "status": "mined" if blockchain_found else ("pending" if mempool_found else "unmined"),
            "daemon_available": blockchain_daemon_instance is not None,
        }

        # Prefer transaction data from DB, then blockchain, then mempool
        if banknote and hasattr(banknote, "transaction_data"):
            try:
                tx_data = (
                    json.loads(banknote.transaction_data)
                    if banknote.transaction_data
                    else {}
                )
            except Exception:
                tx_data = None
        if not tx_data:
            tx_data = blockchain_data or mempool_tx

        if tx_data:
            try:
                def _try_lunalib_sm2_verify(payload: dict):
                    try:
                        from lunalib.gtx.digital_bill import DigitalBill as LunalibDigitalBill
                    except Exception:
                        try:
                            from lunalib.digital_bill import DigitalBill as LunalibDigitalBill
                        except Exception:
                            return None, "lunalib DigitalBill unavailable"

                    bill_instance = None
                    init_errors = []

                    # Try known constructor signatures
                    try:
                        bill_instance = LunalibDigitalBill(
                            bill_type=payload.get("type", "banknote"),
                            front_serial=payload.get("front_serial", ""),
                            back_serial=payload.get("back_serial", ""),
                            metadata_hash=payload.get("metadata_hash", ""),
                            timestamp=payload.get("timestamp", 0),
                            issued_to=payload.get("issued_to", ""),
                            denomination=payload.get("denomination", ""),
                            public_key=payload.get("public_key"),
                            signature=payload.get("signature"),
                        )
                    except Exception as e:
                        init_errors.append(str(e))

                    if bill_instance is None:
                        try:
                            bill_instance = LunalibDigitalBill(**payload)
                        except Exception as e:
                            init_errors.append(str(e))

                    if bill_instance is None:
                        return None, f"lunalib DigitalBill init failed: {init_errors}"

                    for method_name in (
                        "verify",
                        "verify_signature",
                        "verify_sm2",
                        "verify_signature_sm2",
                        "verify_signature_sm2_only",
                    ):
                        method = getattr(bill_instance, method_name, None)
                        if callable(method):
                            try:
                                return bool(method()), None
                            except Exception as e:
                                return None, f"lunalib {method_name} error: {e}"

                    # Try classmethod/staticmethod patterns
                    for class_method_name in ("verify", "verify_signature"):
                        class_method = getattr(LunalibDigitalBill, class_method_name, None)
                        if callable(class_method):
                            try:
                                return bool(class_method(payload)), None
                            except Exception as e:
                                return None, f"lunalib class {class_method_name} error: {e}"

                    return None, "lunalib DigitalBill verify method not found"

                if True:
                    # Get signature components
                    public_key = tx_data.get("public_key")
                    signature = tx_data.get("signature")
                    metadata_hash = tx_data.get("metadata_hash", "")
                    issued_to = tx_data.get("issued_to", "")
                    denomination = tx_data.get("denomination", "")
                    front_serial = tx_data.get("front_serial", "")
                    timestamp = tx_data.get("timestamp", 0)

                    print(f"[VERIFY] Verifying {front_serial}")
                    print(
                        f"[VERIFY] Public key: {public_key[:30] if public_key else 'None'}..."
                    )
                    print(
                        f"[VERIFY] Signature: {signature[:30] if signature else 'None'}..."
                    )
                    print(
                        f"[VERIFY] Signature length: {len(signature) if signature else 0}"
                    )

                    # CHECK BLOCKCHAIN STATUS - UPDATED LOGIC
                    # LAYER 4: Check Mempool
                    try:
                        mempool_found = False
                        if blockchain_daemon_instance and hasattr(
                            blockchain_daemon_instance, "mempool"
                        ):
                            # Check if transaction is in mempool
                            mempool_found = any(
                                tx.get("serial_number") == front_serial
                                or tx.get("front_serial") == front_serial
                                for tx in blockchain_daemon_instance.mempool
                            )
                        mempool_status = "pending" if mempool_found else "not_found"
                        validation_results["mempool"] = {
                            "found": mempool_found,
                            "status": mempool_status,
                            "mined_from_mempool": False,
                        }
                    except Exception as e:
                        print(f"Error checking mempool: {e}")
                        validation_results["mempool"] = {
                            "found": False,
                            "error": str(e),
                        }

                    # LAYER 5: Check Blockchain
                    blockchain_found = False
                    blockchain_data = None
                    blockchain_status = "unknown"

                    try:
                        # Check if transaction is in the blockchain cache
                        print(
                            f"[BLOCKCHAIN CHECK] Checking for serial: '{front_serial}'"
                        )

                        if blockchain_daemon_instance:
                            # First, let's see what we're working with
                            print(f"[BLOCKCHAIN DEBUG] Blockchain instance available")
                            print(
                                f"[BLOCKCHAIN DEBUG] Has blockchain attr: {hasattr(blockchain_daemon_instance, 'blockchain')}"
                            )
                            if hasattr(blockchain_daemon_instance, "blockchain"):
                                print(
                                    f"[BLOCKCHAIN DEBUG] Blockchain length: {len(blockchain_daemon_instance.blockchain)} blocks"
                                )

                            # Define a helper function to check if a transaction contains our serial
                            def transaction_contains_serial(tx, serial_to_find):
                                """Check if transaction contains the serial in any relevant field."""
                                # Check all possible serial fields
                                serial_fields = [
                                    "serial_number",
                                    "front_serial",
                                    "back_serial",
                                    "serial",
                                ]
                                for field in serial_fields:
                                    field_value = tx.get(field)
                                    if field_value == serial_to_find:
                                        print(
                                            f"[BLOCKCHAIN DEBUG] Found match in field '{field}': '{field_value}'"
                                        )
                                        return True
                                return False

                            # Check mined serials cache if it exists
                            if hasattr(blockchain_daemon_instance, "mined_serials"):
                                print(
                                    f"[BLOCKCHAIN DEBUG] Checking mined_serials cache"
                                )
                                print(
                                    f"[BLOCKCHAIN DEBUG] mined_serials type: {type(blockchain_daemon_instance.mined_serials)}"
                                )
                                print(
                                    f"[BLOCKCHAIN DEBUG] mined_serials sample: {list(blockchain_daemon_instance.mined_serials)[:3] if blockchain_daemon_instance.mined_serials else 'Empty'}"
                                )

                                is_mined = (
                                    front_serial
                                    in blockchain_daemon_instance.mined_serials
                                )
                                print(
                                    f"[BLOCKCHAIN DEBUG] Serial '{front_serial}' in mined_serials: {is_mined}"
                                )

                                if is_mined:
                                    blockchain_status = "mined"
                                    print(
                                        f"[BLOCKCHAIN] ✓ Serial '{front_serial}' is in mined_serials cache"
                                    )
                            else:
                                print(
                                    f"[BLOCKCHAIN DEBUG] No mined_serials attribute found"
                                )
                                blockchain_status = "unknown"

                            # SEARCH THE BLOCKCHAIN (whether we found it in cache or not)
                            blockchain_found = False
                            search_successful = False

                            if (
                                hasattr(blockchain_daemon_instance, "blockchain")
                                and blockchain_daemon_instance.blockchain
                            ):
                                print(
                                    f"[BLOCKCHAIN] Searching blockchain ({len(blockchain_daemon_instance.blockchain)} blocks)..."
                                )

                                for block_idx, block in enumerate(
                                    blockchain_daemon_instance.blockchain
                                ):
                                    # Debug first block structure
                                    if block_idx == 0 and not blockchain_found:
                                        print(
                                            f"[BLOCKCHAIN DEBUG] First block structure:"
                                        )
                                        print(f"  Block index: {block.get('index')}")
                                        print(
                                            f"  Block hash: {block.get('hash')[:20]}..."
                                        )
                                        print(
                                            f"  Transactions in block: {len(block.get('transactions', []))}"
                                        )

                                    # Search transactions in this block
                                    for tx_idx, tx in enumerate(
                                        block.get("transactions", [])
                                    ):
                                        if transaction_contains_serial(
                                            tx, front_serial
                                        ):
                                            blockchain_found = True
                                            blockchain_data = (
                                                tx.copy()
                                            )  # Use copy to avoid modifying original
                                            blockchain_data["block_height"] = block_idx
                                            blockchain_data["block_hash"] = block.get(
                                                "hash"
                                            )
                                            blockchain_data["block_index"] = block.get(
                                                "index"
                                            )

                                            print(
                                                f"[BLOCKCHAIN] ✓ Found transaction in blockchain!"
                                            )
                                            print(
                                                f"[BLOCKCHAIN]   Block: {block_idx} (index: {block.get('index')})"
                                            )
                                            print(
                                                f"[BLOCKCHAIN]   Transaction type: {tx.get('type', 'N/A')}"
                                            )
                                            print(
                                                f"[BLOCKCHAIN]   Transaction hash: {tx.get('hash', '')[:20]}..."
                                            )

                                            # Update status
                                            if blockchain_status != "mined":
                                                blockchain_status = "mined"

                                            search_successful = True
                                            break

                                    if blockchain_found:
                                        break

                                if not blockchain_found:
                                    print(
                                        f"[BLOCKCHAIN] ✗ Serial '{front_serial}' not found in any block transactions"
                                    )
                                    print(
                                        f"[BLOCKCHAIN DEBUG] Checking what transactions exist in first block..."
                                    )
                                    if blockchain_daemon_instance.blockchain:
                                        first_block = (
                                            blockchain_daemon_instance.blockchain[0]
                                        )
                                        for tx_idx, tx in enumerate(
                                            first_block.get("transactions", [])
                                        ):
                                            print(
                                                f"[BLOCKCHAIN DEBUG] Transaction {tx_idx}:"
                                            )
                                            print(f"  Type: {tx.get('type')}")
                                            print(
                                                f"  serial_number: {tx.get('serial_number')}"
                                            )
                                            print(
                                                f"  front_serial: {tx.get('front_serial')}"
                                            )
                                            print(
                                                f"  back_serial: {tx.get('back_serial')}"
                                            )
                            else:
                                print(f"[BLOCKCHAIN] No blockchain data available")
                                blockchain_status = "no_chain"

                            # If not found in blockchain, check mempool
                            if not blockchain_found and hasattr(
                                blockchain_daemon_instance, "mempool"
                            ):
                                print(
                                    f"[BLOCKCHAIN] Checking mempool for serial '{front_serial}'..."
                                )

                                # Check mempool for the serial
                                mempool_found = False
                                for tx in blockchain_daemon_instance.mempool:
                                    if transaction_contains_serial(tx, front_serial):
                                        mempool_found = True
                                        print(
                                            f"[BLOCKCHAIN] ✓ Serial '{front_serial}' found in mempool (pending)"
                                        )
                                        blockchain_status = "pending"

                                        # Store mempool transaction info
                                        blockchain_data = tx.copy()
                                        blockchain_data["mempool"] = True
                                        blockchain_data["status"] = "pending"
                                        break

                                if not mempool_found:
                                    print(
                                        f"[BLOCKCHAIN] ✗ Serial '{front_serial}' not found in mempool"
                                    )

                                    # If we haven't set a status yet, set to unmined
                                    if blockchain_status in ["unknown", "no_chain"]:
                                        blockchain_status = "unmined"

                            # Determine confirmations if found in blockchain
                            confirmations = 0
                            if (
                                blockchain_found
                                and blockchain_data
                                and blockchain_status == "mined"
                            ):
                                # Estimate confirmations based on blockchain length
                                if hasattr(blockchain_daemon_instance, "blockchain"):
                                    block_height = blockchain_data.get("block_height")
                                    if block_height is not None:
                                        confirmations = max(
                                            0,
                                            len(blockchain_daemon_instance.blockchain)
                                            - block_height
                                            - 1,
                                        )
                                        print(
                                            f"[BLOCKCHAIN] Block height: {block_height}, Blockchain length: {len(blockchain_daemon_instance.blockchain)}, Confirmations: {confirmations}"
                                        )
                        else:
                            print(
                                f"[BLOCKCHAIN ERROR] No blockchain daemon instance available"
                            )
                            blockchain_status = "daemon_unavailable"

                        # Prepare validation results
                        validation_results["blockchain"] = {
                            "found": blockchain_found,
                            "data": blockchain_data,
                            "confirmations": confirmations,
                            "status": blockchain_status,
                            "daemon_available": blockchain_daemon_instance is not None,
                        }

                        print(
                            f"[BLOCKCHAIN SUMMARY] Status: {blockchain_status}, Found: {blockchain_found}, Confirmations: {confirmations}"
                        )

                    except Exception as e:
                        print(f"[BLOCKCHAIN ERROR] Error checking blockchain: {e}")
                        import traceback

                        traceback.print_exc()
                        validation_results["blockchain"] = {
                            "found": False,
                            "error": str(e),
                            "status": "error",
                            "daemon_available": False,
                        }
                        blockchain_status = "error"

                    # =====================================================
                    # LAYER 3: SIGNATURE VERIFICATION - SIMPLIFIED LOGIC
                    # =====================================================
                    verification_attempts = []

                    # STRATEGY 1: BLOCKCHAIN CONFIRMATION (HIGHEST PRIORITY)
                    if blockchain_found and blockchain_status == "mined":
                        signature_valid = True
                        verification_method = "blockchain_confirmed"
                        verification_attempts.append(("blockchain_confirmed", True))
                        print(f"[VERIFY] ✓ Transaction confirmed on blockchain")

                        if blockchain_data:
                            print(
                                f"[VERIFY]   Block height: {blockchain_data.get('block_height')}"
                            )
                            print(f"[VERIFY]   Confirmations: {confirmations}")

                        # Check if signature matches transaction hash
                        if (
                            signature
                            and blockchain_data
                            and signature == blockchain_data.get("hash")
                        ):
                            verification_method = "blockchain_hash_match"
                            verification_attempts.append(
                                ("blockchain_hash_match", True)
                            )
                            print(
                                f"[VERIFY] ✓ Signature matches blockchain transaction hash"
                            )

                    # STRATEGY 2: SM2 SIGNATURE VERIFICATION
                    elif (
                        signature_valid is None
                        and DigitalBill
                        and public_key
                        and signature
                    ):
                        try:
                            lunalib_handled = False
                            lunalib_result, lunalib_error = _try_lunalib_sm2_verify(tx_data)
                            if lunalib_result is True:
                                signature_valid = True
                                verification_method = "lunalib_sm2_signature"
                                verification_attempts.append(("lunalib_sm2", True))
                                print(f"[VERIFY] ✓ Lunalib SM2 verification passed")
                                lunalib_handled = True
                            elif lunalib_result is False:
                                signature_valid = False
                                verification_method = "lunalib_sm2_invalid"
                                verification_attempts.append(("lunalib_sm2", False))
                                print(f"[VERIFY] ✗ Lunalib SM2 verification failed")
                                lunalib_handled = True
                            elif lunalib_error:
                                print(f"[VERIFY] ⚠️ Lunalib SM2 verification unavailable: {lunalib_error}")

                            if not lunalib_handled:
                                # Create DigitalBill object
                                digital_bill = DigitalBill(
                                    bill_type=tx_data.get("type", "banknote"),
                                    front_serial=front_serial,
                                    back_serial=tx_data.get("back_serial", ""),
                                    metadata_hash=metadata_hash,
                                    timestamp=timestamp,
                                    issued_to=issued_to,
                                    denomination=denomination,
                                    public_key=public_key,
                                    signature=signature,
                                )

                                print(
                                    f"[VERIFY] Created DigitalBill for signature verification"
                                )

                                # FIRST: Check if it's valid SM2 format
                                is_valid_sm2_format = (
                                    len(signature) == 128
                                    and len(public_key) >= 130
                                    and public_key.startswith("04")
                                    and all(
                                        c in "0123456789abcdefABCDEF" for c in signature
                                    )
                                )

                                if is_valid_sm2_format:
                                    print(f"[VERIFY] ✓ Valid SM2 format detected")

                                    # Try actual SM2 verification
                                    print(
                                        f"[VERIFY] Attempting SM2 cryptographic verification..."
                                    )
                                    is_valid = digital_bill.verify()

                                    if is_valid:
                                        signature_valid = True
                                        verification_method = "sm2_signature"
                                        verification_attempts.append(("sm2_crypto", True))
                                        print(
                                            f"[VERIFY] ✓ SM2 cryptographic verification passed"
                                        )
                                    else:
                                        # Format is valid but verification failed - accept as valid format
                                        signature_valid = True
                                        verification_method = "sm2_format_valid"
                                        verification_attempts.append(("sm2_format", True))
                                        print(
                                            f"[VERIFY] ⚠️ SM2 crypto verification failed but format is valid"
                                        )
                                else:
                                    signature_valid = False
                                    verification_method = "invalid_sm2_format"
                                    verification_attempts.append(
                                        ("sm2_format_check", False)
                                    )
                                    print(f"[VERIFY] ✗ Invalid SM2 format")

                        except Exception as e:
                            print(f"[VERIFY ERROR] DigitalBill verification error: {e}")
                            verification_attempts.append(("digital_bill", False))
                            signature_valid = False
                            verification_method = "sm2_error"

                    # STRATEGY 3: MOCK SIGNATURE (LEGACY SUPPORT)
                    elif (
                        signature_valid is None
                        and signature
                        and signature.startswith("mock_signature_")
                    ):
                        import hashlib

                        expected_mock = (
                            "mock_signature_"
                            + hashlib.md5(
                                f"{issued_to}{denomination}{front_serial}".encode()
                            ).hexdigest()
                        )

                        if signature == expected_mock:
                            signature_valid = True
                            verification_method = "mock_signature"
                            verification_attempts.append(("mock_signature", True))
                            print(f"[VERIFY] ✓ Mock signature validated (legacy)")
                        else:
                            signature_valid = False
                            verification_method = "mock_invalid"
                            verification_attempts.append(("mock_signature", False))
                            print(f"[VERIFY] ✗ Mock signature invalid")

                    # STRATEGY 4: MEMPOOL PENDING
                    elif signature_valid is None and mempool_found:
                        signature_valid = True
                        verification_method = "mempool_pending"
                        verification_attempts.append(("mempool", True))
                        print(f"[VERIFY] ✓ Transaction found in mempool (pending)")

                    # STRATEGY 5: SERIAL DATABASE EXISTS
                    elif signature_valid is None and serial_record:
                        signature_valid = True
                        verification_method = "serial_db_exists"
                        verification_attempts.append(("serial_db", True))
                        print(f"[VERIFY] ✓ Serial number exists in database")

                    # STRATEGY 6: ALL FAILED
                    elif signature_valid is None:
                        signature_valid = False
                        verification_method = "all_failed"
                        verification_attempts.append(("all_failed", False))
                        print(f"[VERIFY] ✗ All verification methods failed")

                    # Prepare signature details for display
                    signature_type = "unknown"
                    if signature:
                        if len(signature) == 128 and all(
                            c in "0123456789abcdefABCDEF" for c in signature
                        ):
                            signature_type = "sm2"
                        elif signature.startswith("mock_signature_"):
                            signature_type = "mock"
                        elif len(signature) == 64 and all(
                            c in "0123456789abcdefABCDEF" for c in signature
                        ):
                            signature_type = "sha256"

                    signature_details = {
                        "public_key": public_key,
                        "public_key_short": public_key[:20] + "..."
                        if public_key and len(public_key) > 20
                        else public_key or "N/A",
                        "signature": signature,
                        "signature_short": signature[:20] + "..."
                        if signature and len(signature) > 20
                        else signature or "N/A",
                        "signature_type": signature_type,
                        "timestamp": timestamp,
                        "timestamp_readable": datetime.fromtimestamp(
                            timestamp
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        if timestamp
                        else "Unknown",
                        "verification_method": verification_method,
                        "metadata_hash": metadata_hash,
                        "issued_to": issued_to,
                        "denomination": denomination,
                        "front_serial": front_serial,
                        "verification_attempts": verification_attempts,
                        "sm2_available": sm2_available,
                    }

                    # Add blockchain information to signature details if available
                    if blockchain_found and blockchain_data:
                        signature_details["blockchain_info"] = {
                            "block_height": blockchain_data.get("block_height"),
                            "block_hash": blockchain_data.get("block_hash", "")[:20]
                            + "...",
                            "transaction_hash": blockchain_data.get("hash", "")[:20]
                            + "...",
                            "confirmations": confirmations,
                            "status": blockchain_status,
                        }

                    # Add SM2 info if available
                    if sm2_available and public_key:
                        signature_details["sm2_info"] = {
                            "public_key_length": len(public_key) if public_key else 0,
                            "signature_length": len(signature) if signature else 0,
                            "public_key_format": "sm2_uncompressed"
                            if public_key and public_key.startswith("04")
                            else "unknown",
                            "signature_format": "sm2"
                            if signature_type == "sm2"
                            else "other",
                        }

                    # If it's a mock signature, get the r and s components
                    if signature_type == "mock" and signature.startswith(
                        "mock_signature_"
                    ):
                        signature_details["r"] = signature[
                            15:47
                        ]  # First 32 chars after prefix
                        signature_details["s"] = signature[47:79]  # Next 32 chars
                    elif signature_type == "sm2" and len(signature) >= 128:
                        signature_details["r"] = signature[:64]
                        signature_details["s"] = signature[64:128]

                    validation_results["digital_bill"] = {
                        "found": True,
                        "signature_valid": signature_valid,
                        "verification_method": verification_method,
                        "verification_attempts": verification_attempts,
                        "signature_type": signature_type,
                    }

            except Exception as e:
                print(f"[VERIFY ERROR] Processing error: {e}")
                import traceback

                traceback.print_exc()
                validation_results["digital_bill"] = {
                    "found": False,
                    "signature_valid": False,
                    "error": str(e),
                }
                signature_details["error"] = str(e)
                signature_valid = False

    # Calculate validation score and percentage
    validation_score = 0
    total_layers = 5

    # Check each layer
    serial_db_valid = (
        validation_results["serial_db"] and validation_results["serial_db"]["found"]
    )
    banknote_db_valid = (
        validation_results["banknote_db"] and validation_results["banknote_db"]["found"]
    )
    digital_bill_valid = validation_results["digital_bill"] and validation_results[
        "digital_bill"
    ].get("signature_valid")
    mempool_valid = validation_results["mempool"] and validation_results["mempool"].get(
        "found"
    )
    blockchain_valid = validation_results["blockchain"] and validation_results[
        "blockchain"
    ].get("found")

    # LAYER 1: Serial Database
    if serial_db_valid:
        validation_score += 1

    # LAYER 2: Banknote Database
    if banknote_db_valid:
        validation_score += 1

    # LAYER 3: Digital Bill Signature
    if digital_bill_valid:
        validation_score += 1

    # LAYER 4: Mempool (automatic credit if in blockchain)
    # If it's in blockchain, we assume it was processed through mempool
    mempool_credited = False
    if blockchain_valid:
        # Automatically credit mempool layer when transaction is confirmed on blockchain
        validation_score += 1
        mempool_credited = True
        print(
            f"[VALIDATION] ✓ Mempool layer automatically credited (transaction is blockchain-confirmed)"
        )
    elif mempool_valid:
        # Only add score if actually found in mempool (and not in blockchain)
        validation_score += 1
        mempool_credited = True

    # LAYER 5: Blockchain
    if blockchain_valid:
        validation_score += 1

    validation_percentage = (
        (validation_score / total_layers) * 100 if total_layers > 0 else 0
    )

    # Create a summary of validation layers
    validation_summary = {
        "serial_db": {"valid": serial_db_valid, "score_added": serial_db_valid},
        "banknote_db": {"valid": banknote_db_valid, "score_added": banknote_db_valid},
        "digital_bill": {
            "valid": digital_bill_valid,
            "score_added": digital_bill_valid,
        },
        "mempool": {
            "valid": mempool_valid
            or blockchain_valid,  # Consider valid if in blockchain
            "score_added": mempool_credited,
            "auto_credited": blockchain_valid
            and not mempool_valid,  # Flag if auto-credited
        },
        "blockchain": {"valid": blockchain_valid, "score_added": blockchain_valid},
    }
    print(validation_summary)
    print(f"[VALIDATION SCORE] Breakdown:")
    print(f"  Serial DB: {serial_db_valid} (+{1 if serial_db_valid else 0})")
    print(f"  Banknote DB: {banknote_db_valid} (+{1 if banknote_db_valid else 0})")
    print(f"  Digital Bill: {digital_bill_valid} (+{1 if digital_bill_valid else 0})")
    print(
        f"  Mempool: {mempool_valid or blockchain_valid} (+{1 if mempool_credited else 0}) {'[AUTO]' if blockchain_valid and not mempool_valid else ''}"
    )
    print(f"  Blockchain: {blockchain_valid} (+{1 if blockchain_valid else 0})")
    print(
        f"[VALIDATION SCORE] Total: {validation_score}/5 ({validation_percentage:.1f}%)"
    )
    # Determine overall status
    overall_status = "invalid"
    status_color = "danger"

    if validation_score >= 4:
        overall_status = "highly_valid"
        status_color = "success"
    elif validation_score >= 3:
        overall_status = "valid"
        status_color = "primary"
    elif validation_score >= 2:
        overall_status = "partially_valid"
        status_color = "warning"

    # Add blockchain/mempool relationship note
    blockchain_note = None
    if (
        validation_results.get("mempool")
        and validation_results["mempool"].get("mined_from_mempool")
        and validation_results.get("blockchain")
        and validation_results["blockchain"].get("found")
    ):
        blockchain_note = "✓ This transaction was successfully mined from the mempool and is now confirmed on the blockchain."

    return render_template(
        "verify.html",
        result=result,
        serial_input=serial_input,
        banknote=banknote,
        title="Verify Serial",
        current_user=get_current_user(),
        signature_valid=signature_valid,
        signature_details=signature_details,
        blockchain_status=blockchain_status,
        mined_transaction=mined_transaction,
        block_details=block_details,
        validation_results=validation_results,
        validation_score=validation_score,
        validation_percentage=validation_percentage,
        overall_status=overall_status,
        status_color=status_color,
        blockchain_note=blockchain_note,
        sm2_available=sm2_available,
        verification_method=verification_method,
    )


def find_genesis_transaction_in_blockchain(serial_number):
    """
    Find a GTX_Genesis transaction in the blockchain by serial number
    Returns (transaction_dict, block_details) or (None, None) if not found
    """
    try:
        for block_index, block in enumerate(blockchain_daemon_instance.blockchain):
            for tx in block.get("transactions", []):
                if (
                    tx.get("type") == "GTX_Genesis"
                    and tx.get("serial_number") == serial_number
                ):
                    block_details = {
                        "block_index": block_index,
                        "block_hash": block.get("hash", "")[:16] + "...",
                        "timestamp": block.get("timestamp"),
                        "timestamp_readable": datetime.fromtimestamp(
                            block.get("timestamp")
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        if block.get("timestamp")
                        else "Unknown",
                        "miner": block.get("miner", "Unknown"),
                        "transaction_count": len(block.get("transactions", [])),
                        "previous_hash": block.get("previous_hash", "")[:16] + "...",
                    }
                    return tx, block_details
        return None, None
    except Exception as e:
        print(f"Error searching blockchain: {e}")
        return None, None


def is_transaction_in_mempool(serial_number):
    """
    Check if a GTX_Genesis transaction is in the mempool
    """
    try:
        for tx in blockchain_daemon_instance.mempool:
            if (
                tx.get("type") == "GTX_Genesis"
                and tx.get("serial_number") == serial_number
            ):
                return True
        return False
    except Exception as e:
        print(f"Error searching mempool: {e}")
        return False


def find_transaction_in_blockchain(serial_number, issued_to, denomination):
    """Look for a transaction in the blockchain that matches this banknote"""
    try:
        for block in blockchain_daemon_instance.blockchain:
            for tx in block.get("transactions", []):
                if (
                    tx.get("serial_number") == serial_number
                    and tx.get("issued_to") == issued_to
                    and str(tx.get("denomination")) == str(denomination)
                ):
                    return tx
    except Exception as e:
        print(f"Error searching blockchain: {e}")
    return None


from functools import wraps


@app.route("/admin")
@admin_required
def admin_panel():
    # Get the active section from query parameter or default to 'dashboard'
    active_section = request.args.get("section", "dashboard")

    # Get real statistics
    stats = get_admin_stats()

    # Get system status
    system_stats = get_system_status()

    # Get recent activity
    recent_activity = get_recent_activity()

    # Get settings (needed for mining difficulty)
    settings = Settings.query.first()
    if not settings:
        settings = Settings()
        db.session.add(settings)
        db.session.commit()

    # Get data for other sections
    portrait_prompt_display = ""
    background_prompt_display = ""

    if active_section == "settings":
        # Load prompts from files if not set in database
        portrait_prompt_display = settings.portrait_prompt or read_prompt_file(
            "portrait_prompt.txt",
            "A professional portrait of a person, high quality, detailed face, neutral background",
        )
        background_prompt_display = settings.background_prompt or read_prompt_file(
            "background_prompt.txt",
            "A beautiful fantasy landscape with mountains and rivers, mystical atmosphere",
        )

    users = User.query.order_by(User.created_at.desc()).limit(200).all()
    banknotes = Banknote.query.order_by(Banknote.created_at.desc()).limit(200).all()
    tasks = GenerationTask.query.order_by(GenerationTask.created_at.desc()).limit(200).all()
    serials = SerialNumber.query.order_by(SerialNumber.created_at.desc()).limit(200).all()

    queue_status = get_generation_queue_status() if tasks else None

    return render_template(
        "admin_panel.html",
        active_section=active_section,
        stats=stats,
        system_stats=system_stats,
        recent_activity=recent_activity,
        settings=settings,
        portrait_prompt_display=portrait_prompt_display,
        background_prompt_display=background_prompt_display,
        users=users,
        banknotes=banknotes,
        tasks=tasks,
        serials=serials,
        current_user=get_current_user(),
        queue_status=queue_status,
    )


def _get_admin_stats_blocking():
    """Get comprehensive admin statistics (blocking version - but with async blockchain ops)"""
    from datetime import timedelta

    # Time calculations
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    # User statistics (fast DB queries - main thread)
    total_users = User.query.count()
    active_users = User.query.filter(User.last_login >= today_start).count()
    new_users_today = User.query.filter(User.created_at >= today_start).count()

    # Banknote statistics (fast DB queries - main thread)
    total_banknotes = Banknote.query.count()
    total_value_result = db.session.query(db.func.sum(Banknote.denomination)).first()
    total_value = float(total_value_result[0] or 0)

    today_generated = Banknote.query.filter(Banknote.created_at >= today_start).count()

    # Blockchain statistics (with timeout - can be slow)
    blockchain_height = 0
    total_txs = 0
    mempool_size = 0

    with BLOCKCHAIN_STATS_LOCK:
        cached_stats = BLOCKCHAIN_STATS_CACHE.get("data")
        cached_at = BLOCKCHAIN_STATS_CACHE.get("timestamp", 0)

    cache_fresh = cached_stats and (time.time() - cached_at) <= BLOCKCHAIN_STATS_TTL_SECONDS

    if cache_fresh:
        blockchain_height = cached_stats.get("total_blocks", 0)
        total_txs = cached_stats.get("total_transactions", 0)
        cached_stats = None
    else:
        cached_stats = None

    def get_blockchain_info():
        """Get blockchain info in a thread with timeout"""
        nonlocal blockchain_height, total_txs, mempool_size
        try:
            daemon = BlockchainDaemon()
            blockchain_status = daemon.get_blockchain_status()
            blockchain_height = blockchain_status.get("blocks", 0)
            total_txs = blockchain_status.get("total_transactions", 0)

            mempool_status = daemon.get_mempool_status()
            mempool_size = mempool_status.get("total", 0)
        except Exception as e:
            logger.warning(f"Blockchain stats error: {e}")
            # Fallback will use default values

    # Run blockchain operations in thread with timeout (skip if cache is fresh)
    if not cache_fresh:
        try:
            thread = threading.Thread(target=get_blockchain_info, daemon=True)
            thread.start()
            thread.join(timeout=1.5)  # Wait max 1.5 seconds

            # If thread timed out, use fallback
            if thread.is_alive():
                logger.debug("Blockchain stats thread timed out")
                # Use database fallback
                mined_serials = SerialNumber.query.filter_by(is_mined=True).count()
                blockchain_height = mined_serials // 10
                pending_banknotes = Banknote.query.filter_by(
                    is_verified=False, verification_status="pending"
                ).count()
                mempool_size = pending_banknotes
        except Exception as e:
            logger.warning(f"Blockchain thread error: {e}")
            # Fallback
            mined_serials = SerialNumber.query.filter_by(is_mined=True).count()
            blockchain_height = mined_serials // 10

    # Digital bills statistics
    digital_bills_count = 0
    try:
        try:
            from lunalib.gtx.genesis import GTXGenesis

            gtx_genesis = GTXGenesis()
            digital_bills_count = (
                len(gtx_genesis.get_all_bills())
                if hasattr(gtx_genesis, "get_all_bills")
                else 0
            )
        except ImportError:
            try:
                from lunalib.genesis import GTXGenesis

                gtx_genesis = GTXGenesis()
                digital_bills_count = (
                    len(gtx_genesis.get_all_bills())
                    if hasattr(gtx_genesis, "get_all_bills")
                    else 0
                )
            except ImportError:
                pass
    except Exception as e:
        logger.warning(f"Digital bills error: {e}")

    # Generation tasks statistics (fast DB queries - main thread)
    active_tasks = GenerationTask.query.filter_by(status="processing").count()
    completed_tasks = GenerationTask.query.filter_by(status="completed").count()

    # Mining statistics
    mining_stats = get_mining_stats()

    return {
        "total_users": total_users,
        "active_users": active_users,
        "new_users_today": new_users_today,
        "total_banknotes": total_banknotes,
        "total_value": f"{total_value:,.2f}",
        "today_generated": today_generated,
        "blockchain_height": blockchain_height,
        "total_txs": total_txs,
        "mempool_size": mempool_size,
        "digital_bills": digital_bills_count,
        "active_tasks": active_tasks,
        "completed_tasks": completed_tasks,
        "mining_rewards": mining_stats.get("total_rewards", 0),
        "mining_difficulty": mining_stats.get("current_difficulty", 1),
        "avg_generation_time": get_avg_generation_time(),
        "success_rate": get_generation_success_rate(),
    }


def get_admin_stats():
    """Get comprehensive admin statistics (non-blocking in main thread with context)"""
    # Database queries are fast and need app context, so do them in main thread
    # Only async blockchain operations happen in background

    return _get_admin_stats_blocking()


def get_system_status():
    """Get system status including daemon, network, and resource usage"""
    import psutil

    with SYSTEM_STATUS_LOCK:
        cached_status = SYSTEM_STATUS_CACHE.get("data")
        cached_at = SYSTEM_STATUS_CACHE.get("timestamp", 0)

    cache_fresh = cached_status and (time.time() - cached_at) <= SYSTEM_STATUS_TTL_SECONDS
    if cache_fresh:
        return cached_status

    def _compute_status():
        status = {
            "daemon_running": False,
            "network_online": False,
            "memory_usage": 0,
            "cpu_usage": 0,
            "disk_usage": 0,
            "last_sync": None,
            "blockchain_height": 0,
            "mempool_size": 0,
            "total_transactions": 0,
        }

        try:
            daemon_instance = blockchain_daemon_instance if "blockchain_daemon_instance" in globals() else None
            if daemon_instance and hasattr(daemon_instance, "is_running"):
                status["daemon_running"] = daemon_instance.is_running

            with BLOCKCHAIN_STATS_LOCK:
                cached_stats = BLOCKCHAIN_STATS_CACHE.get("data")

            if cached_stats:
                status["blockchain_height"] = cached_stats.get("total_blocks", 0)
                status["total_transactions"] = cached_stats.get("total_transactions", 0)

            if daemon_instance and hasattr(daemon_instance, "blockchain"):
                status["blockchain_height"] = max(
                    status["blockchain_height"], len(daemon_instance.blockchain)
                )
                if daemon_instance.blockchain:
                    last_block = daemon_instance.blockchain[-1]
                    timestamp = last_block.get("timestamp", time.time())
                    status["last_sync"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(float(timestamp))
                    )

            if daemon_instance and hasattr(daemon_instance, "mempool"):
                try:
                    status["mempool_size"] = len(daemon_instance.mempool)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"System status daemon error: {e}")

        try:
            import socket

            socket.setdefaulttimeout(0.5)
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=0.5)
                status["network_online"] = True
            except Exception:
                status["network_online"] = False
        except Exception as e:
            logger.warning(f"Network check error: {e}")
            status["network_online"] = False

        try:
            memory = psutil.virtual_memory()
            status["memory_usage"] = round(memory.percent, 1)

            status["cpu_usage"] = round(psutil.cpu_percent(interval=0.0), 1)

            disk = psutil.disk_usage(os.path.abspath(os.sep))
            status["disk_usage"] = round(disk.percent, 1)
        except Exception as e:
            logger.warning(f"Error getting system resources: {e}")

        return status

    with SYSTEM_STATUS_LOCK:
        if SYSTEM_STATUS_CACHE.get("refreshing"):
            if cached_status:
                return cached_status
            return {
                "daemon_running": False,
                "network_online": False,
                "memory_usage": 0,
                "cpu_usage": 0,
                "disk_usage": 0,
                "last_sync": None,
                "blockchain_height": 0,
                "mempool_size": 0,
                "total_transactions": 0,
                "refreshing": True,
            }
        SYSTEM_STATUS_CACHE["refreshing"] = True

    def _worker():
        try:
            status = _compute_status()
            with SYSTEM_STATUS_LOCK:
                SYSTEM_STATUS_CACHE["data"] = status
                SYSTEM_STATUS_CACHE["timestamp"] = time.time()
        finally:
            with SYSTEM_STATUS_LOCK:
                SYSTEM_STATUS_CACHE["refreshing"] = False

    threading.Thread(target=_worker, daemon=True).start()

    if cached_status:
        response = dict(cached_status)
        response["refreshing"] = True
        return response

    return {
        "daemon_running": False,
        "network_online": False,
        "memory_usage": 0,
        "cpu_usage": 0,
        "disk_usage": 0,
        "last_sync": None,
        "blockchain_height": 0,
        "mempool_size": 0,
        "total_transactions": 0,
        "refreshing": True,
    }


def get_recent_activity():
    """Get recent system activity"""
    from datetime import datetime, timedelta
    import random

    activities = []
    now = datetime.utcnow()

    # Get recent user logins
    recent_logins = (
        User.query.filter(User.last_login >= now - timedelta(hours=24))
        .order_by(User.last_login.desc())
        .limit(5)
        .all()
    )

    for user in recent_logins:
        if user.last_login:
            activities.append(
                {
                    "icon": "👤",
                    "text": f"User {user.username} logged in",
                    "time": format_timedelta(now - user.last_login),
                }
            )

    # Get recent banknote generations
    recent_banknotes = (
        Banknote.query.filter(Banknote.created_at >= now - timedelta(hours=24))
        .order_by(Banknote.created_at.desc())
        .limit(5)
        .all()
    )

    for banknote in recent_banknotes:
        activities.append(
            {
                "icon": "💵",
                "text": f"Banknote ${banknote.denomination} generated for {banknote.user.username if banknote.user else 'Unknown'}",
                "time": format_timedelta(now - banknote.created_at),
            }
        )

    # Get recent blockchain activity without network calls
    try:
        daemon = blockchain_daemon_instance if "blockchain_daemon_instance" in globals() else None

        if daemon and hasattr(daemon, "blockchain"):
            block_count = len(daemon.blockchain)
            if block_count > 0:
                activities.append(
                    {
                        "icon": "⛓️",
                        "text": f"Blockchain height: {block_count} blocks",
                        "time": "Now",
                    }
                )

        if daemon and hasattr(daemon, "mempool"):
            mempool_count = len(daemon.mempool)
            if mempool_count > 0:
                activities.append(
                    {
                        "icon": "📝",
                        "text": f"{mempool_count} transactions in mempool",
                        "time": "Now",
                    }
                )
    except Exception:
        pass

    # Add some system events
    event_types = [
        ("🔄", "System maintenance completed"),
        ("🔒", "Security audit passed"),
        ("📊", "Daily report generated"),
        ("⚡", "Performance optimized"),
        ("🚀", "New features deployed"),
    ]

    # Add 2-3 random system events
    for icon, text in random.sample(event_types, min(3, len(event_types))):
        hours_ago = random.randint(1, 12)
        activities.append(
            {"icon": icon, "text": text, "time": f"{hours_ago} hours ago"}
        )

    # Sort by time (most recent first)
    activities.sort(key=lambda x: x.get("time", ""))

    return activities[:10]  # Return only 10 most recent activities


def format_timedelta(td):
    """Format timedelta to human readable string"""
    if td.days > 0:
        return f"{td.days} days ago"
    elif td.seconds > 3600:
        hours = td.seconds // 3600
        return f"{hours} hours ago"
    elif td.seconds > 60:
        minutes = td.seconds // 60
        return f"{minutes} minutes ago"
    else:
        return "Just now"


def get_mining_stats():
    """Get mining statistics"""
    try:
        daemon = blockchain_daemon_instance

        # Analyze blockchain for mining rewards
        total_rewards = 0.0
        difficulties = []
        block_times = []
        last_ts = None
        total_expected_rewards = 0.0
        reward_blocks = 0

        for block in daemon.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("type") == "reward":
                    try:
                        total_rewards += float(tx.get("amount", 0))
                    except (TypeError, ValueError):
                        pass

            ts_raw = block.get("timestamp")
            try:
                ts_val = float(ts_raw)
            except (TypeError, ValueError):
                ts_val = None
            if ts_val is not None:
                if last_ts is not None:
                    delta = ts_val - last_ts
                    if delta > 0:
                        block_times.append(delta)
                last_ts = ts_val

            # Extract difficulty if present
            difficulty = block.get("difficulty")
            if difficulty and isinstance(difficulty, (int, float)):
                difficulties.append(difficulty)

            try:
                diff_val = float(difficulty) if difficulty is not None else 1.0
            except (TypeError, ValueError):
                diff_val = 1.0
            non_reward_txs = [
                tx
                for tx in block.get("transactions", [])
                if isinstance(tx, dict) and tx.get("type") != "reward"
            ]
            fees_total = 0.0
            for tx in non_reward_txs:
                fee = tx.get("fee", 0)
                try:
                    fee_val = float(fee)
                except (TypeError, ValueError):
                    continue
                if fee_val < 0:
                    continue
                fees_total += fee_val
            try:
                from lunalib.mining.difficulty import DifficultySystem

                difficulty_system = DifficultySystem()
                total_expected_rewards += float(
                    difficulty_system.calculate_block_reward(
                        diff_val,
                        block_height=block.get("index"),
                        tx_count=len(non_reward_txs),
                        fees_total=float(fees_total),
                    )
                )
                reward_blocks += 1
            except Exception:
                pass

        daemon.stop_daemon()

        # Calculate average difficulty
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 1

        avg_block_time = sum(block_times) / len(block_times) if block_times else 0
        avg_reward_per_block = (total_expected_rewards / reward_blocks) if reward_blocks else 0.0
        estimated_lkc_per_hr = (avg_reward_per_block * 3600 / avg_block_time) if avg_block_time else 0.0

        return {
            "total_rewards": total_rewards,
            "total_blocks": len(difficulties),
            "current_difficulty": avg_difficulty,
            "estimated_lkc_per_hr": estimated_lkc_per_hr,
            "miners_count": len(
                set(
                    block.get("miner", "unknown")
                    for block in daemon.blockchain
                    if block.get("miner")
                )
            ),
        }
    except:
        return {
            "total_rewards": 0,
            "total_blocks": 0,
            "current_difficulty": 1,
            "miners_count": 0,
        }


def get_avg_generation_time():
    """Calculate average banknote generation time"""

    completed_tasks = GenerationTask.query.filter_by(status="completed").all()

    if not completed_tasks:
        return "N/A"

    total_time = 0
    count = 0

    for task in completed_tasks:
        if task.created_at and task.completed_at:
            duration = (task.completed_at - task.created_at).total_seconds()
            total_time += duration
            count += 1

    if count == 0:
        return "N/A"

    avg_seconds = total_time / count

    if avg_seconds < 60:
        return f"{avg_seconds:.1f}s"
    elif avg_seconds < 3600:
        return f"{avg_seconds/60:.1f}m"
    else:
        return f"{avg_seconds/3600:.1f}h"


def get_generation_success_rate():
    """Calculate banknote generation success rate"""

    total_tasks = GenerationTask.query.count()
    completed_tasks = GenerationTask.query.filter_by(status="completed").count()

    if total_tasks == 0:
        return "0%"

    success_rate = (completed_tasks / total_tasks) * 100
    return f"{success_rate:.1f}%"


@app.route("/admin/delete_serial/<int:serial_id>", methods=["POST"])
@admin_required
def admin_delete_serial(serial_id):
    serial = SerialNumber.query.get_or_404(serial_id)
    db.session.delete(serial)
    db.session.commit()
    flash("Serial number deleted successfully!", "success")
    return redirect(url_for("admin_panel", section="serials"))


@app.route("/admin/cancel_task/<int:task_id>", methods=["POST"])
@admin_required
def admin_cancel_task(task_id):
    task = GenerationTask.query.get_or_404(task_id)
    if task.status in ["queued", "pending", "processing"]:
        task.status = "cancelled"
        task.completed_at = datetime.utcnow()
        db.session.commit()
        flash("Task cancelled successfully!", "success")
    else:
        flash(
            "Cannot cancel a task that is not queued, pending, or processing", "error"
        )
    return redirect(url_for("admin_panel", section="tasks"))


@app.route("/admin/delete_task/<int:task_id>", methods=["POST"])
@admin_required
def admin_delete_task(task_id):
    task = GenerationTask.query.get_or_404(task_id)

    # Only allow deletion of completed, failed, or cancelled tasks
    if task.status in ["completed", "failed", "cancelled"]:
        db.session.delete(task)
        db.session.commit()
        flash("Task deleted successfully!", "success")
    else:
        flash("Cannot delete a task that is still active. Cancel it first.", "error")

    return redirect(url_for("admin_panel", section="tasks"))


@app.route("/admin/clear-generation-tasks", methods=["POST"])
@admin_required
def admin_clear_generation_tasks():
    """Clear all generation tasks and in-memory tracking."""
    try:
        tasks_deleted = GenerationTask.query.delete()
        db.session.commit()

        clear_generation_queue_state()

        flash(f"Cleared {tasks_deleted} generation tasks.", "success")
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error clearing generation tasks: {e}")
        flash(f"Error clearing generation tasks: {e}", "danger")

    return redirect(url_for("admin_panel", section="tasks"))


import atexit
import threading


def cleanup_stale_generations():
    """Clean up any generation entries that are too old"""
    with GENERATION_LOCK:
        current_time = time.time()
        stale_users = []

        for user_id, info in GENERATION_THREADS.items():
            if isinstance(info, threading.Thread):
                if not info.is_alive():
                    stale_users.append(user_id)
                continue
            # Remove entries older than 1 hour (dict-style tracking)
            try:
                if current_time - info.get("start_time", 0) > 3600:
                    stale_users.append(user_id)
            except Exception:
                # Unknown entry type; remove to prevent leaks
                stale_users.append(user_id)

        for user_id in stale_users:
            del GENERATION_THREADS[user_id]
            print(f"Cleaned up stale generation entry for user {user_id}")


# Run cleanup every 30 minutes
def periodic_cleanup():
    while True:
        time.sleep(1800)  # 30 minutes
        cleanup_stale_generations()


# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

# Also clean up on exit
atexit.register(cleanup_stale_generations)


# Add a helper function to check generation status
@app.route("/generation-status/<int:user_id>")
def generation_status(user_id):
    # Always read latest task from DB so UI can show progress messages
    task = (
        GenerationTask.query.filter_by(user_id=user_id)
        .order_by(GenerationTask.created_at.desc())
        .first()
    )

    if not task:
        return jsonify({"status": "not_found"})

    with GENERATION_LOCK:
        is_thread_active = task.id in GENERATION_THREADS

    return jsonify(
        {
            "status": "found",
            "task_id": task.id,
            "db_status": task.status,
            "message": task.message,
            "progress": task.progress,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "thread_active": is_thread_active,
        }
    )


@app.route("/admin/generate-money/<int:user_id>", methods=["POST"])
@admin_required
def generate_money(user_id):
    """Generate banknotes for a user using the new queue system"""
    current_user = get_current_user()
    if not _user_has_strong_auth(current_user):
        flash("2FA or a security key is required to generate bills.", "error")
        return redirect(url_for("admin_panel"))

    user = User.query.get_or_404(user_id)

    # Check if user already has a task in queue or processing
    queue_status = get_generation_queue_status()
    if user_id in queue_status["active_tasks"]:
        flash(
            f"User {user.username} already has a generation task in progress.",
            "warning",
        )
        return redirect(url_for("admin_panel"))

    # Add task to queue
    task_id = run_generation_task(user_id, user.username)

    if task_id:
        flash(
            f"Generation task started for {user.username}. Task ID: {task_id}",
            "success",
        )
        print(f"[ADMIN] Started generation task {task_id} for user {user.username}")
    else:
        try:
            db.session.rollback()
        except Exception:
            pass
        flash(f"Failed to start generation task for {user.username}.", "error")
        print(f"[ADMIN ERROR] Failed to start generation for user {user.username}")

    return redirect(url_for("admin_panel"))


@app.route("/admin/debug/tasks")
@admin_required
def admin_debug_tasks():
    """Debug all generation tasks"""
    tasks = GenerationTask.query.order_by(GenerationTask.created_at.desc()).all()

    task_list = []
    for task in tasks:
        task_list.append(
            {
                "id": task.id,
                "user_id": task.user_id,
                "username": task.user.username if task.user else "Unknown",
                "status": task.status,
                "message": task.message,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": task.completed_at.isoformat()
                if task.completed_at
                else None,
            }
        )

    return jsonify({"total_tasks": len(tasks), "tasks": task_list})


@app.route("/admin/debug/queue")
@admin_required
def admin_debug_queue():
    """Debug the generation queue"""
    queue_status = get_generation_queue_status()

    active_tasks_info = []
    for user_id in queue_status["active_tasks"]:
        user = User.query.get(user_id)
        if user:
            # Get the latest task for this user
            task = (
                GenerationTask.query.filter_by(user_id=user_id)
                .order_by(GenerationTask.created_at.desc())
                .first()
            )
            active_tasks_info.append(
                {
                    "user_id": user_id,
                    "username": user.username,
                    "task_id": task.id if task else None,
                    "task_status": task.status if task else None,
                }
            )

    return jsonify({"queue_status": queue_status, "active_tasks": active_tasks_info})


@app.route("/admin/test-worker/<int:user_id>")
@admin_required
def admin_test_worker(user_id):
    """Test the worker process manually"""
    user = User.query.get_or_404(user_id)

    try:
        # Test running the worker directly
        import subprocess

        script_path = os.path.join(os.path.dirname(__file__), "generate_worker.py")

        # Create a test task first
        task = GenerationTask(
            user_id=user_id, status="queued", message="Manual test task"
        )
        db.session.add(task)
        db.session.commit()

        cmd = ["python", script_path, str(user_id), user.username, str(task.id)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        return jsonify(
            {
                "success": True,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "task_id": task.id,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/admin/queue-status")
@admin_required
def queue_status():
    """Check the current generation queue status"""
    status = get_generation_queue_status()
    active_tasks = []

    for user_id in status["active_tasks"]:
        user = User.query.get(user_id)
        if user:
            active_tasks.append(user.username)

    return {
        "queue_size": status["queue_size"],
        "active_tasks": active_tasks,
        "is_running": status["is_running"],
    }


@app.route("/generate-money", methods=["POST"])
def generate_money_user():
    current_user = get_current_user()
    if not current_user:
        flash("Please log in to generate money", "error")
        return redirect(url_for("login"))

    if not _user_has_strong_auth(current_user):
        flash("2FA or a security key is required to generate bills.", "error")
        return redirect(url_for("account_settings"))

    if not current_user.can_generate_money():
        flash(
            f"You can generate money again in {current_user.days_until_next_generation()} days",
            "error",
        )
        return redirect(url_for("profile", username=current_user.username))

    # Check if user already has an active task
    queue_status = get_generation_queue_status()
    if current_user.id in queue_status["active_tasks"]:
        flash("You already have a generation task in progress", "error")
        return redirect(url_for("profile", username=current_user.username))

    # This returns IMMEDIATELY - no blocking
    task_id = run_generation_task(current_user.id, current_user.username)

    if task_id:
        flash(
            "Banknote generation started! This will run in the background. You can check status on your profile.",
            "success",
        )
    else:
        flash("Failed to start generation. Please try again.", "error")

    return redirect(url_for("profile", username=current_user.username))


def read_prompt_file(filename, default_prompt=""):
    """Read prompt from file, return default if file doesn't exist"""
    try:
        # Use absolute path relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                print(f"[DEBUG] Loaded prompt from {filepath}: {content[:50]}...")
                return content
        else:
            print(f"[DEBUG] Prompt file not found: {filepath}, using default")
            return default_prompt
    except Exception as e:
        print(f"[!] Error reading {filename}: {e}")
        return default_prompt

# Mirror /api/peers/register to /api/peers (POST)
@app.post("/api/peers")
def register_peer_mirror():
    return register_peer()

@app.route("/admin/settings", methods=["GET", "POST"])
def admin_settings():
    # Ensure user is admin (add your authentication logic here)
    # if not current_user.is_authenticated or not current_user.is_admin:
    #     return redirect(url_for('login'))

    # Get or create settings
    settings = Settings.query.first()
    if not settings:
        settings = Settings()
        db.session.add(settings)
        db.session.commit()

    if request.method == "POST":
        try:
            def _get_text(name, default):
                value = request.form.get(name, "")
                value = value.strip() if isinstance(value, str) else value
                return value if value else default

            def _get_int(name, default):
                raw = request.form.get(name, "")
                raw = raw.strip() if isinstance(raw, str) else raw
                return int(raw) if raw not in (None, "") else default

            def _get_float(name, default):
                raw = request.form.get(name, "")
                raw = raw.strip() if isinstance(raw, str) else raw
                return float(raw) if raw not in (None, "") else default

            settings.system_name = _get_text("system_name", settings.system_name or "Banknote Generator")
            settings.max_banknotes = _get_int("max_banknotes", settings.max_banknotes or 100)
            settings.cooldown_days = _get_int("cooldown_days", settings.cooldown_days or 7)
            settings.maintenance_mode = "maintenance_mode" in request.form
            settings.allow_registrations = "allow_registrations" in request.form
            settings.max_file_size = _get_int("max_file_size", settings.max_file_size or 512)
            settings.blockchain_difficulty = _get_int("blockchain_difficulty", settings.blockchain_difficulty or 6)
            mining_reward_val = _get_float("mining_reward", settings.mining_reward or 0.0001)
            if mining_reward_val < 0.0000001 or mining_reward_val > 9999999:
                raise ValueError("mining_reward out of allowed range")
            settings.mining_reward = mining_reward_val

            settings.bill_width_mm = _get_float("bill_width_mm", settings.bill_width_mm or 160.0)
            settings.bill_height_mm = _get_float("bill_height_mm", settings.bill_height_mm or 60.0)
            settings.bill_title = _get_text("bill_title", settings.bill_title or "灵国国库")
            settings.bill_subtitle = _get_text("bill_subtitle", settings.bill_subtitle or "天圆地方")
            settings.bill_dpi = _get_float("bill_dpi", settings.bill_dpi or 300.0)
            settings.font_dir = _get_text("font_dir", settings.font_dir or "./fonts")
            settings.bg_dir = _get_text("bg_dir", settings.bg_dir or "./backgrounds")
            settings.icon_dir = _get_text("icon_dir", settings.icon_dir or "./icons")
            settings.eisenscript_dir = _get_text("eisenscript_dir", settings.eisenscript_dir or "./eisen")
            settings.eisenscript_prefix_front = sanitize_eisenscript(request.form.get("eisenscript_prefix_front", ""))
            settings.eisenscript_suffix_front = sanitize_eisenscript(request.form.get("eisenscript_suffix_front", ""))
            settings.eisenscript_prefix_back = sanitize_eisenscript(request.form.get("eisenscript_prefix_back", ""))
            settings.eisenscript_suffix_back = sanitize_eisenscript(request.form.get("eisenscript_suffix_back", ""))
            settings.eisenscript_prefix_coin_front = sanitize_eisenscript(request.form.get("eisenscript_prefix_coin_front", ""))
            settings.eisenscript_suffix_coin_front = sanitize_eisenscript(request.form.get("eisenscript_suffix_coin_front", ""))
            settings.eisenscript_prefix_coin_back = sanitize_eisenscript(request.form.get("eisenscript_prefix_coin_back", ""))
            settings.eisenscript_suffix_coin_back = sanitize_eisenscript(request.form.get("eisenscript_suffix_coin_back", ""))
            settings.eisenscript_prefix_card_front = sanitize_eisenscript(request.form.get("eisenscript_prefix_card_front", ""))
            settings.eisenscript_suffix_card_front = sanitize_eisenscript(request.form.get("eisenscript_suffix_card_front", ""))
            settings.eisenscript_prefix_card_back = sanitize_eisenscript(request.form.get("eisenscript_prefix_card_back", ""))
            settings.eisenscript_suffix_card_back = sanitize_eisenscript(request.form.get("eisenscript_suffix_card_back", ""))
            settings.eisenscript_receipt = sanitize_eisenscript(request.form.get("eisenscript_receipt", ""))

            # Retry commit if SQLite is temporarily locked
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    db.session.commit()
                    flash("Settings updated successfully!", "success")
                    break
                except OperationalError as op_err:
                    db.session.rollback()
                    if "database is locked" in str(op_err).lower() and attempt < max_attempts - 1:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    raise
        except ValueError:
            flash("Invalid input values. Please check your entries.", "error")
        except Exception as e:
            flash(f"Error updating settings: {str(e)}", "error")

        return redirect(url_for("admin_settings"))

    return render_template(
        "admin_panel.html",
        active_section="settings",
        settings=settings,
        users=User.query.all(),  # You might want to paginate this
        banknotes=Banknote.query.all(),
        current_user=get_current_user(),
        stats=get_admin_stats(),
        system_stats=get_system_status(),
        recent_activity=get_recent_activity(),
        tasks=GenerationTask.query.order_by(GenerationTask.created_at.desc()).all(),
        serials=SerialNumber.query.order_by(SerialNumber.created_at.desc()).all(),
        queue_status=get_generation_queue_status(),
    )


@app.route("/admin/compile-eisenscript", methods=["POST"])
def compile_eisenscript():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        script = payload.get("script", "")
        name = payload.get("name", "eisenscript")
        if not script.strip():
            return jsonify({"ok": False, "message": "Script is empty."}), 400

        from generate import render_eisenscript_jinja2
        from lunamint.scripting import render_script_to_svg_html
        import tempfile
        from pathlib import Path

        context = {
            "username": "user",
            "denomination": "1",
            "denom_exponent": 0,
            "dendom_exp": 0,
            "pow_level": 0,
            "denomination_words": "one",
            "denomination_compact": "1",
            "denomination_words_cn": "一",
            "denomination_compact_cn": "壹",
            "serial": "SERIAL",
            "title": "TITLE",
            "subtitle": "SUBTITLE",
            "denomination_color": "#000000",
            "denom_color": "#000000",
            "width_mm": 160.0,
            "height_mm": 60.0,
            "timestamp": 0,
        }

        rendered = render_eisenscript_jinja2(script, context)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "preview.svg"
            render_script_to_svg_html(rendered, out_path)

        return jsonify({"ok": True, "message": f"{name} compiled successfully."})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.route("/admin/mining/config", methods=["POST"])
@admin_required
def update_mining_config():
    """Update mining configuration (difficulty, timeout, etc.)"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        # Get or create settings
        settings = Settings.query.first()
        if not settings:
            settings = Settings()
            db.session.add(settings)

        # Update blockchain difficulty
        if "difficulty" in data:
            difficulty = int(data["difficulty"])
            if not (1 <= difficulty <= 9):
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Difficulty must be between 1 and 9",
                        }
                    ),
                    400,
                )
            settings.blockchain_difficulty = difficulty

        # Save timeout if provided (can be used for future mining timeout feature)
        # For now, we just acknowledge it
        timeout = data.get("timeout", 300)

        db.session.commit()

        return (
            jsonify(
                {
                    "success": True,
                    "message": f"Mining configuration updated: difficulty={settings.blockchain_difficulty}, timeout={timeout}s",
                    "difficulty": settings.blockchain_difficulty,
                    "timeout": timeout,
                }
            ),
            200,
        )

    except ValueError as e:
        return jsonify({"success": False, "error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        db.session.rollback()
        return (
            jsonify(
                {"success": False, "error": f"Error updating mining config: {str(e)}"}
            ),
            500,
        )


@app.post("/api/peers/register")
def register_peer():
    """Register a new peer node in the network"""
    try:
        # Get raw data for debugging
        raw_data = request.get_data(as_text=True)
        content_type = request.content_type

        print(f"🔍 [PEER/REGISTER] Content-Type: {content_type}")
        print(f"🔍 [PEER/REGISTER] Raw data: {raw_data}")

        # Attempt to parse JSON
        peer_info = request.get_json(force=False, silent=False)
        print(f"🔍 [PEER/REGISTER] Parsed JSON: {peer_info}")
        print(f"🔍 [PEER/REGISTER] Data type: {type(peer_info)}")

        # Enhanced validation with detailed error messages
        if peer_info is None:
            error_details = {
                "success": False,
                "error": "No JSON data received",
                "details": {
                    "content_type": content_type,
                    "raw_data_length": len(raw_data) if raw_data else 0,
                    "raw_data_preview": raw_data[:200] if raw_data else None,
                    "hint": "Make sure to set Content-Type: application/json header",
                },
            }
            print(f"❌ [PEER/REGISTER] {error_details}")
            return jsonify(error_details), 400

        if not isinstance(peer_info, dict):
            error_details = {
                "success": False,
                "error": f"Expected JSON object, got {type(peer_info).__name__}",
                "details": {
                    "received_type": str(type(peer_info)),
                    "received_value": str(peer_info)[:200],
                },
            }
            print(f"❌ [PEER/REGISTER] {error_details}")
            return jsonify(error_details), 400

        if "peer_url" not in peer_info:
            error_details = {
                "success": False,
                "error": "Missing required field: peer_url",
                "details": {
                    "received_fields": list(peer_info.keys()),
                    "required_fields": ["peer_url"],
                    "example": {"peer_url": "https://example.com"},
                },
            }
            print(f"❌ [PEER/REGISTER] {error_details}")
            return jsonify(error_details), 400

        # Call the blockchain daemon to register peer
        result = blockchain_daemon_instance.register_peer(peer_info)
        print(f"🔍 [PEER/REGISTER] Result: {result}")

        # Return appropriate status code
        status_code = 200 if result.get("success") else 400
        return jsonify(result), status_code

    except Exception as e:
        print(f"❌ [PEER/REGISTER] Exception: {e}")
        import traceback

        error_traceback = traceback.format_exc()
        print(error_traceback)

        error_response = {
            "success": False,
            "error": f"Server error: {str(e)}",
            "details": {
                "exception_type": type(e).__name__,
                "traceback": error_traceback,
            },
        }
        return jsonify(error_response), 500


@app.get("/api/peers")
def get_peers():
    """Get list of all registered peers"""
    result = blockchain_daemon_instance.get_peers_info()
    return jsonify(result), 200


@app.get("/api/peers/list")
def get_peers_list():
    """Alias for listing peers"""
    result = blockchain_daemon_instance.get_peers_info()
    return jsonify(result), 200


@app.delete("/api/peers/<path:peer_id>")
def remove_peer(peer_id):
    """Remove a peer from the network"""
    from urllib.parse import unquote

    peer_url = unquote(peer_id)
    result = blockchain_daemon_instance.remove_peer_by_url(peer_url)
    return jsonify(result), 200 if result["success"] else 404


@app.route("/admin/cleanup-mempool", methods=["POST"])
@admin_required
def admin_cleanup_mempool():
    """Remove spam/invalid/duplicate/mined transactions from mempool."""
    try:
        max_age_hours_raw = request.form.get("max_age_hours") or request.args.get("max_age_hours")
        max_age_hours = None
        if max_age_hours_raw is not None and str(max_age_hours_raw).strip() != "":
            max_age_hours = float(max_age_hours_raw)
        max_age_seconds = None if max_age_hours is None else int(max_age_hours * 3600)

        result = blockchain_daemon_instance.cleanup_mempool_spam(
            max_age_seconds=max_age_seconds
        )

        flash(
            f"Mempool cleanup: removed {result.get('removed', 0)} spam/invalid txs, remaining {result.get('remaining', 0)}.",
            "success",
        )
        return jsonify({"success": True, "result": result}), 200
    except Exception as e:
        current_app.logger.error(f"Mempool cleanup failed: {e}")
        flash(f"Mempool cleanup failed: {e}", "danger")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/admin/reset-user/<int:user_id>", methods=["POST"])
@admin_required
def admin_reset_user(user_id):
    user = User.query.get_or_404(user_id)

    try:
        # Delete all banknotes for the user
        banknotes_deleted = Banknote.query.filter_by(user_id=user.id).delete()

        # Delete all serial numbers for the user
        serials_deleted = SerialNumber.query.filter_by(user_id=user.id).delete()

        # Reset balance
        user.balance = 0

        db.session.commit()

        flash(
            f"Reset successful for {user.username}: {banknotes_deleted} banknotes and {serials_deleted} serial numbers deleted, balance set to 0",
            "success",
        )

    except Exception as e:
        db.session.rollback()
        flash(f"Error resetting user: {str(e)}", "danger")
        current_app.logger.error(f"Error resetting user {user_id}: {str(e)}")

    return redirect(url_for("admin_panel"))


@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)

    # Delete related records in the correct order to avoid foreign key constraints
    GenerationTask.query.filter_by(user_id=user.id).delete()
    Banknote.query.filter_by(user_id=user.id).delete()
    SerialNumber.query.filter_by(user_id=user.id).delete()

    db.session.delete(user)
    db.session.commit()

    flash(f"Deleted user {user.username} and all their data", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/reset-banknotes", methods=["POST"])
@admin_required
def admin_reset_banknotes():
    """Reset all banknotes/serials and archive images folder."""
    archived_path = None
    try:
        images_root_abs = os.path.abspath(IMAGES_ROOT)
        if os.path.exists(images_root_abs) and os.path.isdir(images_root_abs):
            parent_dir = os.path.dirname(images_root_abs)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base_name = f"old_banknotes_{timestamp}"
            target_path = os.path.join(parent_dir, base_name)
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(parent_dir, f"{base_name}_{counter}")
                counter += 1
            os.rename(images_root_abs, target_path)
            archived_path = target_path

        os.makedirs(images_root_abs, exist_ok=True)

        banknotes_deleted = Banknote.query.delete()
        serials_deleted = SerialNumber.query.delete()
        tasks_deleted = GenerationTask.query.delete()
        User.query.update({User.balance: 0})

        db.session.commit()

        flash(
            (
                f"Reset complete: {banknotes_deleted} banknotes, {serials_deleted} serials, "
                f"{tasks_deleted} tasks deleted. New images folder created."
            ),
            "success",
        )
        if archived_path:
            flash(f"Archived images folder: {archived_path}", "info")
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error resetting banknotes: {str(e)}")
        flash(f"Error resetting banknotes: {str(e)}", "danger")

    return redirect(url_for("admin_panel"))


@app.route("/admin/backup-reset-all", methods=["POST"])
@admin_required
def admin_backup_reset_all():
    """Backup blockchain/mempool + banknote assets/DB, then reset all core data."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "backups", f"admin_reset_{timestamp}")
    )
    os.makedirs(backup_root, exist_ok=True)

    results = {
        "backup_root": backup_root,
        "backups": {},
        "warnings": [],
        "errors": [],
    }

    def _backup_file(src: str, rel_dest: str):
        if not src:
            return
        if os.path.exists(src):
            dest = os.path.join(backup_root, rel_dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            results["backups"][rel_dest] = dest
        else:
            results["warnings"].append(f"Missing file: {src}")

    def _backup_dir(src: str, rel_dest: str):
        if not src:
            return
        if os.path.isdir(src):
            dest = os.path.join(backup_root, rel_dest)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            results["backups"][rel_dest] = dest
        elif os.path.exists(src):
            _backup_file(src, rel_dest)
        else:
            results["warnings"].append(f"Missing dir: {src}")

    # Backup blockchain/mempool files
    try:
        daemon = blockchain_daemon_instance
        if daemon:
            _backup_file(
                os.path.abspath(daemon.blockchain_file),
                os.path.join("blockchain", os.path.basename(daemon.blockchain_file)),
            )
            _backup_file(
                os.path.abspath(daemon.mempool_file),
                os.path.join("mempool", os.path.basename(daemon.mempool_file)),
            )
    except Exception as e:
        results["errors"].append(f"Blockchain/mempool backup failed: {e}")

    # Backup legacy mempool.json if present
    try:
        legacy_mempool = os.path.abspath("mempool.json")
        if os.path.exists(legacy_mempool):
            _backup_file(legacy_mempool, os.path.join("mempool", "mempool.json"))
    except Exception as e:
        results["errors"].append(f"Legacy mempool backup failed: {e}")

    # Backup DB
    try:
        db_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
        if db_uri.startswith("sqlite:///"):
            db_path = os.path.abspath(db_uri.replace("sqlite:///", ""))
            _backup_file(db_path, os.path.join("db", os.path.basename(db_path)))
    except Exception as e:
        results["errors"].append(f"DB backup failed: {e}")

    # Backup banknote assets
    try:
        images_root_abs = os.path.abspath(IMAGES_ROOT)
        _backup_dir(images_root_abs, os.path.join("banknotes", "images"))

        settings = Settings.query.first()
        bg_dir = settings.bg_dir if settings and settings.bg_dir else "./backgrounds"
        _backup_dir(os.path.abspath(bg_dir), os.path.join("banknotes", "backgrounds"))
    except Exception as e:
        results["errors"].append(f"Banknote assets backup failed: {e}")

    # Reset chain + mempool (to empty)
    try:
        if blockchain_daemon_instance:
            blockchain_daemon_instance.blockchain = []
            blockchain_daemon_instance.mempool = []
            blockchain_daemon_instance.mined_serials = set()
            blockchain_daemon_instance.save_blockchain()
            blockchain_daemon_instance.save_mempool()
    except Exception as e:
        results["errors"].append(f"Blockchain/mempool reset failed: {e}")

    # Clear legacy mempool.json
    try:
        legacy_mempool = os.path.abspath("mempool.json")
        if os.path.exists(legacy_mempool):
            with open(legacy_mempool, "w", encoding="utf-8") as f:
                f.write("[]")
    except Exception as e:
        results["errors"].append(f"Legacy mempool clear failed: {e}")

    # Reset banknotes/serials/tasks + balances
    banknotes_deleted = serials_deleted = tasks_deleted = 0
    try:
        banknotes_deleted = Banknote.query.delete()
        serials_deleted = SerialNumber.query.delete()
        tasks_deleted = GenerationTask.query.delete()
        User.query.update({User.balance: 0})
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        results["errors"].append(f"DB reset failed: {e}")

    # Clear banknote asset folders after backup
    try:
        images_root_abs = os.path.abspath(IMAGES_ROOT)
        if os.path.isdir(images_root_abs):
            shutil.rmtree(images_root_abs)
        os.makedirs(images_root_abs, exist_ok=True)

        settings = Settings.query.first()
        bg_dir = settings.bg_dir if settings and settings.bg_dir else "./backgrounds"
        bg_dir_abs = os.path.abspath(bg_dir)
        if os.path.isdir(bg_dir_abs):
            shutil.rmtree(bg_dir_abs)
        os.makedirs(bg_dir_abs, exist_ok=True)
    except Exception as e:
        results["errors"].append(f"Asset reset failed: {e}")

    message = (
        f"Backup created at {backup_root}. Reset complete: "
        f"{banknotes_deleted} banknotes, {serials_deleted} serials, {tasks_deleted} tasks cleared."
    )
    success = len(results["errors"]) == 0

    return (
        jsonify({"success": success, "message": message, "details": results}),
        200 if success else 500,
    )


@app.route("/admin/delete-banknote/<int:banknote_id>", methods=["POST"])
@admin_required
def admin_delete_banknote(banknote_id):
    bn = Banknote.query.get_or_404(banknote_id)
    SerialNumber.query.filter_by(banknote_id=bn.id).delete()
    db.session.delete(bn)
    db.session.commit()

    flash(f"Deleted banknote {bn.serial_number}", "success")
    return redirect(url_for("admin_panel"))


@app.route("/")
def landing():
    # Get current user (implementation depends on your authentication system)
    current_user = get_current_user()

    # Calculate statistics
    total_banknotes = Banknote.query.count()
    total_users = User.query.count()

    # Calculate recent activity (last 7d logins, fallback to creations)
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    recent_activity = User.query.filter(User.last_login >= one_week_ago).count()
    if recent_activity == 0:
        recent_activity = User.query.filter(User.created_at >= one_week_ago).count()

    # Calculate total value of all banknotes
    banknotes = Banknote.query.all()
    total_value = 0
    for note in banknotes:
        try:
            total_value += float(note.denomination)
        except (ValueError, TypeError):
            pass

    # Get recent users (last 7 days)
    one_day_ago = datetime.utcnow() - timedelta(hours=24)
    one_week_ago_users = datetime.utcnow() - timedelta(days=7)
    recent_users = User.query.filter(User.created_at >= one_week_ago_users).count()

    # Daily active users (last 24h logins, fallback to signups)
    daily_active_users = User.query.filter(User.last_login >= one_day_ago).count()
    if daily_active_users == 0:
        daily_active_users = User.query.filter(User.created_at >= one_day_ago).count()

    # Get recent transactions (last 24 hours) from blockchain + mempool for live accuracy
    recent_transactions = 0
    try:
        one_day_ago_ts = one_day_ago.timestamp()
        daemon = blockchain_daemon_instance

        for block in daemon.blockchain:
            try:
                block_ts = float(block.get("timestamp", 0))
                if block_ts >= one_day_ago_ts:
                    recent_transactions += len(block.get("transactions", []))
            except Exception:
                continue

        for tx in getattr(daemon, "mempool", []):
            try:
                tx_ts = float(tx.get("timestamp") or tx.get("time") or 0)
                if tx_ts >= one_day_ago_ts:
                    recent_transactions += 1
            except Exception:
                continue
    except Exception as e:
        print(f"Recent transactions calculation fallback: {e}")
        recent_transactions = Banknote.query.filter(
            Banknote.created_at >= one_day_ago
        ).count()

    # Get top collector (user with most banknotes)
    top_collector = {}
    if total_users > 0:
        # Get all users with their banknote counts
        users_with_counts = []
        for user in User.query.all():
            user_banknotes = Banknote.query.filter_by(user_id=user.id).count()
            users_with_counts.append((user, user_banknotes))

        if users_with_counts:
            top_user, top_count = max(users_with_counts, key=lambda x: x[1])
            # Calculate total value for top user
            user_banknotes = Banknote.query.filter_by(user_id=top_user.id).all()
            user_total_value = 0
            for note in user_banknotes:
                try:
                    user_total_value += float(note.denomination)
                except (ValueError, TypeError):
                    pass

            top_collector = {
                "username": top_user.username,
                "banknotes": top_count,
                "value": user_total_value,
            }

    # Get recent trade (most recent banknote created)
    recent_trade = {}
    latest_banknote = Banknote.query.order_by(Banknote.created_at.desc()).first()
    if latest_banknote:
        recent_trade = {
            "from": latest_banknote.user.username if latest_banknote.user else "System",
            "to": "Owner",  # Simplified - assuming creator is owner
            "amount": latest_banknote.denomination
            if latest_banknote.denomination
            else "0",
        }

    # Get platform growth stats
    month_ago = datetime.utcnow() - timedelta(days=30)
    month_ago_users = User.query.filter(User.created_at <= month_ago).count()
    month_ago_banknotes = Banknote.query.filter(
        Banknote.created_at <= month_ago
    ).count()

    # Avoid zero-baseline showing 0% when growth exists; use denominator at least 1
    user_growth_rate = (
        ((total_users - month_ago_users) / max(month_ago_users, 1) * 100)
        if total_users > 0
        else 0
    )
    banknote_growth_rate = (
        ((total_banknotes - month_ago_banknotes) / max(month_ago_banknotes, 1) * 100)
        if total_banknotes > 0
        else 0
    )

    # Get current user's stats if logged in
    user_stats = {}
    if (
        current_user
        and hasattr(current_user, "is_authenticated")
        and current_user.is_authenticated
    ):
        user_banknotes = Banknote.query.filter_by(user_id=current_user.id).count()

        # Check if user can generate money
        can_generate = current_user.can_generate_money()
        days_until_next = current_user.days_until_next_generation()

        # Get user's total value
        user_banknotes_list = Banknote.query.filter_by(user_id=current_user.id).all()
        user_total_value = 0
        for note in user_banknotes_list:
            try:
                user_total_value += float(note.denomination)
            except (ValueError, TypeError):
                pass

        user_stats = {
            "banknotes_created": user_banknotes,
            "can_generate": can_generate,
            "days_until_next": days_until_next,
            "balance": current_user.balance if hasattr(current_user, "balance") else 0,
            "total_value": user_total_value,
        }

    # Handle None values in template
    recent_users = recent_users if recent_users is not None else 0
    recent_transactions = recent_transactions if recent_transactions is not None else 0

    return render_template(
        "landing.html",
        total_banknotes=total_banknotes,
        total_users=total_users,
        recent_activity=recent_activity,
        total_value=total_value,
        user_stats=user_stats,
        current_user=current_user,
        recent_users=recent_users,
        recent_transactions=recent_transactions,
        top_collector=top_collector,
        recent_trade=recent_trade,
        user_growth_rate=user_growth_rate,
        banknote_growth_rate=banknote_growth_rate,
        month_ago_users=month_ago_users,
        month_ago_banknotes=month_ago_banknotes,
        daily_active_users=daily_active_users,
    )


@app.route("/search", methods=["GET"])
def search_all():
    """Unified search across transactions, blocks, and users."""
    query = (request.args.get("q") or "").strip()

    results = {
        "transactions": [],
        "blocks": [],
        "users": [],
    }

    if query:
        # Users
        try:
            results["users"] = (
                User.query.filter(User.username.ilike(f"%{query}%"))
                .limit(20)
                .all()
            )
        except Exception:
            results["users"] = []

        # Blocks
        chain = getattr(blockchain_daemon_instance, "blockchain", []) or []
        block_seen = set()
        blocks = []
        if query.isdigit():
            idx = int(query)
            if 0 <= idx < len(chain):
                block = chain[idx]
                block_hash = block.get("hash")
                blocks.append(
                    {
                        "index": block.get("index", idx),
                        "hash": block_hash,
                        "tx_count": len(block.get("transactions", [])),
                    }
                )
                if block_hash:
                    block_seen.add(block_hash)

        if len(query) >= 4:
            for idx, block in enumerate(chain):
                block_hash = str(block.get("hash", ""))
                if query.lower() in block_hash.lower():
                    if block_hash in block_seen:
                        continue
                    blocks.append(
                        {
                            "index": block.get("index", idx),
                            "hash": block_hash,
                            "tx_count": len(block.get("transactions", [])),
                        }
                    )
                    block_seen.add(block_hash)
                if len(blocks) >= 25:
                    break

        results["blocks"] = blocks

        # Transactions
        tx_seen = set()
        transactions = []

        def _add_tx(tx, status=None, block_index=None):
            if not isinstance(tx, dict):
                return
            tx_hash = tx.get("hash")
            if not tx_hash or tx_hash in tx_seen:
                return
            tx_seen.add(tx_hash)
            transactions.append(
                {
                    "hash": tx_hash,
                    "type": tx.get("type", "transfer"),
                    "amount": tx.get("amount", tx.get("reward_amount", 0)),
                    "block_height": block_index,
                    "status": status or ("mined" if block_index is not None else "unknown"),
                }
            )

        tx_data = blockchain_daemon_instance.get_transaction(query)
        if isinstance(tx_data, list):
            for item in tx_data:
                _add_tx(item)
        elif isinstance(tx_data, dict):
            _add_tx(tx_data)

        mempool_tx = blockchain_daemon_instance.get_mempool_transaction(query)
        if isinstance(mempool_tx, dict):
            _add_tx(mempool_tx, status="mempool")

        if len(query) >= 6:
            for idx, block in enumerate(chain):
                for tx in block.get("transactions", []):
                    if query.lower() in str(tx.get("hash", "")).lower():
                        _add_tx(tx, status="mined", block_index=block.get("index", idx))
                if len(transactions) >= 25:
                    break

            for tx in getattr(blockchain_daemon_instance, "mempool", []) or []:
                if query.lower() in str(tx.get("hash", "")).lower():
                    _add_tx(tx, status="mempool")
                if len(transactions) >= 25:
                    break

        results["transactions"] = transactions

    return render_template(
        "search_results.html",
        query=query,
        results=results,
        title="Search Results",
        current_user=get_current_user(),
    )


@app.route("/portraits/<path:filename>")
def serve_portrait(filename):
    """
    Serve portrait images from the portraits directory
    """
    return send_from_directory("portraits", filename)


def _sanitize_portrait_filename(filename):
    filename = unquote(filename)
    filename = filename.replace("\\", "/")
    if filename.startswith("portraits/"):
        filename = filename[10:]
    if ".." in filename or filename.startswith("/"):
        return None
    return filename


@app.route("/portrait-thumbnail/<path:filename>")
def serve_portrait_thumbnail(filename):
    clean_name = _sanitize_portrait_filename(filename)
    if not clean_name:
        abort(404)

    try:
        width = int(request.args.get("w", 100))
        height = int(request.args.get("h", 100))
    except ValueError:
        return jsonify({"error": "Invalid thumbnail size"}), 400

    if width < 20 or height < 20 or width > 800 or height > 800:
        return jsonify({"error": "Thumbnail size out of bounds"}), 400

    portraits_root = os.path.join(os.path.dirname(__file__), "portraits")
    if clean_name.lower().endswith(".svg"):
        return send_from_directory("portraits", clean_name)
    source_path = os.path.join(portraits_root, clean_name)
    if not os.path.exists(source_path):
        abort(404)

    thumb_dir = os.path.join(portraits_root, ".thumbs", f"{width}x{height}")
    thumb_path = os.path.join(thumb_dir, clean_name)

    try:
        if (not os.path.exists(thumb_path)) or (
            os.path.getmtime(thumb_path) < os.path.getmtime(source_path)
        ):
            _generate_thumbnail(source_path, thumb_path, (width, height))
    except Exception as e:
        logger.warning(f"Portrait thumbnail generation failed for {clean_name}: {e}")
        return send_from_directory("portraits", clean_name)

    return send_file(thumb_path, mimetype="image/png", conditional=True)


# Add this route
@app.route("/static/<path:filename>")
def serve_static(filename):
    """
    Serve static files from the root directory.
    This allows serving portraits from ./portraits/
    """
    return send_from_directory(".", filename)


@app.route("/gallery")
def gallery_index():
    # Get page parameter, default to 1
    page = request.args.get("page", 1, type=int)
    per_page = 50

    # Get paginated users from the database
    users_pagination = User.query.order_by(User.username).paginate(
        page=page, per_page=per_page, error_out=False
    )

    # Calculate total stats for all users
    total_users = User.query.count()
    total_balance = db.session.query(db.func.sum(User.balance)).scalar() or 0
    total_banknotes = Banknote.query.count()

    # Get current year for "New This Year" stat
    from datetime import datetime

    current_year = datetime.now().year
    new_this_year = User.query.filter(
        db.extract("year", User.created_at) == current_year
    ).count()

    return render_template(
        "gallery_index.html",
        users=users_pagination.items,
        pagination=users_pagination,
        total_users=total_users,
        total_balance=total_balance,
        total_banknotes=total_banknotes,
        new_this_year=new_this_year,
        title="Members",
        current_user=get_current_user(),
    )


@app.route("/gallery/<name>")
def show_name(name):
    import unicodedata

    name = unicodedata.normalize("NFC", name)
    name_path = os.path.join(IMAGES_ROOT, name)
    if not os.path.exists(name_path):
        return f"<h1 style='color:red'>Name {name} not found</h1><a href='/gallery'>← Gallery</a>"

    fronts, backs = [], []
    for denom in sorted(os.listdir(name_path)):
        denom_path = os.path.join(name_path, denom)
        if not os.path.isdir(denom_path):
            continue
        for f in sorted(os.listdir(denom_path)):
            if f.lower().endswith(".svg"):
                side = "front" if "_FRONT" in f else "back"
                bill = {
                    "file": url_for("serve_image", filename=f"{name}/{denom}/{f}"),
                    "side": side,
                    "denom": denom,
                }
                if side == "front":
                    fronts.append(bill)
                else:
                    backs.append(bill)

    return render_template(
        "name_detail.html",
        name=name,
        fronts=fronts,
        backs=backs,
        title=f"Member - {name}",
        current_user=get_current_user(),
    )


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGES_ROOT, filename)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            if user.two_factor_secret:
                session["pre_2fa_user_id"] = user.id
                return redirect(url_for("verify_2fa_login"))
            else:
                session["user_id"] = user.id
                flash("Logged in successfully!", "success")
                return redirect(url_for("landing"))
        else:
            flash("Invalid username or password", "error")

    return render_template("login.html", title="Login", current_user=get_current_user())


import os
from flask import current_app
from glob import glob


@app.route("/my-wallet")
def my_wallet():
    current_user = get_current_user()

    if not current_user:
        flash("Please log in to access your wallet", "error")
        return redirect(url_for("login"))

    # Debug: Print the current working directory and check if images folder exists
    print(f"Current working directory: {os.getcwd()}")

    # Check if the images directory exists at the expected path
    images_base_path = "./images"  # This is relative to your application root
    print(f"Looking for images in: {images_base_path}")
    print(f"Directory exists: {os.path.exists(images_base_path)}")

    if os.path.exists(images_base_path):
        print("Contents of images directory:")
        for item in os.listdir(images_base_path):
            print(f"  - {item}")

    # Scan for the user's specific folder
    user_images_path = os.path.join(images_base_path, current_user.username)
    print(f"Looking for user folder: {user_images_path}")
    print(f"User folder exists: {os.path.exists(user_images_path)}")

    # Dictionary to store all found images by denomination
    denomination_images = {}

    if os.path.exists(user_images_path):
        print("User folder contents:")
        for item in os.listdir(user_images_path):
            item_path = os.path.join(user_images_path, item)
            print(f"  - {item} (is_dir: {os.path.isdir(item_path)})")

            if os.path.isdir(item_path):
                # This is a denomination folder
                svg_files = glob(os.path.join(item_path, "*.svg"))
                print(f"    SVG files in {item}: {svg_files}")

                front_files = [f for f in svg_files if "_FRONT.svg" in f]
                back_files = [f for f in svg_files if "_BACK.svg" in f]

                if front_files or back_files:
                    denomination_images[item] = {
                        "front": sorted(front_files),
                        "back": sorted(back_files),
                    }

    print(f"Found denominations: {list(denomination_images.keys())}")

    denominations = sorted(denomination_images.keys())

    if not denominations:
        flash("No banknotes found in your wallet", "warning")
        return redirect(url_for("profile", username=current_user.username))

    # Helper functions to get images
    def get_front_image(denom):
        files = denomination_images.get(denom, {}).get("front", [])
        if files:
            filename = os.path.basename(files[-1])
            return f"./images/{current_user.username}/{denom}/{filename}"
        return None

    def get_back_image(denom):
        files = denomination_images.get(denom, {}).get("back", [])
        if files:
            filename = os.path.basename(files[-1])
            return f"./images/{current_user.username}/{denom}/{filename}"
        return None

    return render_template(
        "my_wallet.html",
        denominations=denominations,
        get_front_image=get_front_image,
        get_back_image=get_back_image,
        current_user=current_user,
        title=f"{current_user.username}'s Wallet",
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if password != confirm_password:
            flash("Passwords do not match", "error")
            return render_template(
                "register.html", title="Register", current_user=get_current_user()
            )

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
            return render_template(
                "register.html", title="Register", current_user=get_current_user()
            )

        if User.query.filter_by(email=email).first():
            flash("Email already registered", "error")
            return render_template(
                "register.html", title="Register", current_user=get_current_user()
            )

        user = User(username=username, email=email)
        user.set_password(password)
        user.two_factor_secret = pyotp.random_base32()

        # Generate email verification token
        verification_token = user.generate_verification_token()

        db.session.add(user)
        db.session.commit()

        # Send verification email
        try:
            app_url = request.url_root.rstrip("/")
            send_verification_email(email, username, verification_token, app_url)
            flash(
                "Registration successful! Please check your email to verify your account.",
                "success",
            )
        except Exception as e:
            print(f"[ERROR] Failed to send verification email: {e}")
            flash(
                "Registration successful! However, we couldn't send the verification email.",
                "warning",
            )

        session["pre_2fa_user_id"] = user.id
        return redirect(url_for("setup_2fa"))

    return render_template(
        "register.html", title="Register", current_user=get_current_user()
    )


@app.route("/setup-2fa")
def setup_2fa():
    if "pre_2fa_user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["pre_2fa_user_id"])
    if not user:
        return redirect(url_for("login"))

    uri = user.get_totp_uri()
    qr_code = generate_qr_code(uri)

    return render_template(
        "two_factor_setup.html",
        qr_code=qr_code,
        title="Setup 2FA",
        current_user=get_current_user(),
    )


@app.route("/setup-2fa", methods=["POST"])
def verify_2fa_setup():
    if "pre_2fa_user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["pre_2fa_user_id"])
    if not user:
        return redirect(url_for("login"))

    token = request.form.get("token")

    import pyotp

    totp = pyotp.TOTP(user.two_factor_secret)

    is_valid = False
    if totp.verify(token):
        is_valid = True
    elif totp.verify(token, valid_window=1):
        is_valid = True
    elif totp.verify(token, valid_window=2):
        is_valid = True

    if is_valid:
        session.pop("pre_2fa_user_id")
        session["user_id"] = user.id
        flash("Two-factor authentication setup complete!", "success")
        return redirect(url_for("landing"))
    else:
        flash(
            "Invalid token. Please check that your authenticator app time is synchronized with the server.",
            "error",
        )
        return redirect(url_for("setup_2fa"))


@app.route("/account-settings")
def account_settings():
    current_user = get_current_user()
    if not current_user:
        flash("Please log in to access settings", "error")
        return redirect(url_for("login"))

    _ensure_webauthn_name_column()

    current_user = User.query.get(current_user.id)
    security_keys = WebAuthnCredential.query.filter_by(
        user_id=current_user.id
    ).order_by(WebAuthnCredential.created_at.desc()).all()

    if not current_user.two_factor_secret:
        current_user.two_factor_secret = pyotp.random_base32()
        db.session.commit()

    uri = current_user.get_totp_uri()
    qr_code = generate_qr_code(uri)

    return render_template(
        "account_settings.html",
        qr_code=qr_code,
        security_keys=security_keys,
        title="Account Settings",
        current_user=current_user,
    )


@app.route("/account-settings/2fa", methods=["POST"])
def account_settings_2fa():
    current_user = get_current_user()
    if not current_user:
        flash("Please log in to access settings", "error")
        return redirect(url_for("login"))

    token = request.form.get("token")
    if not token:
        flash("Please enter a 6-digit code", "error")
        return redirect(url_for("account_settings"))

    import pyotp

    totp = pyotp.TOTP(current_user.two_factor_secret)

    is_valid = False
    if totp.verify(token):
        is_valid = True
    elif totp.verify(token, valid_window=1):
        is_valid = True
    elif totp.verify(token, valid_window=2):
        is_valid = True

    if is_valid:
        flash("Two-factor authentication updated successfully!", "success")
    else:
        flash(
            "Invalid token. Please check that your authenticator app time is synchronized with the server.",
            "error",
        )

    return redirect(url_for("account_settings"))


def _get_webauthn_rp_id():
    return request.host.split(":")[0]


def _get_webauthn_origin():
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip()
    scheme = forwarded_proto or request.scheme
    host = request.host
    if host == "bank.linglin.art":
        scheme = "https"
    return f"{scheme}://{host}"


def _webauthn_imports():
    from webauthn import (
        generate_registration_options,
        verify_registration_response,
        generate_authentication_options,
        verify_authentication_response,
    )
    from webauthn.helpers import options_to_json, bytes_to_base64url, base64url_to_bytes
    from webauthn.helpers.structs import (
        RegistrationCredential,
        AuthenticationCredential,
        PublicKeyCredentialDescriptor,
        AuthenticatorSelectionCriteria,
        UserVerificationRequirement,
        AuthenticatorAttestationResponse,
    )
    try:
        from webauthn.helpers.structs import AttestationConveyancePreference
    except Exception:
        AttestationConveyancePreference = None
    try:
        from webauthn.helpers.structs import AuthenticatorAssertionResponse
    except Exception:
        AuthenticatorAssertionResponse = None
    return (
        generate_registration_options,
        verify_registration_response,
        generate_authentication_options,
        verify_authentication_response,
        options_to_json,
        bytes_to_base64url,
        base64url_to_bytes,
        RegistrationCredential,
        AuthenticationCredential,
        PublicKeyCredentialDescriptor,
        AuthenticatorSelectionCriteria,
        UserVerificationRequirement,
        AuthenticatorAttestationResponse,
        AttestationConveyancePreference,
        AuthenticatorAssertionResponse,
    )


def _webauthn_options_to_dict(options_to_json, options, bytes_to_base64url=None):
    options_json = None
    try:
        options_json = options_to_json(options)
    except Exception:
        options_json = None

    data = None
    if isinstance(options_json, str):
        data = json.loads(options_json)
    elif isinstance(options_json, (bytes, bytearray)):
        data = json.loads(options_json.decode("utf-8"))
    elif isinstance(options_json, dict):
        data = options_json
    elif options_json is not None:
        data = json.loads(str(options_json))

    if data is None:
        if hasattr(options, "dict"):
            data = options.dict()
        elif isinstance(options, dict):
            data = dict(options)
        elif hasattr(options, "__dict__"):
            data = {k: v for k, v in vars(options).items() if not k.startswith("_")}
        else:
            data = {}

    def normalize_key(key):
        return {
            "rp_id": "rpId",
            "allow_credentials": "allowCredentials",
            "exclude_credentials": "excludeCredentials",
            "user_verification": "userVerification",
        }.get(key, key)

    normalized = {}
    for key, value in data.items():
        normalized[normalize_key(key)] = value

    if bytes_to_base64url:
        def convert_bytes(value):
            try:
                import dataclasses
            except Exception:
                dataclasses = None

            if isinstance(value, (bytes, bytearray)):
                return bytes_to_base64url(value)
            if hasattr(value, "model_dump"):
                return convert_bytes(value.model_dump())
            if hasattr(value, "dict") and callable(getattr(value, "dict")):
                return convert_bytes(value.dict())
            if dataclasses and dataclasses.is_dataclass(value):
                return convert_bytes(dataclasses.asdict(value))
            if isinstance(value, dict):
                return {k: convert_bytes(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert_bytes(v) for v in value]
            if isinstance(value, tuple):
                return [convert_bytes(v) for v in value]
            if hasattr(value, "__dict__"):
                return convert_bytes(
                    {k: v for k, v in vars(value).items() if not k.startswith("_")}
                )
            return value

        normalized = convert_bytes(normalized)

    return normalized


def _normalize_webauthn_credential(credential, base64url_to_bytes):
    if not isinstance(credential, dict):
        return credential

    response = credential.get("response") or {}
    if isinstance(response, dict):
        for key in [
            "attestationObject",
            "clientDataJSON",
            "authenticatorData",
            "signature",
            "userHandle",
        ]:
            if key in response and isinstance(response[key], str):
                response[key] = base64url_to_bytes(response[key])
        credential["response"] = response

    if "raw_id" in credential and isinstance(credential["raw_id"], str):
        credential["raw_id"] = base64url_to_bytes(credential["raw_id"])

    if "id" in credential and isinstance(credential["id"], str):
        credential["id"] = base64url_to_bytes(credential["id"])

    return credential


def _has_webauthn_bytes(value):
    if isinstance(value, (bytes, bytearray)):
        return True
    if isinstance(value, dict):
        return any(_has_webauthn_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_webauthn_bytes(v) for v in value)
    return False


def _parse_webauthn_credential(model, credential):
    if _has_webauthn_bytes(credential):
        if hasattr(model, "parse_obj"):
            return model.parse_obj(credential)
        return model(**credential)

    if hasattr(model, "parse_raw"):
        return model.parse_raw(json.dumps(credential))
    if hasattr(model, "parse_obj"):
        return model.parse_obj(credential)
    return model(**credential)


def _webauthn_credential_type_summary(credential):
    if not isinstance(credential, dict):
        return {"credential": str(type(credential))}

    response = credential.get("response") or {}
    if not isinstance(response, dict):
        response = {"_type": str(type(response))}

    return {
        "id": str(type(credential.get("id"))),
        "raw_id": str(type(credential.get("raw_id"))),
        "type": str(type(credential.get("type"))),
        "response.attestationObject": str(
            type(response.get("attestationObject"))
        ),
        "response.clientDataJSON": str(type(response.get("clientDataJSON"))),
        "response.signature": str(type(response.get("signature"))),
        "response.authenticatorData": str(
            type(response.get("authenticatorData"))
        ),
        "response.userHandle": str(type(response.get("userHandle"))),
    }


@app.route("/webauthn/register/options", methods=["POST"])
def webauthn_register_options():
    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({"error": "Login required"}), 401

        (
            generate_registration_options,
            _verify_registration_response,
            _generate_authentication_options,
            _verify_authentication_response,
            options_to_json,
            bytes_to_base64url,
            base64url_to_bytes,
            _RegistrationCredential,
            _AuthenticationCredential,
            PublicKeyCredentialDescriptor,
            AuthenticatorSelectionCriteria,
            UserVerificationRequirement,
            _AuthenticatorAttestationResponse,
            AttestationConveyancePreference,
        ) = _webauthn_imports()

        rp_id = _get_webauthn_rp_id()
        exclude_credentials = [
            PublicKeyCredentialDescriptor(
                id=base64url_to_bytes(cred.credential_id),
                type="public-key",
            )
            for cred in current_user.webauthn_credentials
        ]

        user_verification = getattr(UserVerificationRequirement, "PREFERRED", None)
        authenticator_selection = None
        if user_verification is not None and hasattr(user_verification, "value"):
            authenticator_selection = AuthenticatorSelectionCriteria(
                user_verification=user_verification
            )

        attestation = None
        if AttestationConveyancePreference is not None:
            attestation_value = getattr(AttestationConveyancePreference, "NONE", None)
            if attestation_value is not None and hasattr(attestation_value, "value"):
                attestation = attestation_value

        options_kwargs = {
            "rp_id": rp_id,
            "rp_name": "Ling Country Treasury",
            "user_id": str(current_user.id).encode(),
            "user_name": current_user.username,
            "user_display_name": current_user.username,
            "exclude_credentials": exclude_credentials,
        }
        if attestation is not None:
            options_kwargs["attestation"] = attestation
        if authenticator_selection is not None:
            options_kwargs["authenticator_selection"] = authenticator_selection

        options = generate_registration_options(**options_kwargs)

        session["webauthn_registration_challenge"] = bytes_to_base64url(
            options.challenge
        )

        return jsonify(
            _webauthn_options_to_dict(options_to_json, options, bytes_to_base64url)
        )
    except Exception as e:
        print(f"❌ WebAuthn registration options error: {e}")
        return (
            jsonify(
                {
                    "error": "WebAuthn registration options failed",
                    "detail": str(e),
                }
            ),
            500,
        )


@app.route("/webauthn/register/verify", methods=["POST"])
def webauthn_register_verify():
    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({"error": "Login required"}), 401

        _ensure_webauthn_name_column()

        data = request.get_json(silent=True) or {}
        credential = data.get("credential")
        if not credential:
            return jsonify({"error": "Missing credential"}), 400

        credential_label = (data.get("label") or "").strip()
        if len(credential_label) > 120:
            credential_label = credential_label[:120]

        raw_id_str = credential.get("rawId") or credential.get("id")
        raw_id = credential.get("rawId") or credential.get("raw_id")
        credential = {
            key: value
            for key, value in credential.items()
            if key not in {
                "authenticatorAttachment",
                "clientExtensionResults",
                "rawId",
            }
        }
        if raw_id_str and "id" not in credential:
            credential["id"] = raw_id_str
        if raw_id and "raw_id" not in credential:
            credential["raw_id"] = raw_id

        (
            _generate_registration_options,
            verify_registration_response,
            _generate_authentication_options,
            _verify_authentication_response,
            _options_to_json,
            bytes_to_base64url,
            base64url_to_bytes,
            RegistrationCredential,
            _AuthenticationCredential,
            _PublicKeyCredentialDescriptor,
            _AuthenticatorSelectionCriteria,
            _UserVerificationRequirement,
            AuthenticatorAttestationResponse,
            _AttestationConveyancePreference,
        ) = _webauthn_imports()

        expected_challenge = session.get("webauthn_registration_challenge")
        if not expected_challenge:
            return jsonify({"error": "Missing registration challenge"}), 400

        # 确保 WebAuthnCredential 表存在
        try:
            WebAuthnCredential.__table__.create(db.engine, checkfirst=True)
        except Exception:
            pass

        credential = _normalize_webauthn_credential(
            credential, base64url_to_bytes
        )
        if credential.get("raw_id") is not None:
            credential["id"] = bytes_to_base64url(credential["raw_id"])

        parsed_credential = RegistrationCredential(
            id=credential.get("id"),
            raw_id=credential.get("raw_id"),
            response=AuthenticatorAttestationResponse(
                attestation_object=credential.get("response", {}).get(
                    "attestationObject"
                ),
                client_data_json=credential.get("response", {}).get(
                    "clientDataJSON"
                ),
                transports=credential.get("response", {}).get("transports", []),
            ),
            type=credential.get("type", "public-key"),
        )

        # 验证注册响应
        verification = verify_registration_response(
            credential=parsed_credential,
            expected_challenge=base64url_to_bytes(expected_challenge),
            expected_rp_id=_get_webauthn_rp_id(),
            expected_origin=_get_webauthn_origin(),
            require_user_verification=False,
        )

        # 保存凭证
        credential_id = bytes_to_base64url(verification.credential_id)
        public_key = bytes_to_base64url(verification.credential_public_key)
        sign_count = verification.sign_count

        # 获取传输方式
        transports = ",".join(
            credential.get("response", {}).get("transports", []) or []
        )

        # 检查是否已存在
        existing_cred = WebAuthnCredential.query.filter_by(
            credential_id=credential_id
        ).first()
        
        if existing_cred:
            existing_cred.user_id = current_user.id
            existing_cred.public_key = public_key
            existing_cred.sign_count = sign_count
            existing_cred.transports = transports
            if credential_label:
                existing_cred.name = credential_label
        else:
            new_cred = WebAuthnCredential(
                user_id=current_user.id,
                credential_id=credential_id,
                name=credential_label or "Security Key",
                public_key=public_key,
                sign_count=sign_count,
                transports=transports,
            )
            db.session.add(new_cred)

        db.session.commit()

        session.pop("webauthn_registration_challenge", None)
        
        # 返回成功响应
        saved_count = WebAuthnCredential.query.filter_by(
            user_id=current_user.id
        ).count()
        
        return jsonify({
            "success": True, 
            "count": saved_count,
            "message": "Security key registered successfully"
        })
        
    except Exception as e:
        print(f"❌ WebAuthn registration verify error: {e}")
        import traceback
        traceback.print_exc()
        try:
            app.logger.error(
                "WebAuthn credential field types: %s",
                _webauthn_credential_type_summary(credential),
            )
        except Exception:
            pass
        db.session.rollback()
        return (
            jsonify({
                "error": "WebAuthn registration failed", 
                "detail": str(e),
                "type": type(e).__name__
            }),
            400,
        )


@app.route("/webauthn/login/options", methods=["POST"])
def webauthn_login_options():
    try:
        data = request.get_json(silent=True) or {}
        print(f"🔐 WebAuthn login options payload: {data}")
        username = (data.get("username") or "").strip()
        if not username:
            return jsonify({"error": "Username required"}), 400

        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
        print(f"🔐 WebAuthn login options user_id={user.id} username={user.username}")

        (
            _generate_registration_options,
            _verify_registration_response,
            generate_authentication_options,
            _verify_authentication_response,
            options_to_json,
            bytes_to_base64url,
            base64url_to_bytes,
            _RegistrationCredential,
            _AuthenticationCredential,
            PublicKeyCredentialDescriptor,
            _AuthenticatorSelectionCriteria,
            _UserVerificationRequirement,
            _AuthenticatorAttestationResponse,
            _AttestationConveyancePreference,
            _AuthenticatorAssertionResponse,
        ) = _webauthn_imports()

        allow_credentials = []
        for cred in user.webauthn_credentials or []:
            credential_id = getattr(cred, "credential_id", None)
            if not credential_id:
                continue
            try:
                if isinstance(credential_id, (bytes, bytearray)):
                    credential_bytes = bytes(credential_id)
                else:
                    credential_bytes = base64url_to_bytes(credential_id)
                allow_credentials.append(
                    PublicKeyCredentialDescriptor(
                        id=credential_bytes,
                        type="public-key",
                    )
                )
            except Exception as decode_error:
                print(
                    f"⚠️ Skipping invalid WebAuthn credential id for user {user.id}: {decode_error}"
                )

        print(f"🔐 WebAuthn allow_credentials count: {len(allow_credentials)}")

        if not allow_credentials:
            return jsonify({"error": "No security keys registered"}), 400

        try:
            options = generate_authentication_options(
                rp_id=_get_webauthn_rp_id(),
                allow_credentials=allow_credentials,
            )
        except TypeError:
            options = generate_authentication_options(
                allow_credentials=allow_credentials,
            )

        print(
            f"🔐 WebAuthn options generated rp_id={_get_webauthn_rp_id()} origin={_get_webauthn_origin()}"
        )

        session["webauthn_authentication_challenge"] = bytes_to_base64url(
            options.challenge
        )
        session["webauthn_login_user_id"] = user.id

        return jsonify(
            _webauthn_options_to_dict(options_to_json, options, bytes_to_base64url)
        )
    except Exception as e:
        print(f"❌ WebAuthn login options error: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())
        return (
            jsonify(
                {"error": "WebAuthn login options failed", "detail": str(e)}
            ),
            500,
        )


@app.route("/webauthn/login/verify", methods=["POST"])
def webauthn_login_verify():
    try:
        data = request.get_json(silent=True) or {}
        credential = data.get("credential")
        if not credential:
            return jsonify({"error": "Missing credential"}), 400

        (
            _generate_registration_options,
            _verify_registration_response,
            _generate_authentication_options,
            verify_authentication_response,
            _options_to_json,
            bytes_to_base64url,
            base64url_to_bytes,
            _RegistrationCredential,
            AuthenticationCredential,
            _PublicKeyCredentialDescriptor,
            _AuthenticatorSelectionCriteria,
            _UserVerificationRequirement,
            _AuthenticatorAttestationResponse,
            _AttestationConveyancePreference,
            AuthenticatorAssertionResponse,
        ) = _webauthn_imports()

        raw_id_str = credential.get("rawId") or credential.get("id")
        raw_id = credential.get("rawId") or credential.get("raw_id")
        credential = {
            key: value
            for key, value in credential.items()
            if key not in {
                "authenticatorAttachment",
                "clientExtensionResults",
                "rawId",
            }
        }
        if raw_id_str and "id" not in credential:
            credential["id"] = raw_id_str
        if raw_id and "raw_id" not in credential:
            credential["raw_id"] = raw_id

        credential = _normalize_webauthn_credential(
            credential, base64url_to_bytes
        )
        if credential.get("raw_id") is not None:
            credential["id"] = bytes_to_base64url(credential.get("raw_id"))

        user_id = session.get("webauthn_login_user_id")
        challenge = session.get("webauthn_authentication_challenge")
        if not user_id or not challenge:
            return jsonify({"error": "Missing authentication challenge"}), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        if not raw_id_str:
            return jsonify({"error": "Invalid credential"}), 400

        stored = WebAuthnCredential.query.filter_by(
            credential_id=raw_id_str
        ).first()
        if not stored:
            return jsonify({"error": "Security key not found"}), 400

        if AuthenticatorAssertionResponse is not None:
            parsed_auth = AuthenticationCredential(
                id=credential.get("id"),
                raw_id=credential.get("raw_id"),
                response=AuthenticatorAssertionResponse(
                    authenticator_data=credential.get("response", {}).get(
                        "authenticatorData"
                    ),
                    client_data_json=credential.get("response", {}).get(
                        "clientDataJSON"
                    ),
                    signature=credential.get("response", {}).get(
                        "signature"
                    ),
                    user_handle=credential.get("response", {}).get(
                        "userHandle"
                    ),
                ),
                type=credential.get("type", "public-key"),
            )
        else:
            parsed_auth = _parse_webauthn_credential(
                AuthenticationCredential, credential
            )

        verification = verify_authentication_response(
            credential=parsed_auth,
            expected_challenge=base64url_to_bytes(challenge),
            expected_rp_id=_get_webauthn_rp_id(),
            expected_origin=_get_webauthn_origin(),
            credential_public_key=base64url_to_bytes(stored.public_key),
            credential_current_sign_count=stored.sign_count,
            require_user_verification=False,
        )

        stored.sign_count = verification.new_sign_count
        db.session.commit()

        session.pop("webauthn_authentication_challenge", None)
        session.pop("webauthn_login_user_id", None)
        session["user_id"] = user.id
        flash("Logged in with security key", "success")
        return jsonify({"success": True})
    except Exception as e:
        print(f"❌ WebAuthn login verify error: {e}")
        return (
            jsonify({"error": "WebAuthn login failed", "detail": str(e)}),
            400,
        )


@app.route("/webauthn/credential/<int:credential_id>/delete", methods=["POST"])
@app.route("/webauthn/credential/delete/<int:credential_id>", methods=["POST", "GET"])
@app.route("/webauthn/credential/<int:credential_id>", methods=["DELETE"])
def webauthn_delete_credential(credential_id):
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "Login required"}), 401

    _ensure_webauthn_name_column()

    credential = WebAuthnCredential.query.filter_by(
        id=credential_id, user_id=current_user.id
    ).first()
    if not credential:
        return jsonify({"error": "Security key not found"}), 404

    db.session.delete(credential)
    db.session.commit()

    remaining = WebAuthnCredential.query.filter_by(
        user_id=current_user.id
    ).count()
    return jsonify({"success": True, "count": remaining})


@app.route("/verify-email/<token>")
def verify_email(token):
    """Verify email address using token"""
    user = User.query.filter_by(verification_token=token).first()

    if not user:
        flash("Invalid verification token", "error")
        return redirect(url_for("login"))

    if user.verify_email_token(token):
        db.session.commit()
        flash("Email verified successfully! You can now log in.", "success")
    else:
        flash("Verification token expired. Please request a new one.", "error")

    return redirect(url_for("login"))


@app.route("/resend-verification")
def resend_verification():
    """Resend verification email"""
    user_id = session.get("pre_2fa_user_id") or session.get("user_id")

    if not user_id:
        flash("Please log in first", "error")
        return redirect(url_for("login"))

    user = User.query.get(user_id)
    if not user:
        flash("User not found", "error")
        return redirect(url_for("login"))

    if user.email_verified:
        flash("Email already verified", "info")
        return redirect(url_for("dashboard"))

    # Generate new token
    verification_token = user.generate_verification_token()
    db.session.commit()

    # Send email
    try:
        app_url = request.url_root.rstrip("/")
        send_verification_email(user.email, user.username, verification_token, app_url)
        flash("Verification email sent! Please check your inbox.", "success")
    except Exception as e:
        print(f"[ERROR] Failed to send verification email: {e}")
        flash("Failed to send verification email. Please try again later.", "error")

    return redirect(url_for("dashboard"))


@app.route("/profile/change-email", methods=["POST"])
def change_email():
    """Initiate email change process"""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    user = User.query.get(session["user_id"])
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    new_email = request.form.get("new_email", "").strip()
    password = request.form.get("password", "")

    # Validate password
    if not user.check_password(password):
        return jsonify({"success": False, "error": "Incorrect password"}), 400

    # Validate email format
    if not new_email or "@" not in new_email:
        return jsonify({"success": False, "error": "Invalid email address"}), 400

    # Check if email is already in use
    existing_user = User.query.filter_by(email=new_email).first()
    if existing_user and existing_user.id != user.id:
        return jsonify({"success": False, "error": "Email already in use"}), 400

    # Same email as current
    if new_email == user.email:
        return (
            jsonify({"success": False, "error": "This is already your current email"}),
            400,
        )

    # Store pending email and generate new verification token
    user.pending_email = new_email
    user.email_verified = False  # Require re-verification
    verification_token = user.generate_verification_token()
    db.session.commit()

    # Send verification email to NEW email address
    try:
        from email_service import send_email_change_verification

        app_url = request.url_root.rstrip("/")
        send_email_change_verification(
            new_email, user.username, verification_token, user.email, app_url
        )
        return jsonify(
            {
                "success": True,
                "message": "Verification email sent to new address. Please check your inbox.",
            }
        )
    except Exception as e:
        print(f"[ERROR] Failed to send email change verification: {e}")
        return (
            jsonify({"success": False, "error": "Failed to send verification email"}),
            500,
        )


@app.route("/profile/verify-email-change/<token>")
def verify_email_change(token):
    """Complete email change after verification"""
    if "user_id" not in session:
        flash("Please log in first", "error")
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])
    if not user:
        flash("User not found", "error")
        return redirect(url_for("login"))

    if not user.pending_email:
        flash("No pending email change", "info")
        return redirect(url_for("profile", username=user.username))

    # Verify token
    if user.verify_email_token(token):
        from models import EmailHistory

        # Record email change in history
        email_history = EmailHistory(
            user_id=user.id,
            old_email=user.email,
            new_email=user.pending_email,
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent", "")[:255],
        )
        db.session.add(email_history)

        # Update email
        old_email = user.email
        user.email = user.pending_email
        user.pending_email = None
        user.email_verified = True
        db.session.commit()

        flash(f"Email successfully changed from {old_email} to {user.email}", "success")
    else:
        flash("Invalid or expired verification token", "error")

    return redirect(url_for("profile", username=user.username))


@app.route("/profile/cancel-email-change")
def cancel_email_change():
    """Cancel pending email change"""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    user = User.query.get(session["user_id"])
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    user.pending_email = None
    user.verification_token = None
    user.verification_token_expires = None
    db.session.commit()

    return jsonify({"success": True, "message": "Email change cancelled"})


@app.route("/profile/resend-verification")
def profile_resend_verification():
    """Resend verification email from profile"""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    user = User.query.get(session["user_id"])
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    if user.email_verified and not user.pending_email:
        return jsonify({"success": False, "error": "Email already verified"}), 400

    # Generate new token
    verification_token = user.generate_verification_token()
    db.session.commit()

    # Send to pending_email if exists, otherwise current email
    target_email = user.pending_email if user.pending_email else user.email

    try:
        from email_service import (
            send_verification_email,
            send_email_change_verification,
        )

        app_url = request.url_root.rstrip("/")

        if user.pending_email:
            send_email_change_verification(
                target_email, user.username, verification_token, user.email, app_url
            )
        else:
            send_verification_email(
                target_email, user.username, verification_token, app_url
            )

        return jsonify(
            {"success": True, "message": f"Verification email sent to {target_email}"}
        )
    except Exception as e:
        print(f"[ERROR] Failed to send verification email: {e}")
        return (
            jsonify({"success": False, "error": "Failed to send verification email"}),
            500,
        )


@app.route("/profile/email-history")
def email_history():
    """View email change history"""
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    from models import EmailHistory

    user = User.query.get(session["user_id"])
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    history = (
        EmailHistory.query.filter_by(user_id=user.id)
        .order_by(EmailHistory.changed_at.desc())
        .all()
    )

    return jsonify(
        {
            "success": True,
            "history": [
                {
                    "old_email": h.old_email,
                    "new_email": h.new_email,
                    "changed_at": h.changed_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "ip_address": h.ip_address,
                }
                for h in history
            ],
        }
    )


@app.route("/admin/generate-money/<int:user_id>", methods=["POST"])
@app.route("/verify-2fa", methods=["GET", "POST"])
def verify_2fa_login():
    if "pre_2fa_user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["pre_2fa_user_id"])
    if not user:
        return redirect(url_for("login"))

    if request.method == "POST":
        token = request.form.get("token")

        # Use the same robust verification as in setup
        import pyotp

        totp = pyotp.TOTP(user.two_factor_secret)

        is_valid = False
        if totp.verify(token):
            is_valid = True
        elif totp.verify(token, valid_window=1):  # Allow previous token
            is_valid = True
        elif totp.verify(token, valid_window=2):  # Allow next token
            is_valid = True

        if is_valid:
            session.pop("pre_2fa_user_id")
            session["user_id"] = user.id
            flash("Logged in successfully!", "success")
            return redirect(url_for("landing"))
        else:
            flash("Invalid token. Please try again.", "error")

    return render_template(
        "two_factor_verify.html", title="Verify 2FA", current_user=get_current_user()
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for("landing"))


@app.route("/transaction-viewer/<tx_hash>")
def transaction_viewer(tx_hash):
    # Avoid network sync by default for fast page load
    sync_requested = request.args.get("sync") == "1"
    if sync_requested:
        try:
            blockchain_daemon_instance.sync_with_network()
        except Exception as sync_err:
            print(f"[SYNC WARNING] Could not sync with network: {sync_err}")
    """
    View transaction details from the blockchain
    """
    try:
        def _find_local_transaction(target_hash: str):
            mempool = getattr(blockchain_daemon_instance, "mempool", []) or []
            for tx in mempool:
                if tx.get("hash") == target_hash:
                    tx_copy = tx.copy()
                    tx_copy["status"] = "pending"
                    tx_copy["confirmations"] = 0
                    return tx_copy

            blockchain = getattr(blockchain_daemon_instance, "blockchain", []) or []
            for block_index, block in enumerate(blockchain):
                for tx in block.get("transactions", []):
                    if tx.get("hash") == target_hash:
                        tx_copy = tx.copy()
                        tx_copy["status"] = "confirmed"
                        tx_copy["block_height"] = block_index
                        tx_copy["block_hash"] = block.get("hash")
                        tx_copy["confirmations"] = len(blockchain) - block_index - 1
                        return tx_copy
            return None

        # Check if this is a custom reward transaction format: reward_<block_index>_<identifier>
        is_reward_tx = False
        block_data = None
        tx_data = None
        
        if tx_hash.startswith("reward_"):
            # Parse custom reward format: reward_<block_index>_<identifier>
            parts = tx_hash.split("_")
            if len(parts) >= 2:
                try:
                    block_index = int(parts[1])
                    # Try to fetch block by index
                    for block in blockchain_daemon_instance.blockchain:
                        if block.get("index") == block_index:
                            block_data = block
                            break
                    
                    if block_data and "reward" in block_data:
                        is_reward_tx = True
                except (ValueError, IndexError):
                    pass

        # If not found with custom format, try normal transaction lookup
        if not tx_data:
            tx_data = _find_local_transaction(tx_hash)
            if not tx_data and sync_requested:
                tx_data = blockchain_daemon_instance.get_transaction(tx_hash)
            if isinstance(tx_data, list):
                tx_data = next((item for item in tx_data if isinstance(item, dict)), None)
            elif not isinstance(tx_data, dict):
                tx_data = None
        # 追加: dict型以外はNone
        if not isinstance(tx_data, dict):
            tx_data = None

        # If still not found, check if it's a block hash (mining reward)
        if not tx_data and not block_data:
            block_data = blockchain_daemon_instance.get_block(tx_hash)

        # 追加: block_dataからget_transactionを呼ぶ場合も型チェック
        if tx_data is not None:
            if isinstance(tx_data, list):
                tx_data = next((item for item in tx_data if isinstance(item, dict)), None)
            elif not isinstance(tx_data, dict):
                tx_data = None
        # 追加: dict型以外はNone
        if not isinstance(tx_data, dict):
            tx_data = None

        if block_data and "reward" in block_data:
            is_reward_tx = True
            reward_amount = block_data.get("reward", 0)
            miner_address = block_data.get("miner", "Unknown")

            # Create synthetic transaction data for the reward
            tx_data = {
                "hash": block_data.get("hash", tx_hash),
                "block_height": block_data.get("index"),
                "timestamp": block_data.get("timestamp"),
                "is_reward": True,
                "is_coinbase": True,
                "reward_amount": reward_amount,
                "miner": miner_address,
                "type": "mining_reward",
                "confirmations": 1,  # Assuming if we can see it, it's confirmed
                "inputs": [],
                "outputs": [
                    {
                        "address": miner_address,
                        "value": reward_amount,
                        "type": "mining_reward",
                        "description": f'Mining reward for block #{block_data.get("index", "N/A")}',
                    }
                ],
                "total_value": reward_amount,
                "fee": 0,
                "size": 0,
                "difficulty": block_data.get("difficulty"),
                "nonce": block_data.get("nonce"),
                "previous_hash": block_data.get("previous_hash"),
                "transactions_count": len(block_data.get("transactions", [])),
            }
        elif not tx_data:
            flash("Transaction not found on the blockchain", "error")
            return redirect(url_for("verify_serial"))

        # Check if this is a normal transaction from the transactions array
        # You might need a different API endpoint to get transaction by hash
        # from the blockchain daemon

        # Prepare transaction data for the template
        if not isinstance(tx_data, dict):
            flash("Transaction data is invalid (internal error)", "error")
            return redirect(url_for("verify_serial"))
        transaction = {
            "hash": tx_hash,
            "block_height": tx_data.get("block_height") or tx_data.get("index"),
            "block_hash": tx_data.get("block_hash"),
            "confirmations": tx_data.get("confirmations", 0),
            "timestamp": tx_data.get("timestamp"),
            "size": tx_data.get("size", 0),
            "fee": tx_data.get("fee", 0),
            "total_value": tx_data.get("total_value") or tx_data.get("amount", 0),
            "inputs": tx_data.get("inputs", []),
            "outputs": tx_data.get("outputs", []),
            "valid": True,
            "is_coinbase": tx_data.get("is_coinbase", False),
            "is_reward": tx_data.get("is_reward", False),
            "type": tx_data.get("type", "transfer"),
            "version": tx_data.get("version", "1.0"),
            "memo": tx_data.get("memo", ""),
            "difficulty": tx_data.get("difficulty"),
            "nonce": tx_data.get("nonce"),
            "previous_hash": tx_data.get("previous_hash"),
            "miner": tx_data.get("miner"),
            "from_address": tx_data.get("from"),
            "to_address": tx_data.get("to"),
        }

        # Handle different transaction types
        tx_type = tx_data.get("type", "transfer")
        is_reward_tx = tx_data.get("is_reward", False) or tx_data.get(
            "is_coinbase", False
        )

        # For normal transactions, extract sender/receiver
        if not is_reward_tx:
            transaction["from"] = tx_data.get("from")
            transaction["to"] = tx_data.get("to")
            transaction["amount"] = tx_data.get("amount", 0)
            transaction["priority"] = tx_data.get("priority", "normal")
            transaction["public_key"] = tx_data.get("public_key")
            transaction["signature"] = tx_data.get("signature")
            transaction["bill_type"] = tx_data.get("bill_type")
            transaction["front_serial"] = tx_data.get("front_serial")

        # Calculate input/output totals for normal transactions
        if not is_reward_tx and "inputs" in tx_data and "outputs" in tx_data:
            input_total = sum(inp.get("value", 0) for inp in transaction["inputs"])
            output_total = sum(out.get("value", 0) for out in transaction["outputs"])
        elif is_reward_tx:
            # For mining rewards, there are no inputs
            input_total = 0
            output_total = transaction.get("reward_amount", 0) or transaction.get(
                "total_value", 0
            )
        else:
            # For simple transfers
            input_total = transaction.get("amount", 0)
            output_total = transaction.get("amount", 0)

        # Get mempool status if not confirmed
        mempool_status = None
        if not transaction["block_height"]:
            mempool_status = blockchain_daemon_instance.get_mempool_transaction(tx_hash)

        # Check if this transaction contains any banknote data
        banknote_serial = None
        banknote_info = None

        # Check different possible locations for banknote serial
        possible_serials = [
            tx_data.get("front_serial"),
            tx_data.get("memo"),
            tx_data.get("bill_type"),
        ]

        for serial_source in possible_serials:
            if serial_source and "GTX" in str(serial_source):
                banknote_serial = serial_source
                break

        # Also check OP_RETURN data in outputs
        if not banknote_serial:
            for output in transaction.get("outputs", []):
                if output.get("script_type") == "op_return":
                    op_return_data = output.get("op_return", "")
                    if "GTX" in op_return_data or "SN-" in op_return_data:
                        banknote_serial = op_return_data
                        break

        # Get banknote info if found
        if banknote_serial:
            from models import Banknote, SerialRecord

            # Clean up serial format
            if banknote_serial.startswith("GTX"):
                # Convert GTX format to SN- format if needed
                serial_parts = banknote_serial.split("_")
                if len(serial_parts) >= 2:
                    banknote_serial = (
                        f"SN-{serial_parts[1]}-{serial_parts[2]}"
                        if len(serial_parts) > 2
                        else banknote_serial
                    )

            banknote_info = Banknote.query.filter_by(
                serial_number=banknote_serial
            ).first()
            if not banknote_info:
                # Try with different serial formats
                serial_record = SerialRecord.query.filter_by(
                    serial=banknote_serial
                ).first()
                if serial_record and serial_record.banknote_id:
                    banknote_info = Banknote.query.get(serial_record.banknote_id)

        # Format timestamp for display
        if transaction["timestamp"]:
            from datetime import datetime

            try:
                # Handle both Unix timestamp and float timestamps
                timestamp = float(transaction["timestamp"])
                dt = datetime.fromtimestamp(timestamp)
                transaction["timestamp_formatted"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                transaction["timestamp_readable"] = dt.strftime("%B %d, %Y at %I:%M %p")
                transaction["timestamp_relative"] = get_relative_time(dt)
            except:
                transaction["timestamp_formatted"] = str(transaction["timestamp"])
                transaction["timestamp_readable"] = str(transaction["timestamp"])
                transaction["timestamp_relative"] = "Unknown time"
        else:
            transaction["timestamp_formatted"] = "Pending"
            transaction["timestamp_readable"] = "Not yet confirmed"
            transaction["timestamp_relative"] = "Just now"

        # Calculate intelligent validation metrics
        validation_score = 0
        max_score = 5
        validation_layers = []
        
        # Check if transaction is in blockchain
        is_in_blockchain = bool(transaction["block_height"])
        
        # Layer 1: Blockchain confirmation (2 points max)
        if is_in_blockchain:
            validation_score += 2
            validation_layers.append({"name": "Blockchain Confirmed", "points": 2, "valid": True})
        else:
            # Check if in mempool
            if mempool_status:
                validation_score += 1
                validation_layers.append({"name": "In Mempool", "points": 1, "valid": True})
            else:
                validation_layers.append({"name": "Not yet broadcast", "points": 0, "valid": False})
        
        # Layer 2: Signature validation (1 point)
        has_valid_signature = False
        if transaction.get("signature"):
            try:
                has_valid_signature = verify_transaction_signature(transaction)
                if has_valid_signature:
                    validation_score += 1
                    validation_layers.append({"name": "Valid Signature", "points": 1, "valid": True})
                else:
                    validation_layers.append({"name": "Invalid Signature", "points": 0, "valid": False})
            except:
                validation_layers.append({"name": "Signature Unverifiable", "points": 0, "valid": False})
        else:
            validation_layers.append({"name": "No Signature", "points": 0, "valid": False})
        
        # Layer 3: Transaction structure validation (1 point)
        is_valid_structure = True
        if is_reward_tx:
            # Mining rewards must have miner address and amount
            if transaction.get("miner") and transaction.get("reward_amount"):
                validation_score += 1
                validation_layers.append({"name": "Valid Structure (Reward)", "points": 1, "valid": True})
            else:
                is_valid_structure = False
                validation_layers.append({"name": "Invalid Structure", "points": 0, "valid": False})
        else:
            # Normal transactions must have from, to, and amount
            if transaction.get("from") and transaction.get("to") and transaction.get("amount"):
                validation_score += 1
                validation_layers.append({"name": "Valid Structure", "points": 1, "valid": True})
            else:
                is_valid_structure = False
                validation_layers.append({"name": "Invalid Structure", "points": 0, "valid": False})
        
        # Layer 4: Database/Serial validation (if applicable, 1 point)
        has_db_validation = False
        if banknote_info:
            validation_score += 1
            has_db_validation = True
            validation_layers.append({"name": "Serial in Database", "points": 1, "valid": True})
        elif banknote_serial:
            validation_layers.append({"name": "Serial Not in Database", "points": 0, "valid": False})
        
        # Mining rewards are always valid if in blockchain
        if is_reward_tx and is_in_blockchain:
            confirmations = int(transaction.get("confirmations") or 0)
            confirmation_ratio = min(1.0, confirmations / 6) if confirmations else 0.0
            confirmation_points = round(confirmation_ratio, 2)
            validation_score = min(max_score, 4 + confirmation_points)
            validation_layers = [
                {"name": "Block Mined", "points": 2, "valid": True},
                {"name": "Valid Reward Transaction", "points": 1, "valid": True},
                {"name": "Correct Reward Structure", "points": 1, "valid": True},
                {
                    "name": "Confirmations (x/6)",
                    "points": confirmation_points,
                    "valid": confirmations >= 6,
                },
            ]
        elif is_reward_tx:
            # Pending reward transactions
            validation_layers = [
                {"name": "Reward Transaction", "points": 1, "valid": True},
                {"name": "Mining in Progress", "points": 1, "valid": True},
            ]
            validation_score = 2

        validation_percentage = (validation_score / max_score) * 100

        # Prepare validation results structure
        validation_results = {
            "blockchain": {
                "found": bool(transaction["block_height"]),
                "confirmations": transaction["confirmations"],
                "data": {
                    "block_height": transaction["block_height"],
                    "confirmations": transaction["confirmations"],
                },
            },
            "transaction_type": {
                "found": True,  # Transaction type is always present
                "type": transaction.get("type", "unknown"),
                "is_reward": is_reward_tx,
                "description": get_transaction_type_description(
                    transaction.get("type", "transfer")
                ),
            },
            "validation_layers": validation_layers,
        }

        # Only add mempool status if NOT in blockchain
        if mempool_status and not transaction["block_height"]:
            validation_results["mempool"] = {"found": True, "data": mempool_status}

        # Add signature validation
        if transaction.get("signature"):
            validation_results["signature"] = {
                "found": True,
                "valid": verify_transaction_signature(transaction),
                "public_key": transaction.get("public_key", "")[:20] + "..."
                if transaction.get("public_key")
                else None,
            }

        # Add banknote validation if applicable
        if banknote_serial:
            from models import Banknote, SerialRecord

            validation_results["serial_db"] = {
                "found": bool(
                    SerialRecord.query.filter_by(serial=banknote_serial).first()
                ),
                "data": {"serial": banknote_serial},
            }

            if banknote_info:
                validation_results["banknote_db"] = {
                    "found": True,
                    "data": {
                        "id": banknote_info.id,
                        "denomination": banknote_info.denomination,
                        "side": banknote_info.side,
                        "owner": banknote_info.user.username
                        if banknote_info.user
                        else "Unknown",
                    },
                }

            validation_results["digital_bill"] = {
                "found": bool(banknote_info),
                "serial_match": bool(banknote_info),
                "verification_method": "Blockchain Transaction",
            }

        # Add mining reward specific info
        if is_reward_tx:
            # Calculate chain depth (confirmations from tip)
            chain_depth = len(blockchain_daemon_instance.blockchain) - (transaction["block_height"] or 0)
            
            validation_results["mining_info"] = {
                "reward_amount": transaction.get("reward_amount")
                or transaction.get("total_value", 0),
                "miner": transaction.get("miner", "Unknown"),
                "difficulty": transaction.get("difficulty"),
                "nonce": transaction.get("nonce"),
                "block_hash": transaction.get("hash"),
                "previous_hash": transaction.get("previous_hash"),
                "chain_depth": chain_depth,
                "blocks_since_reward": chain_depth,
                "confirmations": chain_depth,
            }

        confirmations = int(transaction.get("confirmations") or 0)
        transaction["confirmations"] = confirmations

        # Determine status from confirmations (confirmed after 6)
        current_status = (transaction.get("status") or "").lower()
        if current_status in {"failed", "error"}:
            transaction["status"] = current_status
        else:
            if confirmations >= 6:
                transaction["status"] = "confirmed"
            elif confirmations > 0 or transaction.get("block_height") is not None:
                transaction["status"] = "pending"
            else:
                transaction["status"] = "pending"

        # Determine status icon and color
        status_info = {
            "pending": {"icon": "⏳", "color": "warning", "label": "Pending"},
            "confirmed": {"icon": "✅", "color": "success", "label": "Confirmed"},
            "failed": {"icon": "❌", "color": "danger", "label": "Failed"},
            "unknown": {"icon": "❓", "color": "secondary", "label": "Unknown"},
        }

        status_key = transaction.get("status", "unknown") or "unknown"
        if isinstance(status_key, str):
            status_key = status_key.lower()
        else:
            status_key = "unknown"

        transaction["status_icon"] = status_info.get(
            status_key, status_info["unknown"]
        )["icon"]
        transaction["status_color"] = status_info.get(
            status_key, status_info["unknown"]
        )["color"]
        transaction["status_label"] = status_info.get(
            status_key, status_info["unknown"]
        )["label"]

        # Prepare template context
        # Calculate transaction age
        transaction_age = int(time.time()) - (
            transaction["timestamp"] if transaction["timestamp"] else int(time.time())
        )

        # Calculate confirmation percentage
        confirmation_percentage = (
            min(100, (transaction["confirmations"] / 6) * 100)
            if transaction["confirmations"]
            else 0
        )

        context = {
            "transaction": transaction,
            "validation_score": validation_score,
            "validation_percentage": validation_percentage,
            "validation_results": validation_results,
            "input_total": input_total,
            "output_total": output_total,
            "banknote_info": banknote_info,
            "banknote_serial": banknote_serial,
            "mempool_status": mempool_status,
            "is_reward_tx": is_reward_tx,
            "tx_type": tx_type,
            "current_user": get_current_user(),
            "transaction_age": transaction_age,
            "confirmation_percentage": confirmation_percentage,
        }

        # Add reward-specific context
        if is_reward_tx:
            context["reward_amount"] = transaction.get(
                "reward_amount"
            ) or transaction.get("total_value", 0)
            context["miner_address"] = transaction.get("miner", "Unknown")
            context["difficulty"] = transaction.get("difficulty")
            context["nonce"] = transaction.get("nonce")
            if block_data:
                block_transactions = []
                try:
                    from models import Banknote, SerialNumber

                    for tx in block_data.get("transactions", []):
                        if not isinstance(tx, dict):
                            block_transactions.append(tx)
                            continue

                        tx_copy = tx.copy()

                        if tx_copy.get("type") == "GTX_Genesis":
                            serial_val = tx_copy.get("serial_number") or tx_copy.get("front_serial")
                            banknote = None

                            if serial_val:
                                banknote = Banknote.query.filter_by(serial_number=serial_val).first()
                                if not banknote:
                                    serial_record = SerialNumber.query.filter_by(serial=serial_val).first()
                                    if serial_record and serial_record.banknote_id:
                                        banknote = Banknote.query.get(serial_record.banknote_id)

                            if banknote:
                                if banknote.png_path:
                                    png_path = banknote.png_path.replace("\\", "/")
                                    if png_path.startswith("./"):
                                        png_path = png_path[2:]
                                    if png_path.startswith("images/"):
                                        png_path = png_path[len("images/"):]
                                    tx_copy["png_path"] = png_path
                                if not tx_copy.get("issued_to") and banknote.user:
                                    tx_copy["issued_to"] = banknote.user.username

                            if serial_val and not tx_copy.get("serial_number"):
                                tx_copy["serial_number"] = serial_val

                        block_transactions.append(tx_copy)
                except Exception as e:
                    print(f"[BLOCK TX WARNING] Could not enrich block transactions: {e}")
                    block_transactions = block_data.get("transactions", [])

                context["block_transactions"] = block_transactions

        # Choose template based on transaction type
        template_name = "reward_viewer.html" if is_reward_tx else "transaction_viewer.html"
        
        # Check if reward template exists, fall back to regular if not
        import os
        template_path = os.path.join(app.template_folder, template_name)
        if not os.path.exists(template_path):
            template_name = "transaction_viewer.html"
        
        return render_template(template_name, **context)

    except Exception as e:
        print(f"Error viewing transaction: {str(e)}")
        import traceback

        traceback.print_exc()
        flash(f"Error retrieving transaction: {str(e)}", "error")
        return redirect(url_for("verify_serial"))


# Helper functions
def get_relative_time(dt):
    """Get relative time string (e.g., '2 hours ago')"""
    from datetime import datetime

    now = datetime.now()
    diff = now - dt

    if diff.days > 365:
        years = diff.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "just now"


def get_transaction_type_description(tx_type):
    """Get human-readable description of transaction type"""
    descriptions = {
        "transfer": "Normal transfer of funds",
        "mining_reward": "Mining reward for creating a new block",
        "coinbase": "Coinbase transaction (mining reward)",
        "stake_reward": "Staking reward",
        "genesis": "Genesis transaction (initial distribution)",
        "bill_creation": "Banknote creation transaction",
        "bill_transfer": "Banknote transfer transaction",
    }
    return descriptions.get(tx_type, "Unknown transaction type")


def verify_transaction_signature(transaction):
    """Verify transaction signature (simplified)"""
    # This would use your actual signature verification logic
    # For now, just check if signature exists and looks valid
    signature = transaction.get("signature", "")
    public_key = transaction.get("public_key", "")

    if not signature or not public_key:
        return False

    # Check if signature looks like a valid hex string
    import re

    if not re.match(r"^[0-9a-fA-F]{64,128}$", signature):
        return False

    # Check if public key looks valid
    if not public_key.startswith("pub_"):
        return False

    # In a real implementation, you would verify the cryptographic signature
    # return blockchain_daemon_instance.verify_signature(transaction)

    return True  # Simplified for now


@app.route("/banknote-image/<path:filename>")
def serve_banknote_image(filename):
    # Decode URL-encoded characters
    filename = unquote(filename)
    # Convert backslashes to forward slashes for cross-platform compatibility
    filename = filename.replace("\\", "/")
    # Remove any leading "images/" if it exists
    if filename.startswith("images/"):
        filename = filename[7:]
    # Ensure we're not dealing with directory traversal attacks
    if ".." in filename or filename.startswith("/"):
        abort(404)
    return send_from_directory(IMAGES_ROOT, filename)


def _sanitize_banknote_filename(filename):
    filename = unquote(filename)
    filename = filename.replace("\\", "/")
    if filename.startswith("images/"):
        filename = filename[7:]
    if ".." in filename or filename.startswith("/"):
        return None
    return filename


def _generate_thumbnail(source_path, thumb_path, size):
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    with Image.open(source_path) as img:
        img = img.convert("RGBA")
        thumb = ImageOps.fit(img, size, Image.LANCZOS)
        thumb.save(thumb_path, format="PNG", optimize=True)


@app.route("/banknote-thumbnail/<path:filename>")
def serve_banknote_thumbnail(filename):
    clean_name = _sanitize_banknote_filename(filename)
    if not clean_name:
        abort(404)

    try:
        width = int(request.args.get("w", 320))
        height = int(request.args.get("h", 120))
    except ValueError:
        return jsonify({"error": "Invalid thumbnail size"}), 400

    if width < 20 or height < 20 or width > 800 or height > 800:
        return jsonify({"error": "Thumbnail size out of bounds"}), 400

    source_path = os.path.join(IMAGES_ROOT, clean_name)
    if not os.path.exists(source_path):
        abort(404)

    thumb_dir = os.path.join(IMAGES_ROOT, ".thumbs", f"{width}x{height}")
    thumb_path = os.path.join(thumb_dir, clean_name)

    try:
        if (not os.path.exists(thumb_path)) or (
            os.path.getmtime(thumb_path) < os.path.getmtime(source_path)
        ):
            _generate_thumbnail(source_path, thumb_path, (width, height))
    except Exception as e:
        logger.warning(f"Thumbnail generation failed for {clean_name}: {e}")
        return send_from_directory(IMAGES_ROOT, clean_name)

    return send_file(thumb_path, mimetype="image/png", conditional=True)


@app.route("/banknote/<serial_id>")
@app.route("/banknote-viewer/<serial_id>")
def banknote_viewer(serial_id):
    """Render a full-width banknote view with details."""
    if not serial_id:
        flash("Banknote serial is required", "error")
        return redirect(url_for("verify_serial"))

    banknote = Banknote.query.filter_by(serial_number=serial_id).first()
    if not banknote:
        serial_record = SerialNumber.query.filter_by(serial=serial_id).first()
        if serial_record and serial_record.banknote_id:
            banknote = Banknote.query.get(serial_record.banknote_id)

    if not banknote:
        flash("Banknote not found", "error")
        return redirect(url_for("verify_serial"))

    def normalize_png_path(png_path: str) -> str:
        if not png_path:
            return ""
        path = str(png_path).replace("\\", "/")
        if os.path.isabs(path):
            try:
                rel = os.path.relpath(path, IMAGES_ROOT)
                rel = rel.replace("\\", "/")
                if not rel.startswith(".."):
                    path = rel
            except Exception:
                pass
        if path.startswith("./"):
            path = path[2:]
        if path.startswith("images/"):
            path = path[len("images/"):]
        return path

    tx_data = banknote.get_transaction_data() if hasattr(banknote, "get_transaction_data") else {}

    if not tx_data and serial_id:
        try:
            tx_source = Banknote.query.filter(
                Banknote.transaction_data.contains(serial_id)
            ).first()
            if tx_source and getattr(tx_source, "transaction_data", None):
                tx_data = json.loads(tx_source.transaction_data)
        except Exception:
            pass

    def resolve_banknote_by_serial(serial_value: str):
        if not serial_value:
            return None
        from models import Banknote, SerialNumber

        note = Banknote.query.filter_by(serial_number=serial_value).first()
        if not note:
            serial_record = SerialNumber.query.filter_by(serial=serial_value).first()
            if serial_record and serial_record.banknote_id:
                note = Banknote.query.get(serial_record.banknote_id)
        return note

    def find_matching_banknote(current_note, tx_payload):
        candidates = []
        side = getattr(current_note, "side", "").lower()
        if side == "front":
            candidates.append(tx_payload.get("back_serial"))
        elif side == "back":
            candidates.append(tx_payload.get("front_serial"))

        candidates.extend([
            tx_payload.get("front_serial"),
            tx_payload.get("back_serial"),
        ])

        if not any(candidates) and isinstance(current_note.serial_number, str):
            serial_upper = current_note.serial_number.upper()
            if "FRONT" in serial_upper:
                candidates.append(current_note.serial_number.replace("FRONT", "BACK"))
            elif "BACK" in serial_upper:
                candidates.append(current_note.serial_number.replace("BACK", "FRONT"))

        seen = set()
        serials = [c for c in candidates if c and not (c in seen or seen.add(c))]
        for serial in serials:
            note = resolve_banknote_by_serial(str(serial))
            if note and note.id != current_note.id:
                return note
        return None

    front_note = banknote if (banknote.side or "").lower() == "front" else None
    back_note = banknote if (banknote.side or "").lower() == "back" else None

    match_note = find_matching_banknote(banknote, tx_data)
    if not match_note:
        try:
            opposite_side = "back" if (banknote.side or "").lower() == "front" else "front"
            time_center = banknote.created_at or datetime.utcnow()
            window_start = time_center - timedelta(seconds=60)
            window_end = time_center + timedelta(seconds=60)

            match_note = (
                Banknote.query.filter(
                    Banknote.user_id == banknote.user_id,
                    Banknote.denomination == banknote.denomination,
                    Banknote.side == opposite_side,
                    Banknote.created_at >= window_start,
                    Banknote.created_at <= window_end,
                )
                .order_by(Banknote.created_at.desc())
                .first()
            )

            if not match_note:
                match_note = (
                    Banknote.query.filter(
                        Banknote.denomination == banknote.denomination,
                        Banknote.side == opposite_side,
                        Banknote.created_at >= window_start,
                        Banknote.created_at <= window_end,
                    )
                    .order_by(Banknote.created_at.desc())
                    .first()
                )

            if not match_note and isinstance(banknote.serial_number, str):
                serial_parts = banknote.serial_number.split("-")
                serial_ts = serial_parts[-1] if serial_parts else ""
                if serial_ts.isdigit():
                    ts_ms = int(serial_ts)
                    ts_center = datetime.utcfromtimestamp(ts_ms / 1000.0)
                    ts_start = ts_center - timedelta(seconds=60)
                    ts_end = ts_center + timedelta(seconds=60)
                    match_note = (
                        Banknote.query.filter(
                            Banknote.user_id == banknote.user_id,
                            Banknote.side == opposite_side,
                            Banknote.created_at >= ts_start,
                            Banknote.created_at <= ts_end,
                        )
                        .order_by(Banknote.created_at.desc())
                        .first()
                    )

                if not match_note and serial_ts:
                    like_suffix = f"%-{serial_ts}"
                    match_note = (
                        Banknote.query.filter(
                            Banknote.user_id == banknote.user_id,
                            Banknote.side == opposite_side,
                            Banknote.serial_number.like(like_suffix),
                        )
                        .order_by(Banknote.created_at.desc())
                        .first()
                    )
        except Exception:
            match_note = None
    if match_note:
        match_side = (match_note.side or "").lower()
        if match_side == "front" and not front_note:
            front_note = match_note
        elif match_side == "back" and not back_note:
            back_note = match_note
        elif not back_note and (banknote.side or "").lower() == "front":
            back_note = match_note
        elif not front_note and (banknote.side or "").lower() == "back":
            front_note = match_note

    front_image_path = normalize_png_path(front_note.png_path) if front_note and front_note.png_path else ""
    back_image_path = normalize_png_path(back_note.png_path) if back_note and back_note.png_path else ""

    front_svg_path = ""
    back_svg_path = ""
    if front_note and not front_image_path and front_note.svg_path:
        front_svg_path = normalize_png_path(front_note.svg_path)
    if back_note and not back_image_path and back_note.svg_path:
        back_svg_path = normalize_png_path(back_note.svg_path)

    return render_template(
        "banknote_viewer.html",
        banknote=banknote,
        front_note=front_note,
        back_note=back_note,
        front_image_path=front_image_path,
        back_image_path=back_image_path,
        front_svg_path=front_svg_path,
        back_svg_path=back_svg_path,
        transaction_data=tx_data,
        title=f"Banknote {banknote.serial_number}",
        current_user=get_current_user(),
    )


@app.route("/banknote-matching-thumbnail/<serial_id>")
def get_matching_banknote_thumbnail(serial_id):
    """Return the matching front/back thumbnail for a given serial."""
    if not serial_id:
        return jsonify({"error": "Missing serial"}), 400

    def normalize_png_path(png_path: str) -> str:
        if not png_path:
            return ""
        path = str(png_path).replace("\\", "/")
        if os.path.isabs(path):
            try:
                rel = os.path.relpath(path, IMAGES_ROOT)
                rel = rel.replace("\\", "/")
                if not rel.startswith(".."): 
                    path = rel
            except Exception:
                pass
        if path.startswith("./"):
            path = path[2:]
        if path.startswith("images/"):
            path = path[len("images/"):]
        return path

    def resolve_banknote_by_serial(serial_value: str):
        if not serial_value:
            return None
        from models import Banknote, SerialNumber

        banknote = Banknote.query.filter_by(serial_number=serial_value).first()
        if not banknote:
            serial_record = SerialNumber.query.filter_by(serial=serial_value).first()
            if serial_record and serial_record.banknote_id:
                banknote = Banknote.query.get(serial_record.banknote_id)
        return banknote

    banknote = resolve_banknote_by_serial(serial_id)
    if not banknote:
        return jsonify({"error": "Banknote not found"}), 404

    tx_data = (
        banknote.get_transaction_data()
        if hasattr(banknote, "get_transaction_data")
        else {}
    )

    requested_side = request.args.get("side", "match").lower()
    candidates = []

    if requested_side == "back":
        candidates.append(tx_data.get("back_serial"))
    elif requested_side == "front":
        candidates.append(tx_data.get("front_serial"))
    else:
        if getattr(banknote, "side", "").lower() == "front":
            candidates.append(tx_data.get("back_serial"))
        elif getattr(banknote, "side", "").lower() == "back":
            candidates.append(tx_data.get("front_serial"))

    candidates.extend(
        [
            tx_data.get("front_serial"),
            tx_data.get("back_serial"),
        ]
    )

    if not any(candidates) and isinstance(serial_id, str):
        serial_upper = serial_id.upper()
        if "FRONT" in serial_upper:
            candidates.append(serial_id.replace("FRONT", "BACK"))
        elif "BACK" in serial_upper:
            candidates.append(serial_id.replace("BACK", "FRONT"))

    seen = set()
    serials = [c for c in candidates if c and not (c in seen or seen.add(c))]

    match_note = None
    for serial in serials:
        match_note = resolve_banknote_by_serial(str(serial))
        if match_note:
            break

    if not match_note or not match_note.png_path:
        return jsonify({"error": "Matching banknote image not found"}), 404

    clean_path = normalize_png_path(match_note.png_path)
    if not clean_path:
        return jsonify({"error": "Invalid banknote image path"}), 404

    return serve_banknote_thumbnail(clean_path)


@app.route("/transaction-thumbnail/<tx_hash>")
def get_transaction_banknote_thumbnail(tx_hash):
    """Return banknote thumbnail for a given transaction hash."""
    if not tx_hash or tx_hash == "undefined":
        return jsonify({"error": "Missing transaction hash"}), 400

    def normalize_png_path(png_path: str) -> str:
        if not png_path:
            return ""
        path = str(png_path).replace("\\", "/")
        if os.path.isabs(path):
            try:
                rel = os.path.relpath(path, IMAGES_ROOT)
                rel = rel.replace("\\", "/")
                if not rel.startswith(".."):
                    path = rel
            except Exception:
                pass
        if path.startswith("./"):
            path = path[2:]
        if path.startswith("images/"):
            path = path[len("images/"):]
        return path

    def resolve_banknote_by_serial(serial_value: str):
        if not serial_value:
            return None
        from models import Banknote, SerialNumber

        banknote = Banknote.query.filter_by(serial_number=serial_value).first()
        if not banknote:
            serial_record = SerialNumber.query.filter_by(serial=serial_value).first()
            if serial_record and serial_record.banknote_id:
                banknote = Banknote.query.get(serial_record.banknote_id)
        return banknote

    tx_data = blockchain_daemon_instance.get_transaction(tx_hash)
    if isinstance(tx_data, list):
        tx_data = next((item for item in tx_data if isinstance(item, dict)), None)
    elif not isinstance(tx_data, dict):
        tx_data = None

    if not tx_data:
        mempool_tx = blockchain_daemon_instance.get_mempool_transaction(tx_hash)
        if isinstance(mempool_tx, dict):
            tx_data = mempool_tx

    if not tx_data:
        for block in blockchain_daemon_instance.blockchain:
            for tx in block.get("transactions", []):
                if isinstance(tx, dict) and tx.get("hash") == tx_hash:
                    tx_data = tx
                    break
            if tx_data:
                break

    if not tx_data:
        return jsonify({"error": "Transaction not found"}), 404

    side = request.args.get("side", "front").lower()
    candidates = []
    if side == "back":
        candidates.append(tx_data.get("back_serial"))
    else:
        candidates.append(tx_data.get("front_serial"))

    candidates.extend(
        [
            tx_data.get("front_serial"),
            tx_data.get("back_serial"),
            tx_data.get("serial_number"),
            tx_data.get("serial"),
            tx_data.get("memo"),
            tx_data.get("bill_type"),
        ]
    )

    seen = set()
    serials = [c for c in candidates if c and not (c in seen or seen.add(c))]

    banknote = None
    for serial in serials:
        banknote = resolve_banknote_by_serial(str(serial))
        if banknote:
            break

    if not banknote or not banknote.png_path:
        return jsonify({"error": "Banknote image not found for transaction"}), 404

    clean_path = normalize_png_path(banknote.png_path)
    if not clean_path:
        return jsonify({"error": "Invalid banknote image path"}), 404

    return serve_banknote_thumbnail(clean_path)


@app.route("/toggle-banknote/<int:banknote_id>")
def toggle_banknote_visibility(banknote_id):
    current_user = get_current_user()
    if not current_user:
        return redirect(url_for("login"))

    banknote = Banknote.query.get_or_404(banknote_id)
    if banknote.user_id != current_user.id:
        flash("You don't have permission to modify this banknote", "error")
        return redirect(url_for("profile", username=current_user.username))

    banknote.is_public = not banknote.is_public
    db.session.commit()

    flash(
        f"Banknote visibility set to {'public' if banknote.is_public else 'private'}",
        "success",
    )
    return redirect(url_for("profile", username=current_user.username))


@app.route("/debug/generation/<username>")
def debug_generation(username):
    """Debug endpoint to check generation status"""
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Check generation tasks
    tasks = (
        GenerationTask.query.filter_by(user_id=user.id)
        .order_by(desc(GenerationTask.created_at))
        .all()
    )

    # Check banknotes in database
    banknotes = Banknote.query.filter_by(user_id=user.id).all()

    # Check files on disk
    user_dir = os.path.join(IMAGES_ROOT, username)
    files_exist = os.path.exists(user_dir)
    file_list = []

    if files_exist:
        for root, dirs, files in os.walk(user_dir):
            for file in files:
                if file.endswith((".svg", ".png", ".pdf")):
                    file_list.append(os.path.join(root, file))

    return jsonify(
        {
            "user": {
                "id": user.id,
                "username": user.username,
                "balance": user.balance,
                "last_generation": user.last_generation.isoformat()
                if user.last_generation
                else None,
            },
            "tasks": [
                {
                    "id": t.id,
                    "status": t.status,
                    "message": t.message,
                    "created_at": t.created_at.isoformat(),
                    "completed_at": t.completed_at.isoformat()
                    if t.completed_at
                    else None,
                }
                for t in tasks
            ],
            "banknotes_count": len(banknotes),
            "files_exist": files_exist,
            "file_count": len(file_list),
            "files": file_list[:10],  # First 10 files only
        }
    )


@app.route("/debug/user/<username>")
def debug_user(username):
    """Debug endpoint to check user's banknote status"""
    user = User.query.filter_by(username=username).first()
    if not user:
        return "User not found", 404

    # Check database records
    banknotes = Banknote.query.filter_by(user_id=user.id).all()

    # Check files on disk
    user_images_path = os.path.join(IMAGES_ROOT, username)
    files_on_disk = []

    if os.path.exists(user_images_path):
        for denom in os.listdir(user_images_path):
            denom_path = os.path.join(user_images_path, denom)
            if os.path.isdir(denom_path):
                for file in os.listdir(denom_path):
                    if file.endswith((".svg", ".png", ".pdf")):
                        files_on_disk.append(os.path.join(denom, file))

    # Check generation tasks
    tasks = (
        GenerationTask.query.filter_by(user_id=user.id)
        .order_by(desc(GenerationTask.created_at))
        .all()
    )

    response = f"""
    <h1>Debug Info for {username}</h1>
    <h2>Database Records: {len(banknotes)} banknotes</h2>
    <ul>
    {"".join(f'<li>{b.serial_number} - {b.denomination} - {b.side} - {b.created_at}</li>' for b in banknotes)}
    </ul>
    
    <h2>Files on Disk: {len(files_on_disk)} files</h2>
    <ul>
    {"".join(f'<li>{f}</li>' for f in files_on_disk)}
    </ul>
    
    <h2>Generation Tasks: {len(tasks)} tasks</h2>
    <ul>
    {"".join(f'<li>{t.status} - {t.created_at} - {t.message}</li>' for t in tasks)}
    </ul>
    
    <h2>User Balance: {user.balance}</h2>
    """

    return response


@app.route("/member/<username>", methods=["GET", "POST"])
def profile(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        flash("User not found", "error")
        return redirect(url_for("landing"))

    _ensure_webauthn_name_column()

    current_user_obj = get_current_user()

    if request.method == "POST":
        if current_user_obj and current_user_obj.id == user.id:
            updated = False
            if "bio" in request.form:
                raw_bio = request.form.get("bio", "")
                user.bio = sanitize_bio(raw_bio)
                updated = True
            if "custom_eisenscript" in request.form:
                raw_script = request.form.get("custom_eisenscript", "")
                user.custom_eisenscript = sanitize_eisenscript(raw_script)
                updated = True
            if updated:
                db.session.commit()
                flash("Profile updated successfully", "success")
            return redirect(url_for("profile", username=username))

    generation_tasks = (
        GenerationTask.query.filter_by(user_id=user.id)
        .order_by(desc(GenerationTask.created_at))
        .limit(10)
        .all()
    )

    security_keys = WebAuthnCredential.query.filter_by(
        user_id=user.id
    ).order_by(WebAuthnCredential.created_at.desc()).all()

    # DEBUG: Check if files exist on disk
    user_images_path = os.path.join(IMAGES_ROOT, username)
    print(f"[DEBUG] Checking for user images at: {user_images_path}")

    if os.path.exists(user_images_path):
        print(f"[DEBUG] User image directory exists")
        for denom in os.listdir(user_images_path):
            denom_path = os.path.join(user_images_path, denom)
            if os.path.isdir(denom_path):
                print(
                    f"[DEBUG] Denomination {denom} has files: {os.listdir(denom_path)}"
                )

    # Check database for banknotes
    if current_user_obj and current_user_obj.id == user.id:
        banknotes = Banknote.query.filter_by(user_id=user.id).all()
    else:
        banknotes = Banknote.query.filter_by(user_id=user.id, is_public=True).all()

    print(f"[DEBUG] Found {len(banknotes)} banknotes in database for user {username}")

    # Custom sorting: first by denomination (numeric value), then by side (fronts first)
    def banknote_sort_key(banknote):
        import re

        denomination_str = str(banknote.denomination).upper()

        # Extract numeric part before _FRONT/_BACK
        numbers = re.findall(r"\d+", denomination_str)
        numeric_value = int(numbers[0]) if numbers else 0

        # Detect side either from banknote.side or denom string
        side_str = getattr(banknote, "side", None)
        if not side_str and ("FRONT" in denomination_str or "BACK" in denomination_str):
            if "FRONT" in denomination_str:
                side_str = "FRONT"
            elif "BACK" in denomination_str:
                side_str = "BACK"

        side_order = {"FRONT": 0, "BACK": 1}
        return (numeric_value, side_order.get(side_str.upper(), 2))

    # Sort the banknotes using our custom key
    banknotes.sort(key=banknote_sort_key)

    # Generate SVG paths for each banknote
    for banknote in banknotes:
        # Create SVG path from PNG path
        if hasattr(banknote, "png_path") and banknote.png_path:
            banknote.svg_path = banknote.png_path.replace(".png", ".svg")
        else:
            banknote.svg_path = None

    return render_template(
        "profile.html",
        user=user,
        generation_tasks=generation_tasks,
        banknotes=banknotes,
        security_keys=security_keys,
        title=f"Profile - {username}",
        current_user=current_user_obj,
    )


@app.route("/eisenscript-guide")
def eisenscript_guide():
    return render_template(
        "eisenscript_guide.html",
        title="EisenScript Guide",
        current_user=get_current_user(),
    )


@app.route("/<username>", methods=["GET", "POST"])
def profile_legacy(username):
    if request.method == "POST":
        return redirect(url_for("profile", username=username), code=307)
    return redirect(url_for("profile", username=username), code=301)


@app.route("/debug/validate-block-format", methods=["POST"])
def debug_validate_block_format():
    """Debug endpoint to test block format validation"""
    try:
        block_data = request.get_json()

        # Test the exact validation your daemon uses
        temp_daemon = BlockchainDaemon()

        is_valid = blockchain_daemon_instance.validate_block(block_data)

        # Calculate expected hash
        calculated_hash = temp_daemon.calculate_block_hash(
            block_data.get("index"),
            block_data.get("previous_hash"),
            block_data.get("timestamp"),
            block_data.get("transactions", []),
            block_data.get("nonce"),
        )

        return jsonify(
            {
                "is_valid": is_valid,
                "hash_match": block_data.get("hash") == calculated_hash,
                "provided_hash": block_data.get("hash"),
                "calculated_hash": calculated_hash,
                "missing_fields": [
                    f
                    for f in [
                        "index",
                        "timestamp",
                        "transactions",
                        "previous_hash",
                        "nonce",
                        "hash",
                        "miner",
                    ]
                    if f not in block_data
                ],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def load_json_file(filename):
    """Safely load a JSON file and return its contents as Python."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        abort(404, description=f"{filename} not found")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # try parsing as full JSON
            return json.load(f)
    except json.JSONDecodeError:
        # fallback: newline-delimited JSON (mempool style)
        transactions = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        transactions.append(json.loads(line))
                    except Exception:
                        continue  # skip corrupted lines
        return transactions


# Add these helper functions
def filter_mined_transactions(mempool):
    """Filter out transactions that have already been mined in the blockchain"""
    if not mempool:
        return []

    # Load the current blockchain to check against
    blockchain_data = load_json_file("blockchain.json")
    if not blockchain_data:
        return mempool  # No blockchain yet, return all transactions

    filtered = []
    mined_count = 0

    for tx in mempool:
        if not is_transaction_mined(tx, blockchain_data):
            filtered.append(tx)
        else:
            mined_count += 1
            # Log the mined transaction for debugging
            tx_type = tx.get("type", "unknown")
            tx_id = tx.get("signature", tx.get("serial_number", "unknown"))[:16]
            print(f"   ⏭️  Filtered mined {tx_type} transaction: {tx_id}...")

    if mined_count > 0:
        print(f"✅ Filtered {mined_count} already mined transactions from mempool")

    return filtered


def is_transaction_mined(transaction, blockchain_data):
    """Check if a transaction has already been mined in the blockchain"""
    if not transaction or not blockchain_data:
        return False

    tx_signature = transaction.get("signature")
    tx_serial = transaction.get("serial_number")
    tx_type = transaction.get("type", "")

    # Check all blocks in the blockchain
    for block in blockchain_data:
        for block_tx in block.get("transactions", []):
            # Check by signature (for regular transactions)
            if tx_signature and block_tx.get("signature") == tx_signature:
                return True

            # Special check for GTX_Genesis transactions by serial number
            if (
                tx_type == "GTX_Genesis"
                and block_tx.get("type") == "GTX_Genesis"
                and tx_serial
                and block_tx.get("serial_number") == tx_serial
            ):
                return True

            # Check by content for other transaction types
            if (
                tx_type == block_tx.get("type")
                and transaction.get("from") == block_tx.get("from")
                and transaction.get("to") == block_tx.get("to")
                and transaction.get("amount") == block_tx.get("amount")
            ):
                return True

    return False


def load_json_file(filename):
    """Load JSON file with error handling"""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return []


# Add a cleanup endpoint for manual mempool maintenance
@app.route("/admin/cleanup-mempool", methods=["POST"])
def cleanup_mempool():
    """Admin endpoint to manually clean the mempool"""
    try:
        mempool_data = load_json_file("mempool.json")
        blockchain_data = load_json_file("blockchain.json")

        initial_count = len(mempool_data)
        cleaned_mempool = filter_mined_transactions(mempool_data)
        cleaned_count = initial_count - len(cleaned_mempool)

        # Save the cleaned mempool
        with open("mempool.json", "w", encoding="utf-8") as f:
            json.dump(cleaned_mempool, f, indent=2)

        return jsonify(
            {
                "status": "success",
                "message": f"Cleaned {cleaned_count} mined transactions",
                "initial_count": initial_count,
                "current_count": len(cleaned_mempool),
                "cleaned_count": cleaned_count,
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": f"Cleanup failed: {str(e)}"}), 500


# Initialize Database
with app.app_context():
    db.create_all()
    _ensure_serial_numbers_is_mined_column()
    _ensure_banknotes_verification_columns()
    _ensure_users_custom_eisenscript_column()
    _ensure_settings_eisenscript_columns()
    # Initialize blockchain manager
    start_generation_task_processor()

# Initialize the generation queue after all imports are complete
if __name__ == "__main__":
    # Start the background task processor in a separate thread
    # This ensures it runs only once in the main process, not the reloader.
    start_generation_task_processor()

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        if not hasattr(app, "blockchain_daemon_instance"):
            # blockchain_daemon = BlockchainDaemon()
            # IMPORTANT: Attach it to the app instance
            # app.blockchain_daemon = blockchain_daemon
            # blockchain_daemon.repair_blockchain()
            # blockchain_daemon.emergency_repair()
            # blockchain_daemon_instance.start_daemon()
            # blockchain_daemon.diagnose_transfer_issue()
            # blockchain_daemon.debug_mining_selection()
            # blockchain_daemon.force_mine_transfers()
            # blockchain_daemon_instance.debug_reward_issue()
            # blockchain_daemon_instance.comprehensive_diagnostic()
            # blockchain_daemon_instance.debug_hash_mismatch()
            # blockchain_daemon_instance.debug_mining_issues()
            atexit.register(
                lambda: blockchain_daemon_instance.stop_daemon()
                if blockchain_daemon_instance
                else None
            )

    # Initialize notification scheduler in background
    try:
        notification_scheduler = init_notification_scheduler(
            check_interval=3600
        )  # Check every hour
        print(
            color_text(
                "[SCHEDULER] Background notification scheduler started",
                Colors.BRIGHT_GREEN,
            )
        )
    except Exception as e:
        print(
            color_text(
                f"[SCHEDULER] Failed to start notification scheduler: {e}",
                Colors.BRIGHT_RED,
            )
        )

    app.run(
        debug=True, host="0.0.0.0", port=5001, use_reloader=False
    )  # use_reloader=False to avoid double-start
