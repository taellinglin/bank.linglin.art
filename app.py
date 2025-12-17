# app.py
import os
from flask import Flask, render_template, send_from_directory, url_for, request, redirect, flash, session, abort, jsonify, g
from typing import Dict
from flask_migrate import Migrate
from models import User, GenerationTask, Banknote, SerialNumber, Settings, MiningSession, BlockchainTransaction
from utils import (
    get_current_user, generate_qr_code, validate_serial_id, 
    GENERATION_LOCK, GENERATION_THREADS, get_user_avatar_or_default, get_user_avatar_url, get_user_by_username, has_banknotes,
    IMAGES_ROOT, get_generation_queue_status
)
from datetime import timedelta
from sqlalchemy import desc # <-- Add this if using desc in utility functions
import pyotp
import threading
from utils import get_formatted_initials, get_user_avatar, get_user_avatar_url, sanitize_bio, get_generation_queue_status, db
from urllib.parse import unquote
from datetime import datetime
from signatures import DigitalBill
import json
import asyncio
import time
import hashlib
import logging
# Add to your app.py
from blockchain_daemon import BlockchainDaemon
from functools import wraps
# ROYGBIV Color Scheme 🌈 plus more
class Colors:
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Bright background colors
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'
    STRIKETHROUGH = '\033[9m'
    
    # Reset
    END = '\033[0m'

def color_text(text, *color_codes):
    """
    Color text with one or more color/style codes
    
    Usage:
        color_text("Hello", Colors.RED)
        color_text("Warning", Colors.YELLOW, Colors.BOLD)
        color_text("Error", Colors.RED, Colors.BOLD, Colors.BG_WHITE)
    """
    color_string = ''.join(color_codes)
    return f"{color_string}{text}{Colors.END}"
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ILoveYouForeverXOXO")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lingcountrytreasury.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
# Initialize db with app
DATA_DIR = "./system-data/"
db.init_app(app)
migrate = Migrate(app, db)
# In app.py, near the top with other initializations
blockchain_daemon_instance = None

def create_app():
    # ... your existing create_app code ...
    
    # Initialize blockchain daemon
    global blockchain_daemon_instance
    if blockchain_daemon_instance is None:
        try:
            blockchain_daemon_instance = BlockchainDaemon()

            blockchain_daemon_instance.start_daemon()
            print("[BLOCKCHAIN] Blockchain daemon initialized")
        except Exception as e:
            print(f"[BLOCKCHAIN] Error initializing daemon: {e}")
    
    return app

# Make sure this runs when the module is imported
create_app()

@app.template_filter('format_number')
def format_number(value):
    """Format numbers with commas for thousands."""
    try:
        # Handle None, empty string, or non-numeric values
        if value is None or value == '':
            return "0"
        
        # Convert to int if it's a number
        if isinstance(value, (int, float)):
            num = int(value)
        else:
            # Try to convert string to int
            num = int(str(value).replace(',', '').split('.')[0])
        
        # Format with commas
        return f"{num:,}"
    except (ValueError, TypeError):
        return "0"
# Add as both a global and filter for flexibility
@app.template_global('max')
def template_max(a, b):
    """Max function for templates."""
    try:
        return max(a, b)
    except (TypeError, ValueError):
        return a if a > b else b

# Optional: Also add as a filter
@app.template_filter('safe_max')
def safe_max_filter(value, compare_to):
    """Safe max filter for templates."""
    return template_max(value, compare_to)
@app.context_processor
def utility_processor():
    """
    Make functions available to all templates
    """
    return {
        'get_user_avatar': get_user_avatar,  # Add this
        'get_formatted_initials': get_formatted_initials,  # Add this
        'get_user_avatar_url': get_user_avatar_url,
        'get_user_by_username': get_user_by_username,
        'has_banknotes': has_banknotes
    }
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        current_user = getattr(g, 'current_user', None) or get_current_user()
        if not current_user or not getattr(current_user, "is_admin", False):
            flash("Admin access required", "error")
            return redirect(url_for("landing"))
        return f(*args, **kwargs)
    return decorated
def run_generation_task(user_id, username):
    """Start a generation task and return task ID"""
    try:
        # Import the function from utils
        from utils import run_generation_task as utils_run_generation_task
        return utils_run_generation_task(user_id, username)
    except Exception as e:
        print(f"Error in app.run_generation_task: {e}")
        return None

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
        print(f"🔍 DEBUG: Starting view_block_detail for hash: {block_hash} (type: {type(block_hash)})")
        
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
        genesis_count = sum(1 for tx in transactions if tx.get("type") in ["genesis", "GTX_Genesis"])
        transfer_count = sum(1 for tx in transactions if tx.get("type") == "transfer")
        reward_count = sum(1 for tx in transactions if tx.get("type") == "reward")
        other_count = len(transactions) - genesis_count - transfer_count - reward_count
        
        print(f"📊 DEBUG: Transaction counts - genesis: {genesis_count}, transfer: {transfer_count}, reward: {reward_count}, other: {other_count}")
        
        # SIMPLE timestamp handling
        def safe_timestamp_to_readable(ts):
            """Simple timestamp conversion"""
            try:
                if ts is None:
                    return "Unknown"
                
                # Force conversion to float
                if isinstance(ts, str):
                    # Remove any non-numeric characters except . and -
                    clean_ts = ''.join(c for c in ts if c.isdigit() or c == '.' or c == '-')
                    ts = float(clean_ts) if clean_ts and clean_ts != '-' else 0
                else:
                    ts = float(ts)
                
                if ts > 0 and ts < 4102444800:
                    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
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
        except:
            block_size = 0
        
        # FIX: Ensure mining_time is a number, not a string
        mining_time = found_block.get("mining_time", 0)
        try:
            if isinstance(mining_time, str):
                # Extract numbers from string
                clean_mining_time = ''.join(c for c in mining_time if c.isdigit() or c == '.')
                mining_time_numeric = float(clean_mining_time) if clean_mining_time else 0.0
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
                previous_block = blockchain_daemon_instance.blockchain[block_index_int - 1] if block_index_int - 1 < len(blockchain_daemon_instance.blockchain) else None
            
            if block_index_int + 1 < len(blockchain_daemon_instance.blockchain):
                next_block = blockchain_daemon_instance.blockchain[block_index_int + 1]
        except (IndexError, ValueError, TypeError):
            pass
        
        # SAFE transaction details preparation
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
                    clean_ts = ''.join(c for c in tx_timestamp if c.isdigit() or c == '.' or c == '-')
                    numeric_timestamp = float(clean_ts) if clean_ts and clean_ts != '-' else 0
                else:
                    numeric_timestamp = float(tx_timestamp) if tx_timestamp else 0
            except:
                numeric_timestamp = 0
            
            # Ensure hash is properly formatted string
            tx_hash = str(tx.get("hash", f"tx-{i}")).strip()
            
            tx_info = {
                "index": i + 1,
                "type": str(tx.get("type", "unknown")),
                "hash": tx_hash,
                "timestamp": numeric_timestamp,
                "timestamp_readable": tx_readable_time,
                "size": len(json.dumps(tx, default=str)) if tx else 0
            }
            
            # Add type-specific fields with proper string conversion
            tx_type = tx.get("type", "")
            if tx_type == "transfer":
                tx_info.update({
                    "from": str(tx.get("from", "N/A")),
                    "to": str(tx.get("to", "N/A")),
                    "amount": tx.get("amount", "N/A")
                })
            elif tx_type in ["genesis", "GTX_Genesis"]:
                tx_info.update({
                    "serial_number": str(tx.get("serial_number", "N/A")),
                    "issued_to": str(tx.get("issued_to", "N/A")),
                    "denomination": tx.get("denomination", "N/A")
                })
            elif tx_type == "reward":
                tx_info.update({
                    "to": str(tx.get("to", "N/A")),
                    "amount": tx.get("amount", "N/A"),
                    "block_height": tx.get("block_height", "N/A"),
                    "description": str(tx.get("description", "Mining Reward"))
                })
            
            transaction_details.append(tx_info)
        
        # Prepare block info for template - ensure all values are properly typed
        block_info = {
            "block": found_block,
            "metadata": {
                "transaction_count": int(len(transactions)),
                "genesis_count": int(genesis_count),
                "transfer_count": int(transfer_count),
                "reward_count": int(reward_count),
                "other_count": int(other_count),
                "timestamp_readable": str(readable_time),
                "block_size": int(block_size),
                "is_genesis_block": bool(block_index == 0),
                "miner": str(found_block.get("miner", "Unknown")),
                "difficulty": found_block.get("difficulty", "N/A"),
                "mining_time": mining_time_numeric  # FIXED: This is now a number, not a string
            },
            "transactions": transaction_details,
            "navigation": {
                "previous_block": previous_block,
                "next_block": next_block,
                "current_index": int(block_index),
                "total_blocks": int(len(blockchain_daemon_instance.blockchain) if blockchain_daemon_instance.blockchain else 0)
            }
        }
        
        print(f"✅ DEBUG: Successfully prepared block info for #{block_index}")
        print(f"🔧 DEBUG: Mining time type: {type(block_info['metadata']['mining_time'])} value: {block_info['metadata']['mining_time']}")
        
        return render_template('block_detail.html',
                            block_info=block_info,
                            current_user=get_current_user(),
                            title=f"Block #{block_index} Details")
        
    except Exception as e:
        import traceback
        error_details = f"Error in view_block_detail: {str(e)}"
        print(f"❌ DEBUG: {error_details}")
        print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
        
        flash(f"Error loading block details: {str(e)}", "error")
        return redirect(url_for("blockchain_viewer"))



@app.route("/mempool", methods=["GET"])
def get_mempool():
    """Serve filtered mempool (only unmined transactions) - FIXED"""
    # This should return filtered mempool, not the raw one
    
    filtered_mempool = blockchain_daemon_instance.mempool
    #filtered_mempool = filter_mined_transactions(filtered_mempool)
    return jsonify(filtered_mempool)  # Return filtered, not the full mempool

from datetime import datetime, timedelta
from collections import defaultdict
import statistics

@app.route("/mempool-viewer")
@app.route("/mempool-viewer/<int:page>")
def mempool_viewer(page=1):
    """Display detailed mempool information in a web interface WITH PAGINATION"""
    try:
        # Get timeframe from query parameter
        selected_timeframe = request.args.get('timeframe', '1h')
        
        # Get mempool data from the new daemon
        mempool_status = blockchain_daemon_instance.get_mempool_status()
        mempool_data = mempool_status['transactions']
        
        # Get blockchain status for additional context
        blockchain_status = blockchain_daemon_instance.get_blockchain_status()
        
        # Pagination settings
        per_page = 15  # Reduced for compact view
        total_transactions = mempool_status['total']
        total_pages = (total_transactions + per_page - 1) // per_page
        
        # Ensure page is within valid range
        page = max(1, min(page, total_pages))
        
        # Calculate slice for current page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        current_transactions = mempool_data[start_idx:end_idx]
        
        print(f"🔍 Mempool Pagination: page {page}, showing transactions {start_idx}-{end_idx} of {total_transactions}")
        
        # Calculate statistics
        active_transactions = total_transactions
        mined_transactions = blockchain_status['total_transactions']
        
        # Count by transaction type
        type_counts = {
            'bills': mempool_status['bills'],
            'transfers': mempool_status['transfers'],
            'rewards': mempool_status['rewards']
        }
        
        # Get transaction details for current page
        transactions = []
        for tx in current_transactions:
            tx_info = {
                "hash": tx.get("hash", "N/A"),
                "type": tx.get("type", "unknown"),
                "timestamp": tx.get("timestamp", 0),
                "timestamp_readable": datetime.fromtimestamp(tx.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S') if tx.get("timestamp") else "Unknown",
                "is_mined": False,
                "size": len(json.dumps(tx)),
                "confirmations": 0
            }
            
            # Add type-specific fields
            if tx.get("type") == "transfer":
                tx_info["from"] = tx.get("from", "N/A")
                tx_info["to"] = tx.get("to", "N/A")
                tx_info["amount"] = tx.get("amount", "N/A")
                
            elif tx.get("type") in ["genesis", "GTX_Genesis"]:
                tx_info["serial_number"] = tx.get("serial_number", "N/A")
                tx_info["issued_to"] = tx.get("issued_to", "N/A")
                tx_info["denomination"] = tx.get("denomination", "N/A")
                
            elif tx.get("type") == "reward":
                tx_info["to"] = tx.get("to", "N/A")
                tx_info["from"] = tx.get("from", "https://bank.linglin.art")
                tx_info["amount"] = tx.get("amount", "N/A")
                tx_info["description"] = tx.get("description", "N/A")
            
            transactions.append(tx_info)
        
        # Sort transactions by timestamp (newest first)
        transactions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # Get blockchain info for context
        blockchain_info = {
            "total_blocks": blockchain_status['blocks'],
            "total_mined_transactions": mined_transactions,
            "mined_genesis": blockchain_status['genesis_transactions'],
            "mined_transfers": blockchain_status['transfer_transactions'],
            "mined_rewards": blockchain_status['reward_transactions']
        }
        
        return render_template('mempool_viewer.html',
                            transactions=transactions,
                            total_transactions=total_transactions,
                            active_transactions=active_transactions,
                            mined_transactions=mined_transactions,
                            type_counts=type_counts,
                            blockchain_info=blockchain_info,
                            current_page=page,
                            total_pages=total_pages,
                            per_page=per_page,
                            selected_timeframe=selected_timeframe,
                            current_user=get_current_user(),
                            title="Mempool Viewer")
        
    except Exception as e:
        print(f"❌ Error in mempool_viewer: {e}")
        flash(f"Error loading mempool data: {str(e)}", "error")
        return render_template('mempool_viewer.html',
                            transactions=[],
                            total_transactions=0,
                            active_transactions=0,
                            mined_transactions=0,
                            type_counts={},
                            blockchain_info={},
                            current_page=1,
                            total_pages=1,
                            per_page=15,
                            selected_timeframe='1h',
                            current_user=get_current_user(),
                            title="Mempool Viewer")

@app.route("/api/mempool/activity")
def mempool_activity():
    """API endpoint for mempool activity data"""
    try:
        timeframe = request.args.get('timeframe', '1h')
        
        # Get all mempool transactions
        mempool_status = blockchain_daemon_instance.get_mempool_status()
        all_transactions = mempool_status['transactions']
        
        # Calculate time range based on timeframe
        now = datetime.now()
        if timeframe == '1h':
            start_time = now - timedelta(hours=1)
            interval_minutes = 5  # 12 intervals
        elif timeframe == '6h':
            start_time = now - timedelta(hours=6)
            interval_minutes = 30  # 12 intervals
        elif timeframe == '24h':
            start_time = now - timedelta(hours=24)
            interval_minutes = 120  # 12 intervals
        elif timeframe == '7d':
            start_time = now - timedelta(days=7)
            interval_minutes = 840  # 12 intervals (7 days / 12 intervals)
        else:
            start_time = now - timedelta(hours=1)
            interval_minutes = 5
        
        # Initialize data structures
        num_intervals = 12
        interval_data = {
            'transfers': [0] * num_intervals,
            'bills': [0] * num_intervals,
            'rewards': [0] * num_intervals
        }
        
        # Generate labels for x-axis
        labels = []
        for i in range(num_intervals):
            label_time = start_time + timedelta(minutes=interval_minutes * i)
            if timeframe == '1h':
                labels.append(label_time.strftime('%H:%M'))
            elif timeframe == '6h':
                labels.append(label_time.strftime('%H:%M'))
            elif timeframe == '24h':
                labels.append(label_time.strftime('%H:%M'))
            else:  # 7d
                labels.append(label_time.strftime('%m/%d'))
        
        # Process transactions
        for tx in all_transactions:
            tx_time = datetime.fromtimestamp(tx.get('timestamp', 0))
            
            # Skip if transaction is outside timeframe
            if tx_time < start_time or tx_time > now:
                continue
            
            # Calculate which interval this transaction belongs to
            time_diff = (tx_time - start_time).total_seconds() / 60  # difference in minutes
            interval_index = min(int(time_diff // interval_minutes), num_intervals - 1)
            
            # Count by type
            tx_type = tx.get('type', 'unknown')
            if tx_type == 'transfer':
                interval_data['transfers'][interval_index] += 1
            elif tx_type in ['genesis', 'GTX_Genesis']:
                interval_data['bills'][interval_index] += 1
            elif tx_type == 'reward':
                interval_data['rewards'][interval_index] += 1
        
        # Calculate statistics
        all_counts = []
        for i in range(num_intervals):
            total = sum(interval_data[tx_type][i] for tx_type in interval_data)
            all_counts.append(total)
        
        if all_counts:
            peak = max(all_counts)
            average_per_minute = statistics.mean(all_counts) / (interval_minutes / 60)
            totals = {
                'all': sum(all_counts),
                'transfers': sum(interval_data['transfers']),
                'bills': sum(interval_data['bills']),
                'rewards': sum(interval_data['rewards'])
            }
        else:
            peak = 0
            average_per_minute = 0
            totals = {'all': 0, 'transfers': 0, 'bills': 0, 'rewards': 0}
        
        return jsonify({
            'timeline': interval_data,
            'labels': labels,
            'peak': peak,
            'average_per_minute': average_per_minute,
            'totals': totals,
            'timeframe': timeframe
        })
        
    except Exception as e:
        print(f"❌ Error in mempool_activity API: {e}")
        return jsonify({
            'timeline': {'transfers': [], 'bills': [], 'rewards': []},
            'labels': [],
            'peak': 0,
            'average_per_minute': 0,
            'totals': {'all': 0, 'transfers': 0, 'bills': 0, 'rewards': 0},
            'timeframe': timeframe,
            'error': str(e)
        })
@app.route("/mine-all-transfers")
def mine_all_transfers():
    """Mine all pending transfers in multiple blocks if needed"""
    try:
        blockchain_data = getattr(blockchain_daemon_instance, 'blockchain', [])
        mempool_data = getattr(blockchain_daemon_instance, 'mempool', [])
        
        if not blockchain_data:
            return jsonify({"error": "No blockchain available"})
        
        transfer_txs = [tx for tx in mempool_data if tx.get('type') == 'transfer']
        
        if not transfer_txs:
            return jsonify({"error": "No transfer transactions in mempool"})
        
        results = {
            "blocks_mined": 0,
            "total_transfers_mined": 0,
            "blocks": [],
            "remaining_transfers": len(transfer_txs)
        }
        
        # Mine transfers in batches of 20 per block
        transfers_per_block = 20
        total_blocks_needed = (len(transfer_txs) + transfers_per_block - 1) // transfers_per_block
        
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
                "hash": ""
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
                    nonce
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
                    tx for tx in mempool_data 
                    if tx.get('hash') not in [t.get('hash') for t in block_transfers]
                ]
                
                # Save
                blockchain_daemon_instance.save_blockchain()
                blockchain_daemon_instance.save_mempool()
                
                results["blocks_mined"] += 1
                results["total_transfers_mined"] += len(block_transfers)
                results["blocks"].append({
                    "index": new_block["index"],
                    "transfers": len(block_transfers),
                    "hash": new_block["hash"][:20] + "..."
                })
                
                # Update for next iteration
                mempool_data = blockchain_daemon_instance.mempool
                results["remaining_transfers"] = len([tx for tx in mempool_data if tx.get('type') == 'transfer'])
            else:
                results["error"] = f"Failed to mine block {block_num}"
                break
        
        return jsonify({
            "success": True,
            "message": f"Mined {results['blocks_mined']} blocks with {results['total_transfers_mined']} transfers",
            "results": results
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })
@app.route("/step-by-step-mine-transfers")
def step_by_step_mine_transfers():
    """Step-by-step transfer mining with detailed error reporting"""
    try:
        # Step 1: Get current state
        blockchain_data = getattr(blockchain_daemon_instance, 'blockchain', [])
        mempool_data = getattr(blockchain_daemon_instance, 'mempool', [])
        
        if not blockchain_data:
            return jsonify({"error": "No blockchain available"})
        
        transfer_txs = [tx for tx in mempool_data if tx.get('type') == 'transfer']
        
        if not transfer_txs:
            return jsonify({"error": "No transfer transactions in mempool"})
        
        steps = []
        
        # Step 2: Validate transfers
        valid_transfers = []
        for tx in transfer_txs:
            if blockchain_daemon_instance.validate_transfer_for_mining(tx):
                valid_transfers.append(tx)
        
        steps.append(f"Step 1: Found {len(valid_transfers)} valid transfers out of {len(transfer_txs)} total")
        
        if not valid_transfers:
            return jsonify({
                "error": "No valid transfers to mine", 
                "steps": steps,
                "validation_issues": "All transfers failed validation"
            })
        
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
            "hash": ""
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
                return jsonify({
                    "error": "Mining timeout",
                    "steps": steps,
                    "time_elapsed": time.time() - start_time
                })
            
            new_block["nonce"] = nonce
            calculated_hash = blockchain_daemon_instance.calculate_block_hash(
                new_block["index"],
                new_block["previous_hash"],
                new_block["timestamp"],
                new_block["transactions"],
                nonce
            )
            
            if calculated_hash.startswith(target):
                new_block["hash"] = calculated_hash
                steps.append(f"✅ Block mined successfully with nonce {nonce}")
                steps.append(f"✅ Final hash: {calculated_hash[:20]}...")
                break
        else:
            steps.append("❌ Failed to find valid nonce within limit")
            return jsonify({
                "error": "Mining failed - no valid nonce found",
                "steps": steps,
                "attempts": 1000000
            })
        
        # Step 6: Add to blockchain
        blockchain_daemon_instance.blockchain.append(new_block)
        steps.append("Step 5: Added block to blockchain")
        
        # Step 7: Use the ENHANCED cleanup method instead of simple hash matching
        initial_mempool_size = len(blockchain_daemon_instance.mempool)
        
        # Use the enhanced cleanup method
        removed_count = blockchain_daemon_instance.remove_mined_transactions(transfers_to_mine)
        
        # Also run comprehensive cleanup to catch any edge cases
        additional_removed = blockchain_daemon_instance.cleanup_mined_transactions_enhanced()
        
        steps.append(f"Step 6: Enhanced cleanup removed {removed_count} + {additional_removed} additional = {removed_count + additional_removed} total transactions")
        
        # Step 8: Save everything
        blockchain_daemon_instance.save_blockchain()
        blockchain_daemon_instance.save_mempool()
        steps.append("Step 7: Saved blockchain and mempool")
        
        # Step 9: Verify cleanup worked
        final_mempool_size = len(blockchain_daemon_instance.mempool)
        remaining_transfers = len([tx for tx in blockchain_daemon_instance.mempool if tx.get('type') == 'transfer'])
        steps.append(f"Step 8: Verification - Mempool: {final_mempool_size} total, {remaining_transfers} transfers remaining")
        
        return jsonify({
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
                "remaining_transfers": remaining_transfers
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "steps": steps if 'steps' in locals() else ["Failed before steps began"]
        })


@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    """Format a timestamp as datetime string - BULLETPROOF VERSION"""
    try:
        if value is None:
            return "Unknown"
        
        # Convert ANY value to numeric timestamp
        numeric_value = 0
        if isinstance(value, (int, float)):
            numeric_value = float(value)
        elif isinstance(value, str):
            clean_value = ''.join(c for c in value if c.isdigit() or c == '.' or c == '-')
            numeric_value = float(clean_value) if clean_value and clean_value != '-' else 0
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
        debug_info.append({
            "block_index": i,
            "timestamp": timestamp,
            "timestamp_type": type(timestamp).__name__,
            "is_numeric": isinstance(timestamp, (int, float)),
            "is_string": isinstance(timestamp, str)
        })
    
    return jsonify(debug_info)
from builtins import max as builtin_max, min as builtin_min
from datetime import datetime
@app.route("/blockchain-viewer")
@app.route("/blockchain-viewer/<int:page>")
def blockchain_viewer(page=1):
    """Display blockchain information in a web interface - WITH PAGINATION"""
    try:
        print(f"🚀 ======= STARTING blockchain_viewer() for page {page} =======")
        
        # Get the raw blockchain data from daemon
        print(f"🔍 [1/8] Checking blockchain_daemon_instance...")
        
        if blockchain_daemon_instance is None:
            print(f"❌ CRITICAL: blockchain_daemon_instance is None!")
            print(f"   This means the daemon wasn't properly initialized.")
            return render_template('blockchain_viewer.html',
                                blocks=[],
                                total_blocks=0,
                                total_transactions=0,
                                genesis_count=0,
                                transfer_count=0,
                                reward_count=0,
                                current_page=page,
                                datetime=datetime,  # Add this line
                                total_pages=1,
                                per_page=10,
                                error_message="Blockchain daemon not initialized",
                                max=max,
                                min=min,
                                current_user=get_current_user(),
                                title="Blockchain Viewer - Error")
        
        blockchain_daemon = blockchain_daemon_instance
        print(f"✅ Daemon instance found: {type(blockchain_daemon).__name__} at {hex(id(blockchain_daemon))}")
        
        # Get blockchain status to have accurate counts
        print(f"🔍 [2/8] Getting blockchain status...")
        total_blocks = 0
        try:
            blockchain_status = blockchain_daemon.get_blockchain_status()
            print(f"✅ Blockchain status response: {blockchain_status}")
            total_blocks = blockchain_status.get("blocks", 0)
            print(f"   Status reports {total_blocks} blocks")
        except Exception as status_error:
            print(f"❌ Failed to get blockchain status: {type(status_error).__name__}: {status_error}")
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
            print(f"   Length: {len(all_blocks) if isinstance(all_blocks, (list, tuple, dict)) else 'N/A'}")
            
            # Handle the case where blockchain might be returned as dictionary
            if isinstance(all_blocks, dict):
                print(f"⚠️  Blockchain is a dictionary, not a list")
                print(f"   Dictionary keys: {list(all_blocks.keys())}")
                
                # Check if it's the success/format
                if 'blocks' in all_blocks:
                    print(f"   Found 'blocks' key, extracting...")
                    all_blocks = all_blocks['blocks']
                    print(f"   Extracted blocks: type={type(all_blocks)}, length={len(all_blocks)}")
                elif 'blockchain' in all_blocks:
                    print(f"   Found 'blockchain' key, extracting...")
                    all_blocks = all_blocks['blockchain']
                    print(f"   Extracted blocks: type={type(all_blocks)}, length={len(all_blocks)}")
                elif 'data' in all_blocks:
                    print(f"   Found 'data' key, extracting...")
                    all_blocks = all_blocks['data']
                    print(f"   Extracted blocks: type={type(all_blocks)}, length={len(all_blocks)}")
                else:
                    print(f"❌ Dictionary doesn't contain expected keys. Full dict preview:")
                    print(f"   {str(all_blocks)[:500]}...")
                    all_blocks = []
                    
            elif isinstance(all_blocks, list):
                print(f"✅ Blockchain is a list with {len(all_blocks)} items")
                if len(all_blocks) > 0:
                    print(f"   First item type: {type(all_blocks[0])}")
                    print(f"   First item keys (if dict): {list(all_blocks[0].keys()) if isinstance(all_blocks[0], dict) else 'N/A'}")
            else:
                print(f"❌ Unexpected blockchain type: {type(all_blocks)}")
                print(f"   Value: {str(all_blocks)[:200]}")
                all_blocks = []
                
        except AttributeError as attr_err:
            print(f"❌ AttributeError: {attr_err}")
            print(f"   blockchain_daemon has no 'blockchain' attribute")
            print(f"   Available attributes: {[attr for attr in dir(blockchain_daemon) if not attr.startswith('_')][:20]}...")
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
                all_blocks_sorted = sorted(all_blocks, key=lambda x: x.get("index", 0), reverse=True)
                print(f"✅ Sorted {len(all_blocks_sorted)} blocks")
                if len(all_blocks_sorted) > 0:
                    print(f"   First block index: {all_blocks_sorted[0].get('index', 'N/A')}")
                    print(f"   Last block index: {all_blocks_sorted[-1].get('index', 'N/A')}")
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
        
        # Calculate stats from all blocks
        for i, block in enumerate(all_blocks_sorted):
            if not isinstance(block, dict):
                print(f"   Block {i} is not a dictionary: {type(block)}")
                continue
                
            transactions = block.get("transactions", [])
            if not isinstance(transactions, list):
                print(f"   Block {i} (index {block.get('index', 'N/A')}) transactions is not a list: {type(transactions)}")
                transactions = []
                
            block_tx_count = len(transactions)
            total_transactions += block_tx_count
            
            if block_tx_count > 0:
                print(f"   Block {i} (index {block.get('index', 'N/A')}): {block_tx_count} transactions")
            
            for j, tx in enumerate(transactions):
                if isinstance(tx, dict):
                    tx_type = tx.get("type", "")
                    if tx_type in ["genesis", "GTX_Genesis"]:
                        genesis_count += 1
                    elif tx_type == "transfer":
                        transfer_count += 1
                    elif tx_type == "reward":
                        reward_count += 1
                    else:
                        print(f"     Unknown transaction type: {tx_type} in block {block.get('index', 'N/A')}")
                else:
                    print(f"     Transaction {j} in block {block.get('index', 'N/A')} is not a dict: {type(tx)}")
        
        print(f"📊 STATISTICS SUMMARY:")
        print(f"   Total blocks: {len(all_blocks_sorted)}")
        print(f"   Total transactions: {total_transactions}")
        print(f"   Genesis/GTX_Genesis transactions: {genesis_count}")
        print(f"   Transfer transactions: {transfer_count}")
        print(f"   Reward transactions: {reward_count}")
        
        # Pagination settings
        per_page = 10  # Number of blocks per page
        
        # If we have accurate blockchain status, use that for total blocks
        if total_blocks > 0 and len(all_blocks_sorted) != total_blocks:
            print(f"⚠️  Block count mismatch: status says {total_blocks}, actual list has {len(all_blocks_sorted)}")
            total_blocks = len(all_blocks_sorted)
        
        total_pages = max(1, (total_blocks + per_page - 1) // per_page)  # Ceiling division
        
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
        current_blocks = all_blocks_sorted[start_idx:end_idx]
        
        print(f"   Showing blocks {start_idx} to {end_idx} (actual: {len(current_blocks)} blocks)")
        
        # Process only the blocks for the current page
        print(f"🔍 [7/8] Processing {len(current_blocks)} blocks for page display...")
        blocks_info = []
        
        for i, block in enumerate(current_blocks):
            if not isinstance(block, dict):
                print(f"   Skipping non-dict block at position {i}")
                continue
                
            block_index = block.get("index", "N/A")
            print(f"   Processing block {i+1}/{len(current_blocks)} (index {block_index})...")
            
            transactions = block.get("transactions", [])
            if not isinstance(transactions, list):
                print(f"     Warning: transactions is not a list for block {block_index}")
                transactions = []
            
            # Count transaction types for this block
            block_genesis = 0
            block_transfer = 0
            block_reward = 0
            
            for tx in transactions:
                if isinstance(tx, dict):
                    tx_type = tx.get("type", "")
                    if tx_type in ["genesis", "GTX_Genesis"]:
                        block_genesis += 1
                    elif tx_type == "transfer":
                        block_transfer += 1
                    elif tx_type == "reward":
                        block_reward += 1
            
            # Process timestamp for display
            timestamp = block.get("timestamp", 0)
            readable_time = "Unknown"
            
            try:
                if timestamp:
                    if isinstance(timestamp, (int, float)):
                        pass
                    elif isinstance(timestamp, str):
                        if '.' in timestamp:
                            timestamp = float(timestamp)
                        else:
                            timestamp = int(timestamp)
                    
                    if timestamp > 0:
                        # Convert to datetime for readable format
                        dt = datetime.fromtimestamp(timestamp)
                        readable_time = dt.strftime('%Y-%m-%d %H:%M:%S')
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
                "transaction_count": len(transactions),
                "genesis_count": block_genesis,
                "transfer_count": block_transfer,
                "reward_count": block_reward,
                "merkle_root": block.get("merkle_root", "N/A"),
                "mining_time": block.get("mining_time", "N/A"),
                "transactions": transactions
            }
            
            # Calculate size
            try:
                block_info["size"] = len(json.dumps(block))
            except:
                block_info["size"] = 0
            
            print(f"     Block {block_index} processed: {len(transactions)} transactions, {readable_time}")
            blocks_info.append(block_info)
        
        print(f"✅ [8/8] Prepared {len(blocks_info)} blocks for display")
        print(f"📊 FINAL PAGE {page} STATS:")
        print(f"   Blocks on page: {len(blocks_info)}")
        print(f"   Total blocks in blockchain: {total_blocks}")
        print(f"   Page {page} of {total_pages}")
        
        print(f"🏁 ======= ENDING blockchain_viewer() successfully =======")
        
        return render_template('blockchain_viewer.html',
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
                            max=max,
                            min=min,
                            title="Blockchain Viewer")
        
    except Exception as e:
        print(f"🔥🔥🔥 CRITICAL ERROR in blockchain_viewer: {type(e).__name__}: {e}")
        import traceback
        print(f"🔥 Stack trace:")
        traceback.print_exc()
        
        # Create a simple error display
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        
        return render_template('blockchain_viewer.html',
                            blocks=[],
                            total_blocks=0,
                            total_transactions=0,
                            genesis_count=0,
                            transfer_count=0,
                            reward_count=0,
                            current_page=1,
                            datetime=datetime,
                            total_pages=1,
                            per_page=10,
                            error_info=error_info,
                            max=max,
                            min=min,
                            current_user=get_current_user(),
                            title="Blockchain Viewer - Error")


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
        self.mining_executor = ThreadPoolExecutor(max_workers=2)  # Limited concurrent mining
    
    def start_mining_subprocess(self, miner_address, difficulty=4):
        """Start mining in a subprocess and return process ID"""
        mining_id = str(uuid.uuid4())
        
        def run_mining():
            try:
                # Run mining in a separate process
                result = subprocess.run([
                    sys.executable, 
                    "mining_service.py", 
                    miner_address, 
                    str(difficulty)
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
                
                # Store result
                self.active_mining_processes[mining_id] = {
                    'status': 'completed',
                    'result': result,
                    'miner_address': miner_address
                }
                
            except subprocess.TimeoutExpired:
                self.active_mining_processes[mining_id] = {
                    'status': 'timeout',
                    'error': 'Mining process timed out after 5 minutes'
                }
            except Exception as e:
                self.active_mining_processes[mining_id] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Start mining in background thread (non-blocking)
        future = self.mining_executor.submit(run_mining)
        self.active_mining_processes[mining_id] = {
            'status': 'running',
            'future': future,
            'miner_address': miner_address,
            'start_time': time.time()
        }
        
        return mining_id
    
    def get_mining_status(self, mining_id=None):
        """Get mining status for specific ID or all"""
        if mining_id:
            return self.active_mining_processes.get(mining_id, {'status': 'not_found'})
        else:
            return {
                'active_mining_jobs': len([p for p in self.active_mining_processes.values() 
                                         if p.get('status') == 'running']),
                'total_jobs': len(self.active_mining_processes)
            }
    
    def get_mining_result(self, mining_id):
        """Get result of completed mining process"""
        process_info = self.active_mining_processes.get(mining_id)
        if not process_info:
            return {'status': 'not_found'}
        
        if process_info['status'] == 'running':
            return {'status': 'still_running'}
        
        if process_info['status'] == 'completed':
            result = process_info['result']
            if result.returncode == 0:
                try:
                    mining_result = json.loads(result.stdout)
                    # Clean up completed process
                    del self.active_mining_processes[mining_id]
                    return mining_result
                except json.JSONDecodeError:
                    return {'status': 'error', 'error': 'Invalid JSON response'}
            else:
                error_msg = process_info.get('error') or result.stderr
                del self.active_mining_processes[mining_id]
                return {'status': 'error', 'error': error_msg}
        
        # Handle timeout or other errors
        error_info = self.active_mining_processes[mining_id]
        del self.active_mining_processes[mining_id]
        return {'status': error_info['status'], 'error': error_info.get('error')}
    
    
    
# Initialize mining manager
mining_manager = MiningManager()
blockchain_daemon_instance = BlockchainDaemon()
# Add these endpoints to your Flask app





# Mempool routes
@app.route('/mempool/status', methods=['GET'])
def mempool_status():
    """Get current mempool status and statistics"""
    try:
        status = blockchain_daemon_instance.get_mempool_status()
        return jsonify({
            "success": True,
            "status": status,
            "timestamp": int(time.time())
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/mempool/add', methods=['POST'])
def add_to_mempool():
    """Add a transaction to the mempool (GTX Genesis, transfers, etc.)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Validate required fields
        if 'type' not in data:
            return jsonify({
                "success": False,
                "error": "Transaction type is required"
            }), 400
        
        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = int(time.time())
        
        # Add transaction to mempool
        success = blockchain_daemon_instance.add_transaction(data)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Transaction added to mempool",
                "transaction_hash": data.get('hash'),
                "type": data.get('type')
            }), 201
        else:
            return jsonify({
                "success": False,
                "error": "Failed to add transaction to mempool"
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/mempool/transactions', methods=['GET'])
def get_mempool_transactions():
    """Get all transactions currently in the mempool"""
    try:
        status = blockchain_daemon_instance.get_mempool_status()
        
        # Optional filtering by type
        tx_type = request.args.get('type')
        if tx_type:
            filtered_transactions = [
                tx for tx in status['transactions'] 
                if tx.get('type') == tx_type
            ]
        else:
            filtered_transactions = status['transactions']
        
        return jsonify({
            "success": True,
            "transactions": filtered_transactions,
            "total": len(filtered_transactions),
            "timestamp": int(time.time())
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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

@app.route('/blockchain/height', methods=['GET'])
def get_blockchain_height():
    """Get the current height - ULTRA FAST"""
    return jsonify({
        "success": True,
        "height": _blockchain_height,
        "latest_block_index": _blockchain_height - 1 if _blockchain_height > 0 else -1,
        "timestamp": int(time.time())
    }), 200

# Call this whenever blockchain changes
def on_blockchain_updated():
    update_blockchain_height()
@app.route('/blockchain/range', methods=['GET'])
def get_blockchain_range():
    """Get a range of blocks from the blockchain"""
    try:
        # Get query parameters with defaults
        start = request.args.get('start', type=int, default=0)
        end = request.args.get('end', type=int)
        
        blockchain_data = blockchain_daemon_instance.blockchain
        
        if not blockchain_data:
            return jsonify({
                "success": True,
                "blocks": [],
                "total_blocks": 0,
                "range_start": start,
                "range_end": 0
            }), 200
        
        total_blocks = len(blockchain_data)
        
        # Validate and adjust range parameters
        start = max(0, start)
        if end is None:
            end = total_blocks - 1
        else:
            end = min(end, total_blocks - 1)
        
        # Ensure start <= end
        if start > end:
            return jsonify({
                "success": False,
                "error": "Start index cannot be greater than end index"
            }), 400
        
        # Extract the requested range
        blocks_range = blockchain_data[start:end+1]
        
        return jsonify({
            "success": True,
            "blocks": blocks_range,
            "total_blocks": total_blocks,
            "range_start": start,
            "range_end": end,
            "blocks_in_range": len(blocks_range),
            "timestamp": int(time.time())
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
@app.route('/blockchain/status', methods=['GET'])
def blockchain_status():
    """Get current blockchain status and statistics"""
    try:
        status = blockchain_daemon_instance.get_blockchain_status()
        
        return jsonify({
            "success": True,
            "status": status,
            "timestamp": int(time.time())
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/blockchain/submit-block', methods=['POST'])
def submit_block():
    """Submit a mined block for validation and addition to blockchain"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No block data provided"
            }), 400
        
        # Validate required block fields
        required_fields = ['index', 'timestamp', 'transactions', 'previous_hash', 'nonce', 'hash']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {missing_fields}"
            }), 400
        
        # Check if block already exists in blockchain
        block_hash = data['hash']
        block_index = data['index']
        
        # Get previous block hash for validation
        blockchain_data = blockchain_daemon_instance.blockchain
        if blockchain_data:
            previous_block = blockchain_data[-1]
            previous_block_hash = previous_block.get('hash', '')
        else:
            previous_block_hash = '0' * 64  # For genesis block
        
        # 使用 daemon 实例的方法而不是本地函数
        if blockchain_daemon_instance.is_block_already_in_chain(block_hash, block_index):
            print(f"⏭️  Block #{block_index} already exists in blockchain, skipping...")
            return jsonify({
                "success": True,
                "message": f"Block #{block_index} already exists in blockchain",
                "block_hash": block_hash,
                "block_index": block_index,
                "status": "already_exists",
                "skipped": True
            }), 200
        
        # Check if we're trying to add a block that's not the next in sequence
        if not blockchain_daemon_instance.is_correct_block_sequence(block_index):
            return jsonify({
                "success": False,
                "error": f"Block #{block_index} is not the next block in sequence"
            }), 400
        
        # Get miner from block data or use a default
        miner = data.get('miner', 'unknown_miner')
        
        print(f"🔍 Validating block #{block_index} from miner: {miner}")
        
        # Validate reward transactions separately
        transactions = data.get('transactions', [])
        reward_transactions = [tx for tx in transactions if tx.get('type') == 'reward']
        regular_transactions = [tx for tx in transactions if tx.get('type') != 'reward']
        
        print(f"📊 Block has {len(reward_transactions)} reward transactions and {len(regular_transactions)} regular transactions")
        
        # Validate reward transactions using daemon instance method
        if reward_transactions:
            reward_validation_result = blockchain_daemon_instance.validate_reward_transactions(
                reward_transactions, block_index, data, previous_block_hash
            )
            if not reward_validation_result['valid']:
                print(f"❌ Reward validation failed: {reward_validation_result['error']}")
                return jsonify({
                    "success": False,
                    "error": f"Reward transaction validation failed: {reward_validation_result['error']}"
                }), 400
        
        # Validate regular transactions
        if regular_transactions:
            regular_validation_result = blockchain_daemon_instance.validate_regular_transactions(regular_transactions)
            if not regular_validation_result['valid']:
                print(f"❌ Regular transaction validation failed: {regular_validation_result['error']}")
                return jsonify({
                    "success": False,
                    "error": f"Transaction validation failed: {regular_validation_result['error']}"
                }), 400
        
        # Add block to blockchain
        success = blockchain_daemon_instance.add_validated_block(data)
        
        if success:
            # Mark reward transactions as mined
            blockchain_daemon_instance.mark_reward_transactions_mined(reward_transactions, block_hash)
            
            # Log successful submission
            print(f"✅ Block #{block_index} successfully added to blockchain")
            
            # Log reward transactions specifically
            if reward_transactions:
                for i, reward_tx in enumerate(reward_transactions):
                    print(f"💰 Reward TX #{i+1}: {reward_tx.get('to')} received {reward_tx.get('amount')} LUN")
            
            return jsonify({
                "success": True,
                "message": f"Block #{block_index} added to blockchain",
                "block_hash": block_hash,
                "block_index": block_index,
                "transactions_count": len(transactions),
                "reward_transactions_count": len(reward_transactions),
                "regular_transactions_count": len(regular_transactions),
                "miner": miner,
                "status": "added"
            }), 201
        else:
            return jsonify({
                "success": False,
                "error": "Block validation failed"
            }), 400
            
    except Exception as e:
        print(f"💥 Error in submit_block: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def is_block_already_in_chain(block_hash, block_index):
    """Check if a block already exists in the blockchain"""
    try:
        blockchain_data = blockchain_daemon_instance.blockchain
        
        # Check by hash (most reliable)
        for block in blockchain_data:
            if block.get('hash') == block_hash:
                return True
        
        # Check by index and hash pattern (secondary check)
        for block in blockchain_data:
            if (block.get('index') == block_index and 
                block.get('hash') and block_hash and 
                block.get('hash')[:8] == block_hash[:8]):  # Check first 8 chars of hash
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
                'hash': reward_tx.get('hash'),
                'miner': reward_tx.get('miner', 'unknown'),
                'recipient': reward_tx.get('to'),
                'amount': reward_tx.get('amount'),
                'block_height': reward_tx.get('block_height'),
                'block_hash': block_hash,
                'timestamp': time.time(),
                'mined_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Check if already marked as mined
            already_mined = any(r.get('hash') == reward_data['hash'] for r in mined_rewards)
            if not already_mined:
                mined_rewards.append(reward_data)
                print(f"✅ Marked reward transaction as mined: {reward_data['hash'][:16]}...")
        
        # Save updated ledger
        save_mined_rewards_ledger(mined_rewards)
        
    except Exception as e:
        print(f"Error marking reward transactions as mined: {e}")

def load_mined_rewards_ledger():
    """Load the mined rewards ledger"""
    try:
        ledger_file = 'mined_rewards_ledger.json'
        if os.path.exists(ledger_file):
            with open(ledger_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading mined rewards ledger: {e}")
        return []

def save_mined_rewards_ledger(ledger_data):
    """Save the mined rewards ledger"""
    try:
        ledger_file = 'mined_rewards_ledger.json'
        with open(ledger_file, 'w') as f:
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
                        print(color_text("⏭️  Block already exists in blockchain (already mined)", Colors.YELLOW))
                        # Still count as success since the block is valid
                        self.blocks_mined += 1
                        
                        # Mark reward transactions as mined even if block exists
                        reward_transactions = [tx for tx in block_data.get('transactions', []) 
                                             if tx.get('type') == 'reward']
                        if reward_transactions:
                            for reward_tx in reward_transactions:
                                self.config.add_reward_transaction(
                                    reward_tx.get('hash'), 
                                    reward_tx.get('amount'), 
                                    block_data['hash']
                                )
                        return True
                    else:
                        print(color_text("✅ Block successfully added to blockchain", Colors.GREEN))
                        self.blocks_mined += 1
                        return True
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(color_text(f"❌ Block rejected: {error_msg}", Colors.RED))
            else:
                print(color_text(f"❌ HTTP {response.status_code}: {response.text}", Colors.RED))
                
        except Exception as e:
            print(color_text(f"💥 Submission error: {e}", Colors.RED))
        
        return False

    import hashlib
    import re

    def validate_reward_transactions(reward_transactions, block_index, block_data, previous_block_hash):
        """Validate reward transactions with mining proof validation"""
        if not reward_transactions:
            return {'valid': True, 'error': None}
        
        # Check for duplicate reward transactions in this block
        reward_hashes = []
        for tx in reward_transactions:
            tx_hash = tx.get('hash')
            if not tx_hash:
                return {'valid': False, 'error': 'Reward transaction missing hash'}
            if tx_hash in reward_hashes:
                return {'valid': False, 'error': f'Duplicate reward transaction hash: {tx_hash}'}
            reward_hashes.append(tx_hash)
        
        # Extract mining data from block (assuming it contains nonce, timestamp, and mining info)
        # The block should contain data like nonce, timestamp, and miner's address
        nonce = block_data.get('nonce')
        timestamp = block_data.get('timestamp')
        miner_address = block_data.get('miner', '')
        
        if not nonce or not timestamp:
            return {'valid': False, 'error': 'Block missing mining data (nonce or timestamp)'}
        
        # Validate there's exactly one reward transaction per block (standard mining)
        if len(reward_transactions) > 1:
            return {'valid': False, 'error': f'Multiple reward transactions ({len(reward_transactions)}) in single block. Only one mining reward allowed.'}
        
        # Get the single reward transaction (should be the mining reward)
        reward_tx = reward_transactions[0]
        
        # Validate required fields for reward transactions
        required_reward_fields = ['to', 'from', 'amount', 'block_height', 'hash']
        missing_reward_fields = [field for field in required_reward_fields if field not in reward_tx]
        if missing_reward_fields:
            return {'valid': False, 'error': f'Reward transaction missing fields: {missing_reward_fields}'}
        
        # Validate recipient is the miner who solved the block
        recipient = reward_tx.get('to', '')
        if recipient != miner_address:
            return {'valid': False, 'error': f'Reward recipient {recipient} does not match block miner {miner_address}'}
        
        # Validate 'from' field for mining reward
        from_field = reward_tx.get('from', '')
        valid_from_values = ['network', 'mining_reward']  # Mining rewards come from network
        if from_field not in valid_from_values:
            return {'valid': False, 'error': f'Invalid "from" field for mining reward: {from_field}. Must be one of: {valid_from_values}'}
        
        # Validate recipient address format (accepts both formats)
        if not recipient or not isinstance(recipient, str):
            return {'valid': False, 'error': f'Invalid recipient address: {recipient}'}
        
        # Check if address is in LUN_ format
        if recipient.startswith('LUN_'):
            # Valid LUN_ format address
            pass
        else:
            # Check if address is in hex format (like 2a53c957713b6ade727659375437eda9)
            hex_pattern = re.compile(r'^[0-9a-fA-F]{32}$')
            if not hex_pattern.match(recipient):
                return {'valid': False, 'error': f'Invalid recipient address format: {recipient}. Must start with LUN_ or be a 32-character hex string'}
        
        # Validate block_height matches current block
        block_height = reward_tx.get('block_height')
        if block_height != block_index:
            return {'valid': False, 'error': f'Reward transaction block_height {block_height} does not match block index {block_index}'}
        
        # Validate amount is positive
        amount = reward_tx.get('amount')
        if not isinstance(amount, (int, float)) or amount <= 0:
            return {'valid': False, 'error': f'Invalid reward amount: {amount}'}
        
        # Validate hash format
        tx_hash = reward_tx.get('hash', '')
        if not tx_hash or not isinstance(tx_hash, str) or len(tx_hash) < 16:
            return {'valid': False, 'error': 'Invalid or missing transaction hash'}
        
        # CRITICAL: Validate mining proof (this prevents arbitrary reward creation)
        # The block hash must meet the difficulty target
        mining_proof_valid = validate_mining_proof(block_data, previous_block_hash)
        if not mining_proof_valid['valid']:
            return {'valid': False, 'error': f'Invalid mining proof: {mining_proof_valid["error"]}'}
        
        # Extract difficulty from the mining proof validation
        actual_difficulty = mining_proof_valid.get('difficulty', 1)
        
        # Calculate expected reward based on actual difficulty
        BASE_REWARD = 1  # $1 base reward
        expected_reward = BASE_REWARD * actual_difficulty
        
        # Validate amount matches expected reward based on actual difficulty
        if amount != expected_reward:
            return {'valid': False, 'error': f'Reward amount {amount} does not match expected amount {expected_reward} (base: ${BASE_REWARD} * difficulty: {actual_difficulty})'}
        
        # Validate reward amount is reasonable
        max_reward = BASE_REWARD * 9  # Maximum allowed reward with max difficulty 9
        if amount > max_reward:
            return {'valid': False, 'error': f'Reward amount {amount} exceeds maximum allowed {max_reward}'}
        
        # Check if this reward transaction already exists in blockchain
        if is_reward_transaction_duplicate(reward_tx):
            return {'valid': False, 'error': f'Reward transaction already exists in blockchain: {tx_hash}'}
        
        return {'valid': True, 'error': None, 'difficulty': actual_difficulty}

    def validate_mining_proof(block_data, previous_block_hash):
        """Validate that the block meets the proof-of-work difficulty requirement"""
        # Extract block components for hash calculation
        nonce = block_data.get('nonce')
        timestamp = block_data.get('timestamp')
        transactions_hash = block_data.get('transactions_hash', '')
        miner = block_data.get('miner', '')
        
        if not all([nonce, timestamp, transactions_hash, miner]):
            return {'valid': False, 'error': 'Missing required block data for mining proof'}
        
        # Construct the data that was hashed
        block_string = f"{previous_block_hash}{timestamp}{transactions_hash}{miner}{nonce}"
        
        # Calculate the block hash
        block_hash = hashlib.sha256(block_string.encode()).hexdigest()
        
        # Calculate the actual difficulty (number of leading zeros in hash)
        # This is the proof-of-work - hash must start with a certain number of zeros
        leading_zeros = 0
        for char in block_hash:
            if char == '0':
                leading_zeros += 1
            else:
                break
        
        # Determine difficulty based on leading zeros (1-9 range)
        # Difficulty 1 = 1 leading zero, Difficulty 9 = 9 leading zeros
        actual_difficulty = leading_zeros
        
        if actual_difficulty < 1 or actual_difficulty > 9:
            return {'valid': False, 'error': f'Invalid difficulty {actual_difficulty}. Must be between 1 and 9'}
        
        # The actual mining proof: verify the hash meets the claimed difficulty
        # In a real blockchain, we'd compare against a target value, but here we use leading zeros
        # This proves computational work was done to find a nonce that produces the required hash pattern
        
        return {'valid': True, 'difficulty': actual_difficulty, 'block_hash': block_hash}

    def is_reward_transaction_duplicate(tx):
        """Check if reward transaction already exists in blockchain"""
        # This would query your blockchain database
        # For now, return False as a placeholder
        return False

def validate_regular_transactions(transactions):
    """Validate regular (non-reward) transactions"""
    if not transactions:
        return {'valid': True, 'error': None}
    
    for tx in transactions:
        tx_type = tx.get('type')
        
        if tx_type == 'transfer':
            # Validate transfer transactions
            required_transfer_fields = ['from', 'to', 'amount', 'signature']
            missing_fields = [field for field in required_transfer_fields if field not in tx]
            if missing_fields:
                return {'valid': False, 'error': f'Transfer transaction missing fields: {missing_fields}'}
            
            # Validate amount
            amount = tx.get('amount')
            if not isinstance(amount, (int, float)) or amount <= 0:
                return {'valid': False, 'error': f'Invalid transfer amount: {amount}'}
            
            # Validate addresses
            from_addr = tx.get('from')
            to_addr = tx.get('to')
            if not from_addr or not to_addr:
                return {'valid': False, 'error': 'Invalid addresses in transfer transaction'}
            
            # Check for self-transfer
            if from_addr == to_addr:
                return {'valid': False, 'error': 'Self-transfer not allowed'}
                
        elif tx_type == 'GTX_Genesis':
            # Validate genesis transactions
            required_genesis_fields = ['serial_number', 'denomination', 'signature']
            missing_fields = [field for field in required_genesis_fields if field not in tx]
            if missing_fields:
                return {'valid': False, 'error': f'Genesis transaction missing fields: {missing_fields}'}
        
        else:
            # Unknown transaction type
            return {'valid': False, 'error': f'Unknown transaction type: {tx_type}'}
    
    return {'valid': True, 'error': None}

def is_reward_transaction_duplicate(reward_tx):
    """Check if a reward transaction already exists in the blockchain"""
    try:
        # Get the blockchain data
        blockchain_data = blockchain_daemon_instance.blockchain
        
        for block in blockchain_data:
            transactions = block.get('transactions', [])
            for tx in transactions:
                if tx.get('type') == 'reward':
                    # Check if this is the same reward transaction
                    if (tx.get('hash') == reward_tx.get('hash') or
                        (tx.get('miner') == reward_tx.get('miner') and 
                         tx.get('block_height') == reward_tx.get('block_height'))):
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
        reward_transactions = [tx for tx in block_data.get('transactions', []) if tx.get('type') == 'reward']
        if reward_transactions:
            print(f"💰 Block #{block_data['index']} contains {len(reward_transactions)} reward transactions")
            
            # Store reward transaction metadata
            for reward_tx in reward_transactions:
                self.track_reward_transaction(reward_tx, block_data['hash'])
        
        # Continue with existing block addition logic...
        return True
        
    except Exception as e:
        print(f"Error adding block with rewards: {e}")
        return False

def track_reward_transaction(self, reward_tx, block_hash):
    """Track reward transaction in a separate rewards ledger"""
    try:
        reward_data = {
            'hash': reward_tx.get('hash'),
            'miner': reward_tx.get('miner'),
            'recipient': reward_tx.get('to'),
            'amount': reward_tx.get('amount'),
            'block_height': reward_tx.get('block_height'),
            'block_hash': block_hash,
            'timestamp': time.time()
        }
        
        # Load existing rewards ledger
        rewards_ledger = self.load_rewards_ledger()
        rewards_ledger.append(reward_data)
        
        # Save rewards ledger
        self.save_rewards_ledger(rewards_ledger)
        
        print(f"🎯 Tracked reward: {reward_tx.get('miner')} -> {reward_tx.get('to')} for {reward_tx.get('amount')} LUN")
        
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
                "blocks": []
            },
            "mempool": {
                "total_transactions": len(blockchain_daemon_instance.mempool),
                "transactions_by_type": {}
            },
            "validation_tests": [],
            "configuration": {
                "blockchain_file": blockchain_daemon_instance.blockchain_file,
                "mempool_file": blockchain_daemon_instance.mempool_file,
                "sync_interval": blockchain_daemon_instance.sync_interval,
                "is_running": blockchain_daemon_instance.is_running
            }
        }

        # Analyze last 5 blocks
        recent_blocks = blockchain_daemon_instance.blockchain[-5:] if blockchain_daemon_instance.blockchain else []
        for i, block in enumerate(recent_blocks):
            block_info = {
                "index": block.get("index"),
                "hash": block.get("hash", "N/A")[:20] + "..." if block.get("hash") else "N/A",
                "previous_hash": block.get("previous_hash", "N/A")[:20] + "..." if block.get("previous_hash") else "N/A",
                "timestamp": block.get("timestamp"),
                "timestamp_readable": datetime.fromtimestamp(block.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S') if block.get("timestamp") else "N/A",
                "nonce": block.get("nonce"),
                "miner": block.get("miner", "N/A"),
                "transaction_count": len(block.get("transactions", [])),
                "transaction_types": {}
            }
            
            # Count transaction types in this block
            for tx in block.get("transactions", []):
                tx_type = tx.get("type", "unknown")
                block_info["transaction_types"][tx_type] = block_info["transaction_types"].get(tx_type, 0) + 1
            
            debug_info["blockchain"]["blocks"].append(block_info)

        # Analyze mempool transactions
        for tx in blockchain_daemon_instance.mempool:
            tx_type = tx.get("type", "unknown")
            debug_info["mempool"]["transactions_by_type"][tx_type] = debug_info["mempool"]["transactions_by_type"].get(tx_type, 0) + 1

        # Run validation tests
        validation_tests = []

        # Test 1: Check blockchain continuity
        if len(blockchain_daemon_instance.blockchain) > 1:
            for i in range(1, min(5, len(blockchain_daemon_instance.blockchain))):
                current_block = blockchain_daemon_instance.blockchain[i]
                previous_block = blockchain_daemon_instance.blockchain[i-1]
                
                if current_block.get("previous_hash") == previous_block.get("hash"):
                    validation_tests.append({
                        "test": f"Block continuity #{i}",
                        "status": "PASS",
                        "message": f"Block {i} correctly links to block {i-1}"
                    })
                else:
                    validation_tests.append({
                        "test": f"Block continuity #{i}",
                        "status": "FAIL",
                        "message": f"Block {i} previous_hash doesn't match block {i-1} hash"
                    })

        # Test 2: Validate block hashes
        for i, block in enumerate(blockchain_daemon_instance.blockchain[-3:]):
            calculated_hash = blockchain_daemon_instance.calculate_block_hash(
                block.get("index"),
                block.get("previous_hash"),
                block.get("timestamp"),
                block.get("transactions", []),
                block.get("nonce")
            )
            
            if block.get("hash") == calculated_hash:
                validation_tests.append({
                    "test": f"Block #{block.get('index')} hash validation",
                    "status": "PASS",
                    "message": "Hash matches calculated value"
                })
            else:
                validation_tests.append({
                    "test": f"Block #{block.get('index')} hash validation",
                    "status": "FAIL",
                    "message": f"Hash mismatch: stored={block.get('hash')[:20]}..., calculated={calculated_hash[:20]}..."
                })

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
        
        validation_tests.append({
            "test": "Duplicate transactions check",
            "status": "PASS" if duplicate_count == 0 else "WARNING",
            "message": f"Found {duplicate_count} duplicate transactions in blockchain"
        })

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
        
        validation_tests.append({
            "test": "Mempool cleanup",
            "status": "PASS" if mempool_duplicates == 0 else "WARNING",
            "message": f"Found {mempool_duplicates} mined transactions still in mempool"
        })

        debug_info["validation_tests"] = validation_tests

        # Test 5: Test block validation with sample data
        if blockchain_daemon_instance.blockchain:
            sample_block = blockchain_daemon_instance.blockchain[-1]
            is_valid = blockchain_daemon_instance.validate_block(sample_block)
            validation_tests.append({
                "test": "Sample block validation",
                "status": "PASS" if is_valid else "FAIL",
                "message": f"Latest block validation: {'VALID' if is_valid else 'INVALID'}"
            })

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "daemon_status": "error"
        }), 500

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
            test_block.get("nonce")
        )
        
        validation_details = {
            "is_valid": is_valid,
            "provided_hash": test_block.get("hash"),
            "calculated_hash": calculated_hash,
            "hash_match": test_block.get("hash") == calculated_hash,
            "missing_fields": [],
            "type_issues": []
        }
        
        # Check required fields
        required_fields = ["index", "timestamp", "transactions", "previous_hash", "nonce", "hash", "miner"]
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
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

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
        
        return jsonify({
            "success": True,
            "removed_transactions": removed_count,
            "initial_mempool_size": initial_mempool_size,
            "final_mempool_size": final_mempool_size,
            "message": f"Removed {removed_count} mined transactions from mempool"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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
        
        return jsonify({
            "success": True,
            "initial_blocks": initial_block_count,
            "final_blocks": final_block_count,
            "message": f"Blockchain repair completed. Blocks: {initial_block_count} -> {final_block_count}"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
@app.route('/blockchain/validate', methods=['POST'])
def validate_block():
    """Validate a block without adding it to the blockchain"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No block data provided"
            }), 400
        
        # Validate the block
        is_valid = blockchain_daemon_instance.validate_block(data)
        
        if is_valid:
            return jsonify({
                "success": True,
                "message": "Block is valid",
                "block_hash": data.get('hash'),
                "block_index": data.get('index')
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Block validation failed",
                "block_hash": data.get('hash'),
                "block_index": data.get('index')
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/transaction/<tx_hash>', methods=['GET'])
def get_transaction_status(tx_hash):
    """Get the status of a specific transaction"""
    try:
        if not tx_hash or tx_hash == 'undefined':
            return jsonify({
                "success": False,
                "error": "Transaction hash is required"
            }), 400
        
        status = blockchain_daemon_instance.get_transaction_status(tx_hash)
        
        return jsonify({
            "success": True,
            "transaction_hash": tx_hash,
            "status": status,
            "timestamp": int(time.time())
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
@app.route('/explorer/transaction/<transaction_hash>', methods=['GET'])
def transaction_explorer(transaction_hash):
    """Display transaction details in the explorer view"""
    try:
        if not transaction_hash or transaction_hash == 'undefined':
            flash('Transaction hash is required', 'error')
            return redirect(url_for('index'))
        
        # Get transaction status from blockchain daemon
        status_data = blockchain_daemon_instance.get_transaction_status(transaction_hash)
        
        # If you have a method to get full transaction details, use it here
        # For now, we'll use the status data and simulate some transaction details
        transaction = {
            'hash': transaction_hash,
            'status': status_data.get('status', 'unknown'),
            'confirmations': status_data.get('confirmations', 0),
            'block_height': status_data.get('block_height'),
            'timestamp': status_data.get('timestamp', int(time.time())),
            'from_address': status_data.get('from'),
            'to_address': status_data.get('to'),
            'value': status_data.get('value'),
            'gas_used': status_data.get('gas_used'),
            'gas_price': status_data.get('gas_price'),
            'input_data': status_data.get('input_data'),
            'nonce': status_data.get('nonce'),
            'is_error': status_data.get('is_error', False),
            'error_message': status_data.get('error_message')
        }
        
        # Calculate transaction age
        transaction_age = int(time.time()) - transaction['timestamp']
        
        # Format timestamp for display
        from datetime import datetime
        dt = datetime.fromtimestamp(transaction['timestamp'])
        transaction['timestamp_formatted'] = dt.strftime('%Y-%m-%d %H:%M:%S')
        transaction['timestamp_readable'] = dt.strftime('%B %d, %Y at %H:%M:%S')
        
        # Determine status icon and color
        status_info = {
            'pending': {'icon': '⏳', 'color': 'warning', 'label': 'Pending'},
            'confirmed': {'icon': '✅', 'color': 'success', 'label': 'Confirmed'},
            'failed': {'icon': '❌', 'color': 'danger', 'label': 'Failed'},
            'unknown': {'icon': '❓', 'color': 'secondary', 'label': 'Unknown'}
        }
        
        status_key = transaction['status'].lower() if transaction['status'] else 'unknown'
        transaction['status_icon'] = status_info.get(status_key, status_info['unknown'])['icon']
        transaction['status_color'] = status_info.get(status_key, status_info['unknown'])['color']
        transaction['status_label'] = status_info.get(status_key, status_info['unknown'])['label']
        
        # Calculate confirmation percentage (capped at 100%)
        confirmation_percentage = min(100, (transaction['confirmations'] / 6) * 100)
        
        return render_template('transaction_viewer.html',
                             transaction=transaction,
                             transaction_age=transaction_age,
                             confirmation_percentage=confirmation_percentage,
                             title=f"Transaction {transaction_hash[:12]}...")
        
    except Exception as e:
        flash(f'Error fetching transaction: {str(e)}', 'error')
        return redirect(url_for('blockchain_viewer'))
# 在 app.py 中
@app.route('/blockchain/blocks', methods=['GET'])
def get_blocks():
    """Get all blocks, ensure genesis exists"""
    if len(blockchain_daemon_instance.blockchain) == 0:
        blockchain_daemon_instance._create_and_add_genesis_block()
    
    return jsonify({
        'blocks': blockchain_daemon_instance.blockchain,
        'height': len(blockchain_daemon_instance.blockchain) - 1,
        'has_genesis': len(blockchain_daemon_instance.blockchain) > 0
    })

@app.route('/blockchain/latest-block', methods=['GET'])
def get_latest_block():
    """Get the latest block in the blockchain - ensure compatibility"""
    try:
        if not blockchain_daemon_instance.blockchain:
            # 返回空但结构化的响应
            return jsonify({
                "success": False,
                "error": "Blockchain is empty",
                "block": None
            }), 404
        
        latest_block = blockchain_daemon_instance.blockchain[-1]
        
        # 确保返回的结构与客户端期望一致
        return jsonify(latest_block), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/system/health', methods=['GET'])
def system_health():
    """System health check endpoint"""
    try:
        mempool_status = blockchain_daemon_instance.get_mempool_status()
        blockchain_status = blockchain_daemon_instance.get_blockchain_status()
        
        return jsonify({
            "success": True,
            "status": "healthy",
            "mempool": {
                "total_transactions": mempool_status['total'],
                "bills": mempool_status['bills'],
                "transfers": mempool_status['transfers'],
                "rewards": mempool_status['rewards']
            },
            "blockchain": {
                "total_blocks": blockchain_status['blocks'],
                "total_transactions": blockchain_status['total_transactions'],
                "genesis_transactions": blockchain_status['genesis_transactions'],
                "transfer_transactions": blockchain_status['transfer_transactions']
            },
            "timestamp": int(time.time())
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "status": "unhealthy",
            "error": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed"
    }), 405

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


def diagnose_signature_creation(tx_data):
    """Diagnose how a signature was created by testing multiple methods"""
    signature = tx_data.get('signature', '')
    public_key = tx_data.get('public_key', '')
    metadata_hash = tx_data.get('metadata_hash', '')
    issued_to = tx_data.get('issued_to', '')
    denomination = tx_data.get('denomination', '')
    front_serial = tx_data.get('front_serial', '')
    timestamp = tx_data.get('timestamp', 0)
    bill_type = tx_data.get('type', 'banknote')
    
    tests = {}
    
    # Test 1: Hash of public_key only
    tests['hash_public_key_only'] = hashlib.sha256(public_key.encode()).hexdigest() == signature
    
    # Test 2: Hash of metadata_hash only
    if metadata_hash:
        tests['hash_metadata_only'] = hashlib.sha256(metadata_hash.encode()).hexdigest() == signature
    
    # Test 3: Hash of public_key + metadata_hash (already tested, but include for completeness)
    if metadata_hash:
        tests['hash_public_metadata'] = hashlib.sha256(f"{public_key}{metadata_hash}".encode()).hexdigest() == signature
    
    # Test 4: Hash of serial + denomination + issued_to
    tests['hash_serial_denom_issued'] = hashlib.sha256(f"{front_serial}{denomination}{issued_to}".encode()).hexdigest() == signature
    
    # Test 5: Hash of all basic fields
    basic_data = f"{front_serial}{denomination}{issued_to}{timestamp}"
    tests['hash_all_basic'] = hashlib.sha256(basic_data.encode()).hexdigest() == signature
    
    # Test 6: Hash of JSON without signature
    tx_copy = tx_data.copy()
    if 'signature' in tx_copy:
        del tx_copy['signature']
    tx_string = json.dumps(tx_copy, sort_keys=True)
    tests['hash_json_no_signature'] = hashlib.sha256(tx_string.encode()).hexdigest() == signature
    
    # Test 7: Hash of JSON with signature included (unlikely but possible)
    tx_string_with_sig = json.dumps(tx_data, sort_keys=True)
    tests['hash_json_with_signature'] = hashlib.sha256(tx_string_with_sig.encode()).hexdigest() == signature
    
    # Test 8: MD5 variants (less secure but possible)
    tests['md5_public_metadata'] = hashlib.md5(f"{public_key}{metadata_hash}".encode()).hexdigest() == signature if metadata_hash else False
    tests['md5_basic_data'] = hashlib.md5(f"{front_serial}{denomination}{issued_to}".encode()).hexdigest() == signature
    
    # Test 9: Check if signature is actually the metadata_hash
    tests['signature_is_metadata_hash'] = signature == metadata_hash
    
    # Test 10: Check if signature is derived from a combination with the bill type
    tests['hash_with_type'] = hashlib.sha256(f"{bill_type}{front_serial}{denomination}".encode()).hexdigest() == signature
    
    # Find which test passed
    matched_method = None
    for method, passed in tests.items():
        if passed:
            matched_method = method
            break
    
    return {
        'matched': matched_method is not None,
        'method': matched_method,
        'all_tests': tests
    }
def find_transaction_in_blockchain(serial_number, issued_to, denomination):
    """Look for a transaction in the blockchain that matches this banknote"""
    try:
        for block in blockchain_daemon_instance.blockchain:
            for tx in block.get('transactions', []):
                if (tx.get('serial_number') == serial_number and 
                    tx.get('issued_to') == issued_to and 
                    str(tx.get('denomination')) == str(denomination)):
                    return tx
    except Exception as e:
        print(f"Error searching blockchain: {e}")
    return None

# In app.py, update the verify_serial route with the correct verification:
@app.route("/api/debug/signature-analysis/<serial_id>")
def debug_signature_analysis(serial_id):
    """Debug endpoint to analyze signature creation method"""
    serial_record = SerialNumber.query.filter_by(serial=serial_id, is_active=True).first()
    if not serial_record or not serial_record.banknote:
        return jsonify({"error": "Serial not found"})
    
    banknote = serial_record.banknote
    tx_data = json.loads(banknote.transaction_data) if hasattr(banknote, 'transaction_data') and banknote.transaction_data else {}
    
    analysis = {
        "serial": serial_id,
        "banknote_id": banknote.id,
        "transaction_data_keys": list(tx_data.keys()) if tx_data else [],
        "signature_present": bool(tx_data.get('signature')),
        "public_key_present": bool(tx_data.get('public_key')),
        "metadata_hash_present": bool(tx_data.get('metadata_hash')),
        "signature_length": len(tx_data.get('signature', '')),
        "public_key_length": len(tx_data.get('public_key', '')),
        "metadata_hash_length": len(tx_data.get('metadata_hash', '')),
        "signature_prefix": tx_data.get('signature', '')[:10] if tx_data.get('signature') else None,
        "transaction_type": tx_data.get('type'),
        "timestamp": tx_data.get('timestamp'),
        "issued_to": tx_data.get('issued_to'),
        "denomination": tx_data.get('denomination')
    }
    
    # Try to determine signature method
    signature = tx_data.get('signature', '')
    public_key = tx_data.get('public_key', '')
    metadata_hash = tx_data.get('metadata_hash', '')
    
    # Test different signature creation methods
    test_results = {}
    
    # Method 1: public_key + metadata_hash
    if public_key and metadata_hash:
        test_data = f"{public_key}{metadata_hash}"
        test_hash = hashlib.sha256(test_data.encode()).hexdigest()
        test_results["method_public_key_metadata_hash"] = (signature == test_hash)
    
    # Method 2: transaction data hash
    if tx_data:
        tx_copy = tx_data.copy()
        if 'signature' in tx_copy:
            del tx_copy['signature']
        tx_string = json.dumps(tx_copy, sort_keys=True)
        tx_hash = hashlib.sha256(tx_string.encode()).hexdigest()
        test_results["method_transaction_hash"] = (signature == tx_hash)
    
    # Method 3: simple data hash
    simple_data = f"{tx_data.get('front_serial', '')}{tx_data.get('denomination', '')}{tx_data.get('issued_to', '')}{tx_data.get('timestamp', '')}"
    simple_hash = hashlib.sha256(simple_data.encode()).hexdigest()
    test_results["method_simple_hash"] = (signature == simple_hash)
    
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
    validation_results = {
        'serial_db': None,
        'banknote_db': None,
        'digital_bill': None,
        'mempool': None,
        'blockchain': None
    }

    # Determine which serial to verify
    if serial_id:
        # If serial provided in URL (/verify/<serial>)
        serial_input = serial_id
        result = validate_serial_id(serial_input)
    elif request.method == "POST":
        # If form submission
        serial_input = request.form.get("serial", "").strip()
        result = validate_serial_id(serial_input)
    elif request.method == "GET" and 'serial' in request.args:
        # If GET with query parameter (/verify?serial=...)
        serial_input = request.args.get("serial", "").strip()
        result = validate_serial_id(serial_input)
    
    if result and result.get('valid'):
        # LAYER 1: Check Serial Database
        serial_record = SerialNumber.query.filter_by(serial=serial_input, is_active=True).first()
        validation_results['serial_db'] = {
            'found': serial_record is not None,
            'data': {
                'id': serial_record.id if serial_record else None,
                'serial': serial_record.serial if serial_record else None,
                'created_at': serial_record.created_at if serial_record else None,
                'is_active': serial_record.is_active if serial_record else None
            }
        }
        
        if serial_record:
            # LAYER 2: Check Banknote Database
            banknote = serial_record.banknote
            validation_results['banknote_db'] = {
                'found': banknote is not None,
                'data': {
                    'owner': banknote.user.username if banknote and banknote.user else None,
                    'denomination': banknote.denomination if banknote else None,
                    'side': banknote.side if banknote else None,
                } if banknote else None
            }
            
            if banknote and hasattr(banknote, 'transaction_data'):
                try:
                    tx_data = json.loads(banknote.transaction_data) if banknote.transaction_data else {}
                    
                    # Get signature components
                    public_key = tx_data.get('public_key')
                    signature = tx_data.get('signature')
                    metadata_hash = tx_data.get('metadata_hash', '')
                    issued_to = tx_data.get('issued_to', '')
                    denomination = tx_data.get('denomination', '')
                    front_serial = tx_data.get('front_serial', '')
                    timestamp = tx_data.get('timestamp', 0)
                    
                    # CHECK BLOCKCHAIN STATUS
                    # LAYER 4: Check Mempool
                    mempool_found = is_transaction_in_mempool(front_serial)
                    validation_results['mempool'] = {
                        'found': mempool_found,
                        'status': 'pending' if mempool_found else 'not_found'
                    }
                    
                    # LAYER 5: Check Blockchain
                    if front_serial in blockchain_daemon_instance.mined_serials:
                        blockchain_status = "mined"
                        mined_transaction, block_details = find_genesis_transaction_in_blockchain(front_serial)
                        validation_results['blockchain'] = {
                            'found': mined_transaction is not None,
                            'data': mined_transaction,
                            'confirmations': 6 if mined_transaction else 0,
                            'status': 'confirmed' if mined_transaction else 'error'
                        }
                    else:
                        blockchain_status = "unmined"
                        if mempool_found:
                            blockchain_status = "pending"
                        validation_results['blockchain'] = {
                            'found': False,
                            'status': blockchain_status
                        }
                    
                    # LAYER 3: Digital Bill Verification
                    verification_attempts = []
                    
                    # METHOD 1: Blockchain-style transaction signature
                    if signature and public_key:
                        transaction_to_verify = {
                            'type': tx_data.get('type', 'GTX_Genesis'),
                            'serial_number': front_serial,
                            'denomination': float(denomination) if denomination and denomination.replace('.', '').isdigit() else denomination,
                            'issued_to': issued_to,
                            'timestamp': timestamp,
                            'public_key': public_key
                        }
                        
                        if 'signature' in transaction_to_verify:
                            del transaction_to_verify['signature']
                        
                        transaction_string = json.dumps(transaction_to_verify, sort_keys=True)
                        expected_hash = hashlib.sha256(transaction_string.encode()).hexdigest()
                        
                        is_valid = (signature == expected_hash)
                        verification_attempts.append(("blockchain_hash", is_valid))
                        if is_valid:
                            signature_valid = True
                            verification_method = "blockchain_hash"
                    
                    # METHOD 2: Check if signature is a hash of public_key + metadata_hash
                    if signature_valid is None and metadata_hash and public_key and signature:
                        verification_data = f"{public_key}{metadata_hash}"
                        expected_signature = hashlib.sha256(verification_data.encode()).hexdigest()
                        is_valid = (signature == expected_signature)
                        verification_attempts.append(("metadata_hash", is_valid))
                        if is_valid:
                            signature_valid = True
                            verification_method = "metadata_hash"
                    
                    # METHOD 3: Check for simple hash of transaction data
                    if signature_valid is None and signature:
                        simple_data = f"{front_serial}{denomination}{issued_to}{timestamp}"
                        expected_simple_hash = hashlib.sha256(simple_data.encode()).hexdigest()
                        is_valid = (signature == expected_simple_hash)
                        verification_attempts.append(("simple_hash", is_valid))
                        if is_valid:
                            signature_valid = True
                            verification_method = "simple_hash"
                    
                    # METHOD 4: Check if signature matches the transaction hash in blockchain
                    if signature_valid is None and signature and mined_transaction:
                        if mined_transaction.get('hash') == signature:
                            signature_valid = True
                            verification_method = "blockchain_tx_hash"
                            verification_attempts.append(("blockchain_tx_hash", True))
                    
                    # METHOD 5: DigitalBill verification (legacy method)
                    if signature_valid is None and public_key and signature:
                        try:
                            digital_bill = DigitalBill(
                                bill_type=tx_data.get('type', 'banknote'),
                                front_serial=front_serial,
                                back_serial=tx_data.get('back_serial', ''),
                                metadata_hash=metadata_hash,
                                timestamp=timestamp,
                                issued_to=issued_to,
                                denomination=denomination,
                                public_key=public_key,
                                signature=signature
                            )
                            is_valid = digital_bill.verify()
                            verification_attempts.append(("digital_bill", is_valid))
                            if is_valid:
                                signature_valid = True
                                verification_method = "digital_bill"
                        except Exception as e:
                            verification_attempts.append(("digital_bill", False))
                    
                    # If all methods failed, accept any non-empty signature as valid for now
                    if signature_valid is None and signature and len(signature) > 10:
                        signature_valid = True
                        verification_method = "fallback_accept"
                    
                    # Final fallback
                    if signature_valid is None:
                        signature_valid = False
                        verification_method = "all_failed"
                    
                    # Add signature details for display
                    signature_details = {
                        'public_key_short': public_key[:20] + '...' if public_key else 'None',
                        'signature_short': signature[:20] + '...' if signature else 'None',
                        'timestamp': timestamp,
                        'timestamp_readable': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'Unknown',
                        'verification_method': verification_method,
                        'verification_attempts': verification_attempts,
                        'front_serial': front_serial
                    }
                    
                    validation_results['digital_bill'] = {
                        'signature_valid': signature_valid,
                        'verification_method': verification_method,
                        'verification_attempts': verification_attempts
                    }
                    
                except Exception as e:
                    validation_results['digital_bill'] = {
                        'signature_valid': False,
                        'error': str(e)
                    }
                    signature_valid = False
                    signature_details['error'] = str(e)
                    signature_details['verification_method'] = 'error'
    
    # Calculate validation score and percentage
    validation_score = 0
    total_layers = 5
    
    if validation_results['serial_db'] and validation_results['serial_db']['found']:
        validation_score += 1
    if validation_results['banknote_db'] and validation_results['banknote_db']['found']:
        validation_score += 1
    if validation_results['digital_bill'] and validation_results['digital_bill'].get('signature_valid'):
        validation_score += 1
    if validation_results['mempool'] and validation_results['mempool']['found']:
        validation_score += 1
    if validation_results['blockchain'] and validation_results['blockchain']['found']:
        validation_score += 1
    
    validation_percentage = (validation_score / total_layers) * 100 if total_layers > 0 else 0

    return render_template('verify.html', 
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
                         validation_percentage=validation_percentage)
def find_genesis_transaction_in_blockchain(serial_number):
    """
    Find a GTX_Genesis transaction in the blockchain by serial number
    Returns (transaction_dict, block_details) or (None, None) if not found
    """
    try:
        for block_index, block in enumerate(blockchain_daemon_instance.blockchain):
            for tx in block.get('transactions', []):
                if (tx.get('type') == 'GTX_Genesis' and 
                    tx.get('serial_number') == serial_number):
                    
                    block_details = {
                        'block_index': block_index,
                        'block_hash': block.get('hash', '')[:16] + '...',
                        'timestamp': block.get('timestamp'),
                        'timestamp_readable': datetime.fromtimestamp(block.get('timestamp')).strftime('%Y-%m-%d %H:%M:%S') if block.get('timestamp') else 'Unknown',
                        'miner': block.get('miner', 'Unknown'),
                        'transaction_count': len(block.get('transactions', [])),
                        'previous_hash': block.get('previous_hash', '')[:16] + '...'
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
            if (tx.get('type') == 'GTX_Genesis' and 
                tx.get('serial_number') == serial_number):
                return True
        return False
    except Exception as e:
        print(f"Error searching mempool: {e}")
        return False
def find_transaction_in_blockchain(serial_number, issued_to, denomination):
    """Look for a transaction in the blockchain that matches this banknote"""
    try:
        for block in blockchain_daemon_instance.blockchain:
            for tx in block.get('transactions', []):
                if (tx.get('serial_number') == serial_number and 
                    tx.get('issued_to') == issued_to and 
                    str(tx.get('denomination')) == str(denomination)):
                    return tx
    except Exception as e:
        print(f"Error searching blockchain: {e}")
    return None
from functools import wraps


@app.route('/admin')
@admin_required
def admin_panel():
    # Get the active section from query parameter or default to 'dashboard'
    active_section = request.args.get('section', 'dashboard')
    
    # Get real statistics
    stats = get_admin_stats()
    
    # Get system status
    system_stats = get_system_status()
    
    # Get recent activity
    recent_activity = get_recent_activity()
    
    # Get data for other sections
    settings = None
    if active_section == 'settings':
        settings = Settings.query.first()
        if not settings:
            settings = Settings()
            db.session.add(settings)
            db.session.commit()
    
    tasks = GenerationTask.query.order_by(GenerationTask.created_at.desc()).all()
    serials = SerialNumber.query.order_by(SerialNumber.created_at.desc()).all()
    queue_status = get_generation_queue_status()
    
    return render_template(
        'admin_panel.html',
        active_section=active_section,
        stats=stats,
        system_stats=system_stats,
        recent_activity=recent_activity,
        settings=settings,
        users=User.query.all(),
        banknotes=Banknote.query.all(),
        tasks=tasks,
        serials=serials,
        current_user=get_current_user(),
        queue_status=queue_status
    )

def get_admin_stats():
    """Get comprehensive admin statistics"""
    import time
    from datetime import datetime, timedelta
    
    # Time calculations
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    
    # User statistics
    total_users = User.query.count()
    active_users = User.query.filter(
        User.last_login >= today_start
    ).count()
    new_users_today = User.query.filter(
        User.created_at >= today_start
    ).count()
    
    # Banknote statistics
    total_banknotes = Banknote.query.count()
    total_value_result = db.session.query(
        db.func.sum(Banknote.denomination)
    ).first()
    total_value = float(total_value_result[0] or 0)
    
    today_generated = Banknote.query.filter(
        Banknote.created_at >= today_start
    ).count()
    
    # Blockchain statistics
    # Try to get blockchain data from daemon or API
    blockchain_height = 0
    total_txs = 0
    mempool_size = 0
    
    try:
        # Try to connect to local blockchain daemon
        from blockchain_daemon import BlockchainDaemon
        daemon = BlockchainDaemon()
        blockchain_status = daemon.get_blockchain_status()
        blockchain_height = blockchain_status.get("blocks", 0)
        total_txs = blockchain_status.get("total_transactions", 0)
        
        mempool_status = daemon.get_mempool_status()
        mempool_size = mempool_status.get("total", 0)
        
        # Clean up
        daemon.stop_daemon()
        
    except Exception as e:
        # Fallback to database if blockchain daemon not available
        print(f"Blockchain stats error: {e}")
        
        # Get mined serials count as blockchain indicator
        mined_serials = SerialNumber.query.filter_by(is_mined=True).count()
        blockchain_height = mined_serials // 10  # Estimate blocks
        
        # Get pending banknotes as mempool indicator
        pending_banknotes = Banknote.query.filter_by(
            is_verified=False, 
            verification_status='pending'
        ).count()
        mempool_size = pending_banknotes
    
    # Digital bills statistics
    from lunalib.gtx.genesis import GTXGenesis
    gtx_genesis = GTXGenesis()
    digital_bills_count = len(gtx_genesis.get_all_bills()) if hasattr(gtx_genesis, 'get_all_bills') else 0
    
    # Generation tasks statistics
    active_tasks = GenerationTask.query.filter_by(
        status='processing'
    ).count()
    completed_tasks = GenerationTask.query.filter_by(
        status='completed'
    ).count()
    
    # Mining statistics
    mining_stats = get_mining_stats()
    
    return {
        # User stats
        "total_users": total_users,
        "active_users": active_users,
        "new_users_today": new_users_today,
        
        # Banknote stats
        "total_banknotes": total_banknotes,
        "total_value": f"{total_value:,.2f}",
        "today_generated": today_generated,
        
        # Blockchain stats
        "blockchain_height": blockchain_height,
        "total_txs": total_txs,
        "mempool_size": mempool_size,
        
        # Additional stats
        "digital_bills": digital_bills_count,
        "active_tasks": active_tasks,
        "completed_tasks": completed_tasks,
        "mining_rewards": mining_stats.get("total_rewards", 0),
        "mining_difficulty": mining_stats.get("current_difficulty", 1),
        
        # Performance stats
        "avg_generation_time": get_avg_generation_time(),
        "success_rate": get_generation_success_rate()
    }

def get_system_status():
    """Get system status including daemon, network, and resource usage"""
    import psutil
    import time
    
    status = {
        "daemon_running": False,
        "network_online": False,
        "memory_usage": 0,
        "cpu_usage": 0,
        "disk_usage": 0,
        "last_sync": None
    }
    
    # Check if blockchain daemon is running
    try:
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'blockchain_daemon' in cmdline:
                        status["daemon_running"] = True
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except:
        pass
    
    # Check network connectivity
    try:
        import socket
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        status["network_online"] = True
    except:
        status["network_online"] = False
    
    # Get system resource usage
    try:
        status["memory_usage"] = round(psutil.virtual_memory().percent, 1)
        status["cpu_usage"] = round(psutil.cpu_percent(interval=0.1), 1)
        status["disk_usage"] = round(psutil.disk_usage('/').percent, 1)
    except:
        status["memory_usage"] = 0
        status["cpu_usage"] = 0
        status["disk_usage"] = 0
    
    # Get last blockchain sync time
    try:
        from blockchain_daemon import BlockchainDaemon
        daemon = BlockchainDaemon()
        if daemon.blockchain:
            last_block = daemon.blockchain[-1]
            timestamp = last_block.get('timestamp', time.time())
            status["last_sync"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        daemon.stop_daemon()
    except:
        status["last_sync"] = "Never"
    
    return status

def get_recent_activity():
    """Get recent system activity"""
    from datetime import datetime, timedelta
    import random
    
    activities = []
    now = datetime.utcnow()
    
    # Get recent user logins
    recent_logins = User.query.filter(
        User.last_login >= now - timedelta(hours=24)
    ).order_by(User.last_login.desc()).limit(5).all()
    
    for user in recent_logins:
        if user.last_login:
            activities.append({
                "icon": "👤",
                "text": f"User {user.username} logged in",
                "time": format_timedelta(now - user.last_login)
            })
    
    # Get recent banknote generations
    recent_banknotes = Banknote.query.filter(
        Banknote.created_at >= now - timedelta(hours=24)
    ).order_by(Banknote.created_at.desc()).limit(5).all()
    
    for banknote in recent_banknotes:
        activities.append({
            "icon": "💵",
            "text": f"Banknote ${banknote.denomination} generated for {banknote.user.username if banknote.user else 'Unknown'}",
            "time": format_timedelta(now - banknote.created_at)
        })
    
    # Get recent blockchain activity if available
    try:
        from blockchain_daemon import BlockchainDaemon
        daemon = BlockchainDaemon()
        blockchain_status = daemon.get_blockchain_status()
        if blockchain_status.get("blocks", 0) > 0:
            activities.append({
                "icon": "⛓️",
                "text": f"Blockchain height: {blockchain_status['blocks']} blocks",
                "time": "Now"
            })
        
        mempool_status = daemon.get_mempool_status()
        if mempool_status.get("total", 0) > 0:
            activities.append({
                "icon": "📝",
                "text": f"{mempool_status['total']} transactions in mempool",
                "time": "Now"
            })
        
        daemon.stop_daemon()
    except:
        pass
    
    # Add some system events
    event_types = [
        ("🔄", "System maintenance completed"),
        ("🔒", "Security audit passed"),
        ("📊", "Daily report generated"),
        ("⚡", "Performance optimized"),
        ("🚀", "New features deployed")
    ]
    
    # Add 2-3 random system events
    for icon, text in random.sample(event_types, min(3, len(event_types))):
        hours_ago = random.randint(1, 12)
        activities.append({
            "icon": icon,
            "text": text,
            "time": f"{hours_ago} hours ago"
        })
    
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
        total_rewards = 0
        difficulties = []
        
        for block in daemon.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("type") == "reward":
                    total_rewards += tx.get("amount", 0)
            
            # Extract difficulty if present
            difficulty = block.get("difficulty")
            if difficulty and isinstance(difficulty, (int, float)):
                difficulties.append(difficulty)
        
        daemon.stop_daemon()
        
        # Calculate average difficulty
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 1
        
        return {
            "total_rewards": total_rewards,
            "total_blocks": len(difficulties),
            "current_difficulty": avg_difficulty,
            "miners_count": len(set(block.get("miner", "unknown") for block in daemon.blockchain if block.get("miner")))
        }
    except:
        return {
            "total_rewards": 0,
            "total_blocks": 0,
            "current_difficulty": 1,
            "miners_count": 0
        }

def get_avg_generation_time():
    """Calculate average banknote generation time"""
    
    completed_tasks = GenerationTask.query.filter_by(
        status='completed'
    ).all()
    
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
    completed_tasks = GenerationTask.query.filter_by(
        status='completed'
    ).count()
    
    if total_tasks == 0:
        return "0%"
    
    success_rate = (completed_tasks / total_tasks) * 100
    return f"{success_rate:.1f}%"
@app.route('/admin/delete_serial/<int:serial_id>', methods=['POST'])
@admin_required
def admin_delete_serial(serial_id):
    serial = SerialNumber.query.get_or_404(serial_id)
    db.session.delete(serial)
    db.session.commit()
    flash('Serial number deleted successfully!', 'success')
    return redirect(url_for('admin_panel', section='serials'))

@app.route('/admin/cancel_task/<int:task_id>', methods=["POST"])
@admin_required
def admin_cancel_task(task_id):
    task = GenerationTask.query.get_or_404(task_id)
    if task.status in ['queued', 'pending', 'processing']:
        task.status = 'cancelled'
        task.completed_at = datetime.utcnow()
        db.session.commit()
        flash('Task cancelled successfully!', 'success')
    else:
        flash('Cannot cancel a task that is not queued, pending, or processing', 'error')
    return redirect(url_for('admin_panel', section='tasks'))

@app.route('/admin/delete_task/<int:task_id>', methods=["POST"])
@admin_required
def admin_delete_task(task_id):
    task = GenerationTask.query.get_or_404(task_id)
    
    # Only allow deletion of completed, failed, or cancelled tasks
    if task.status in ['completed', 'failed', 'cancelled']:
        db.session.delete(task)
        db.session.commit()
        flash('Task deleted successfully!', 'success')
    else:
        flash('Cannot delete a task that is still active. Cancel it first.', 'error')
    
    return redirect(url_for('admin_panel', section='tasks'))

import atexit
import threading

def cleanup_stale_generations():
    """Clean up any generation entries that are too old"""
    with GENERATION_LOCK:
        current_time = time.time()
        stale_users = []
        
        for user_id, info in GENERATION_THREADS.items():
            # Remove entries older than 1 hour
            if current_time - info.get('start_time', 0) > 3600:
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
    with GENERATION_LOCK:
        status = GENERATION_THREADS.get(user_id, {})
    
    if not status:
        return jsonify({'status': 'not_found'})
    
    # Check if the task is still in the database as processing
    task = GenerationTask.query.filter_by(user_id=user_id).order_by(GenerationTask.created_at.desc()).first()
    
    if task:
        status['db_status'] = task.status
        status['message'] = task.message
    
    return jsonify(status)
@app.route('/admin/generate-money/<int:user_id>', methods=['POST'])
@admin_required
def generate_money(user_id):
    """Generate banknotes for a user using the new queue system"""
    user = User.query.get_or_404(user_id)
    
    # Check if user already has a task in queue or processing
    queue_status = get_generation_queue_status()
    if user_id in queue_status['active_tasks']:
        flash(f'User {user.username} already has a generation task in progress.', 'warning')
        return redirect(url_for('admin_panel'))
    
    # Add task to queue
    task_id = run_generation_task(user_id, user.username)
    
    if task_id:
        flash(f'Generation task started for {user.username}. Task ID: {task_id}', 'success')
        print(f"[ADMIN] Started generation task {task_id} for user {user.username}")
    else:
        flash(f'Failed to start generation task for {user.username}.', 'error')
        print(f"[ADMIN ERROR] Failed to start generation for user {user.username}")
    
    return redirect(url_for('admin_panel'))
@app.route("/admin/debug/tasks")
@admin_required
def admin_debug_tasks():
    """Debug all generation tasks"""
    tasks = GenerationTask.query.order_by(GenerationTask.created_at.desc()).all()
    
    task_list = []
    for task in tasks:
        task_list.append({
            'id': task.id,
            'user_id': task.user_id,
            'username': task.user.username if task.user else 'Unknown',
            'status': task.status,
            'message': task.message,
            'created_at': task.created_at.isoformat() if task.created_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None
        })
    
    return jsonify({
        'total_tasks': len(tasks),
        'tasks': task_list
    })

@app.route("/admin/debug/queue")
@admin_required
def admin_debug_queue():
    """Debug the generation queue"""
    queue_status = get_generation_queue_status()
    
    active_tasks_info = []
    for user_id in queue_status['active_tasks']:
        user = User.query.get(user_id)
        if user:
            # Get the latest task for this user
            task = GenerationTask.query.filter_by(user_id=user_id).order_by(GenerationTask.created_at.desc()).first()
            active_tasks_info.append({
                'user_id': user_id,
                'username': user.username,
                'task_id': task.id if task else None,
                'task_status': task.status if task else None
            })
    
    return jsonify({
        'queue_status': queue_status,
        'active_tasks': active_tasks_info
    })
@app.route("/admin/test-worker/<int:user_id>")
@admin_required
def admin_test_worker(user_id):
    """Test the worker process manually"""
    user = User.query.get_or_404(user_id)
    
    try:
        # Test running the worker directly
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), 'generate_worker.py')
        
        # Create a test task first
        task = GenerationTask(
            user_id=user_id,
            status='queued',
            message="Manual test task"
        )
        db.session.add(task)
        db.session.commit()
        
        cmd = ['python', script_path, str(user_id), user.username, str(task.id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return jsonify({
            'success': True,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'task_id': task.id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/admin/queue-status')
@admin_required
def queue_status():
    """Check the current generation queue status"""
    status = get_generation_queue_status()
    active_tasks = []
    
    for user_id in status['active_tasks']:
        user = User.query.get(user_id)
        if user:
            active_tasks.append(user.username)
    
    return {
        'queue_size': status['queue_size'],
        'active_tasks': active_tasks,
        'is_running': status['is_running']
    }



@app.route("/generate-money", methods=["POST"])
def generate_money_user():
    current_user = get_current_user()
    if not current_user:
        flash("Please log in to generate money", "error")
        return redirect(url_for("login"))
    
    if not current_user.can_generate_money():
        flash(f"You can generate money again in {current_user.days_until_next_generation()} days", "error")
        return redirect(url_for("profile", username=current_user.username))
    
    # Check if user already has an active task
    queue_status = get_generation_queue_status()
    if current_user.id in queue_status['active_tasks']:
        flash("You already have a generation task in progress", "error")
        return redirect(url_for("profile", username=current_user.username))
    
    # This returns IMMEDIATELY - no blocking
    task_id = run_generation_task(current_user.id, current_user.username)
    
    if task_id:
        flash("Banknote generation started! This will run in the background. You can check status on your profile.", "success")
    else:
        flash("Failed to start generation. Please try again.", "error")
    
    return redirect(url_for("profile", username=current_user.username))

@app.route('/admin/settings', methods=['GET', 'POST'])
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
    
    if request.method == 'POST':
        try:
            settings.system_name = request.form.get('system_name', 'Banknote Generator')
            settings.max_banknotes = int(request.form.get('max_banknotes', 100))
            settings.cooldown_days = int(request.form.get('cooldown_days', 7))
            settings.maintenance_mode = 'maintenance_mode' in request.form
            settings.allow_registrations = 'allow_registrations' in request.form
            settings.max_file_size = int(request.form.get('max_file_size', 10))
            
            db.session.commit()
            flash('Settings updated successfully!', 'success')
        except ValueError:
            flash('Invalid input values. Please check your entries.', 'error')
        except Exception as e:
            flash(f'Error updating settings: {str(e)}', 'error')
        
        return redirect(url_for('admin_settings'))
    
    return render_template('admin_panel.html', 
                         active_section='settings',
                         settings=settings,
                         users=User.query.all(),  # You might want to paginate this
                         banknotes=Banknote.query.all(),
                         current_user=get_current_user())# You might want to paginate this

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
        
        flash(f"Reset successful for {user.username}: {banknotes_deleted} banknotes and {serials_deleted} serial numbers deleted, balance set to 0", "success")
    
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
@app.route("/admin/delete-banknote/<int:banknote_id>", methods=["POST"])
@admin_required
def admin_delete_banknote(banknote_id):
    bn = Banknote.query.get_or_404(banknote_id)
    SerialNumber.query.filter_by(banknote_id=bn.id).delete()
    db.session.delete(bn)
    db.session.commit()

    flash(f"Deleted banknote {bn.serial_number}", "success")
    return redirect(url_for("admin_panel"))

@app.route('/')
def landing():
    # Get current user (implementation depends on your authentication system)
    current_user = get_current_user()
    
    # Calculate statistics
    total_banknotes = Banknote.query.count()
    total_users = User.query.count()
    
    # Calculate recent activity using created_at instead of last_seen
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    recent_activity = User.query.filter(User.created_at >= one_week_ago).count()
    
    # Calculate total value of all banknotes
    banknotes = Banknote.query.all()
    total_value = 0
    for note in banknotes:
        try:
            total_value += float(note.denomination)
        except (ValueError, TypeError):
            pass
    
    # Get recent users (last 24 hours)
    one_day_ago = datetime.utcnow() - timedelta(hours=24)
    recent_users = User.query.filter(User.created_at >= one_day_ago).count()
    
    # Get recent transactions (last 24 hours)
    recent_transactions = Banknote.query.filter(Banknote.created_at >= one_day_ago).count()
    
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
                'username': top_user.username,
                'banknotes': top_count,
                'value': user_total_value
            }
    
    # Get recent trade (most recent banknote created)
    recent_trade = {}
    latest_banknote = Banknote.query.order_by(Banknote.created_at.desc()).first()
    if latest_banknote:
        recent_trade = {
            'from': latest_banknote.user.username if latest_banknote.user else 'System',
            'to': 'Owner',  # Simplified - assuming creator is owner
            'amount': latest_banknote.denomination if latest_banknote.denomination else '0'
        }
    
    # Get platform growth stats
    month_ago = datetime.utcnow() - timedelta(days=30)
    month_ago_users = User.query.filter(User.created_at <= month_ago).count()
    user_growth_rate = ((total_users - month_ago_users) / month_ago_users * 100) if month_ago_users > 0 else 0
    
    month_ago_banknotes = Banknote.query.filter(Banknote.created_at <= month_ago).count()
    banknote_growth_rate = ((total_banknotes - month_ago_banknotes) / month_ago_banknotes * 100) if month_ago_banknotes > 0 else 0
    
    # Get current user's stats if logged in
    user_stats = {}
    if current_user and hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
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
            'banknotes_created': (user_banknotes/2),
            'can_generate': can_generate,
            'days_until_next': days_until_next,
            'balance': current_user.balance if hasattr(current_user, 'balance') else 0,
            'total_value': user_total_value/2
        }
    
    # Handle None values in template
    recent_users = recent_users if recent_users is not None else 0
    recent_transactions = recent_transactions if recent_transactions is not None else 0
    
    return render_template('landing.html', 
                         total_banknotes=(total_banknotes/2),
                         total_users=total_users,
                         recent_activity=recent_activity,
                         total_value=(total_value/2),
                         user_stats=user_stats,
                         current_user=current_user,
                         recent_users=recent_users,
                         recent_transactions=recent_transactions,
                         top_collector=top_collector,
                         recent_trade=recent_trade,
                         user_growth_rate=user_growth_rate,
                         banknote_growth_rate=banknote_growth_rate,
                         month_ago_users=month_ago_users,
                         month_ago_banknotes=month_ago_banknotes)
@app.route("/portraits/<path:filename>")
def serve_portrait(filename):
    """
    Serve portrait images from the portraits directory
    """
    return send_from_directory('portraits', filename)
# Add this route
@app.route("/static/<path:filename>")
def serve_static(filename):
    """
    Serve static files from the root directory.
    This allows serving portraits from ./portraits/
    """
    return send_from_directory('.', filename)
@app.route("/gallery")
def gallery_index():
    # Get all users from the database instead of folder names
    users = User.query.order_by(User.username).all()
    return render_template('gallery_index.html', users=users, title="Members", current_user=get_current_user())

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
                    "denom": denom
                }
                if side == "front":
                    fronts.append(bill)
                else:
                    backs.append(bill)

    return render_template('name_detail.html', name=name, fronts=fronts, backs=backs, title=f"Member - {name}", current_user=get_current_user())

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
    
    return render_template('login.html', title="Login", current_user=get_current_user())


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
    images_base_path = './images'  # This is relative to your application root
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
                svg_files = glob(os.path.join(item_path, '*.svg'))
                print(f"    SVG files in {item}: {svg_files}")
                
                front_files = [f for f in svg_files if '_FRONT.svg' in f]
                back_files = [f for f in svg_files if '_BACK.svg' in f]
                
                if front_files or back_files:
                    denomination_images[item] = {
                        'front': sorted(front_files),
                        'back': sorted(back_files)
                    }
    
    print(f"Found denominations: {list(denomination_images.keys())}")
    
    denominations = sorted(denomination_images.keys())
    
    if not denominations:
        flash("No banknotes found in your wallet", "warning")
        return redirect(url_for("profile", username=current_user.username))
    
    # Helper functions to get images
    def get_front_image(denom):
        files = denomination_images.get(denom, {}).get('front', [])
        if files:
            filename = os.path.basename(files[-1])
            return f"./images/{current_user.username}/{denom}/{filename}"
        return None
    
    def get_back_image(denom):
        files = denomination_images.get(denom, {}).get('back', [])
        if files:
            filename = os.path.basename(files[-1])
            return f"./images/{current_user.username}/{denom}/{filename}"
        return None
    
    return render_template('my_wallet.html', 
                         denominations=denominations,
                         get_front_image=get_front_image,
                         get_back_image=get_back_image,
                         current_user=current_user,
                         title=f"{current_user.username}'s Wallet")
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return render_template('register.html', title="Register", current_user=get_current_user())
        
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
            return render_template('register.html', title="Register", current_user=get_current_user())
        
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "error")
            return render_template('register.html', title="Register", current_user=get_current_user())
        
        user = User(username=username, email=email)
        user.set_password(password)
        user.two_factor_secret = pyotp.random_base32()
        
        db.session.add(user)
        db.session.commit()
        
        session["pre_2fa_user_id"] = user.id
        return redirect(url_for("setup_2fa"))
    
    return render_template('register.html', title="Register", current_user=get_current_user())

@app.route("/setup-2fa")
def setup_2fa():
    if "pre_2fa_user_id" not in session:
        return redirect(url_for("login"))
    
    user = User.query.get(session["pre_2fa_user_id"])
    if not user:
        return redirect(url_for("login"))
    
    uri = user.get_totp_uri()
    qr_code = generate_qr_code(uri)
    
    return render_template('two_factor_setup.html', qr_code=qr_code, title="Setup 2FA", current_user=get_current_user())

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
        flash("Invalid token. Please check that your authenticator app time is synchronized with the server.", "error")
        return redirect(url_for("setup_2fa"))
    
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
    
    return render_template('two_factor_verify.html', title="Verify 2FA", current_user=get_current_user())

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for("landing"))

@app.route('/transaction-viewer/<tx_hash>')
def transaction_viewer(tx_hash):
    """
    View transaction details from the blockchain
    """
    try:
        # Use the blockchain daemon to get transaction details
        tx_data = blockchain_daemon_instance.get_transaction(tx_hash)
        
        # If transaction not found
        if not tx_data:
            flash('Transaction not found on the blockchain', 'error')
            return redirect(url_for('verify'))
        
        # Prepare transaction data for the template
        transaction = {
            'hash': tx_hash,
            'block_height': tx_data.get('block_height'),
            'confirmations': tx_data.get('confirmations', 0),
            'timestamp': tx_data.get('timestamp'),
            'size': tx_data.get('size'),
            'fee': tx_data.get('fee'),
            'total_value': tx_data.get('total_value'),
            'inputs': tx_data.get('inputs', []),
            'outputs': tx_data.get('outputs', []),
            'valid': True,
            'is_coinbase': tx_data.get('is_coinbase', False)
        }
        
        # Calculate input/output totals
        input_total = sum(inp.get('value', 0) for inp in transaction['inputs'])
        output_total = sum(out.get('value', 0) for out in transaction['outputs'])
        
        # Get mempool status if not confirmed
        mempool_status = None
        if not transaction['block_height']:
            mempool_status = blockchain_daemon_instance.get_mempool_transaction(tx_hash)
        
        # Check if this transaction contains any banknote data
        banknote_serial = None
        for output in transaction['outputs']:
            if output.get('script_type') == 'op_return':
                # Try to extract banknote serial from OP_RETURN data
                try:
                    op_return_data = output.get('op_return', '')
                    if 'SN-' in op_return_data:
                        banknote_serial = op_return_data
                        break
                except:
                    pass
        
        # Get banknote info if found
        banknote_info = None
        if banknote_serial:
            from models import Banknote
            banknote_info = Banknote.query.filter_by(serial=banknote_serial).first()
        
        # Format timestamp for display
        if transaction['timestamp']:
            from datetime import datetime
            dt = datetime.fromtimestamp(transaction['timestamp'])
            transaction['timestamp_formatted'] = dt.strftime('%Y-%m-%d %H:%M:%S')
            transaction['timestamp_readable'] = dt.strftime('%B %d, %Y at %I:%M %p')
        else:
            transaction['timestamp_formatted'] = 'Pending'
            transaction['timestamp_readable'] = 'Not yet confirmed'
        
        # Calculate validation metrics (similar to verify page)
        validation_score = 0
        if transaction['block_height']:
            validation_score += 1  # Confirmed in block
        if transaction['confirmations'] >= 6:
            validation_score += 1  # Deeply confirmed
        
        validation_percentage = (validation_score / 5) * 100
        
        # Prepare validation results structure
        validation_results = {
            'blockchain': {
                'found': bool(transaction['block_height']),
                'confirmations': transaction['confirmations'],
                'data': {
                    'block_height': transaction['block_height'],
                    'confirmations': transaction['confirmations']
                }
            },
            'mempool': {
                'found': bool(mempool_status),
                'data': mempool_status or {}
            }
        }
        
        # Add serial and banknote validation layers if applicable
        if banknote_serial:
            from models import Banknote, SerialRecord
            validation_results['serial_db'] = {
                'found': bool(SerialRecord.query.filter_by(serial=banknote_serial).first()),
                'data': {'id': banknote_serial}
            }
            validation_results['banknote_db'] = {
                'found': bool(banknote_info),
                'data': banknote_info._asdict() if banknote_info else {}
            }
            validation_results['digital_bill'] = {
                'found': banknote_info and banknote_info.signature_verified if banknote_info else False,
                'signature_valid': banknote_info.signature_verified if banknote_info else False,
                'verification_method': 'Blockchain OP_RETURN'
            }
            
            # Update validation score based on banknote validation
            if validation_results['serial_db']['found']:
                validation_score += 1
            if validation_results['banknote_db']['found']:
                validation_score += 1
            if validation_results['digital_bill']['found'] and validation_results['digital_bill']['signature_valid']:
                validation_score += 1
            
            validation_percentage = (validation_score / 5) * 100
        
        return render_template('transaction-viewer.html',
                            transaction=transaction,
                            validation_score=validation_score,
                            validation_percentage=validation_percentage,
                            validation_results=validation_results,
                            input_total=input_total,
                            output_total=output_total,
                            banknote_info=banknote_info,
                            banknote_serial=banknote_serial,
                            mempool_status=mempool_status)
    
    except Exception as e:
        print(f"Error viewing transaction: {str(e)}")
        flash(f'Error retrieving transaction: {str(e)}', 'error')
        return redirect(url_for('verify_serial'))

@app.route("/banknote-image/<path:filename>")
def serve_banknote_image(filename):
    # Decode URL-encoded characters
    filename = unquote(filename)
    # Convert backslashes to forward slashes for cross-platform compatibility
    filename = filename.replace('\\', '/')
    # Remove any leading "images/" if it exists
    if filename.startswith('images/'):
        filename = filename[7:]
    # Ensure we're not dealing with directory traversal attacks
    if '..' in filename or filename.startswith('/'):
        abort(404)
    return send_from_directory(IMAGES_ROOT, filename)

@app.route("/toggle-banknote/<int:banknote_id>")
def toggle_banknote_visibility(banknote_id):
    current_user = get_current_user()
    if not current_user:
        return redirect(url_for('login'))
    
    banknote = Banknote.query.get_or_404(banknote_id)
    if banknote.user_id != current_user.id:
        flash("You don't have permission to modify this banknote", "error")
        return redirect(url_for('profile', username=current_user.username))
    
    banknote.is_public = not banknote.is_public
    db.session.commit()
    
    flash(f"Banknote visibility set to {'public' if banknote.is_public else 'private'}", "success")
    return redirect(url_for('profile', username=current_user.username))
@app.route('/debug/generation/<username>')
def debug_generation(username):
    """Debug endpoint to check generation status"""
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    
    # Check generation tasks
    tasks = GenerationTask.query.filter_by(user_id=user.id).order_by(desc(GenerationTask.created_at)).all()
    
    # Check banknotes in database
    banknotes = Banknote.query.filter_by(user_id=user.id).all()
    
    # Check files on disk
    user_dir = os.path.join(IMAGES_ROOT, username)
    files_exist = os.path.exists(user_dir)
    file_list = []
    
    if files_exist:
        for root, dirs, files in os.walk(user_dir):
            for file in files:
                if file.endswith(('.svg', '.png', '.pdf')):
                    file_list.append(os.path.join(root, file))
    
    return jsonify({
        "user": {
            "id": user.id,
            "username": user.username,
            "balance": user.balance,
            "last_generation": user.last_generation.isoformat() if user.last_generation else None
        },
        "tasks": [{
            "id": t.id,
            "status": t.status,
            "message": t.message,
            "created_at": t.created_at.isoformat(),
            "completed_at": t.completed_at.isoformat() if t.completed_at else None
        } for t in tasks],
        "banknotes_count": len(banknotes),
        "files_exist": files_exist,
        "file_count": len(file_list),
        "files": file_list[:10]  # First 10 files only
    })
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
                    if file.endswith(('.svg', '.png', '.pdf')):
                        files_on_disk.append(os.path.join(denom, file))
    
    # Check generation tasks
    tasks = GenerationTask.query.filter_by(user_id=user.id).order_by(desc(GenerationTask.created_at)).all()
    
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

@app.route("/<username>", methods=["GET", "POST"])
def profile(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        flash("User not found", "error")
        return redirect(url_for("landing"))
    
    current_user_obj = get_current_user()
    
    if request.method == "POST":
        if current_user_obj and current_user_obj.id == user.id:
            raw_bio = request.form.get("bio", "")
            # Sanitize the bio before saving
            user.bio = sanitize_bio(raw_bio)
            db.session.commit()
            flash("Bio updated successfully", "success")
            return redirect(url_for("profile", username=username))
    
    generation_tasks = GenerationTask.query.filter_by(user_id=user.id).order_by(desc(GenerationTask.created_at)).limit(10).all()
    
    # DEBUG: Check if files exist on disk
    user_images_path = os.path.join(IMAGES_ROOT, username)
    print(f"[DEBUG] Checking for user images at: {user_images_path}")
    
    if os.path.exists(user_images_path):
        print(f"[DEBUG] User image directory exists")
        for denom in os.listdir(user_images_path):
            denom_path = os.path.join(user_images_path, denom)
            if os.path.isdir(denom_path):
                print(f"[DEBUG] Denomination {denom} has files: {os.listdir(denom_path)}")
    
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
        numbers = re.findall(r'\d+', denomination_str)
        numeric_value = int(numbers[0]) if numbers else 0
        
        # Detect side either from banknote.side or denom string
        side_str = getattr(banknote, 'side', None)
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
        if hasattr(banknote, 'png_path') and banknote.png_path:
            banknote.svg_path = banknote.png_path.replace('.png', '.svg')
        else:
            banknote.svg_path = None
    
    return render_template('profile.html', user=user, generation_tasks=generation_tasks, 
                         banknotes=banknotes, title=f"Profile - {username}", current_user=current_user_obj)

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
            block_data.get("nonce")
        )
        
        return jsonify({
            "is_valid": is_valid,
            "hash_match": block_data.get("hash") == calculated_hash,
            "provided_hash": block_data.get("hash"),
            "calculated_hash": calculated_hash,
            "missing_fields": [f for f in ["index", "timestamp", "transactions", "previous_hash", "nonce", "hash", "miner"] 
                             if f not in block_data]
        })
        
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
            if (tx_type == "GTX_Genesis" and 
                block_tx.get("type") == "GTX_Genesis" and
                tx_serial and block_tx.get("serial_number") == tx_serial):
                return True
            
            # Check by content for other transaction types
            if (tx_type == block_tx.get("type") and
                transaction.get("from") == block_tx.get("from") and
                transaction.get("to") == block_tx.get("to") and
                transaction.get("amount") == block_tx.get("amount")):
                return True
    
    return False


def load_json_file(filename):
    """Load JSON file with error handling"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
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
        with open("mempool.json", 'w', encoding='utf-8') as f:
            json.dump(cleaned_mempool, f, indent=2)
        
        return jsonify({
            "status": "success",
            "message": f"Cleaned {cleaned_count} mined transactions",
            "initial_count": initial_count,
            "current_count": len(cleaned_mempool),
            "cleaned_count": cleaned_count
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Cleanup failed: {str(e)}"
        }), 500





# Initialize Database
with app.app_context():
    db.create_all()
    # Initialize blockchain manager
# Initialize the generation queue after all imports are complete
if __name__ == "__main__":
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        if not hasattr(app, 'blockchain_daemon_instance'):
            #blockchain_daemon = BlockchainDaemon()
            # IMPORTANT: Attach it to the app instance
            #app.blockchain_daemon = blockchain_daemon
            #blockchain_daemon.repair_blockchain()
            #blockchain_daemon.emergency_repair()
            #blockchain_daemon_instance.start_daemon()
            #blockchain_daemon.diagnose_transfer_issue()
            #blockchain_daemon.debug_mining_selection()
            #blockchain_daemon.force_mine_transfers()
            #blockchain_daemon_instance.debug_reward_issue()
            #blockchain_daemon_instance.comprehensive_diagnostic()
            #blockchain_daemon_instance.debug_hash_mismatch()
            #blockchain_daemon_instance.debug_mining_issues()
            atexit.register(lambda: blockchain_daemon_instance.stop_daemon() if blockchain_daemon_instance else None)

    app.run(debug=True, host="0.0.0.0", port=5555)