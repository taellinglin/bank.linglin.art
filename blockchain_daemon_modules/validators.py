# blockchain_daemon_modules/validators.py
"""
Validation functions for blocks, transactions, and mining proofs
Refactored to use lunalib 1.6.6 validation methods
"""

import json
import hashlib
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Dict, List, Optional

from blockchain_daemon_modules.mining import calculate_expected_reward

try:
    from lunalib.gtx.digital_bill import DigitalBill
    from lunalib.gtx.genesis import GTXGenesis
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("lunalib not available, using fallback validation")

logger = logging.getLogger(__name__)

_SM3_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lunalib-sm3-")


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


def validate_transaction_structure(transaction: Dict) -> bool:
    """Validate transaction structure using lunalib 1.6.6 validation"""
    tx_type = transaction.get("type")
    
    logger.debug(f"Validating {tx_type} transaction structure")
    logger.debug(f"Transaction keys: {list(transaction.keys())}")
    
    if tx_type == "transfer":
        required_fields = ["from", "to", "amount", "timestamp", "signature"]
    
    elif tx_type == "GTX_Genesis":
        # Genesis banknote validation - use lunalib's validation
        required_fields = ["serial_number", "denomination", "issued_to", "timestamp"]
        
        # Check required fields exist
        missing_fields = [field for field in required_fields if field not in transaction]
        if missing_fields:
            logger.error(f"GTX_Genesis missing fields: {missing_fields}")
            logger.error(f"Available fields: {list(transaction.keys())}")
            return False
        
        # Validate serial number
        serial_number = transaction.get("serial_number")
        if not serial_number or not isinstance(serial_number, str):
            logger.error(f"Invalid serial_number: {serial_number}")
            return False
            
        # Validate denomination
        try:
            denomination = float(transaction.get("denomination", 0))
            if denomination <= 0:
                logger.error(f"Invalid denomination: {denomination}")
                return False
        except (ValueError, TypeError) as e:
            logger.error(f"Denomination conversion error: {e}")
            return False
        
        # Validate issued_to
        issued_to = transaction.get("issued_to")
        if not issued_to or not isinstance(issued_to, str):
            logger.error(f"Invalid issued_to: {issued_to}")
            return False
        
        # Signature and public_key are optional during creation
        # They will be added by the GTXGenesis system
        
        logger.info(f"GTX_Genesis transaction validated: {serial_number}")
        return True
    
    elif tx_type == "genesis":
        required_fields = ["message", "timestamp", "hash"]
    
    elif tx_type == "reward":
        required_fields = ["to", "amount", "timestamp", "block_height", "hash"]
    
    else:
        logger.warning(f"Unknown transaction type: {tx_type}")
        return False
    
    # Check if all required fields are present
    missing_fields = [field for field in required_fields if field not in transaction]
    if missing_fields:
        logger.warning(f"Missing fields for {tx_type}: {missing_fields}")
        return False
        
    return True


def validate_regular_transactions(transactions: List[Dict]) -> Dict:
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
            
            # Check for self-transfer (allow but flag if needed)
            if from_addr == to_addr:
                # Allow self-transfer; mining wallets may self-send
                pass
                
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


def validate_reward_transactions(reward_transactions: List[Dict], block_index: int, block_data: Dict, 
                                 previous_block_hash: str, is_transaction_mined_func, 
                                 mempool_mgr) -> Dict:
    """Validate reward transactions with mining proof validation"""
    print("=" * 100)
    print("🔥 DEBUG: validate_reward_transactions ENTERED")
    print("=" * 100)
    
    # Log everything
    print(f"Block #{block_index} received for validation")
    print(f"Block hash: {block_data.get('hash', '')[:16]}...")
    print(f"Previous hash: {previous_block_hash[:16]}...")
    print(f"Reward transactions: {len(reward_transactions)}")
    
    if not reward_transactions:
        print("✅ No reward transactions to validate")
        return {'valid': True, 'error': None}

    if len(reward_transactions) != 1:
        return {
            'valid': False,
            'error': f'Expected exactly 1 reward transaction, got {len(reward_transactions)}'
        }
    
    # Quick logging of what we're validating
    print(f"📊 Validating {len(reward_transactions)} reward transaction(s)")
    print(f"📦 Block #{block_index}, Hash: {block_data.get('hash', '')[:16]}...")
    
    # Extract basic block data
    block_hash = block_data.get('hash', '')
    nonce = block_data.get('nonce')
    timestamp = block_data.get('timestamp')
    miner_address = block_data.get('miner', '')
    difficulty = block_data.get('difficulty', 0)
    
    # Basic checks
    if not block_hash:
        return {'valid': False, 'error': 'No block hash provided'}
    if not nonce or not timestamp:
        return {'valid': False, 'error': 'Block missing mining data (nonce or timestamp)'}
    
    # Get all transactions
    all_transactions = block_data.get('transactions', [])
    non_reward_txs = [tx for tx in all_transactions if tx.get('type') != 'reward']

    reward_aliases = {"reward", "coinbase", "mining_reward", "mining", "block_reward"}
    for tx in non_reward_txs:
        tx_type = str(tx.get('type') or '').lower()
        if tx_type in reward_aliases:
            return {
                'valid': False,
                'error': f'Found reward-like transaction type in non-reward list: {tx_type}'
            }
    
    print(f"📈 Transaction breakdown:")
    print(f"  Total: {len(all_transactions)}")
    print(f"  Non-reward: {len(non_reward_txs)}")
    print(f"  Reward: {len(reward_transactions)}")
    
    # ====== STEP 1: VALIDATE MINING PROOF ======
    print("\n🔬 STEP 1: Validating mining proof...")
    
    # First, check difficulty requirement
    if not block_hash.startswith('0' * difficulty):
        return {'valid': False, 'error': f'Hash {block_hash[:16]}... doesn\'t start with {difficulty} zeros'}
    
    print(f"✅ Difficulty check passed: {difficulty} zeros")
    
    # Try to validate the mining proof
    mining_proof_result = validate_mining_proof_internal(
        block_hash, difficulty, block_data, previous_block_hash, non_reward_txs
    )

    if not mining_proof_result['valid']:
        return {'valid': False, 'error': f'Invalid mining proof: {mining_proof_result["error"]}'}
    
    print(f"✅ Mining proof validated via: {mining_proof_result.get('method', 'unknown')}")
    
    # ====== STEP 2: VALIDATE REWARD TRANSACTION ======
    print("\n💰 STEP 2: Validating reward transaction...")
    
    # Get the reward transaction (should be only one)
    reward_tx = reward_transactions[0]
    
    # Basic reward transaction validation
    required_fields = ['to', 'from', 'amount', 'block_height', 'hash', 'timestamp']
    missing_fields = [field for field in required_fields if field not in reward_tx]
    if missing_fields:
        return {'valid': False, 'error': f'Reward transaction missing fields: {missing_fields}'}
    
    # Validate recipient matches miner
    if reward_tx.get('to') != miner_address:
        return {'valid': False, 'error': f'Reward recipient {reward_tx.get("to")} != miner {miner_address}'}

    # Validate sender is network/mining reward
    from_field = reward_tx.get('from')
    valid_from_values = {'ling country', 'network', 'mining_reward', 'block_reward', 'coinbase'}
    if str(from_field).strip().lower() not in valid_from_values:
        return {
            'valid': False,
            'error': f'Invalid reward sender: {from_field}. Must be one of {sorted(valid_from_values)}'
        }

    # Validate recipient/miner address format (reject placeholders)
    if not _is_valid_luna_address(miner_address):
        return {
            'valid': False,
            'error': f'Invalid miner address format: {miner_address}'
        }
    if not _is_valid_luna_address(reward_tx.get('to')):
        return {
            'valid': False,
            'error': f'Invalid reward recipient address format: {reward_tx.get("to")}'
        }
    
    # Validate block height
    if reward_tx.get('block_height') != block_index:
        return {'valid': False, 'error': f'Reward block_height {reward_tx.get("block_height")} != block index {block_index}'}

    # Validate reward transaction hash format (64 hex)
    tx_hash = str(reward_tx.get('hash', '')).lower()
    if len(tx_hash) != 64 or any(ch not in "0123456789abcdef" for ch in tx_hash):
        return {'valid': False, 'error': 'Invalid reward transaction hash format'}

    # Validate reward amount is positive numeric
    amount = reward_tx.get('amount', 0)
    try:
        amount = float(amount)
        if amount <= 0:
            return {'valid': False, 'error': f'Invalid reward amount: {amount}'}
    except (ValueError, TypeError):
        return {'valid': False, 'error': f'Invalid reward amount format: {amount}'}

    # Validate reward timestamp close to block timestamp
    try:
        reward_ts = float(reward_tx.get('timestamp'))
        block_ts = float(block_data.get('timestamp'))
        if abs(block_ts - reward_ts) > 600:
            return {
                'valid': False,
                'error': f'Reward timestamp too far from block timestamp (diff={abs(block_ts - reward_ts)}s)'
            }
    except (TypeError, ValueError):
        return {'valid': False, 'error': 'Invalid reward or block timestamp'}
    
    # ====== STEP 3: CALCULATE & VALIDATE REWARD AMOUNT ======
    print("\n📊 STEP 3: Calculating reward amount...")
    
    # Determine expected reward using lunalib-compatible calculation
    
    print(f"\n🔍 REWARD TRANSACTION DEBUG:")
    print(f"   Full reward TX: {json.dumps(reward_tx, indent=2)}")
    print(f"   Reward TX keys: {list(reward_tx.keys())}")
    print(f"   Difficulty from block: {difficulty}")
    print(f"   Difficulty in reward TX: {reward_tx.get('difficulty', 'NOT SET')}")
    
    total_fees = 0
    for tx in non_reward_txs:
        fee = tx.get('fee', 0)
        try:
            fee_val = float(fee)
        except (ValueError, TypeError):
            return {'valid': False, 'error': f'Invalid transaction fee: {fee}'}
        if fee_val < 0:
            return {'valid': False, 'error': f'Negative transaction fee: {fee_val}'}
        total_fees += fee_val
    empty_block = len(non_reward_txs) == 0
    base_reward_for_calc = None
    try:
        from models import Settings
        settings = Settings.query.first()
        if settings and settings.mining_reward is not None:
            base_reward_for_calc = float(settings.mining_reward)
    except Exception:
        base_reward_for_calc = None

    expected_reward = None
    try:
        from lunalib.mining.difficulty import DifficultySystem

        difficulty_system = DifficultySystem()
        expected_reward = float(
            difficulty_system.calculate_block_reward(
                difficulty,
                block_height=block_index,
                tx_count=len(non_reward_txs),
                fees_total=float(total_fees),
                **({"base_reward": float(base_reward_for_calc)} if base_reward_for_calc is not None else {}),
            )
        )
    except Exception:
        expected_reward = None

    if expected_reward is None:
        expected_reward = calculate_expected_reward(
            difficulty,
            fees=total_fees,
            empty_block=empty_block,
            block_height=block_index,
            tx_count=len(non_reward_txs),
            base_reward=base_reward_for_calc,
        )
    if empty_block:
        print(f"🌑 Empty block: BASE_REWARD * {difficulty} = {expected_reward}")
    else:
        print(
            f"📦 Regular block: (BASE_REWARD * {difficulty}) + {total_fees} fees = {expected_reward}"
        )
    
    print(f"\n💰 REWARD COMPARISON:")
    print(f"   Expected: {expected_reward} LKC")
    print(f"   Provided: {amount} LKC")
    print(f"   Difference: {abs(amount - expected_reward)} LKC")
    print(f"   Match: {abs(amount - expected_reward) <= 0.000001}")
    
    # Allow small floating point differences
    if abs(amount - expected_reward) > 0.000001:
        error_details = {
            'reward_transaction': reward_tx,
            'expected_reward': expected_reward,
            'provided_reward': amount,
            'difficulty': difficulty,
            'calculation': (
                f'BASE_REWARD * {difficulty} = {expected_reward}'
                if empty_block
                else f'BASE_REWARD * {difficulty} + fees = {expected_reward}'
            ),
            'block_hash': block_data.get('hash', '')[:16] + '...',
            'block_timestamp': block_data.get('timestamp'),
            'reward_timestamp': reward_tx.get('timestamp'),
            'timestamp_diff': abs(block_data.get('timestamp', 0) - reward_tx.get('timestamp', 0))
        }
        print(f"\n❌ REWARD VALIDATION FAILED:")
        print(f"   Error details: {json.dumps(error_details, indent=2)}")
        
        return {
            'valid': False,
            'error': (
                f'Reward amount {amount} != expected {expected_reward} (BASE_REWARD * {difficulty})'
                if empty_block
                else f'Reward amount {amount} != expected {expected_reward} (BASE_REWARD * {difficulty} + fees)'
            ),
            'debug': error_details
        }
    
    print(f"✅ Reward amount validated")
    
    # ====== STEP 4: FINAL CHECKS ======
    print("\n✅ STEP 4: Final validation...")
    
    # Check for duplicates
    if is_reward_transaction_duplicate(reward_tx, is_transaction_mined_func):
        return {'valid': False, 'error': f'Duplicate reward transaction: {reward_tx.get("hash")[:16]}...'}
    
    print("=" * 80)
    print("🎉 ALL VALIDATIONS PASSED!")
    print("=" * 80)
    
    return {
        'valid': True, 
        'error': None, 
        'difficulty': difficulty,
        'empty_block': len(non_reward_txs) == 0
    }


def validate_mining_proof_internal(block_hash: str, difficulty: int, block_data: Dict,
                                   previous_block_hash: str, non_reward_txs: List[Dict]) -> Dict:
    """Internal mining proof validation using lunalib only."""
    print("=" * 80)
    print("🔍 DEBUG: validate_mining_proof_internal CALLED - LUNALIB ONLY")
    print("=" * 80)

    print(f"🔍 Validating hash: {block_hash[:16]}...")
    print(f"   Difficulty: {difficulty}")
    print(f"   Non-reward txs: {len(non_reward_txs)}")
    print(f"   Block index: {block_data.get('index')}")

    index = block_data.get('index')
    timestamp = block_data.get('timestamp')
    nonce = block_data.get('nonce')

    def _resolve_lunalib_callable(module_names: List[str], method_names: List[str]):
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

    def _run_threaded(func, *args):
        future = _SM3_EXECUTOR.submit(func, *args)
        return future.result(timeout=2.5)

    def _normalize_hash(value):
        if value is None:
            return None
        if hasattr(value, "hexdigest"):
            return value.hexdigest()
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, str):
            return value.strip()
        return str(value)

    # --- Lunalib hash calculation (optional, threaded) ---
    hash_calc = _resolve_lunalib_callable(
        [
            "lunalib.blockchain",
            "lunalib.core.blockchain",
            "lunalib.mining",
            "lunalib.hashing",
            "lunalib.utils",
        ],
        [
            "calculate_block_hash",
            "calculate_hash",
            "hash_block",
            "compute_block_hash",
        ],
    )

    sm3_modules = [
        "lunalib.crypto.sm3",
        "lunalib.hashing.sm3",
        "lunalib.core.sm3",
        "lunalib.sm3",
        "lunalib.crypto",
        "lunalib.hashing",
        "lunalib.utils",
    ]
    sm3_hash = _resolve_lunalib_callable(
        sm3_modules,
        [
            "sm3_hash",
            "hash_sm3",
            "sm3",
            "hash",
            "sm3_hash_hex",
            "sm3_hexdigest",
        ],
    )

    sm3_class = None
    for module_name in sm3_modules:
        try:
            module = __import__(module_name, fromlist=["*"])
        except Exception:
            continue
        sm3_class = getattr(module, "SM3Hash", None)
        if sm3_class:
            break

    expected_hash = None
    if hash_calc is not None:
        for args in (
            (block_data,),
            (index, previous_block_hash, timestamp, block_data.get('transactions', []), nonce),
            (index, previous_block_hash, timestamp, non_reward_txs, nonce),
        ):
            try:
                expected_hash = _normalize_hash(_run_threaded(hash_calc, *args))
                if expected_hash:
                    break
            except (FutureTimeout, Exception):
                continue

    def _try_sm3_payload(payload: str) -> Optional[str]:
        if sm3_hash is not None:
            try:
                return _normalize_hash(_run_threaded(sm3_hash, payload))
            except Exception:
                try:
                    return _normalize_hash(_run_threaded(sm3_hash, payload.encode("utf-8")))
                except Exception:
                    return None
        if sm3_class is not None:
            try:
                hasher = sm3_class()
                if hasattr(hasher, "update"):
                    hasher.update(payload.encode("utf-8"))
                if hasattr(hasher, "hexdigest"):
                    return _normalize_hash(hasher.hexdigest())
                if hasattr(hasher, "digest"):
                    return _normalize_hash(hasher.digest())
            except Exception:
                return None
        return None

    if expected_hash is None:
        payloads = []
        payloads.append(
            json.dumps(
                {
                    'index': index,
                    'previous_hash': previous_block_hash,
                    'timestamp': timestamp,
                    'transactions': block_data.get('transactions', []),
                    'nonce': nonce,
                },
                sort_keys=True,
                separators=(',', ':'),
            )
        )
        payloads.append(
            json.dumps(
                {
                    'index': index,
                    'previous_hash': previous_block_hash,
                    'timestamp': timestamp,
                    'transactions': non_reward_txs,
                    'nonce': nonce,
                },
                sort_keys=True,
                separators=(',', ':'),
            )
        )
        transactions_hash = block_data.get("transactions_hash")
        miner = block_data.get("miner")
        if transactions_hash and miner and timestamp is not None and nonce is not None:
            payloads.append(f"{previous_block_hash}{timestamp}{transactions_hash}{miner}{nonce}")

        for payload in payloads:
            expected_hash = _try_sm3_payload(payload)
            if expected_hash:
                break

    if expected_hash:
        if expected_hash != block_hash:
            return {
                'valid': False,
                'error': f'Hash mismatch (lunalib): expected {expected_hash[:16]}..., got {block_hash[:16]}...'
            }

    # --- Lunalib mining proof validation ---
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
        if expected_hash and expected_hash == block_hash:
            if block_hash.startswith('0' * difficulty):
                return {
                    'valid': True,
                    'method': 'lunalib_hash_difficulty'
                }
            return {
                'valid': False,
                'error': f'Hash does not meet difficulty {difficulty}'
            }
        return {
            'valid': False,
            'error': 'Lunalib mining proof validator unavailable'
        }

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
        return {
            'valid': False,
            'error': f'Lunalib mining proof validation error: {e}'
        }

    if isinstance(result, dict):
        if "valid" in result:
            return {
                'valid': bool(result.get('valid')),
                'method': result.get('method') or result.get('validation_method') or 'lunalib'
            }
        if "success" in result:
            return {
                'valid': bool(result.get('success')),
                'method': result.get('method') or result.get('validation_method') or 'lunalib'
            }

    if isinstance(result, bool):
        return {
            'valid': result,
            'method': 'lunalib'
        }

    return {
        'valid': False,
        'error': 'Unexpected response from lunalib mining proof validator'
    }


def is_reward_transaction_duplicate(reward_tx: Dict, is_transaction_mined_func) -> bool:
    """Check if reward transaction already exists in blockchain"""
    try:
        # Use the provided function to check if already mined
        return is_transaction_mined_func(reward_tx)
    except Exception as e:
        logger.error(f"Error checking reward transaction duplicate: {e}")
        return False


def validate_block_for_submission(block: Dict, get_network_height_func, 
                                  get_last_network_hash_func, calculate_hash_func) -> Dict:
    """Comprehensive validation for block submission"""
    validation_result = {
        "valid": False,
        "errors": [],
        "warnings": []
    }
    
    try:
        # 1. Validate block index
        block_index = block.get('index')
        network_height = get_network_height_func()
        
        if block_index != network_height + 1:
            validation_result["errors"].append(
                f"Block index mismatch: expected {network_height + 1}, got {block_index}"
            )
        
        # 2. Validate previous hash
        expected_previous_hash = get_last_network_hash_func()
        block_previous_hash = block.get('previous_hash')
        
        if block_previous_hash != expected_previous_hash:
            validation_result["errors"].append(
                f"Previous hash mismatch: expected {expected_previous_hash[:16]}..., got {block_previous_hash[:16]}..."
            )
        
        # 3. Validate timestamp
        block_timestamp = block.get('timestamp')
        current_time = time.time()
        
        # Timestamp cannot be in the future, nor too old (e.g., over 2 hours ago)
        if block_timestamp > current_time + 300:  # 5 minute tolerance
            validation_result["errors"].append(f"Block timestamp is in the future: {block_timestamp}")
        
        if current_time - block_timestamp > 7200:  # 2 hours
            validation_result["warnings"].append(f"Block timestamp is very old: {block_timestamp}")
        
        # 4. Validate difficulty
        block_difficulty = block.get('difficulty', 1)
        if not (1 <= block_difficulty <= 9):
            validation_result["errors"].append(f"Invalid difficulty: {block_difficulty}")
        
        # 5. Validate block hash
        calculated_hash = calculate_hash_func(
            block.get('index'),
            block.get('previous_hash'),
            block.get('timestamp'),
            block.get('transactions', []),
            block.get('nonce')
        )
        
        if calculated_hash != block.get('hash'):
            validation_result["errors"].append(f"Block hash mismatch. Calculated: {calculated_hash[:16]}..., Provided: {block.get('hash', '')[:16]}...")
        
        # 6. Validate transactions
        transactions = block.get('transactions', [])
        if not transactions:
            validation_result["warnings"].append("Block contains no transactions")
        
        # Check for reward transactions
        reward_txs = [tx for tx in transactions if tx.get('type') == 'reward']
        if not reward_txs:
            validation_result["errors"].append("Block must contain at least one reward transaction")
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
    except Exception as e:
        validation_result["errors"].append(f"Validation error: {e}")
    
    return validation_result


def validate_block(block: Dict, blockchain: List[Dict], validate_block_structure_func, 
                  validate_transaction_for_block_func) -> bool:
    """Validate a mined block from external miner"""
    try:
        # Use blockchain manager's validation method
        validation_result = validate_block_structure_func(block)
        
        if not validation_result['valid']:
            logger.error(f"Block validation failed: {validation_result['issues']}")
            return False
        
        # Additional custom validation
        current_height = len(blockchain)
        
        # Check block index
        if block["index"] != current_height:
            logger.error(f"Block index mismatch: expected {current_height}, got {block['index']}")
            return False
        
        # Check previous hash
        if current_height > 0:
            previous_block = blockchain[-1]
            if block["previous_hash"] != previous_block["hash"]:
                logger.error("Previous hash mismatch")
                return False
        else:
            # Genesis block
            if block["previous_hash"] != "0" * 64 and block["previous_hash"] != "0":
                logger.error("Genesis block invalid previous hash")
                return False
        
        # Validate all transactions
        valid_transactions = 0
        for tx in block.get("transactions", []):
            if validate_transaction_for_block_func(tx, block["index"]):
                valid_transactions += 1
        
        if valid_transactions == 0:
            logger.error("No valid transactions in block")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Block validation error: {e}")
        return False


def calculate_block_hash(index: int, previous_hash: str, timestamp: float, 
                        transactions: List[Dict], nonce: int) -> str:
    """Calculate SHA-256 hash of a block"""
    try:
        index = int(index)
        nonce = int(nonce)
        
        if isinstance(timestamp, float):
            timestamp = timestamp
        else:
            timestamp = float(timestamp)
        
        block_data = {
            'index': index,
            'previous_hash': previous_hash,
            'timestamp': timestamp,
            'transactions': transactions,
            'nonce': nonce
        }
        
        block_string = json.dumps(block_data, sort_keys=True, separators=(',', ':'))
        calculated_hash = hashlib.sha256(block_string.encode()).hexdigest()
        
        return calculated_hash
        
    except Exception as e:
        logger.error(f"Hash calculation error: {e}")
        return "0" * 64


def validate_transaction_for_block(transaction: Dict, block_index: int, mempool: List[Dict], 
                                   mined_serials: set, gtx_genesis, 
                                   is_transaction_mined_func) -> bool:
    """Validate transaction for inclusion in block"""
    try:
        tx_type = transaction.get("type")
        
        # Reward transactions
        if tx_type == "reward":
            required = ["to", "amount", "timestamp", "block_height", "hash"]
            if not all(field in transaction for field in required):
                return False
            
            tx_hash = transaction.get("hash")
            if tx_hash and is_transaction_mined_func(transaction):
                return False
            
            tx_block_height = transaction.get("block_height")
            if tx_block_height not in [block_index, block_index - 1, block_index + 1]:
                return False
            
            reward_amount = transaction.get("amount", 0)
            if reward_amount <= 0 or reward_amount > 1000:
                return False
                
            return True
        
        # Genesis transaction
        elif tx_type == "genesis":
            if block_index != 0:
                return False
            required = ["message", "timestamp", "hash"]
            if not all(field in transaction for field in required):
                return False
            # Check no other genesis exists (would need blockchain reference)
            return True
        
        # GTX_Genesis transactions
        elif tx_type == "GTX_Genesis":
            serial = transaction.get("serial_number")
            if not serial:
                return False
            
            # Check required fields
            required = ["serial_number", "denomination", "issued_to", "timestamp", "signature", "public_key"]
            if not all(field in transaction for field in required):
                return False
            
            # Validate data types
            try:
                denomination = float(transaction.get("denomination", 0))
                if denomination <= 0:
                    return False
            except (ValueError, TypeError):
                return False
            
            # Check signature format
            signature = transaction.get("signature", "")
            if not signature or len(signature) < 10:
                return False
            
            # Check public key format
            public_key = transaction.get("public_key", "")
            if not public_key or len(public_key) < 10:
                return False
            
            # Check for duplicates in mempool
            for existing_tx in mempool:
                if (existing_tx.get("type") == "GTX_Genesis" and 
                    existing_tx.get("serial_number") == serial and
                    existing_tx.get("hash") != transaction.get("hash")):
                    return False
            
            # Check if already mined (double spend protection)
            if serial and serial in mined_serials:
                return False
            
            return True
        
        # Transfer transactions
        elif tx_type == "transfer":
            required = ["from", "to", "amount", "timestamp", "signature"]
            if not all(field in transaction for field in required):
                return False
            
            tx_hash = transaction.get("hash")
            if not tx_hash:
                return False
            
            # Check if in mempool
            in_mempool = any(tx.get("hash") == tx_hash for tx in mempool)
            if mempool and not in_mempool:
                return False
            
            # Check if already mined
            if is_transaction_mined_func(transaction):
                return False
            
            return True
        
        # Unknown transaction type
        else:
            return False
            
    except Exception as e:
        logger.error(f"Transaction validation error: {e}")
        return False
