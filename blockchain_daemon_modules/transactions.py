# blockchain_daemon_modules/transactions.py
"""
Transaction handling functions
"""

import json
import time
import hashlib
import secrets
import logging
import traceback
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_transaction_hash(transaction_data: Dict) -> str:
    """Generate a unique hash for any transaction"""
    hash_data = {
        'type': transaction_data.get('type'),
        'to': transaction_data.get('to'),
        'amount': transaction_data.get('amount'),
        'timestamp': transaction_data.get('timestamp', None),
        'block_height': transaction_data.get('block_height'),
        'miner': transaction_data.get('miner'),
        'nonce': secrets.randbelow(1000000)
    }
    
    hash_string = json.dumps(hash_data, sort_keys=True)
    return hashlib.sha256(hash_string.encode()).hexdigest()


# NOTE: create_reward_transaction has been moved to mining.py with exponential reward calculation
# The new function is: mining.create_reward_transaction(miner_address, block_height, difficulty, transaction_count)
# Old function removed to prevent using incorrect (non-exponential) reward amounts


def add_genesis_transaction(serial_number: str, denomination: float, issued_to: str,
                           gtx_genesis, mempool: List[Dict], add_transaction_func) -> bool:
    """
    Add a genesis transaction for a banknote using GTXGenesis system
    """
    try:
        from lunalib.gtx.digital_bill import DigitalBill
        
        logger.info(f"Creating genesis transaction for serial: {serial_number}")
        
        # Use GTXGenesis to create the bill
        denomination_int = int(denomination)
        digital_bill = gtx_genesis.create_genesis_bill(
            denomination=denomination_int,
            user_address=issued_to,
            custom_data={
                "serial_number": serial_number,
                "issued_to": issued_to
            }
        )
        
        # Generate keys for signing
        private_key, public_key = DigitalBill.generate_key_pair()
        
        # Sign the bill
        signature = digital_bill.sign(private_key)
        
        # Create genesis transaction
        genesis_transaction = {
            "type": "GTX_Genesis",
            "serial_number": serial_number,
            "denomination": denomination,
            "issued_to": issued_to,
            "timestamp": time.time(),
            "signature": signature,
            "public_key": public_key,
            "metadata_hash": digital_bill.metadata_hash,
            "front_serial": digital_bill.front_serial,
            "back_serial": digital_bill.back_serial,
            "bill_type": digital_bill.bill_type
        }
        
        # Calculate hash
        tx_string = json.dumps(genesis_transaction, sort_keys=True)
        genesis_transaction["hash"] = hashlib.sha256(tx_string.encode()).hexdigest()
        
        # Add to local mempool
        success = add_transaction_func(genesis_transaction)
        
        if success:
            logger.info(f"✅ Created genesis transaction: {genesis_transaction['hash'][:16]}...")
            return True
        else:
            logger.error(f"Failed to add genesis transaction to mempool")
        
        return success
        
    except Exception as e:
        logger.error(f"Error creating genesis transaction: {e}")
        logger.error(traceback.format_exc())
        return False


def get_transactions_by_block(blockchain: List[Dict], block_height: int) -> List[Dict]:
    """
    Get all transactions from a specific block.
    
    Args:
        block_height (int): Block index
        
    Returns:
        list: List of transactions in the block
    """
    try:
        if 0 <= block_height < len(blockchain):
            block = blockchain[block_height]
            transactions = block.get('transactions', [])
            return transactions
        return []
    except Exception as e:
        logger.error(f"Error getting transactions for block {block_height}: {e}")
        return []


def get_transaction(mempool: List[Dict], blockchain: List[Dict], tx_hash: str, 
                   blockchain_mgr, mempool_mgr) -> Optional[Dict]:
    """
    Get transaction details by hash from mempool or blockchain.
    
    Args:
        tx_hash (str): Transaction hash to look up
        
    Returns:
        dict: Transaction details or None if not found
    """
    try:
        # 1. First check mempool (pending transactions)
        for tx in mempool:
            if tx.get('hash') == tx_hash:
                logger.info(f"Found transaction in mempool: {tx_hash[:16]}...")
                tx_copy = tx.copy()
                tx_copy['status'] = 'pending'
                tx_copy['confirmations'] = 0
                return tx_copy
        
        # 2. Check blockchain (confirmed transactions)
        for block_index, block in enumerate(blockchain):
            for tx in block.get('transactions', []):
                if tx.get('hash') == tx_hash:
                    logger.info(f"Found transaction in block #{block_index}: {tx_hash[:16]}...")
                    tx_copy = tx.copy()
                    tx_copy['status'] = 'confirmed'
                    tx_copy['block_index'] = block_index
                    tx_copy['block_hash'] = block.get('hash')
                    tx_copy['confirmations'] = len(blockchain) - block_index - 1
                    return tx_copy
        
        # 3. Try to sync with network and check again
        try:
            if mempool_mgr.test_connection():
                # Sync and check again
                network_mempool = mempool_mgr.get_mempool()
                if network_mempool:
                    for tx in network_mempool:
                        if tx.get('hash') == tx_hash:
                            logger.info(f"Found transaction in network mempool: {tx_hash[:16]}...")
                            tx_copy = tx.copy()
                            tx_copy['status'] = 'pending'
                            tx_copy['confirmations'] = 0
                            return tx_copy
        
        except Exception as network_error:
            logger.debug(f"Network check failed: {network_error}")
        
        # 4. Not found anywhere
        logger.warning(f"Transaction not found: {tx_hash}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting transaction {tx_hash}: {e}")
        logger.error(traceback.format_exc())
        return None


def add_transaction(transaction: Dict, mempool: List[Dict], validate_transaction_structure_func,
                   is_transaction_mined_func, blockchain: List[Dict], 
                   mempool_mgr, save_mempool_func) -> bool:
    """Add a transaction to mempool with enhanced logging"""
    try:
        tx_type = transaction.get('type', 'unknown')
        logger.info(f"🔍 [ADD_TX] Starting: {tx_type} transaction")
        logger.info(f"🔍 [ADD_TX] Transaction keys: {list(transaction.keys())}")
        logger.debug(f"🔍 [ADD_TX] Full transaction data: {json.dumps(transaction, indent=2, default=str)}")
        
        # Validate transaction structure
        logger.info(f"🔍 [ADD_TX] Step 1: Validating structure...")
        if not validate_transaction_structure_func(transaction):
            logger.error(f"❌ [ADD_TX] FAILED at validation")
            logger.error(f"❌ [ADD_TX] Transaction fields: {list(transaction.keys())}")
            return False
        
        logger.info(f"✅ [ADD_TX] Step 1: Structure validated")
        
        # Calculate hash if not present
        logger.info(f"🔍 [ADD_TX] Step 2: Checking/generating hash...")
        if not transaction.get("hash"):
            logger.info(f"🔍 [ADD_TX] No hash present, generating...")
            # For GTX_Genesis, ensure all fields are present before hashing
            if tx_type == "GTX_Genesis":
                # Make sure we have all the metadata
                required_for_hash = ['serial_number', 'denomination', 'issued_to', 'timestamp']
                missing = [k for k in required_for_hash if k not in transaction]
                if missing:
                    logger.error(f"❌ [ADD_TX] FAILED: Cannot create hash, missing fields: {missing}")
                    return False
            
            tx_string = json.dumps(transaction, sort_keys=True)
            transaction["hash"] = hashlib.sha256(tx_string.encode()).hexdigest()
            logger.info(f"✅ [ADD_TX] Generated hash: {transaction['hash'][:16]}...")
        else:
            logger.info(f"✅ [ADD_TX] Hash already present: {transaction['hash'][:16]}...")
        
        # Check for duplicates in mempool
        tx_hash = transaction["hash"]
        logger.info(f"🔍 [ADD_TX] Step 3: Checking for duplicates in mempool (size: {len(mempool)})...")
        for existing_tx in mempool:
            if existing_tx.get("hash") == tx_hash:
                logger.info(f"ℹ️ [ADD_TX] Transaction already in local mempool: {tx_hash[:16]}...")
                logger.info(f"ℹ️ [ADD_TX] This is OK - transaction is already queued for mining")
                # Return "duplicate" to indicate success but already exists (no broadcast needed)
                return "duplicate"
        logger.info(f"✅ [ADD_TX] No duplicates found")
        
        # Check if already mined
        logger.info(f"🔍 [ADD_TX] Step 4: Checking if already mined...")
        if is_transaction_mined_func(transaction):
            logger.warning(f"⚠️ [ADD_TX] ALREADY MINED: {tx_hash}")
            return False
        logger.info(f"✅ [ADD_TX] Not yet mined")
        
        # Special check for genesis transaction
        if tx_type == "genesis":
            logger.info(f"🔍 [ADD_TX] Step 5: Genesis block check...")
            if blockchain and len(blockchain) > 0:
                logger.warning("⚠️ [ADD_TX] REJECTED: Genesis transaction can only be in the first block")
                return False
        
        # Add to mempool
        logger.info(f"🔍 [ADD_TX] Step 6: Adding to local mempool...")
        mempool.append(transaction)
        logger.info(f"✅ [ADD_TX] Added to local mempool list")
        
        logger.info(f"🔍 [ADD_TX] Step 7: Saving mempool to disk...")
        try:
            save_mempool_func()
            logger.info(f"✅ [ADD_TX] Mempool saved to disk")
        except Exception as save_err:
            logger.error(f"❌ [ADD_TX] Failed to save mempool: {save_err}")
            # Remove from mempool if save failed
            mempool.remove(transaction)
            return False
        
        # Also add to lunalib mempool manager
        logger.info(f"🔍 [ADD_TX] Step 8: Adding to lunalib mempool manager...")
        try:
            mempool_mgr.add_transaction(transaction)
            logger.info(f"✅ [ADD_TX] Added to lunalib mempool manager")
        except Exception as mempool_err:
            logger.warning(f"⚠️ [ADD_TX] Failed to add to lunalib mempool manager: {mempool_err}")
            # Don't fail the whole operation if lunalib fails
        
        logger.info(f"✅✅✅ [ADD_TX] SUCCESS: Added {tx_type} transaction: {tx_hash[:16]}...")
        return True
        
    except Exception as e:
        logger.error(f"❌❌❌ [ADD_TX] EXCEPTION: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def is_transaction_mined(transaction: Dict, blockchain: List[Dict]) -> bool:
    """Check if transaction has been mined"""
    tx_hash = transaction.get("hash")
    if not tx_hash:
        return False
    
    # Check all blocks for this transaction
    for block in blockchain:
        for tx in block.get("transactions", []):
            if tx.get("hash") == tx_hash:
                return True
    return False


def get_mempool_status(mempool: List[Dict]) -> Dict:
    """Get mempool status and statistics"""
    bills = [tx for tx in mempool if tx.get("type") == "GTX_Genesis"]
    genesis = [tx for tx in mempool if tx.get("type") == "genesis"]
    transfers = [tx for tx in mempool if tx.get("type") == "transfer"]
    rewards = [tx for tx in mempool if tx.get("type") == "reward"]
    
    return {
        "total": len(mempool),
        "bills": len(bills),
        "genesis": len(genesis),
        "transfers": len(transfers),
        "rewards": len(rewards),
        "transactions": mempool
    }


def get_blockchain_status(blockchain: List[Dict], mined_serials: set) -> Dict:
    """Get blockchain status and statistics"""
    total_transactions = 0
    genesis_count = 0
    gtx_genesis_count = 0
    transfer_count = 0
    reward_count = 0
    
    for block in blockchain:
        for tx in block.get("transactions", []):
            total_transactions += 1
            tx_type = tx.get("type")
            if tx_type == "genesis":
                genesis_count += 1
            elif tx_type == "GTX_Genesis":
                gtx_genesis_count += 1
            elif tx_type == "transfer":
                transfer_count += 1
            elif tx_type == "reward":
                reward_count += 1
    
    return {
        "blocks": len(blockchain),
        "total_transactions": total_transactions,
        "genesis_transactions": genesis_count,
        "gtx_genesis_transactions": gtx_genesis_count,
        "transfer_transactions": transfer_count,
        "reward_transactions": reward_count,
        "mined_serials": len(mined_serials)
    }


def remove_mined_transactions(mempool: List[Dict], mined_transactions: List[Dict], 
                              save_mempool_func) -> int:
    """Remove mined transactions from the mempool"""
    initial_count = len(mempool)
    mined_hashes = {tx['hash'] for tx in mined_transactions if 'hash' in tx}

    # Filter out mined transactions
    mempool[:] = [tx for tx in mempool if tx.get('hash') not in mined_hashes]

    removed_count = initial_count - len(mempool)
    if removed_count > 0:
        logger.info(f"✅ Removed {removed_count} mined transactions from mempool")
        save_mempool_func()
    else:
        logger.info("⚠️ No mined transactions found in mempool to remove")

    return removed_count


def mark_reward_transactions_mined(reward_transactions: List[Dict], block_index: int, 
                                   mempool_mgr) -> bool:
    """Mark reward transactions as mined in the system"""
    try:
        if not reward_transactions:
            return True
        
        print(f"📝 Marking {len(reward_transactions)} reward transaction(s) as mined")
        
        for reward_tx in reward_transactions:
            # Extract key info from reward transaction
            tx_hash = reward_tx.get('hash')
            miner_address = reward_tx.get('to')
            block_height = reward_tx.get('block_height')
            amount = reward_tx.get('amount', 0)
            
            if tx_hash:
                print(f"  ✓ Reward tx mined: {tx_hash[:16]}...")
                print(f"    Miner: {miner_address}")
                print(f"    Block: {block_height}")
                print(f"    Amount: {amount}")
                print(f"    Marking in mempool manager...")
                mempool_mgr.mark_transaction_mined(tx_hash, block_height)
                print(f"    ✓ Marked as mined")
                
        return True
        
    except Exception as e:
        print(f"❌ Error marking reward transactions as mined: {e}")
        traceback.print_exc()
        return False


def build_mined_indexes(blockchain: List[Dict], mined_serials: set):
    """Build indexes of all mined serial numbers"""
    mined_serials.clear()
    for block in blockchain:
        for tx in block.get("transactions", []):
            if tx.get("type") == "GTX_Genesis":
                serial = tx.get("serial_number")
                if serial:
                    mined_serials.add(serial)


def update_mined_indexes(block: Dict, mined_serials: set):
    """Update mined indexes with transactions from a new block"""
    for tx in block.get("transactions", []):
        if tx.get("type") == "GTX_Genesis":
            serial = tx.get("serial_number")
            if serial:
                mined_serials.add(serial)
