# blockchain_daemon.py
import json
import time
import hashlib
import threading
import os
from typing import List, Dict, Set
from datetime import datetime
import logging
import sys
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, Future

class BlockchainDaemon:
    def __init__(self, blockchain_file="blockchain.json", mempool_file="mempool.json"):
        self.blockchain_file = blockchain_file
        self.mempool_file = mempool_file
        self.blockchain = []
        self.mempool = []
        self.mined_serials: Set[str] = set()
        self.mined_signatures: Set[str] = set()  # ADD THIS LINE
        self.is_running = False
        self.sync_interval = 10  # seconds
        self.cleanup_interval = 10  # seconds
        # Threading attributes
        self.mining_executor = ThreadPoolExecutor(max_workers=1)  # Only one mining operation at a time
        self.current_mining_future = None
        self.mining_lock = threading.Lock()
        self.is_mining = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create initial files first
        self.create_initial_files()
        
        # Then load data
        self.load_data()
    @property
    def is_mining(self):
        """Check if mining is in progress"""
        return getattr(self, '_is_mining', False)

    @is_mining.setter
    def is_mining(self, value):
        """Set mining status"""
        self._is_mining = value

    def get_available_bills_to_mine(self):
        """Get transactions from mempool that haven't been mined yet"""
        available_transactions = []
        
        for tx in self.mempool:
            # Check if transaction has already been mined
            if not self.is_transaction_mined(tx):
                available_transactions.append(tx)
        
        return available_transactions

    # Update the is_transaction_mined method to handle rewards
    def is_transaction_mined(self, transaction: Dict) -> bool:
        """Check if a transaction has already been mined"""
        tx_hash = transaction.get("hash")
        if not tx_hash:
            return False
        
        # Check ALL blocks for this transaction hash
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("hash") == tx_hash:
                    return True
        return False

    def load_data(self):
        """Load blockchain and mempool from files with validation - ENHANCED VERSION"""
        try:
            self.logger.info("=== STARTING LOAD_DATA ===")
            
            # Load blockchain
            if os.path.exists(self.blockchain_file):
                self.logger.info(f"📁 Blockchain file exists: {self.blockchain_file}")
                try:
                    with open(self.blockchain_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            self.logger.warning("Blockchain file is empty, creating new chain")
                            blockchain_data = []
                        else:
                            blockchain_data = json.loads(content)
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Blockchain JSON decode error: {e}")
                    self.handle_corrupt_files()
                    return
                        
                self.logger.info(f"📊 Loaded blockchain data: {len(blockchain_data)} blocks")
                
                # Validate the loaded blockchain
                if blockchain_data:
                    self.logger.info("🔍 Validating blockchain...")
                    if self.validate_chain(blockchain_data):
                        self.blockchain = blockchain_data
                        self.logger.info(f"✅ Loaded validated blockchain with {len(self.blockchain)} blocks")
                    else:
                        self.logger.error("❌ Loaded blockchain failed validation! Attempting emergency repair.")
                        self.blockchain = blockchain_data  # Set it first so we can repair
                        if not self.emergency_repair():
                            self.logger.error("❌ Emergency repair failed! Creating new chain.")
                            self.blockchain = []
                            self.save_blockchain()
                else:
                    self.logger.info("📁 Blockchain file was empty, creating genesis block")
                    self.blockchain = []
                    self.save_blockchain()
            else:
                self.logger.info("📁 Blockchain file does not exist, creating genesis block")
                self.blockchain = []
                self.save_blockchain()
            
            # Load mempool with better error handling
            if os.path.exists(self.mempool_file):
                self.logger.info(f"📁 Mempool file exists: {self.mempool_file}")
                try:
                    with open(self.mempool_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:  # Handle empty file
                            self.logger.warning("Mempool file is empty, starting with empty mempool")
                            self.mempool = []
                        else:
                            self.mempool = json.loads(content)
                    self.logger.info(f"📊 Loaded mempool: {len(self.mempool)} transactions")
                    
                    # Log first few transactions for debugging
                    for i, tx in enumerate(self.mempool[:3]):
                        self.logger.info(f"   TX {i}: {tx.get('type', 'unknown')} - {tx.get('serial_number', 'no-serial')}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Mempool JSON decode error: {e}")
                    # Don't reset everything for mempool corruption, just start fresh
                    self.mempool = []
                    self.save_mempool()
            else:
                self.logger.info("📁 Mempool file does not exist, creating empty mempool")
                self.mempool = []
                self.save_mempool()
            
            # Build indexes of mined transactions
            self.logger.info("🔨 Building mined indexes...")
            self.build_mined_indexes()
            
            # NEW: Check if we need to create initial genesis transactions
            if self.needs_initial_genesis():
                self.logger.info("🆕 Blockchain has no genesis transactions, creating initial set...")
                self.create_initial_genesis_transactions()
            else:
                self.logger.info("✅ Blockchain already contains genesis transactions")
            
            self.logger.info("✅ load_data completed successfully")
            
        except Exception as e:
            self.logger.error(f"💥 Error loading data: {e}")
            import traceback
            self.logger.error(f"💥 Full traceback: {traceback.format_exc()}")
            # Initialize with empty data but don't corrupt existing files
            if not self.blockchain:
                self.blockchain = []
            self.mempool = []  # Always start with empty mempool on error
    def mine_pending_transactions(self, miner_address: str, difficulty: int = 4, async_mode: bool = False) -> Dict:
        """Main mining function with both sync and async support"""
        if async_mode:
            # Use async version for web requests
            return self.mine_pending_transactions_async(miner_address, difficulty)
        else:
            # Use sync version for normal mining operations
            return self._mine_pending_transactions_sync(miner_address, difficulty)

    def _mine_pending_transactions_sync(self, miner_address: str, difficulty: int = 4) -> Dict:
        """Synchronous mining - this should be called by your normal miners"""
        try:
            # Clean up any mined transactions first
            self.cleanup_mined_transactions()
            
            # Get available transactions of all types
            available_bills = self.get_available_bills_to_mine()
            available_rewards = self.get_available_rewards_to_mine()
            available_transfers = self.get_available_transfers_to_mine()
            
            # DEBUG: Log what's available
            self.logger.info(f"⛏️ [SYNC] Mining - Bills: {len(available_bills)}, Rewards: {len(available_rewards)}, Transfers: {len(available_transfers)}")
            
            # CRITICAL FIX: Return None if no transactions available
            if not available_bills and not available_rewards and not available_transfers:
                self.logger.info("❌ [SYNC] No transactions available to mine - stopping")
                return None
            
            # FIXED REWARD CREATION: Create reward BEFORE transaction selection
            next_block_height = len(self.blockchain)
            reward_created = False
            
            # Only create reward if there are bills to mine
            if available_bills:
                # FIX: Pass the actual bills list, not the count
                reward_created = self.create_and_add_reward_transaction(
                    miner_address, 
                    available_bills[:20],
                    next_block_height
                )
                if reward_created:
                    # Refresh available rewards list
                    available_rewards = self.get_available_rewards_to_mine()
                    self.logger.info(f"💰 [SYNC] Created reward transaction for {miner_address}")
            
            # ENHANCED TRANSACTION SELECTION: Prioritize transfers when backlog exists
            max_per_block = 50
            transactions_to_mine = []
            
            # CRITICAL FIX: Check if we have a transfer backlog and prioritize accordingly
            if len(available_transfers) >= 10:
                self.logger.info("🎯 [SYNC] HIGH TRANSFER BACKLOG DETECTED - PRIORITIZING TRANSFERS")
                transfers_to_include = available_transfers[:30]
                bills_to_include = available_bills[:10]
                rewards_to_include = available_rewards[:10]
                all_candidates = transfers_to_include + bills_to_include + rewards_to_include
            else:
                # Normal allocation when no major backlog
                bills_to_include = available_bills[:20]
                rewards_to_include = available_rewards[:15]  
                transfers_to_include = available_transfers[:15]
                all_candidates = bills_to_include + rewards_to_include + transfers_to_include
            
            # VALIDATE all candidate transactions before including
            valid_candidates = []
            validation_stats = {"bills": 0, "rewards": 0, "transfers": 0, "invalid": 0}
            
            for tx in all_candidates:
                if self.validate_transaction(tx, skip_mined_check=True):
                    valid_candidates.append(tx)
                    tx_type = tx.get("type", "unknown")
                    if tx_type in ["GTX_Genesis", "genesis"]:
                        validation_stats["bills"] += 1
                    elif tx_type == "reward":
                        validation_stats["rewards"] += 1
                    elif tx_type == "transfer":
                        validation_stats["transfers"] += 1
                else:
                    validation_stats["invalid"] += 1
                    self.logger.warning(f"❌ [SYNC] Excluding invalid transaction: {tx.get('type')} - {tx.get('hash', 'no-hash')[:16]}...")
            
            # Log validation results
            self.logger.info(f"📊 [SYNC] Validation: {validation_stats['bills']} bills, {validation_stats['rewards']} rewards, {validation_stats['transfers']} transfers valid")
            
            # Sort by timestamp (oldest first) to prioritize stuck transactions
            valid_candidates.sort(key=lambda tx: tx.get('timestamp', 0))
            transactions_to_mine = valid_candidates[:max_per_block]
            
            # DEBUG: Log the block composition
            bill_count_final = sum(1 for tx in transactions_to_mine if tx.get("type") in ["GTX_Genesis", "genesis"])
            reward_count_final = sum(1 for tx in transactions_to_mine if tx.get("type") == "reward")
            transfer_count_final = sum(1 for tx in transactions_to_mine if tx.get("type") == "transfer")
            
            self.logger.info(f"📦 [SYNC] Block composition - Bills: {bill_count_final}, Rewards: {reward_count_final}, Transfers: {transfer_count_final}")
            
            # CRITICAL CHECK: If we have transfers available but none made it into the block, debug why
            if available_transfers and transfer_count_final == 0:
                self.logger.warning("⚠️ [SYNC] Transfers available but none selected for mining - investigating...")
                self.debug_transfer_mining()
            
            if not transactions_to_mine:
                self.logger.info("[SYNC] No valid transactions to mine after validation")
                return None
            
            # Ensure we have a previous block
            if not self.blockchain:
                self.logger.error("[SYNC] No blockchain available - cannot mine without genesis block")
                return None
                
            previous_block = self.blockchain[-1]
            if not previous_block or "hash" not in previous_block:
                self.logger.error("[SYNC] Previous block is invalid - cannot mine")
                return None
            
            previous_hash = previous_block["hash"]
            
            # Create new block with proper structure
            new_block = {
                "index": len(self.blockchain),
                "timestamp": int(time.time()),
                "transactions": transactions_to_mine,
                "previous_hash": previous_hash,
                "nonce": 0,
                "miner": miner_address,
                "difficulty": difficulty,
                "hash": ""
            }
            
            # Mine the block (proof of work)
            self.logger.info(f"🔨 [SYNC] Starting proof-of-work for block #{new_block['index']}...")
            start_time = time.time()
            mined_block = self.mine_block(new_block, difficulty)
            mining_time = time.time() - start_time
            
            if not mined_block:
                self.logger.error("❌ [SYNC] Failed to mine block - proof of work unsuccessful")
                return None
            
            # Add mining time to block for analytics
            mined_block["mining_time"] = mining_time
            
            # VALIDATE the mined block before adding to chain
            if not self.validate_block_structure(mined_block):
                self.logger.error("❌ [SYNC] Mined block has invalid structure")
                return None
                
            if not self.validate_block_transactions(mined_block.get("transactions", [])):
                self.logger.error("❌ [SYNC] Mined block contains invalid transactions")
                return None
            
            # Check chain continuity
            if self.blockchain and mined_block["previous_hash"] != previous_hash:
                self.logger.error("❌ [SYNC] Block continuity broken - previous hash mismatch")
                return None
            
            # Add to blockchain
            self.blockchain.append(mined_block)
            
            # IMMEDIATELY remove mined transactions from mempool using enhanced cleanup
            self.comprehensive_mempool_cleanup()            
            
            # Update mined indexes
            self.update_mined_indexes(mined_block)
            
            # Save changes with validation
            if not self.validate_chain(self.blockchain):
                self.logger.error("❌ [SYNC] Blockchain validation failed after adding new block - rolling back")
                self.blockchain.pop()
                self.mempool.extend(transactions_to_mine)
                self.save_mempool()
                return None
            
            # Save the validated blockchain
            self.save_blockchain()
            self.save_mempool()
            
            self.logger.info(f"✅ [SYNC] Successfully mined block #{mined_block['index']} with {len(transactions_to_mine)} transactions")
            self.logger.info(f"   - Bills: {bill_count_final}, Rewards: {reward_count_final}, Transfers: {transfer_count_final}")
            self.logger.info(f"   - Mining Time: {mining_time:.2f}s")
            
            return mined_block
            
        except Exception as e:
            self.logger.error(f"💥 [SYNC] Critical error in mine_pending_transactions: {e}")
            import traceback
            self.logger.error(f"💥 [SYNC] Traceback: {traceback.format_exc()}")
            return None
    def needs_initial_genesis(self) -> bool:
        """Check if blockchain needs initial genesis transactions"""
        # If no blocks exist, we need genesis
        if not self.blockchain:
            return True
        
        # Count genesis transactions in blockchain
        genesis_count = 0
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("type") in ["GTX_Genesis", "genesis"]:
                    genesis_count += 1
        
        # If no genesis transactions found, we need to create them
        return genesis_count == 0
    def create_initial_files(self):
        """Create valid initial JSON files if they don't exist or are empty"""
        # Create blockchain file if empty or doesn't exist
        if not os.path.exists(self.blockchain_file) or os.path.getsize(self.blockchain_file) == 0:
            self.logger.info("Creating initial blockchain file")
            self.blockchain = []
            self.save_blockchain()
        
        # Create mempool file if empty or doesn't exist
        if not os.path.exists(self.mempool_file) or os.path.getsize(self.mempool_file) == 0:
            self.logger.info("Creating initial mempool file")
            self.mempool = []
            self.save_mempool()
    def handle_corrupt_files(self):
        """Handle corrupt blockchain or mempool files - LESS AGGRESSIVE"""
        self.logger.warning("🔄 Handling potentially corrupt files...")
        
        # Backup corrupt files
        import shutil
        timestamp = int(time.time())
        
        # Only backup and reset blockchain if it's actually corrupt
        if os.path.exists(self.blockchain_file):
            try:
                with open(self.blockchain_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only backup if file has content
                        json.loads(content)  # Test if it's valid JSON
                        # If we get here, JSON is valid, no need to reset
                        self.logger.info("Blockchain file is actually valid, no reset needed")
                        return
            except (json.JSONDecodeError, Exception):
                # File is corrupt, backup and reset
                backup_name = f"{self.blockchain_file}.corrupt.{timestamp}"
                shutil.copy2(self.blockchain_file, backup_name)
                self.logger.info(f"📁 Backed up corrupt blockchain to: {backup_name}")
                self.blockchain = []
                self.save_blockchain()
        
        # Handle mempool corruption separately
        if os.path.exists(self.mempool_file):
            try:
                with open(self.mempool_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        json.loads(content)  # Test if valid
            except (json.JSONDecodeError, Exception):
                backup_name = f"{self.mempool_file}.corrupt.{timestamp}"
                shutil.copy2(self.mempool_file, backup_name)
                self.logger.info(f"📁 Backed up corrupt mempool to: {backup_name}")
        
        # Always reset mempool on corruption (it's safe to lose mempool transactions)
        self.mempool = []
        self.save_mempool()
        
        self.logger.info("✅ Handled file corruption")
    
    def validate_chain(self, chain: List[Dict]) -> bool:
        """Validate blockchain server-side"""
        if chain is None:
            return False

        if len(chain) == 0:
            self.logger.info("Blockchain is empty (no genesis yet) ✅")
            return True
        
        # If not empty, still validate normally
        genesis = chain[0]
        if genesis.get("index") != 0:
            self.logger.error("First block must start at index 0")
            return False
        
        # Validate each block
        for i in range(1,len(chain)):
            current_block = chain[i]
            
            # Check block structure
            if not self.validate_block_structure(current_block):
                self.logger.error(f"Block {i} has invalid structure")
                return False
            
            # Check hash validity - include difficulty if present
            difficulty = current_block.get("difficulty")
            calculated_hash = self.calculate_block_hash(
                current_block["index"],
                current_block["previous_hash"],
                current_block["timestamp"],
                current_block["transactions"],
                current_block["nonce"],
                difficulty  # Include difficulty in hash calculation
            )
            
            if current_block["hash"] != calculated_hash:
                self.logger.error(f"❌ Block {i} hash is invalid!")
                self.logger.error(f"Expected: {calculated_hash}")
                self.logger.error(f"Got: {current_block['hash']}")
                self.logger.error(f"Block data: index={current_block['index']}, "
                                f"prev_hash={current_block['previous_hash'][:16]}..., "
                                f"timestamp={current_block['timestamp']}, "
                                f"tx_count={len(current_block.get('transactions', []))}, "
                                f"nonce={current_block['nonce']}, "
                                f"difficulty={difficulty}")
                return False
            
            # Check chain continuity (except for genesis block)
            if i > 0:
                previous_block = chain[i-1]
                if current_block["previous_hash"] != previous_block["hash"]:
                    self.logger.error(f"❌ Block {i} previous hash doesn't match!")
                    self.logger.error(f"Expected: {previous_block['hash']}")
                    self.logger.error(f"Got: {current_block['previous_hash']}")
                    return False
        
        self.logger.info("✅ Blockchain validation passed")
        return True
    
    def validate_block_structure(self, block: Dict) -> bool:
        self.logger.info(f"   🔍 Validating block structure for index {block.get('index')}")
    
        # Required fields for ALL blocks
        required_fields = ["index", "timestamp", "transactions", "previous_hash", "nonce", "hash"]
        missing_fields = [field for field in required_fields if field not in block]
        
        if missing_fields:
            self.logger.error(f"❌ Block missing required fields: {missing_fields}")
            self.logger.error(f"   Block keys: {list(block.keys())}")
            return False
        
        # Optional fields that newer blocks might have
        optional_fields = ["miner", "difficulty", "mining_time"]
        
        if not isinstance(block["transactions"], list):
            self.logger.error("❌ Block transactions must be a list")
            self.logger.error(f"   Transactions type: {type(block['transactions'])}")
            return False
        
        if block["index"] < 0:
            self.logger.error("❌ Block index cannot be negative")
            return False
        
        self.logger.info(f"   ✅ Block structure valid")
        return True

    def validate_block_transactions(self, transactions: List[Dict]) -> bool:
        """Validate all transactions in a block - FIXED VERSION"""
        self.logger.info(f"   🔍 Validating {len(transactions)} transactions in block")
        
        for i, tx in enumerate(transactions):
            self.logger.info(f"     Transaction {i}: type={tx.get('type')}, hash={tx.get('hash', 'no-hash')[:16]}...")
            
            # Skip mined checks when validating blocks (transactions are already mined!)
            if not self.validate_transaction(tx, skip_mined_check=True):
                self.logger.error(f"❌ Invalid transaction at index {i}: {tx}")
                return False
            
            # Check for double-spending within the same block
            tx_id = tx.get("signature") or tx.get("serial_number") or tx.get("hash")
            if tx_id:
                occurrences = sum(1 for t in transactions 
                                if (t.get("signature") == tx_id or 
                                    t.get("serial_number") == tx_id or
                                    t.get("hash") == tx_id))
                if occurrences > 1:
                    self.logger.error(f"❌ Transaction {tx_id} appears {occurrences} times in block")
                    return False
        
        self.logger.info(f"   ✅ All transactions valid")
        return True
    
    def build_mined_indexes(self):
        """Build indexes of all mined serial numbers to prevent double-spending"""
        self.mined_serials.clear()
        
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                # Index by serial number (for GTX_Genesis and other bill-based transactions)
                if tx.get("serial_number"):
                    self.mined_serials.add(tx["serial_number"])
        
        self.logger.info(f"Built index: {len(self.mined_serials)} mined serials")
    
    def create_genesis_block(self):
        """Create the genesis block"""
        genesis_transactions = [{
            "type": "genesis",
            "message": "Luna Coin Genesis Block",
            "timestamp": time.time(),
            "hash": "genesis_0000000000000000"
        }]
        
        genesis_block = {
            "index": 0,
            "timestamp": time.time(),
            "transactions": genesis_transactions,
            "previous_hash": "0",
            "nonce": 0,
            "hash": self.calculate_block_hash(0, "0", time.time(), genesis_transactions, 0)
        }
        
        self.logger.info("✅ Created genesis block")
        return genesis_block
    
    def calculate_block_hash(self, index, previous_hash, timestamp, transactions, nonce, difficulty=None):
        """Calculate SHA-256 hash of a block - FIXED VERSION"""
        # ALWAYS convert timestamp to integer for consistency
        if isinstance(timestamp, float):
            timestamp = int(timestamp)
        
        # Ensure other values are the right type
        index = int(index)
        nonce = int(nonce)
        
        # Create a stable representation of transactions
        if transactions:
            # Sort transactions by hash for consistent ordering
            sorted_transactions = sorted(transactions, key=lambda x: x.get('hash', ''))
            transactions_string = json.dumps(sorted_transactions, sort_keys=True, separators=(',', ':'))
        else:
            transactions_string = "[]"
        
        # Build the block string - use consistent formatting
        block_data = {
            'index': index,
            'previous_hash': previous_hash,
            'timestamp': timestamp,
            'transactions': transactions_string,
            'nonce': nonce
        }
        
        # Include difficulty if provided
        if difficulty is not None:
            block_data['difficulty'] = int(difficulty)
        
        # Create a consistent string representation
        block_string = json.dumps(block_data, sort_keys=True, separators=(',', ':'))
        
        calculated_hash = hashlib.sha256(block_string.encode()).hexdigest()
        
        
        return calculated_hash
    
    def get_available_bills_to_mine(self):
        """Get bills from mempool that haven't been mined yet - FIXED VERSION"""
        available_bills = []
        
        for tx in self.mempool:
            # Include both "GTX_Genesis" and "genesis" type transactions
            if tx.get("type") in ["GTX_Genesis", "genesis"]:
                serial = tx.get("serial_number")
                # Check both by serial AND by transaction hash
                if serial and serial not in self.mined_serials and not self.is_transaction_mined(tx):
                    available_bills.append(tx)
        
        self.logger.info(f"📊 Available bills to mine: {len(available_bills)}")
        return available_bills
            
    def get_available_transfers_to_mine(self):
        """Get transfer transactions from mempool that haven't been mined yet - WITH DEBUG"""
        available_transfers = []
        
        self.logger.info(f"🔍 Checking available transfers (mempool size: {len(self.mempool)})")
        
        transfer_count = 0
        for tx in self.mempool:
            if tx.get("type") == "transfer":
                transfer_count += 1
                is_mined = self.is_transaction_mined(tx)
                is_valid = self.validate_transfer_for_mining(tx)
                
                self.logger.info(f"   Transfer {tx.get('hash', 'no-hash')[:16]}...: "
                            f"mined={is_mined}, valid={is_valid}")
                
                if not is_mined and is_valid:
                    available_transfers.append(tx)
        
        self.logger.info(f"📊 Transfer analysis: {transfer_count} total, {len(available_transfers)} available to mine")
        return available_transfers
    def debug_transfer_mining(self):
        """Debug method to see why transfers aren't being mined"""
        available_transfers = []
        rejected_transfers = []
        
        for tx in self.mempool:
            if tx.get("type") == "transfer":
                if not self.is_transaction_mined(tx):
                    if self.validate_transfer_for_mining(tx):
                        available_transfers.append(tx)
                    else:
                        rejected_transfers.append(tx)
        
        self.logger.info(f"🔍 TRANSFER DEBUG:")
        self.logger.info(f"   Available to mine: {len(available_transfers)}")
        self.logger.info(f"   Rejected: {len(rejected_transfers)}")
        
        for tx in rejected_transfers:
            self.logger.info(f"   Rejected transfer: {tx.get('from')} -> {tx.get('to')} amount: {tx.get('amount')}")
        
        return available_transfers
    def validate_transfer_for_mining(self, transfer_tx: Dict) -> bool:
        """Validate transfer transaction before mining - WITH DEBUG"""
        try:
            self.logger.info(f"   Validating transfer for mining: {transfer_tx.get('hash', 'no-hash')[:16]}...")
            
            # Basic structure validation only - balance checking happens during confirmation
            required_fields = ["from", "to", "amount", "timestamp", "signature", "hash"]
            missing_fields = [field for field in required_fields if field not in transfer_tx]
            
            if missing_fields:
                self.logger.error(f"   ❌ Missing fields: {missing_fields}")
                return False
            
            # Validate amount
            if not isinstance(transfer_tx["amount"], (int, float)) or transfer_tx["amount"] <= 0:
                self.logger.error(f"   ❌ Invalid amount: {transfer_tx['amount']}")
                return False
                
            self.logger.info("   ✅ Transfer valid for mining")
            return True
                
        except Exception as e:
            self.logger.error(f"   ❌ Error validating transfer: {e}")
            return False

    def validate_transfer_transaction(self, transaction: Dict) -> bool:
        """Specific validation for transfer transactions"""
        required_fields = ["from", "to", "amount", "timestamp", "signature", "hash"]
        missing_fields = [field for field in required_fields if field not in transaction]
        
        if missing_fields:
            self.logger.error(f"Transfer transaction missing fields: {missing_fields}")
            return False
        
        # Validate amount
        if not isinstance(transaction["amount"], (int, float)) or transaction["amount"] <= 0:
            self.logger.error("Invalid amount in transfer transaction")
            return False
        
        # Validate addresses
        if not transaction["from"] or not transaction["to"]:
            self.logger.error("Invalid addresses in transfer transaction")
            return False
        
        return True
    
    def add_transaction(self, transaction: Dict) -> bool:
        """Add a new transaction to mempool (thread-safe) - UPDATED FOR REWARDS"""
        try:
            # Get transaction type FIRST
            tx_type = transaction.get("type", "transaction")
            
            # Validate transaction structure with detailed logging
            if not self.validate_transaction_structure(transaction):
                self.logger.error("❌ Transaction structure validation failed")
                return False
                
            # Log the transaction for debugging
            self.logger.info(f"🔍 Processing {tx_type} transaction:")
            self.logger.info(f"   Hash: {transaction.get('hash')}")
            
            if tx_type == "transfer":
                self.logger.info(f"   From: {transaction.get('from', 'N/A')}")
                self.logger.info(f"   To: {transaction.get('to', 'N/A')}")
                self.logger.info(f"   Amount: {transaction.get('amount', 'N/A')}")
                self.logger.info(f"   Signature: {transaction.get('signature', 'N/A')[:20]}...")
            elif tx_type == "reward":
                self.logger.info(f"   To: {transaction.get('to', 'N/A')}")
                self.logger.info(f"   Amount: {transaction.get('amount', 'N/A')}")
                self.logger.info(f"   Block Height: {transaction.get('block_height', 'N/A')}")
            elif tx_type in ["GTX_Genesis", "genesis"]:
                self.logger.info(f"   Serial: {transaction.get('serial_number', 'N/A')}")
                self.logger.info(f"   Issued To: {transaction.get('issued_to', 'N/A')}")
                self.logger.info(f"   Denomination: {transaction.get('denomination', 'N/A')}")
            
            # When adding to mempool, DO check for mined duplicates
            if not self.validate_transaction(transaction, skip_mined_check=False):
                self.logger.error("❌ Transaction validation failed")
                return False
                
            # Ensure transaction has required fields based on type
            # Calculate hash for the transaction
            if not transaction.get("hash"):
                transaction_string = json.dumps(transaction, sort_keys=True)
                transaction["hash"] = hashlib.sha256(transaction_string.encode()).hexdigest()
            
            # Check for duplicates using hash
            tx_hash = transaction["hash"]
            for existing_tx in self.mempool:
                if existing_tx.get("hash") == tx_hash:
                    self.logger.warning(f"⚠️ Transaction already in mempool: {tx_hash}")
                    return False
            
            # Special duplicate check for reward transactions
            if tx_type == "reward":
                block_height = transaction.get("block_height")
                miner_address = transaction.get("to")
                if self.is_reward_already_exists(block_height, miner_address):
                    self.logger.warning(f"⚠️ Reward for block {block_height} to {miner_address} already exists")
                    return False
            
            # Special duplicate check for genesis transactions
            if tx_type in ["GTX_Genesis", "genesis"]:
                serial = transaction.get("serial_number")
                if serial:
                    # Check mempool for duplicate serials
                    for existing_tx in self.mempool:
                        if existing_tx.get("type") in ["GTX_Genesis", "genesis"] and existing_tx.get("serial_number") == serial:
                            self.logger.warning(f"⚠️ Genesis transaction with serial {serial} already in mempool")
                            return False
            
            # Check if already mined
            if self.is_transaction_mined(transaction):
                self.logger.warning(f"⚠️ Transaction already mined: {tx_hash}")
                return False
            
            # Add to mempool
            self.mempool.append(transaction)
            self.save_mempool()
            self.logger.info(f"✅ Added {tx_type} transaction to mempool: {tx_hash}")
            self.logger.info(f"📊 Mempool size: {len(self.mempool)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Error in add_transaction: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    def validate_transaction_structure(self, transaction):
        """Validate basic transaction structure - FIXED FOR GENESIS"""
        tx_type = transaction.get("type")
        
        # For transfer transactions
        if tx_type == "transfer":
            required_fields = ["type", "from", "to", "amount", "timestamp", "signature"]
        # For genesis transactions
        elif tx_type in ["GTX_Genesis", "genesis"]:
            required_fields = ["type", "timestamp", "serial_number", "denomination", "issued_to"]
        # For reward transactions
        elif tx_type == "reward":
            required_fields = ["type", "to", "amount", "timestamp", "block_height"]
        # For regular transactions
        elif tx_type == "transaction":
            required_fields = ["type", "timestamp"]
        else:
            # For other types, be more flexible
            required_fields = ["type", "timestamp"]
        
        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in transaction]
        if missing_fields:
            self.logger.error(f"Missing required fields for {tx_type} transaction: {missing_fields}")
            self.logger.error(f"Transaction: {transaction}")
            return False
        
        return True
    def get_denomination_stats(self):
        """Get statistics about bill denominations in mempool"""
        stats = {
            "total_bills": 0,
            "total_value": 0,
            "denomination_counts": {},
            "difficulty_estimate": 0
        }
        
        available_bills = self.get_available_bills_to_mine()
        
        for bill in available_bills:
            denomination = bill.get('denomination', 0)
            stats["total_bills"] += 1
            stats["total_value"] += denomination
            
            if denomination in stats["denomination_counts"]:
                stats["denomination_counts"][denomination] += 1
            else:
                stats["denomination_counts"][denomination] = 1
        
        # Calculate estimated difficulty
        if available_bills:
            stats["difficulty_estimate"] = self.calculate_difficulty_from_bills(available_bills)
        
        return stats
    def create_and_add_reward_transaction(self, miner_address: str, gtx_transactions: List[Dict], block_height: int) -> bool:
        """Create a reward transaction based on GTX Genesis transactions being mined"""
        try:
            # Handle case where we might get bill_count instead of transactions list
            if isinstance(gtx_transactions, int):
                # If we got bill_count instead of transactions, we can't create a proper reward
                self.logger.error(f"❌ Received integer instead of transactions list: {gtx_transactions}")
                return False
            
            if not gtx_transactions:
                self.logger.warning("No GTX transactions provided for reward")
                return False
            
            # Calculate reward based on GTX Genesis transactions being included in this block
            base_reward_per_bill = 1
            total_reward = base_reward_per_bill * len(gtx_transactions)
            
            # Create a descriptive list of the bills being mined
            bill_descriptions = []
            total_bill_value = 0
            
            for tx in gtx_transactions:
                if tx.get("type") in ["GTX_Genesis", "genesis"]:
                    denomination = tx.get('denomination', 0)
                    serial = tx.get('serial_number', 'unknown')
                    desc = f"${denomination}-{serial}"
                    bill_descriptions.append(desc)
                    total_bill_value += denomination
            
            reward_tx = {
                "type": "reward",
                "to": miner_address,
                "amount": total_reward,
                "timestamp": int(time.time()),
                "block_height": block_height,
                "description": f"Mining reward for {len(gtx_transactions)} bills (value: ${total_bill_value})",
                "bills_mined": [tx.get('serial_number') for tx in gtx_transactions if tx.get('serial_number')],
                "bill_count": len(gtx_transactions),
                "total_bill_value": total_bill_value,
                "hash": ""  # Will be calculated
            }
            
            # Calculate hash for the reward transaction
            reward_string = json.dumps(reward_tx, sort_keys=True)
            reward_tx["hash"] = hashlib.sha256(reward_string.encode()).hexdigest()
            
            # Check if reward for this block height already exists
            if self.is_reward_already_exists(block_height, miner_address):
                self.logger.warning(f"Reward for block {block_height} already exists for miner {miner_address}")
                return False
            
            # Add to mempool
            return self.add_transaction(reward_tx)
            
        except Exception as e:
            self.logger.error(f"Error creating reward transaction: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    def is_reward_already_exists(self, block_height: int, miner_address: str) -> bool:
        """Check if a reward transaction for this block height and miner already exists"""
        # Check mempool
        for tx in self.mempool:
            if (tx.get("type") == "reward" and 
                tx.get("block_height") == block_height and 
                tx.get("to") == miner_address):
                return True
        
        # Check blockchain
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if (tx.get("type") == "reward" and 
                    tx.get("block_height") == block_height and 
                    tx.get("to") == miner_address):
                    return True
        
        return False
    def cleanup_mined_transactions(self):
        """Remove any transactions from mempool that have already been mined"""
        initial_count = len(self.mempool)
        
        # Create a list of all transaction hashes in the blockchain
        mined_hashes = set()
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("hash"):
                    mined_hashes.add(tx["hash"])
        
        # Remove mined transactions from mempool
        self.mempool = [tx for tx in self.mempool if tx.get("hash") not in mined_hashes]
        
        removed_count = initial_count - len(self.mempool)
        if removed_count > 0:
            self.logger.info(f"🧹 Cleaned up {removed_count} already mined transactions from mempool")
            self.save_mempool()
        
        return removed_count
    def get_available_rewards_to_mine(self):
        """Get reward transactions from mempool that haven't been mined yet"""
        available_rewards = []
        
        for tx in self.mempool:
            if tx.get("type") == "reward" and not self.is_transaction_mined(tx):
                available_rewards.append(tx)
        
        return available_rewards
    def get_transaction_confirmations(self, tx_hash: str) -> int:
        """Get number of confirmations for a transaction"""
        # Find the block containing the transaction
        tx_block_index = -1
        for i, block in enumerate(self.blockchain):
            for tx in block.get("transactions", []):
                if tx.get("hash") == tx_hash:
                    tx_block_index = i
                    break
            if tx_block_index != -1:
                break
        
        if tx_block_index == -1:
            return 0  # Transaction not found in blockchain
        
        # Confirmations = current block height - transaction block index
        current_height = len(self.blockchain) - 1  # 0-based indexing
        confirmations = current_height - tx_block_index
        
        return max(0, confirmations)  # Ensure non-negative

    def update_transaction_confirmations(self):
        """Update confirmation counts for all transactions in mempool and recent blocks"""
        current_height = len(self.blockchain) - 1
        
        # Update mempool transactions (they should have 0 confirmations)
        for tx in self.mempool:
            if "confirmations" not in tx:
                tx["confirmations"] = 0
            else:
                tx["confirmations"] = 0  # Mempool transactions are unconfirmed
        
        # Update transactions in blockchain
        for i, block in enumerate(self.blockchain):
            confirmations = current_height - i
            for tx in block.get("transactions", []):
                tx["confirmations"] = max(0, confirmations)

    def get_transaction_with_confirmations(self, tx_hash: str) -> Dict:
        """Get transaction with current confirmation count"""
        # Check mempool first
        for tx in self.mempool:
            if tx.get("hash") == tx_hash:
                return {**tx, "confirmations": 0, "status": "pending"}
        
        # Check blockchain
        for i, block in enumerate(self.blockchain):
            for tx in block.get("transactions", []):
                if tx.get("hash") == tx_hash:
                    confirmations = (len(self.blockchain) - 1) - i
                    return {
                        **tx, 
                        "confirmations": confirmations,
                        "status": "confirmed",
                        "block_height": i,
                        "block_hash": block.get("hash")
                    }
        
        return {"error": "Transaction not found"}

    def start_confirmation_updater(self):
        """Start background thread to update confirmations periodically"""
        def confirmation_loop():
            while self.is_running:
                try:
                    self.update_transaction_confirmations()
                    # Save updated state
                    self.save_blockchain()
                    self.save_mempool()
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    self.logger.error(f"Error in confirmation updater: {e}")
                    time.sleep(60)
        
        self.confirmation_thread = threading.Thread(target=confirmation_loop, daemon=True)
        self.confirmation_thread.start()
    def calculate_difficulty_from_bills(self, bills: List[Dict], base_difficulty: int = 2) -> int:
        """Calculate mining difficulty based on bill denominations"""
        if not bills:
            return base_difficulty
        
        # Define difficulty tiers based on denominations
        difficulty_tiers = {
            1: 1,      
            10: 2,      
            100: 3,     
            1000: 4,     
            10000: 5,     
            100000: 6,    
            1000000: 7,    
            10000000: 8,
            100000000: 9   
        }
        
        # Calculate average denomination
        denominations = [bill.get('denomination', 0) for bill in bills if bill.get('denomination')]
        if not denominations:
            return base_difficulty
        
        avg_denomination = sum(denominations) / len(denominations)
        max_denomination = max(denominations)
        
        # Calculate difficulty based on highest denomination and average
        base_difficulty_from_max = difficulty_tiers.get(max_denomination, base_difficulty)
        
        # Adjust based on quantity and value
        quantity_factor = min(len(bills) / 10, 2)  # Scale with quantity, max 2x
        value_factor = min(avg_denomination / 50, 3)  # Scale with value, max 3x
        
        dynamic_difficulty = max(
            base_difficulty,
            int(base_difficulty_from_max * quantity_factor * value_factor)
        )
        
        # Cap the maximum difficulty
        max_allowed_difficulty = 9
        final_difficulty = min(dynamic_difficulty, max_allowed_difficulty)
        
        self.logger.info(f"   💰 Difficulty calculation: {len(bills)} bills, avg: ${avg_denomination:.2f}, max: ${max_denomination}")
        self.logger.info(f"   📊 Factors: base={base_difficulty_from_max}, quantity={quantity_factor:.2f}, value={value_factor:.2f}")
        
        return final_difficulty
    def select_bills_by_denomination(self, bills: List[Dict], max_count: int = 35) -> List[Dict]:
        """Select bills for mining, prioritizing higher denominations"""
        if not bills:
            return []
        
        # Sort bills by denomination (highest first)
        sorted_bills = sorted(bills, key=lambda x: x.get('denomination', 0), reverse=True)
        
        # Take the top bills
        selected_bills = sorted_bills[:max_count]
        
        # Log selection info
        if selected_bills:
            denominations = [bill.get('denomination', 0) for bill in selected_bills]
            self.logger.info(f"   🏆 Selected {len(selected_bills)} bills: min=${min(denominations)}, max=${max(denominations)}, avg=${sum(denominations)/len(denominations):.2f}")
        
        return selected_bills
    def debug_reward_issue(self):
        """Debug method to check what's happening with reward creation"""
        available_bills = self.get_available_bills_to_mine()
        self.logger.info(f"🔍 DEBUG: Available bills count: {len(available_bills)}")
        self.logger.info(f"🔍 DEBUG: Available bills type: {type(available_bills)}")
        
        if available_bills:
            sample_bill = available_bills[0]
            self.logger.info(f"🔍 DEBUG: Sample bill: {sample_bill}")
            self.logger.info(f"🔍 DEBUG: Sample bill type: {type(sample_bill)}")
        
        # Test reward creation
        if available_bills:
            test_result = self.create_and_add_reward_transaction(
                "test_miner",
                available_bills[:3],  # Pass actual transactions
                len(self.blockchain)
            )
            self.logger.info(f"🔍 DEBUG: Test reward creation result: {test_result}")
    def mine_pending_transactions_async(self, miner_address: str, difficulty: int = 4) -> Future:
        """
        Start mining asynchronously and return a Future object
        Usage: future = daemon.mine_pending_transactions_async("miner_address")
               result = future.result()  # This will block until mining completes
        """
        with self.mining_lock:
            if self.is_mining:
                raise Exception("Mining already in progress")
            
            self.is_mining = True
            self.logger.info("🚀 Starting asynchronous mining...")
            
            # Submit mining task to thread pool
            future = self.mining_executor.submit(
                self._mine_pending_transactions_sync, 
                miner_address, 
                difficulty
            )
            
            # Add callback to handle completion
            future.add_done_callback(self._mining_complete_callback)
            self.current_mining_future = future
            
            return future

    def _mining_complete_callback(self, future: Future):
        """Callback when mining completes"""
        with self.mining_lock:
            self.is_mining = False
            self.current_mining_future = None
            
        try:
            result = future.result()
            if result:
                self.logger.info(f"✅ Asynchronous mining completed: Block #{result.get('index')}")
            else:
                self.logger.info("❌ Asynchronous mining completed: No block mined")
        except Exception as e:
            self.logger.error(f"💥 Asynchronous mining failed: {e}")

    def _mine_pending_transactions_sync(self, miner_address: str, difficulty: int = 4) -> Dict:
        """
        Synchronous mining function that runs in background thread
        This is your original mining logic but running in a separate thread
        """
        try:
            # Clean up any mined transactions first
            self.cleanup_mined_transactions()
            
            # Get available transactions of all types
            available_bills = self.get_available_bills_to_mine()
            available_rewards = self.get_available_rewards_to_mine()
            available_transfers = self.get_available_transfers_to_mine()
            
            # DEBUG: Log what's available
            self.logger.info(f"⛏️ [ASYNC] Mining - Bills: {len(available_bills)}, Rewards: {len(available_rewards)}, Transfers: {len(available_transfers)}")
            
            # CRITICAL FIX: Return None if no transactions available
            if not available_bills and not available_rewards and not available_transfers:
                self.logger.info("❌ [ASYNC] No transactions available to mine - stopping")
                return None
            
            # FIXED REWARD CREATION: Create reward BEFORE transaction selection
            next_block_height = len(self.blockchain)
            reward_created = False
            
            # Only create reward if there are bills to mine
            if available_bills:
                # FIX: Pass the actual bills list, not the count
                reward_created = self.create_and_add_reward_transaction(
                    miner_address, 
                    available_bills[:20],  # Pass the actual transactions list, not the count
                    next_block_height
                )
                if reward_created:
                    # Refresh available rewards list
                    available_rewards = self.get_available_rewards_to_mine()
                    self.logger.info(f"💰 [ASYNC] Created reward transaction for {miner_address}")
            
            # ENHANCED TRANSACTION SELECTION: Prioritize transfers when backlog exists
            max_per_block = 50
            transactions_to_mine = []
            
            # CRITICAL FIX: Check if we have a transfer backlog and prioritize accordingly
            if len(available_transfers) >= 10:
                self.logger.info("🎯 [ASYNC] HIGH TRANSFER BACKLOG DETECTED - PRIORITIZING TRANSFERS")
                # Strategy: Clear transfer backlog first
                transfers_to_include = available_transfers[:30]  # Increased from 15 to 30
                bills_to_include = available_bills[:10]  # Reduced from 20 to 10
                rewards_to_include = available_rewards[:10]  # Reduced from 15 to 10
                
                # Combine with transfer priority
                all_candidates = transfers_to_include + bills_to_include + rewards_to_include
            else:
                # Normal allocation when no major backlog
                bills_to_include = available_bills[:20]  # Up to 20 bills per block
                rewards_to_include = available_rewards[:15]  # Up to 15 rewards per block  
                transfers_to_include = available_transfers[:15]  # Up to 15 transfers per block
                
                # Combine all transaction types
                all_candidates = bills_to_include + rewards_to_include + transfers_to_include
            
            # VALIDATE all candidate transactions before including
            valid_candidates = []
            validation_stats = {"bills": 0, "rewards": 0, "transfers": 0, "invalid": 0}
            
            for tx in all_candidates:
                # Use skip_mined_check=True since we're about to mine them
                if self.validate_transaction(tx, skip_mined_check=True):
                    valid_candidates.append(tx)
                    # Track validation stats
                    tx_type = tx.get("type", "unknown")
                    if tx_type in ["GTX_Genesis", "genesis"]:
                        validation_stats["bills"] += 1
                    elif tx_type == "reward":
                        validation_stats["rewards"] += 1
                    elif tx_type == "transfer":
                        validation_stats["transfers"] += 1
                else:
                    validation_stats["invalid"] += 1
                    self.logger.warning(f"❌ [ASYNC] Excluding invalid transaction from mining: {tx.get('type')} - {tx.get('hash', 'no-hash')[:16]}...")
            
            # Log validation results
            self.logger.info(f"📊 [ASYNC] Validation: {validation_stats['bills']} bills, {validation_stats['rewards']} rewards, {validation_stats['transfers']} transfers valid, {validation_stats['invalid']} invalid")
            
            # Sort by timestamp (oldest first) to prioritize stuck transactions
            valid_candidates.sort(key=lambda tx: tx.get('timestamp', 0))
            
            # Take the oldest transactions up to block limit
            transactions_to_mine = valid_candidates[:max_per_block]
            
            # DEBUG: Log the block composition
            bill_count_final = sum(1 for tx in transactions_to_mine if tx.get("type") in ["GTX_Genesis", "genesis"])
            reward_count_final = sum(1 for tx in transactions_to_mine if tx.get("type") == "reward")
            transfer_count_final = sum(1 for tx in transactions_to_mine if tx.get("type") == "transfer")
            
            self.logger.info(f"📦 [ASYNC] Block composition - Bills: {bill_count_final}, Rewards: {reward_count_final}, Transfers: {transfer_count_final}")
            self.logger.info(f"📊 [ASYNC] Transaction selection - Candidates: {len(all_candidates)}, Valid: {len(valid_candidates)}, Final: {len(transactions_to_mine)}")
            
            # CRITICAL CHECK: If we have transfers available but none made it into the block, debug why
            if available_transfers and transfer_count_final == 0:
                self.logger.warning("⚠️ [ASYNC] Transfers available but none selected for mining - investigating...")
                self.debug_transfer_mining()
            
            if not transactions_to_mine:
                self.logger.info("[ASYNC] No valid transactions to mine after validation")
                return None
            
            # Ensure we have a previous block
            if not self.blockchain:
                self.logger.error("[ASYNC] No blockchain available - cannot mine without genesis block")
                return None
                
            previous_block = self.blockchain[-1]
            if not previous_block or "hash" not in previous_block:
                self.logger.error("[ASYNC] Previous block is invalid - cannot mine")
                return None
            
            previous_hash = previous_block["hash"]
            
            # Create new block with proper structure
            new_block = {
                "index": len(self.blockchain),
                "timestamp": int(time.time()),  # Ensure integer timestamp
                "transactions": transactions_to_mine,
                "previous_hash": previous_hash,
                "nonce": 0,
                "miner": miner_address,
                "difficulty": difficulty,
                "hash": ""  # Will be calculated during mining
            }
            
            # Mine the block (proof of work)
            self.logger.info(f"🔨 [ASYNC] Starting proof-of-work for block #{new_block['index']}...")
            start_time = time.time()
            mined_block = self.mine_block(new_block, difficulty)
            mining_time = time.time() - start_time
            
            if not mined_block:
                self.logger.error("❌ [ASYNC] Failed to mine block - proof of work unsuccessful")
                return None
            
            # Add mining time to block for analytics
            mined_block["mining_time"] = mining_time
            
            # VALIDATE the mined block before adding to chain
            if not self.validate_block_structure(mined_block):
                self.logger.error("❌ [ASYNC] Mined block has invalid structure")
                return None
                
            if not self.validate_block_transactions(mined_block.get("transactions", [])):
                self.logger.error("❌ [ASYNC] Mined block contains invalid transactions")
                return None
            
            # Check chain continuity
            if self.blockchain and mined_block["previous_hash"] != previous_hash:
                self.logger.error("❌ [ASYNC] Block continuity broken - previous hash mismatch")
                return None
            
            # Add to blockchain
            self.blockchain.append(mined_block)
            
            # Remove mined transactions from mempool
            removed_count = self.remove_mined_transactions(transactions_to_mine)
            
            # Update mined indexes
            self.update_mined_indexes(mined_block)
            
            # Save changes with validation
            if not self.validate_chain(self.blockchain):
                self.logger.error("❌ [ASYNC] Blockchain validation failed after adding new block - rolling back")
                # Rollback the block
                self.blockchain.pop()
                # Restore mempool transactions
                self.mempool.extend(transactions_to_mine)
                self.save_mempool()
                return None
            
            # Save the validated blockchain
            self.save_blockchain()
            self.save_mempool()
            
            self.logger.info(f"✅ [ASYNC] Successfully mined block #{mined_block['index']} with {len(transactions_to_mine)} transactions")
            self.logger.info(f"   - Bills: {bill_count_final}, Rewards: {reward_count_final}, Transfers: {transfer_count_final}")
            self.logger.info(f"   - Hash: {mined_block['hash'][:16]}...")
            self.logger.info(f"   - Nonce: {mined_block['nonce']}, Difficulty: {mined_block['difficulty']}")
            self.logger.info(f"   - Mining Time: {mining_time:.2f}s")
            self.logger.info(f"   - Removed {removed_count} transactions from mempool")
            
            # Special logging if we cleared transfer backlog
            if transfer_count_final > 0 and len(available_transfers) >= 10:
                remaining_transfers = len(self.get_available_transfers_to_mine())
                self.logger.info(f"   💸 [ASYNC] Transfer Progress: Mined {transfer_count_final}, {remaining_transfers} remaining in mempool")
            
            return mined_block
            
        except Exception as e:
            self.logger.error(f"💥 [ASYNC] Critical error in mine_pending_transactions: {e}")
            import traceback
            self.logger.error(f"💥 [ASYNC] Traceback: {traceback.format_exc()}")
            return None

    def get_mining_status(self):
        """Get current mining status"""
        with self.mining_lock:
            return {
                "is_mining": self.is_mining,
                "has_future": self.current_mining_future is not None,
                "future_done": self.current_mining_future.done() if self.current_mining_future else False,
                "future_running": self.current_mining_future.running() if self.current_mining_future else False
            }

    def wait_for_mining_completion(self, timeout: float = None):
        """Wait for current mining operation to complete"""
        with self.mining_lock:
            if self.current_mining_future:
                return self.current_mining_future.result(timeout=timeout)
        return None

    def stop_mining(self):
        """Stop current mining operation (if possible)"""
        with self.mining_lock:
            if self.current_mining_future and not self.current_mining_future.done():
                self.current_mining_future.cancel()
                self.is_mining = False
                return True
        return False
    
    def enhanced_mempool_cleanup(self):
        """Comprehensive mempool cleanup that finds and removes all mined transactions"""
        initial_count = len(self.mempool)
        
        # Get ALL mined transaction hashes from entire blockchain
        all_mined_hashes = set()
        all_mined_serials = set()
        all_mined_signatures = set()  # For transfers
        
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                # Track by hash
                if tx.get("hash"):
                    all_mined_hashes.add(tx["hash"])
                
                # Track by serial number
                if tx.get("serial_number"):
                    all_mined_serials.add(tx["serial_number"])
                
                # Track by signature (for transfers)
                if tx.get("signature"):
                    all_mined_signatures.add(tx["signature"])
        
        # Remove any transaction from mempool that appears in blockchain
        new_mempool = []
        for tx in self.mempool:
            should_remove = False
            
            # Check by hash
            if tx.get("hash") and tx["hash"] in all_mined_hashes:
                should_remove = True
                self.logger.info(f"🧹 Removing by hash: {tx['hash'][:16]}...")
            
            # Check by serial number
            elif tx.get("serial_number") and tx["serial_number"] in all_mined_serials:
                should_remove = True
                self.logger.info(f"🧹 Removing by serial: {tx['serial_number']}")
            
            # Check by signature (transfers)
            elif tx.get("signature") and tx["signature"] in all_mined_signatures:
                should_remove = True
                self.logger.info(f"🧹 Removing by signature: {tx['signature'][:20]}...")
            
            # Deep check: calculate hash and compare
            else:
                tx_string = json.dumps(tx, sort_keys=True)
                calculated_hash = hashlib.sha256(tx_string.encode()).hexdigest()
                if calculated_hash in all_mined_hashes:
                    should_remove = True
                    self.logger.info(f"🧹 Removing by calculated hash: {calculated_hash[:16]}...")
            
            if not should_remove:
                new_mempool.append(tx)
        
        removed_count = initial_count - len(new_mempool)
        if removed_count > 0:
            self.mempool = new_mempool
            self.save_mempool()
            self.logger.info(f"🚀 Enhanced cleanup removed {removed_count} mined transactions")
        
        return removed_count
    def force_mine_transfers(self, miner_address: str = "transfer_miner"):
        """Force mine a block containing only transfers to clear the backlog"""
        available_transfers = self.get_available_transfers_to_mine()
        
        if not available_transfers:
            self.logger.info("No transfers to mine")
            return None
        
        self.logger.info(f"🔄 Force mining {len(available_transfers)} transfers")
        
        # Create a transfer-only block
        previous_block = self.blockchain[-1]
        transfer_block = {
            "index": len(self.blockchain),
            "timestamp": time.time(),
            "transactions": available_transfers[:20],  # Limit transfers per block
            "previous_hash": previous_block["hash"],
            "nonce": 0,
            "miner": miner_address,
            "difficulty": 2,  # Lower difficulty for faster mining
            "hash": ""
        }
        
        # Mine the block
        mined_block = self.mine_block(transfer_block, 2)
        if mined_block:
            self.blockchain.append(mined_block)
            self.remove_mined_transactions(available_transfers[:20])
            self.update_mined_indexes(mined_block)
            self.save_blockchain()
            self.save_mempool()
            self.logger.info(f"✅ Force-mined transfer block #{mined_block['index']}")
            return mined_block
        
        return None
    def add_genesis_transaction(self, serial_number: str, denomination: float, issued_to: str) -> bool:
        """Add a genesis transaction to mempool (simplified method)"""
        try:
            # Create genesis transaction
            genesis_tx = {
                "type": "GTX_Genesis",
                "serial_number": serial_number,
                "denomination": denomination,
                "issued_to": issued_to,
                "timestamp": time.time(),
                "hash": ""  # Will be calculated
            }
            
            # Calculate hash
            tx_string = json.dumps(genesis_tx, sort_keys=True)
            genesis_tx["hash"] = hashlib.sha256(tx_string.encode()).hexdigest()
            
            self.logger.info(f"🔍 Adding genesis transaction:")
            self.logger.info(f"   Serial: {serial_number}")
            self.logger.info(f"   Denomination: {denomination}")
            self.logger.info(f"   Issued to: {issued_to}")
            self.logger.info(f"   Hash: {genesis_tx['hash']}")
            
            # Use the main add_transaction method
            return self.add_transaction(genesis_tx)
            
        except Exception as e:
            self.logger.error(f"Error adding genesis transaction: {e}")
            return False
    def validate_transaction(self, transaction: Dict, skip_mined_check: bool = False) -> bool:
        """Validate transaction structure - FIXED VERSION"""
        if not isinstance(transaction, dict):
            self.logger.error("Transaction is not a dictionary")
            return False
        
        tx_type = transaction.get("type")
        
        if tx_type in ["GTX_Genesis", "genesis"]:
            required = ["serial_number", "denomination", "issued_to", "timestamp"]
            if not all(key in transaction for key in required):
                missing = [key for key in required if key not in transaction]
                self.logger.error(f"Genesis transaction missing required fields: {missing}")
                self.logger.error(f"Transaction: {transaction}")
                return False
            
            # ONLY check for duplicates when adding to mempool, not when validating blocks
            if not skip_mined_check:
                serial = transaction.get("serial_number")
                if serial and serial in self.mined_serials:
                    self.logger.error(f"Serial number {serial} already mined")
                    return False
                    
                # Check if already in mempool (only when adding new transactions)
                for tx in self.mempool:
                    if tx.get("serial_number") == serial:
                        self.logger.warning(f"Genesis transaction with serial {serial} already in mempool")
                        return False
                        
            return True
        
        elif tx_type == "transfer":
            required = ["from", "to", "amount", "timestamp", "signature", "hash"]  # Add "signature"
            if not all(key in transaction for key in required):
                missing = [key for key in required if key not in transaction]
                self.logger.error(f"Transfer transaction missing required fields: {missing}")
                return False
            
            # Validate amount
            amount = transaction.get("amount")
            if not isinstance(amount, (int, float)) or amount <= 0:
                self.logger.error(f"Invalid amount in transfer transaction: {amount}")
                return False
            
            # Validate signature format (basic check)
            signature = transaction.get("signature")
            if not signature or not isinstance(signature, str) or len(signature) < 10:
                self.logger.error(f"Invalid signature in transfer transaction: {signature}")
                return False
        
        elif tx_type == "reward":
            # Validate reward transaction
            required = ["to", "amount", "timestamp", "block_height", "hash"]
            if not all(key in transaction for key in required):
                missing = [key for key in required if key not in transaction]
                self.logger.error(f"Reward transaction missing required fields: {missing}")
                return False
            
            # Validate amount
            amount = transaction.get("amount")
            if not isinstance(amount, (int, float)) or amount <= 0:
                self.logger.error(f"Invalid amount in reward transaction: {amount}")
                return False
            
            # Validate block_height
            block_height = transaction.get("block_height")
            if not isinstance(block_height, int) or block_height <= 0:
                self.logger.error(f"Invalid block height in reward transaction: {block_height}")
                return False
        
        elif tx_type == "transaction":
            required = ["timestamp", "hash"]
            if not all(key in transaction for key in required):
                self.logger.error("Transaction missing required fields")
                return False
        
        else:
            self.logger.error(f"Unknown transaction type: {tx_type}")
            return False
        
        return True



    def process_transfer_transaction(self, transaction: Dict) -> bool:
        """Process wallet-to-wallet transfer transaction"""
        try:
            # Check if sender has sufficient balance
            sender_balance = self.get_address_balance(transaction["from"])
            amount = transaction["amount"]
            
            if sender_balance < amount:
                self.logger.error(f"Insufficient balance: {transaction['from']} has {sender_balance}, needs {amount}")
                return False
            
            # Check if transaction is already in blockchain (double spend)
            if self.is_transaction_mined(transaction):
                self.logger.error("Transaction already mined (double spend attempt)")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing transfer transaction: {e}")
            return False

    def get_address_balance(self, address: str) -> float:
        """Calculate balance for an address by scanning the blockchain"""
        balance = 0.0
        
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                tx_type = tx.get("type")
                
                if tx_type == "transfer":
                    # Subtract if sender
                    if tx.get("from") == address:
                        balance -= tx.get("amount", 0)
                    # Add if receiver
                    if tx.get("to") == address:
                        balance += tx.get("amount", 0)
                
                elif tx_type == "reward":
                    # Add if reward receiver
                    if tx.get("to") == address:
                        balance += tx.get("amount", 0)
        
        return balance

    def rebuild_blockchain_from_scratch(self):
        """Completely rebuild blockchain from scratch preserving mined transactions"""
        self.logger.info("🔄 COMPLETE BLOCKCHAIN REBUILD")
        
        # Create backup
        import shutil
        backup_name = f"blockchain.json.corrupted.{int(time.time())}"
        if os.path.exists(self.blockchain_file):
            shutil.copy2(self.blockchain_file, backup_name)
            self.logger.info(f"📁 Created backup: {backup_name}")
        
        # Extract all valid transactions from the corrupted blockchain
        all_valid_transactions = []
        mined_serials = set()
        
        if self.blockchain:
            for block in self.blockchain:
                for tx in block.get("transactions", []):
                    if self.validate_transaction(tx, skip_mined_check=True):
                        all_valid_transactions.append(tx)
                        if tx.get("serial_number"):
                            mined_serials.add(tx["serial_number"])
        
        self.logger.info(f"Recovered {len(all_valid_transactions)} valid transactions")
        self.logger.info(f"Recovered {len(mined_serials)} mined serials")
        
        # Create fresh blockchain
        self.blockchain = []
        self.mined_serials = mined_serials
        
        # Mine all recovered transactions in new blocks
        transactions_per_block = 10
        for i in range(0, len(all_valid_transactions), transactions_per_block):
            block_transactions = all_valid_transactions[i:i + transactions_per_block]
            
            previous_block = self.blockchain[-1]
            new_block = {
                "index": len(self.blockchain),
                "timestamp": int(time.time()),
                "transactions": block_transactions,
                "previous_hash": previous_block["hash"],
                "nonce": 0,
                "miner": "system_rebuild",
                "difficulty": 2,  # Low difficulty for fast mining
                "hash": ""
            }
            
            # Mine the block
            mined_block = self.mine_block(new_block, 2)
            if mined_block:
                self.blockchain.append(mined_block)
                self.logger.info(f"Rebuilt block #{mined_block['index']} with {len(block_transactions)} transactions")
            else:
                self.logger.error(f"Failed to mine block during rebuild")
        
        self.save_blockchain()
        self.logger.info("✅ Blockchain completely rebuilt")
        return True
    def mine_block(self, block: Dict, difficulty: int) -> Dict:
        """Perform proof of work mining - FIXED VERSION"""
        target = "0" * difficulty
        start_time = time.time()
        
        # Ensure timestamp is integer for consistency
        if isinstance(block["timestamp"], float):
            block["timestamp"] = int(block["timestamp"])
        
        # Reset nonce and hash
        block["nonce"] = 0
        block["hash"] = ""
        
        # Store the original difficulty in the block
        block["difficulty"] = difficulty
        
        self.logger.info(f"⛏️ Mining block #{block['index']} with difficulty {difficulty} (target: {target})")
        
        while True:
            # Calculate hash with current nonce
            calculated_hash = self.calculate_block_hash(
                block["index"],
                block["previous_hash"],
                block["timestamp"],
                block["transactions"],
                block["nonce"]
            )
            
            # Check if hash meets difficulty target
            if calculated_hash.startswith(target):
                block["hash"] = calculated_hash
                break
            
            block["nonce"] += 1
            
            # Safety check to prevent infinite loop
            if block["nonce"] % 10000 == 0:
                self.logger.debug(f"Tried {block['nonce']} nonces...")
            
            if block["nonce"] > 1000000:
                self.logger.error(f"Mining timeout after {block['nonce']} attempts")
                return None
        
        mining_time = time.time() - start_time
        block["mining_time"] = mining_time
        self.logger.info(f"✅ Block #{block['index']} mined in {mining_time:.2f}s with nonce {block['nonce']}")
        self.logger.info(f"   Final hash: {block['hash']}")
        
        return block
    def get_genesis_transactions(self):
        """Get all genesis transactions from the blockchain"""
        genesis_transactions = []
        
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("type") in ["genesis", "GTX_Genesis"]:
                    genesis_transactions.append(tx)
        
        return genesis_transactions

    def get_blockchain_stats(self):
        """Get comprehensive blockchain statistics"""
        total_blocks = len(self.blockchain)
        total_transactions = 0
        genesis_count = 0
        gtx_genesis_count = 0
        transfer_count = 0
        reward_count = 0
        
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                total_transactions += 1
                tx_type = tx.get("type", "")
                if tx_type == "genesis":
                    genesis_count += 1
                elif tx_type == "GTX_Genesis":
                    gtx_genesis_count += 1
                elif tx_type == "transfer":
                    transfer_count += 1
                elif tx_type == "reward":
                    reward_count += 1
        
        return {
            "total_blocks": total_blocks,
            "total_transactions": total_transactions,
            "genesis_transactions": genesis_count,
            "gtx_genesis_transactions": gtx_genesis_count,
            "transfer_transactions": transfer_count,
            "reward_transactions": reward_count,
            "mempool_size": len(self.mempool),
            "mined_serials": len(self.mined_serials)
        }
    

    def is_reward_already_exists(self, block_height: int, miner_address: str) -> bool:
        """Check if a reward transaction for this block height and miner already exists"""
        # Check mempool
        for tx in self.mempool:
            if (tx.get("type") == "reward" and 
                tx.get("block_height") == block_height and 
                tx.get("to") == miner_address):
                return True
        
        # Check blockchain
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if (tx.get("type") == "reward" and 
                    tx.get("block_height") == block_height and 
                    tx.get("to") == miner_address):
                    return True
        
        return False

    def get_available_rewards_to_mine(self):
        """Get reward transactions from mempool that haven't been mined yet"""
        available_rewards = []
        
        for tx in self.mempool:
            if tx.get("type") == "reward" and not self.is_transaction_mined(tx):
                available_rewards.append(tx)
        
        return available_rewards
    # Add this method to your BlockchainDaemon class
    def repair_blockchain(self):
        """Repair blockchain by recalculating all hashes"""
        self.logger.info("🛠️ Repairing blockchain...")
        
        # Start with genesis block
        if self.blockchain:
            # Recalculate hash for genesis block
            genesis = self.blockchain[0]
            genesis["hash"] = self.calculate_block_hash(
                genesis["index"],
                genesis["previous_hash"], 
                genesis["timestamp"],
                genesis["transactions"],
                genesis["nonce"]
            )
            
            # Recalculate hashes for all subsequent blocks
            for i in range(1, len(self.blockchain)):
                block = self.blockchain[i]
                previous_block = self.blockchain[i-1]
                
                # Update previous_hash to match fixed previous block
                block["previous_hash"] = previous_block["hash"]
                
                # Recalculate current block hash
                block["hash"] = self.calculate_block_hash(
                    block["index"],
                    block["previous_hash"],
                    block["timestamp"], 
                    block["transactions"],
                    block["nonce"]
                )
        
        self.save_blockchain()
        self.logger.info("✅ Blockchain repaired")

    def get_step_by_step_mining_status(self) -> Dict:
        """Get current status for step-by-step mining"""
        available_bills = self.get_available_bills_to_mine()
        available_rewards = self.get_available_rewards_to_mine()
        available_transfers = self.get_available_transfers_to_mine()
        
        total_available = len(available_bills) + len(available_rewards) + len(available_transfers)
        
        return {
            "available_transactions": {
                "bills": len(available_bills),
                "rewards": len(available_rewards),
                "transfers": len(available_transfers),
                "total": total_available
            },
            "blockchain_status": {
                "blocks": len(self.blockchain),
                "needs_genesis": self.needs_initial_genesis() if hasattr(self, 'needs_initial_genesis') else True
            },
            "mempool_status": {
                "total": len(self.mempool),
                "should_cleanup": total_available < len(self.mempool)  # Indicates stuck transactions
            }
        }
    def create_reward_transaction(self, miner_address: str, bill_count: int):
        """Create mining reward transaction"""
        base_reward = 1
        total_reward = base_reward * bill_count
        
        return {
            "type": "reward",
            "to": miner_address,
            "amount": total_reward,
            "timestamp": time.time(),
            "block_height": len(self.blockchain) + 1,
            "description": f"Mining reward for {bill_count} bills"
        }
    def force_mempool_cleanup(self):
        """Force a comprehensive mempool cleanup"""
        self.logger.info("🔄 FORCING COMPREHENSIVE MEMPOOL CLEANUP")
        
        # Run all cleanup methods
        result1 = self.cleanup_mined_transactions()
        result2 = self.cleanup_mined_transactions_enhanced()
        result3 = self.enhanced_mempool_cleanup()
        
        # Final check using the hash mismatch debug
        stuck_txs = self.debug_hash_mismatch()
        
        self.logger.info(f"📊 Cleanup results:")
        self.logger.info(f"   - Basic cleanup: {result1} removed")
        self.logger.info(f"   - Enhanced cleanup: {result2} removed") 
        self.logger.info(f"   - Comprehensive cleanup: {result3} removed")
        self.logger.info(f"   - Still stuck: {len(stuck_txs)} transactions")
        
        return {
            "basic_removed": result1,
            "enhanced_removed": result2,
            "comprehensive_removed": result3,
            "still_stuck": len(stuck_txs)
        }
    def immediate_mempool_cleanup(self, mined_block: Dict):
        """Immediately clean mempool after successful mining"""
        if not mined_block:
            return
        
        mined_transactions = mined_block.get("transactions", [])
        self.logger.info(f"🔍 Immediate cleanup for block #{mined_block['index']} with {len(mined_transactions)} transactions")
        
        # Use the enhanced removal
        removed = self.remove_mined_transactions(mined_transactions)
        
        # Also run the enhanced cleanup to catch any others
        additional_removed = self.cleanup_mined_transactions_enhanced()
        
        self.logger.info(f"🧹 Cleanup results: {removed} immediate + {additional_removed} additional = {removed + additional_removed} total")
    def remove_mined_transactions(self, mined_transactions: List[Dict]) -> int:
        """Enhanced version that handles hash variations and recalculations"""
        initial_count = len(self.mempool)
        
        if not mined_transactions:
            return 0
        
        # Strategy 1: Remove by exact hash match
        mined_hashes = {tx.get("hash") for tx in mined_transactions if tx.get("hash")}
        self.mempool = [tx for tx in self.mempool if tx.get("hash") not in mined_hashes]
        
        removed_count = initial_count - len(self.mempool)
        
        # Strategy 2: Remove by recalculated hash (in case hashes changed)
        if removed_count < len(mined_transactions):
            self.logger.info("🔍 Some transactions not removed by hash, trying recalculated hashes...")
            
            # Get all hashes from blockchain for comparison
            blockchain_hashes = set()
            for block in self.blockchain:
                for tx in block.get("transactions", []):
                    if tx.get("hash"):
                        blockchain_hashes.add(tx["hash"])
            
            # Remove any mempool transaction that exists in blockchain
            new_mempool = []
            for tx in self.mempool:
                tx_hash = tx.get("hash")
                if not tx_hash or tx_hash not in blockchain_hashes:
                    new_mempool.append(tx)
                else:
                    self.logger.info(f"🧹 Removing by blockchain match: {tx_hash[:16]}...")
            
            additional_removed = len(self.mempool) - len(new_mempool)
            self.mempool = new_mempool
            removed_count += additional_removed
        
        # Strategy 3: For genesis transactions, also check by serial number
        if removed_count < len(mined_transactions):
            self.logger.info("🔍 Checking by serial number for genesis transactions...")
            
            # Get all mined serials from blockchain
            mined_serials = set()
            for block in self.blockchain:
                for tx in block.get("transactions", []):
                    if tx.get("serial_number"):
                        mined_serials.add(tx["serial_number"])
            
            new_mempool = []
            for tx in self.mempool:
                tx_serial = tx.get("serial_number")
                if not tx_serial or tx_serial not in mined_serials:
                    new_mempool.append(tx)
                else:
                    self.logger.info(f"🧹 Removing by serial: {tx_serial}")
            
            additional_removed = len(self.mempool) - len(new_mempool)
            self.mempool = new_mempool
            removed_count += additional_removed
        
        if removed_count > 0:
            self.logger.info(f"✅ Removed {removed_count} mined transactions from mempool")
            self.save_mempool()
        
        return removed_count
    
    def update_mined_indexes(self, block: Dict):
        """Update mined indexes with transactions from new block"""
        for tx in block.get("transactions", []):
            if tx.get("serial_number"):
                self.mined_serials.add(tx["serial_number"])
    
    def ensure_backup_dir(self, dir_path: str):
        """Ensure that a backup directory exists"""
        os.makedirs(dir_path, exist_ok=True)

    def backup_file(self, src_file: str, backup_dir: str):
        """Make a timestamped backup of a file"""
        try:
            if os.path.exists(src_file):
                self.ensure_backup_dir(backup_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.basename(src_file)
                backup_name = f"{base_name}.{timestamp}.bak"
                backup_path = os.path.join(backup_dir, backup_name)
                shutil.copy2(src_file, backup_path)
                self.logger.info(f"📁 Backup created: {backup_path}")
        except Exception as e:
            self.logger.error(f"⚠️ Failed to create backup for {src_file}: {e}")

    def save_blockchain(self):
        """Save blockchain to file + backup"""
        try:
            if not self.validate_chain(self.blockchain):
                self.logger.error("Refusing to save invalid blockchain!")
                return

            with open(self.blockchain_file, 'w', encoding='utf-8') as f:
                json.dump(self.blockchain, f, indent=2)
            self.logger.info("✅ Blockchain saved successfully")

            # Backup
            self.backup_file(self.blockchain_file, "./back-up/blockchain")

        except Exception as e:
            self.logger.error(f"Error saving blockchain: {e}")

    def save_mempool(self):
        """Save mempool to file + backup"""
        import tempfile

        try:
            if not isinstance(self.mempool, list):
                self.logger.error("Mempool is not a list, resetting to []")
                self.mempool = []

            temp_dir = os.path.dirname(self.mempool_file) or '.'
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=temp_dir, delete=False, suffix='.tmp') as f:
                json.dump(self.mempool, f, indent=2)
                temp_filename = f.name

            if os.path.exists(self.mempool_file):
                os.remove(self.mempool_file)
            shutil.move(temp_filename, self.mempool_file)

            self.logger.info(f"✅ Mempool saved successfully ({len(self.mempool)} tx)")

            # Backup
            self.backup_file(self.mempool_file, "./back-up/mempool")

        except Exception as e:
            self.logger.error(f"Unexpected error saving mempool: {e}")
    def scan_and_add_banknotes_from_db(self, db_connection=None, db_config=None):
        """
        Scan database for banknotes and add them to mempool as genesis transactions.
        Each banknote is only added once based on serial number.
        
        Args:
            db_connection: Optional existing database connection
            db_config: Database configuration if new connection needed
            
        Returns:
            dict: Statistics about added banknotes
        """
        try:
            self.logger.info("🔍 Scanning database for banknotes to add to mempool...")
            
            # Get all banknotes from database
            banknotes = self._fetch_banknotes_from_db(db_connection, db_config)
            
            if not banknotes:
                self.logger.info("📭 No banknotes found in database")
                return {"added": 0, "skipped": 0, "errors": 0, "total": 0}
            
            self.logger.info(f"📄 Found {len(banknotes)} banknotes in database")
            
            # Track results
            results = {
                "added": 0,
                "skipped": 0, 
                "errors": 0,
                "total": len(banknotes)
            }
            
            # Get already mined serials for quick lookup
            mined_serials = self.mined_serials.copy()
            
            # Get serials already in mempool
            mempool_serials = set()
            for tx in self.mempool:
                if tx.get("type") in ["GTX_Genesis", "genesis"] and tx.get("serial_number"):
                    mempool_serials.add(tx["serial_number"])
            
            # Process each banknote
            for banknote in banknotes:
                try:
                    serial_number = banknote.get('serial_number')
                    denomination = banknote.get('denomination')
                    issued_to = banknote.get('owner', 'Unknown')  # Adjust field name as needed
                    
                    if not serial_number or not denomination:
                        self.logger.warning(f"⚠️ Skipping banknote with missing data: {banknote}")
                        results["errors"] += 1
                        continue
                    
                    # Check if already mined
                    if serial_number in mined_serials:
                        self.logger.debug(f"⏭️ Skipping already mined banknote: {serial_number}")
                        results["skipped"] += 1
                        continue
                    
                    # Check if already in mempool
                    if serial_number in mempool_serials:
                        self.logger.debug(f"⏭️ Skipping banknote already in mempool: {serial_number}")
                        results["skipped"] += 1
                        continue
                    
                    # Create genesis transaction
                    genesis_tx = {
                        "type": "GTX_Genesis",
                        "serial_number": serial_number,
                        "denomination": float(denomination),
                        "issued_to": issued_to,
                        "timestamp": int(time.time()),
                        "description": f"Banknote {serial_number} - ${denomination}",
                        "source": "database_import",
                        "hash": ""  # Will be calculated
                    }
                    
                    # Calculate hash
                    tx_string = json.dumps(genesis_tx, sort_keys=True)
                    genesis_tx["hash"] = hashlib.sha256(tx_string.encode()).hexdigest()
                    
                    # Add to mempool (bypassing some checks since we already validated)
                    self.mempool.append(genesis_tx)
                    mempool_serials.add(serial_number)  # Track in current session
                    results["added"] += 1
                    
                    self.logger.info(f"✅ Added banknote to mempool: {serial_number} - ${denomination}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing banknote {banknote}: {e}")
                    results["errors"] += 1
                    continue
            
            # Save mempool if any transactions were added
            if results["added"] > 0:
                self.save_mempool()
                self.logger.info(f"💾 Saved mempool with {len(self.mempool)} total transactions")
            
            # Log summary
            self.logger.info(f"📊 Banknote import complete: {results['added']} added, "
                            f"{results['skipped']} skipped, {results['errors']} errors")
            
            return results
            
        except Exception as e:
            self.logger.error(f"💥 Error in scan_and_add_banknotes_from_db: {e}")
            import traceback
            self.logger.error(f"💥 Traceback: {traceback.format_exc()}")
            return {"added": 0, "skipped": 0, "errors": 0, "total": 0, "error": str(e)}
        
    def cleanup_mined_transactions_enhanced(self):
        """Enhanced mempool cleanup that properly removes mined transactions"""
        initial_count = len(self.mempool)
        
        # Get all mined transaction hashes from blockchain
        mined_hashes = set()
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("hash"):
                    mined_hashes.add(tx["hash"])
        
        # Also check by serial number for genesis transactions
        mined_serials = set()
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("serial_number"):
                    mined_serials.add(tx["serial_number"])
        
        # Remove transactions from mempool that are already mined
        new_mempool = []
        for tx in self.mempool:
            tx_hash = tx.get("hash")
            tx_serial = tx.get("serial_number")
            
            # Skip if transaction hash is mined
            if tx_hash and tx_hash in mined_hashes:
                self.logger.info(f"🧹 Removing mined transaction: {tx_hash[:16]}...")
                continue
                
            # Skip if genesis transaction serial is mined
            if tx_serial and tx_serial in mined_serials:
                self.logger.info(f"🧹 Removing mined genesis: {tx_serial}")
                continue
                
            new_mempool.append(tx)
    
        removed_count = initial_count - len(new_mempool)
        if removed_count > 0:
            self.mempool = new_mempool
            self.save_mempool()
            self.logger.info(f"✅ Enhanced cleanup removed {removed_count} mined transactions")
        
        return removed_count
    def comprehensive_diagnostic(self):
        """Run comprehensive diagnostics on the blockchain"""
        print("=== COMPREHENSIVE BLOCKCHAIN DIAGNOSTIC ===")
        
        # Basic blockchain info
        print(f"Total blocks: {len(self.blockchain)}")
        print(f"Mempool size: {len(self.mempool)}")
        print(f"Mined serials: {len(self.mined_serials)}")
        
        # Block-by-block analysis
        print("\n=== BLOCK ANALYSIS ===")
        for i, block in enumerate(self.blockchain):
            transactions = block.get('transactions', [])
            tx_types = {}
            for tx in transactions:
                tx_type = tx.get('type', 'unknown')
                tx_types[tx_type] = tx_types.get(tx_type, 0) + 1
            
            print(f"Block #{i}: {len(transactions)} transactions - {tx_types}")
            
            # Validate block hash
            calculated_hash = self.calculate_block_hash(
                block['index'],
                block['previous_hash'],
                block['timestamp'],
                block['transactions'],
                block['nonce'],
                block.get('difficulty')
            )
            
            if block['hash'] != calculated_hash:
                print(f"  ❌ BLOCK HASH INVALID!")
                print(f"     Expected: {calculated_hash}")
                print(f"     Actual:   {block['hash']}")
            else:
                print(f"  ✅ Block hash valid")
        
        # Validate chain continuity
        print("\n=== CHAIN CONTINUITY ===")
        valid_chain = True
        for i in range(1, len(self.blockchain)):
            current_block = self.blockchain[i]
            previous_block = self.blockchain[i-1]
            
            if current_block['previous_hash'] != previous_block['hash']:
                print(f"❌ Chain broken between block {i-1} and {i}")
                valid_chain = False
            else:
                print(f"✅ Block {i} correctly links to block {i-1}")
        
        # Transaction analysis
        print("\n=== TRANSACTION ANALYSIS ===")
        all_transactions = []
        for block in self.blockchain:
            all_transactions.extend(block.get('transactions', []))
        
        tx_by_type = {}
        for tx in all_transactions:
            tx_type = tx.get('type', 'unknown')
            tx_by_type[tx_type] = tx_by_type.get(tx_type, 0) + 1
        
        print("Transaction types in blockchain:")
        for tx_type, count in tx_by_type.items():
            print(f"  {tx_type}: {count}")
        
        # Check for duplicates
        print("\n=== DUPLICATE CHECK ===")
        all_hashes = [tx.get('hash') for tx in all_transactions if tx.get('hash')]
        unique_hashes = set(all_hashes)
        
        if len(all_hashes) != len(unique_hashes):
            print(f"❌ Found {len(all_hashes) - len(unique_hashes)} duplicate transactions!")
        else:
            print("✅ No duplicate transactions found")
        
        # Mempool analysis
        print("\n=== MEMPOOL ANALYSIS ===")
        mempool_by_type = {}
        for tx in self.mempool:
            tx_type = tx.get('type', 'unknown')
            mempool_by_type[tx_type] = mempool_by_type.get(tx_type, 0) + 1
        
        print("Mempool transactions by type:")
        for tx_type, count in mempool_by_type.items():
            print(f"  {tx_type}: {count}")
        
        # Final validation
        print("\n=== FINAL VALIDATION ===")
        if self.validate_chain(self.blockchain):
            print("✅ BLOCKCHAIN IS VALID")
        else:
            print("❌ BLOCKCHAIN VALIDATION FAILED")
        
        return valid_chain

    def _fetch_banknotes_from_db(self, db_connection=None, db_config=None):
        """
        Fetch banknotes from database. You'll need to customize this for your DB schema.
        
        Returns:
            List of banknote dictionaries with serial_number, denomination, and owner
        """
        try:
            # Example implementation - ADJUST THIS FOR YOUR DATABASE SCHEMA
            
            # Option 1: If using SQLite
            if db_connection:
                cursor = db_connection.cursor()
                cursor.execute("SELECT serial_number, denomination, owner FROM banknotes WHERE status = 'active'")
                banknotes = [
                    {
                        'serial_number': row[0],
                        'denomination': row[1], 
                        'owner': row[2]
                    }
                    for row in cursor.fetchall()
                ]
                return banknotes
            
            # Option 2: If using another database (MySQL, PostgreSQL, etc.)
            elif db_config:
                # Implement your database connection logic here
                # Example for MySQL:
                # import mysql.connector
                # conn = mysql.connector.connect(**db_config)
                # cursor = conn.cursor()
                # cursor.execute("SELECT serial_number, denomination, owner FROM banknotes")
                # ... process results
                pass
            
            # Option 3: Mock data for testing (remove in production)
            else:
                self.logger.warning("Using mock banknote data - implement proper database connection")
                return [
                    {'serial_number': 'SN001', 'denomination': 100.0, 'owner': 'Test User 1'},
                    {'serial_number': 'SN002', 'denomination': 50.0, 'owner': 'Test User 2'},
                    {'serial_number': 'SN003', 'denomination': 20.0, 'owner': 'Test User 3'},
                ]
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching banknotes from database: {e}")
            return []

    def add_single_banknote(self, serial_number: str, denomination: float, owner: str = "Unknown") -> bool:
        """
        Add a single banknote to mempool if it doesn't already exist.
        
        Args:
            serial_number: Unique serial number of the banknote
            denomination: Value of the banknote
            owner: Owner of the banknote
            
        Returns:
            bool: True if added successfully, False if already exists or error
        """
        try:
            # Check if already mined
            if serial_number in self.mined_serials:
                self.logger.warning(f"⚠️ Banknote {serial_number} already mined")
                return False
            
            # Check if already in mempool
            for tx in self.mempool:
                if tx.get("type") in ["GTX_Genesis", "genesis"] and tx.get("serial_number") == serial_number:
                    self.logger.warning(f"⚠️ Banknote {serial_number} already in mempool")
                    return False
            
            # Create and add genesis transaction
            genesis_tx = {
                "type": "GTX_Genesis",
                "serial_number": serial_number,
                "denomination": float(denomination),
                "issued_to": owner,
                "timestamp": int(time.time()),
                "description": f"Banknote {serial_number} - ${denomination}",
                "source": "manual_add",
                "hash": ""
            }
            
            # Calculate hash
            tx_string = json.dumps(genesis_tx, sort_keys=True)
            genesis_tx["hash"] = hashlib.sha256(tx_string.encode()).hexdigest()
            
            # Add to mempool
            success = self.add_transaction(genesis_tx)
            
            if success:
                self.logger.info(f"✅ Added banknote to mempool: {serial_number} - ${denomination}")
            else:
                self.logger.error(f"❌ Failed to add banknote: {serial_number}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"💥 Error adding single banknote: {e}")
            return False

    def get_banknote_status(self, serial_number: str) -> str:
        """
        Check the status of a banknote by serial number.
        
        Returns:
            str: "mined", "in_mempool", or "not_found"
        """
        # Check if mined
        if serial_number in self.mined_serials:
            return "mined"
        
        # Check if in mempool
        for tx in self.mempool:
            if tx.get("type") in ["GTX_Genesis", "genesis"] and tx.get("serial_number") == serial_number:
                return "in_mempool"
        
        return "not_found"
    def auto_mine(self, miner_address: str):
        """Auto-mine transactions when conditions are met - FIXED VERSION"""
        try:
            available_bills = self.get_available_bills_to_mine()
            available_rewards = self.get_available_rewards_to_mine()
            available_transfers = self.get_available_transfers_to_mine()
            
            total_available = len(available_bills) + len(available_rewards) + len(available_transfers)
            
            # Only mine if we have at least 1 transaction
            if total_available >= 1:
                self.logger.info(f"🤖 Auto-mining {total_available} available transactions")
                result = self.mine_pending_transactions(miner_address, None, async_mode=True)
                
                if result:
                    self.logger.info(f"✅ Auto-mined block #{result.get('index')}")
                else:
                    self.logger.info("❌ Auto-mining failed (no valid transactions)")
            else:
                self.logger.debug("🤖 No transactions available for auto-mining")
                    
        except Exception as e:
            self.logger.error(f"🤖 Auto-mining error: {e}")
    
    def create_initial_genesis_transactions(self) -> bool:
        """Create initial genesis transactions for an empty blockchain"""
        try:
            self.logger.info("🌱 Creating initial genesis transactions...")
            
            # Create some sample genesis transactions to bootstrap the system
            initial_transactions = [
                {
                    "type": "GTX_Genesis",
                    "serial_number": "SN-INITIAL-0001",
                    "denomination": 1000.0,
                    "issued_to": "System Bootstrap",
                    "timestamp": int(time.time()),
                    "description": "Initial system bootstrap transaction"
                },
                {
                    "type": "GTX_Genesis", 
                    "serial_number": "SN-INITIAL-0002",
                    "denomination": 10000.0,
                    "issued_to": "System Bootstrap",
                    "timestamp": int(time.time()),
                    "description": "Secondary bootstrap transaction"
                },
                {
                    "type": "GTX_Genesis",
                    "serial_number": "SN-INITIAL-0003", 
                    "denomination": 100000.0,
                    "issued_to": "System Bootstrap",
                    "timestamp": int(time.time()),
                    "description": "Tertiary bootstrap transaction"
                }
            ]
            
            added_count = 0
            for tx_data in initial_transactions:
                # Calculate hash for each transaction
                tx_string = json.dumps(tx_data, sort_keys=True)
                tx_data["hash"] = hashlib.sha256(tx_string.encode()).hexdigest()
                
                # Add to mempool
                if self.add_transaction(tx_data):
                    added_count += 1
                    self.logger.info(f"✅ Added initial genesis: ${tx_data['denomination']} - {tx_data['serial_number']}")
            
            self.logger.info(f"🌱 Created {added_count} initial genesis transactions")
            return added_count > 0
            
        except Exception as e:
            self.logger.error(f"❌ Error creating initial genesis transactions: {e}")
            return False
    def get_genesis_block_transactions(self):
        """Get transactions from the genesis block (block 0)"""
        if self.blockchain and len(self.blockchain) > 0:
            return self.blockchain[0].get("transactions", [])
        return []
    def get_transaction_status(self, tx_hash: str) -> str:
        """Get transaction status: pending, confirmed, or not found"""
        # Check if in mempool
        for tx in self.mempool:
            if tx.get("hash") == tx_hash:
                return "pending"
        
        # Check if in blockchain (confirmed)
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("hash") == tx_hash:
                    return "confirmed"
        
        return "not found"
    def debug_mining_selection(self):
        """Debug why transfers aren't being selected for mining"""
        self.logger.info("=== MINING SELECTION DEBUG ===")
        
        # Check what's available for mining
        available_bills = self.get_available_bills_to_mine()
        available_rewards = self.get_available_rewards_to_mine()
        available_transfers = self.get_available_transfers_to_mine()
        
        self.logger.info(f"Available to mine:")
        self.logger.info(f"  - Bills: {len(available_bills)}")
        self.logger.info(f"  - Rewards: {len(available_rewards)}")
        self.logger.info(f"  - Transfers: {len(available_transfers)}")
        
        # Check if transfers pass validation for mining
        valid_transfers = []
        invalid_transfers = []
        
        for tx in available_transfers:
            if self.validate_transfer_for_mining(tx):
                valid_transfers.append(tx)
            else:
                invalid_transfers.append(tx)
        
        self.logger.info(f"Transfer validation:")
        self.logger.info(f"  - Valid: {len(valid_transfers)}")
        self.logger.info(f"  - Invalid: {len(invalid_transfers)}")
        
        # Show why invalid transfers are rejected
        for tx in invalid_transfers:
            self.logger.info(f"  Invalid transfer: {tx.get('hash', 'no-hash')[:16]}...")
            # Check specific validation issues
            if not all(field in tx for field in ["from", "to", "amount", "timestamp", "signature", "hash"]):
                self.logger.info("    Missing required fields")
            if not isinstance(tx.get("amount"), (int, float)) or tx.get("amount", 0) <= 0:
                self.logger.info(f"    Invalid amount: {tx.get('amount')}")
        
        # Check recent blocks to see what's actually being mined
        self.logger.info("Recent blocks content:")
        for i, block in enumerate(self.blockchain[-5:]):  # Last 5 blocks
            block_index = block.get("index", i)
            transactions = block.get("transactions", [])
            bill_count = sum(1 for tx in transactions if tx.get("type") in ["GTX_Genesis", "genesis"])
            reward_count = sum(1 for tx in transactions if tx.get("type") == "reward")
            transfer_count = sum(1 for tx in transactions if tx.get("type") == "transfer")
            
            self.logger.info(f"  Block #{block_index}: {len(transactions)} tx "
                            f"(B:{bill_count} R:{reward_count} T:{transfer_count})")
    def diagnose_transfer_issue(self):
        """Debug why transfers are stuck"""
        pending_transfers = [tx for tx in self.mempool if tx.get("type") == "transfer"]
        
        self.logger.info("=== TRANSFER DIAGNOSIS ===")
        self.logger.info(f"Pending transfers in mempool: {len(pending_transfers)}")
        
        for tx in pending_transfers:
            tx_hash = tx.get("hash", "no-hash")
            is_mined = self.is_transaction_mined(tx)
            self.logger.info(f"Transfer {tx_hash[:16]}...: mined={is_mined}")
            
            if is_mined:
                self.logger.info("   ⚠️  TRANSFER IS MINED BUT STILL IN MEMPOOL!")
                # Find which block it's in
                for i, block in enumerate(self.blockchain):
                    for block_tx in block.get("transactions", []):
                        if block_tx.get("hash") == tx_hash:
                            self.logger.info(f"   Found in block #{i}")
        
        # Also check mempool cleanup
        total_mempool = len(self.mempool)
        transfer_mempool = len(pending_transfers)
        self.logger.info(f"Total mempool size: {total_mempool}")
        self.logger.info(f"Transfer mempool size: {transfer_mempool}")
    def emergency_repair(self):
        """Emergency repair for corrupted blockchain"""
        self.logger.info("🚨 EMERGENCY BLOCKCHAIN REPAIR")
        
        # Create backup of current file
        import shutil
        backup_name = f"blockchain.json.backup.{int(time.time())}"
        if os.path.exists(self.blockchain_file):
            shutil.copy2(self.blockchain_file, backup_name)
            self.logger.info(f"📁 Created backup: {backup_name}")
        
        # Try to repair the existing blockchain first
        if self.blockchain and len(self.blockchain) > 1:
            self.logger.info("Attempting to repair existing blockchain...")
            self.repair_blockchain()
            
            # Validate after repair
            if self.validate_chain(self.blockchain):
                self.logger.info("✅ Repair successful!")
                self.save_blockchain()
                return True
            else:
                self.logger.error("❌ Repair failed, creating new blockchain")
        
        # If repair fails, create fresh blockchain but preserve mined serials
        self.logger.info("Creating fresh blockchain...")
        
        # Save the mined serials from the old blockchain
        old_mined_serials = self.mined_serials.copy()
        
        # Create fresh blockchain
        self.blockchain = []
        self.save_blockchain()
        
        # Restore mined serials
        self.mined_serials = old_mined_serials
        
        self.logger.info("✅ Fresh blockchain created with preserved mined serials")
        return True
    def display_genesis_info(self):
        """Display information about the genesis block"""
        if not self.blockchain:
            self.logger.info("No blockchain data available")
            return
        
        genesis_block = self.blockchain[0]
        genesis_txs = genesis_block.get("transactions", [])
        
        self.logger.info(f"Genesis Block #{genesis_block['index']}")
        self.logger.info(f"Hash: {genesis_block['hash']}")
        self.logger.info(f"Timestamp: {datetime.fromtimestamp(genesis_block['timestamp'])}")
        self.logger.info(f"Transactions: {len(genesis_txs)}")
        
        for i, tx in enumerate(genesis_txs):
            self.logger.info(f"  TX {i}: {tx.get('type')} - {tx.get('message', 'No message')}")
    def cleanup_old_transactions(self):
        """Remove very old transactions from mempool"""
        current_time = time.time()
        max_age = 24 * 60 * 60  # 24 hours
        
        initial_count = len(self.mempool)
        self.mempool = [tx for tx in self.mempool 
                       if current_time - tx.get('timestamp', 0) < max_age]
        
        removed_count = initial_count - len(self.mempool)
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old transactions")
            self.save_mempool()
    def bootstrap_blockchain(self) -> bool:
        """Manually bootstrap the blockchain with genesis transactions"""
        self.logger.info("🚀 Manually bootstrapping blockchain...")
        
        # Ensure genesis block exists
        if not self.blockchain:
            genesis_block = self.create_genesis_block()
            self.blockchain.append(genesis_block)
            self.save_blockchain()
            self.logger.info("✅ Created genesis block")
        
        # Create initial genesis transactions
        success = self.create_initial_genesis_transactions()
        
        if success:
            self.logger.info("✅ Blockchain bootstrapped successfully")
            # Try to immediately mine the first block
            self.auto_mine("bootstrap_miner")
        else:
            self.logger.error("❌ Blockchain bootstrap failed")
        
        return success
    def debug_hash_mismatch(self):
        """Check if transaction hashes change between mempool and blocks"""
        self.logger.info("=== HASH MISMATCH DEBUG ===")
        
        # Get all transaction hashes from blockchain
        blockchain_hashes = set()
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("hash"):
                    blockchain_hashes.add(tx["hash"])
        
        # Check mempool transactions against blockchain
        stuck_transactions = []
        for tx in self.mempool:
            tx_hash = tx.get("hash")
            if tx_hash and tx_hash in blockchain_hashes:
                self.logger.info(f"✅ Found in blockchain: {tx_hash[:16]}...")
            else:
                stuck_transactions.append(tx)
                self.logger.info(f"❌ NOT in blockchain: {tx_hash[:16]}...")
        
        self.logger.info(f"Stuck transactions: {len(stuck_transactions)}")
        return stuck_transactions
    def step_by_step_mine_all_transactions(self, miner_address: str = "step_by_step_miner") -> Dict:
        """
        Step-by-step mining that processes ALL available transactions with detailed logging
        and comprehensive cleanup. This method ensures everything gets processed correctly.
        """
        try:
            self.logger.info("🚀 STEP-BY-STEP MINING: Starting comprehensive mining process")
            
            steps = []
            steps.append("Step 1: Initializing mining process")
            
            # Step 1: Clean up any already mined transactions first
            steps.append("Step 2: Cleaning up already mined transactions from mempool")
            initial_cleanup = self.cleanup_mined_transactions_enhanced()
            steps.append(f"   - Removed {initial_cleanup} already mined transactions")
            
            # Step 2: Get ALL available transactions
            steps.append("Step 3: Gathering all available transactions")
            available_bills = self.get_available_bills_to_mine()
            available_rewards = self.get_available_rewards_to_mine()
            available_transfers = self.get_available_transfers_to_mine()
            
            total_available = len(available_bills) + len(available_rewards) + len(available_transfers)
            steps.append(f"   - Bills: {len(available_bills)}")
            steps.append(f"   - Rewards: {len(available_rewards)}")
            steps.append(f"   - Transfers: {len(available_transfers)}")
            steps.append(f"   - TOTAL: {total_available} transactions available")
            
            if total_available == 0:
                steps.append("❌ No transactions available to mine")
                return {
                    "success": False,
                    "error": "No transactions available to mine",
                    "steps": steps
                }
            
            # Step 3: Create reward transaction if we have bills
            steps.append("Step 4: Creating reward transaction")
            reward_created = False
            if available_bills:
                reward_created = self.create_and_add_reward_transaction(
                    miner_address,
                    available_bills[:20],  # Use first 20 bills for reward calculation
                    len(self.blockchain)
                )
                if reward_created:
                    steps.append("   ✅ Reward transaction created and added to mempool")
                    # Refresh available rewards
                    available_rewards = self.get_available_rewards_to_mine()
                else:
                    steps.append("   ⚠️ Reward transaction creation failed (may already exist)")
            
            # Step 4: Combine ALL transactions
            steps.append("Step 5: Combining all transaction types")
            all_transactions = available_bills + available_rewards + available_transfers
            steps.append(f"   - Combined {len(all_transactions)} total transactions")
            
            # Step 5: Validate all transactions
            steps.append("Step 6: Validating all transactions")
            valid_transactions = []
            validation_stats = {"valid": 0, "invalid": 0}
            
            for tx in all_transactions:
                if self.validate_transaction(tx, skip_mined_check=True):
                    valid_transactions.append(tx)
                    validation_stats["valid"] += 1
                else:
                    validation_stats["invalid"] += 1
                    self.logger.warning(f"❌ Invalid transaction excluded: {tx.get('type')} - {tx.get('hash', 'no-hash')[:16]}...")
            
            steps.append(f"   - Valid: {validation_stats['valid']}, Invalid: {validation_stats['invalid']}")
            
            if not valid_transactions:
                steps.append("❌ No valid transactions to mine after validation")
                return {
                    "success": False,
                    "error": "No valid transactions after validation",
                    "steps": steps
                }
            
            # Step 6: Sort by timestamp (oldest first) to clear backlog
            steps.append("Step 7: Sorting transactions by age (oldest first)")
            valid_transactions.sort(key=lambda tx: tx.get('timestamp', 0))
            steps.append(f"   - Oldest transaction: {datetime.fromtimestamp(valid_transactions[0].get('timestamp', 0))}")
            steps.append(f"   - Newest transaction: {datetime.fromtimestamp(valid_transactions[-1].get('timestamp', 0))}")
            
            # Step 7: Create the block
            steps.append("Step 8: Creating block structure")
            if not self.blockchain:
                steps.append("❌ No blockchain available - cannot create block")
                return {
                    "success": False,
                    "error": "No blockchain available",
                    "steps": steps
                }
            
            previous_block = self.blockchain[-1]
            new_block = {
                "index": len(self.blockchain),
                "timestamp": int(time.time()),
                "transactions": valid_transactions,  # Include ALL valid transactions
                "previous_hash": previous_block["hash"],
                "nonce": 0,
                "miner": miner_address,
                "difficulty": 2,  # Low difficulty for faster processing
                "hash": ""
            }
            
            # Step 8: Mine the block
            steps.append("Step 9: Starting proof-of-work mining")
            start_time = time.time()
            
            target = "0" * new_block["difficulty"]
            steps.append(f"   - Difficulty: {new_block['difficulty']} (target: {target})")
            steps.append(f"   - Max transactions in block: {len(valid_transactions)}")
            
            # Mining loop with progress updates
            for nonce in range(10000000):  # Increased limit for larger blocks
                if nonce % 50000 == 0 and nonce > 0:
                    self.logger.info(f"   ⛏️  Trying nonce {nonce}...")
                
                new_block["nonce"] = nonce
                calculated_hash = self.calculate_block_hash(
                    new_block["index"],
                    new_block["previous_hash"],
                    new_block["timestamp"],
                    new_block["transactions"],
                    nonce
                )
                
                if calculated_hash.startswith(target):
                    new_block["hash"] = calculated_hash
                    mining_time = time.time() - start_time
                    steps.append(f"✅ Block mined successfully!")
                    steps.append(f"   - Nonce found: {nonce}")
                    steps.append(f"   - Mining time: {mining_time:.2f} seconds")
                    steps.append(f"   - Final hash: {calculated_hash[:20]}...")
                    break
            else:
                steps.append("❌ Failed to mine block - no valid nonce found")
                return {
                    "success": False,
                    "error": "Mining failed - no valid nonce found",
                    "steps": steps,
                    "attempts": 10000000
                }
            
            # Step 9: Add mining time to block
            new_block["mining_time"] = mining_time
            
            # Step 10: Validate the mined block
            steps.append("Step 10: Validating mined block")
            if not self.validate_block_structure(new_block):
                steps.append("❌ Mined block has invalid structure")
                return {
                    "success": False,
                    "error": "Mined block validation failed",
                    "steps": steps
                }
            
            if not self.validate_block_transactions(new_block["transactions"]):
                steps.append("❌ Mined block contains invalid transactions")
                return {
                    "success": False,
                    "error": "Block transactions validation failed",
                    "steps": steps
                }
            
            steps.append("✅ Block validation passed")
            
            # Step 11: Add to blockchain
            steps.append("Step 11: Adding block to blockchain")
            self.blockchain.append(new_block)
            steps.append(f"   - Blockchain now has {len(self.blockchain)} blocks")
            
            # Step 12: COMPREHENSIVE CLEANUP - Remove mined transactions
            steps.append("Step 12: Removing mined transactions from mempool")
            initial_mempool_size = len(self.mempool)
            
            # Use enhanced cleanup method
            removed_count = self.remove_mined_transactions(valid_transactions)
            
            # Run additional cleanup to catch any edge cases
            additional_removed = self.cleanup_mined_transactions_enhanced()
            
            total_removed = removed_count + additional_removed
            steps.append(f"   - Initial mempool: {initial_mempool_size} transactions")
            steps.append(f"   - Removed by direct cleanup: {removed_count}")
            steps.append(f"   - Removed by enhanced cleanup: {additional_removed}")
            steps.append(f"   - Total removed: {total_removed}")
            steps.append(f"   - Final mempool: {len(self.mempool)} transactions")
            
            # Step 13: Update mined indexes
            steps.append("Step 13: Updating mined indexes")
            self.update_mined_indexes(new_block)
            steps.append(f"   - Mined serials index now has {len(self.mined_serials)} entries")
            
            # Step 14: Validate blockchain integrity
            steps.append("Step 14: Validating blockchain integrity")
            if not self.validate_chain(self.blockchain):
                steps.append("❌ Blockchain validation failed after adding block - rolling back")
                # Rollback
                self.blockchain.pop()
                # Restore mempool
                self.mempool.extend(valid_transactions)
                self.save_mempool()
                return {
                    "success": False,
                    "error": "Blockchain validation failed - rollback performed",
                    "steps": steps
                }
            steps.append("✅ Blockchain validation passed")
            
            # Step 15: Save everything
            steps.append("Step 15: Saving blockchain and mempool")
            self.save_blockchain()
            self.save_mempool()
            steps.append("✅ All data saved successfully")
            
            # Step 16: Final statistics
            steps.append("Step 16: Generating final statistics")
            bill_count = sum(1 for tx in valid_transactions if tx.get("type") in ["GTX_Genesis", "genesis"])
            reward_count = sum(1 for tx in valid_transactions if tx.get("type") == "reward")
            transfer_count = sum(1 for tx in valid_transactions if tx.get("type") == "transfer")
            
            steps.append(f"📊 BLOCK COMPOSITION:")
            steps.append(f"   - Bills: {bill_count}")
            steps.append(f"   - Rewards: {reward_count}")
            steps.append(f"   - Transfers: {transfer_count}")
            steps.append(f"   - Total: {len(valid_transactions)} transactions")
            
            # Check if mempool is now empty
            remaining_bills = len(self.get_available_bills_to_mine())
            remaining_rewards = len(self.get_available_rewards_to_mine())
            remaining_transfers = len(self.get_available_transfers_to_mine())
            
            steps.append(f"📊 REMAINING IN MEMPOOL:")
            steps.append(f"   - Bills: {remaining_bills}")
            steps.append(f"   - Rewards: {remaining_rewards}")
            steps.append(f"   - Transfers: {remaining_transfers}")
            steps.append(f"   - Total: {remaining_bills + remaining_rewards + remaining_transfers}")
            
            self.logger.info(f"✅ STEP-BY-STEP MINING COMPLETED: Block #{new_block['index']} with {len(valid_transactions)} transactions")
            
            return {
                "success": True,
                "message": f"Successfully mined block #{new_block['index']} with {len(valid_transactions)} transactions",
                "block_index": new_block["index"],
                "block_hash": new_block["hash"],
                "transactions_mined": len(valid_transactions),
                "mining_time": mining_time,
                "composition": {
                    "bills": bill_count,
                    "rewards": reward_count,
                    "transfers": transfer_count
                },
                "cleanup_results": {
                    "initial_mempool": initial_mempool_size,
                    "final_mempool": len(self.mempool),
                    "removed_count": total_removed
                },
                "steps": steps
            }
            
        except Exception as e:
            self.logger.error(f"💥 STEP-BY-STEP MINING ERROR: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(f"💥 Full traceback: {error_traceback}")
            
            steps.append(f"❌ CRITICAL ERROR: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "traceback": error_traceback,
                "steps": steps if 'steps' in locals() else ["Failed before steps began"]
            }
    def lightweight_auto_mine(self, miner_address: str = "daemon_miner") -> Dict:
        """
        Lightweight mining for daemon use - processes transactions without extensive logging
        """
        try:
            # Quick cleanup first
            self.cleanup_mined_transactions_enhanced()
            
            # Get available transactions
            available_bills = self.get_available_bills_to_mine()
            available_rewards = self.get_available_rewards_to_mine()
            available_transfers = self.get_available_transfers_to_mine()
            
            total_available = len(available_bills) + len(available_rewards) + len(available_transfers)
            
            if total_available == 0:
                return {"success": False, "reason": "No transactions available"}
            
            # Combine and validate transactions
            all_transactions = available_bills + available_rewards + available_transfers
            valid_transactions = [tx for tx in all_transactions if self.validate_transaction(tx, skip_mined_check=True)]
            
            if not valid_transactions:
                return {"success": False, "reason": "No valid transactions"}
            
            # Limit to reasonable block size
            if len(valid_transactions) > 50:
                valid_transactions = valid_transactions[:50]
            
            # Create and mine block
            previous_block = self.blockchain[-1]
            new_block = {
                "index": len(self.blockchain),
                "timestamp": int(time.time()),
                "transactions": valid_transactions,
                "previous_hash": previous_block["hash"],
                "nonce": 0,
                "miner": miner_address,
                "difficulty": 2,
                "hash": ""
            }
            
            # Mine with timeout
            target = "0" * 2
            start_time = time.time()
            
            for nonce in range(1000000):
                if time.time() - start_time > 30:  # 30 second timeout
                    return {"success": False, "reason": "Mining timeout"}
                
                new_block["nonce"] = nonce
                calculated_hash = self.calculate_block_hash(
                    new_block["index"],
                    new_block["previous_hash"],
                    new_block["timestamp"],
                    new_block["transactions"],
                    nonce
                )
                
                if calculated_hash.startswith(target):
                    new_block["hash"] = calculated_hash
                    break
            else:
                return {"success": False, "reason": "No valid nonce found"}
            
            # Add to blockchain
            self.blockchain.append(new_block)
            
            # Cleanup mempool
            self.remove_mined_transactions(valid_transactions)
            
            # Save
            self.save_blockchain()
            self.save_mempool()
            
            return {
                "success": True,
                "block_index": new_block["index"],
                "transactions_mined": len(valid_transactions),
                "mining_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Lightweight mining error: {e}")
            return {"success": False, "reason": str(e)}
    def start_daemon(self, miner_address: str = "server_miner"):
        """Start the background daemon - FIXED VERSION"""
        self.is_running = True
        self.logger.info("Starting blockchain daemon...")
        
        # Ensure we have genesis block
        if not self.blockchain:
            self.logger.info("🌱 Creating genesis block...")
            genesis_block = self.create_genesis_block()
            self.blockchain.append(genesis_block)
            self.save_blockchain()
            self.logger.info("✅ Genesis block created")
        
        # Ensure we have genesis transactions for mining
        if self.needs_initial_genesis():
            self.logger.info("🆕 No genesis transactions found, creating initial set...")
            self.create_initial_genesis_transactions()
        
        def daemon_loop():
            iteration = 0
            while self.is_running:
                try:
                    # Update confirmations every iteration
                    self.update_transaction_confirmations()
                    
                    # Run comprehensive step-by-step mining every 100 iterations
                    if iteration % 1 == 0:
                        self.logger.info("🚀 Daemon: Running comprehensive step-by-step mining...")
                        status = self.get_step_by_step_mining_status()
                        if status["available_transactions"]["total"] > 0:
                            result = self.step_by_step_mine_all_transactions(miner_address=miner_address)
                            if result.get("success"):
                                self.logger.info(f"✅ Daemon: Comprehensive mining completed - Block #{result.get('block_index')}")
                            else:
                                self.logger.warning(f"⚠️ Daemon: Comprehensive mining failed: {result.get('error')}")
                        else:
                            self.logger.debug("🤖 Daemon: No transactions for comprehensive mining")
                    
                    # Cleanup every 30 iterations
                    if iteration % 1 == 0:
                        self.logger.debug("🧹 Daemon: Running periodic cleanup...")
                        cleanup_result = self.cleanup_mined_transactions_enhanced()
                        if cleanup_result > 0:
                            self.logger.info(f"✅ Daemon: Cleaned up {cleanup_result} mined transactions")
                    
                    # Force comprehensive cleanup every 200 iterations - FIXED
                    if iteration % 1 == 0:
                        self.logger.info("🔄 Daemon: Running forced comprehensive cleanup...")
                        cleanup_result = self.force_mempool_cleanup()  # This returns dict now
                        
                        # FIX: Handle both dict and int return types for backward compatibility
                        if isinstance(cleanup_result, dict):
                            still_stuck = cleanup_result.get("still_stuck", 0)
                        else:
                            still_stuck = cleanup_result  # For backward compatibility
                        
                        if still_stuck > 0:
                            self.logger.warning(f"⚠️ Daemon: {still_stuck} transactions still stuck")
                    
                    # Reload data every 50 iterations
                    if iteration % 1 == 0:
                        self.logger.debug("🔄 Daemon: Reloading blockchain data...")
                        self.load_data()
                    
                    iteration += 1
                    time.sleep(self.sync_interval)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in daemon loop: {e}")
                    import traceback
                    self.logger.error(f"💥 Traceback: {traceback.format_exc()}")
                    time.sleep(self.sync_interval)
        
        # Start daemon thread
        self.daemon_thread = threading.Thread(target=daemon_loop, daemon=True)
        self.daemon_thread.start()
        
        # Start confirmation updater
        self.start_confirmation_updater()
        
        self.logger.info("Blockchain daemon started")
    try:
        import fcntl  # For Unix/Linux
    # For Windows, we'll use a different approach
    except Exception as e:
        print(f"Error:{e}")
    def save_mempool_thread_safe(self):
        """Thread-safe mempool saving with file locking"""
        import tempfile
        import os
        
        try:
            if not isinstance(self.mempool, list):
                self.logger.error("Mempool is not a list, resetting to []")
                self.mempool = []

            # Create temporary file
            temp_dir = os.path.dirname(self.mempool_file) or '.'
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                        dir=temp_dir, delete=False, suffix='.tmp') as f:
                json.dump(self.mempool, f, indent=2)
                temp_filename = f.name

            # Atomic replace using rename (works on Windows)
            try:
                if os.path.exists(self.mempool_file):
                    os.remove(self.mempool_file)
                os.rename(temp_filename, self.mempool_file)
            except Exception as e:
                # If rename fails, clean up temp file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                raise e

            self.logger.info(f"✅ Mempool saved successfully ({len(self.mempool)} tx)")

            # Backup
            self.backup_file(self.mempool_file, "./back-up/mempool")

        except Exception as e:
            self.logger.error(f"Unexpected error saving mempool: {e}")
            # Don't re-raise, just log the error
    def comprehensive_mempool_cleanup(self):
        """Most thorough mempool cleanup that catches all edge cases"""
        initial_count = len(self.mempool)
        
        if initial_count == 0:
            return 0
        
        self.logger.info(f"🧹 COMPREHENSIVE CLEANUP: Starting with {initial_count} transactions")
        
        # Get ALL transaction hashes from entire blockchain
        blockchain_hashes = set()
        blockchain_serials = set()
        blockchain_signatures = set()
        
        # Build comprehensive indexes from blockchain
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                # By hash
                if tx.get("hash"):
                    blockchain_hashes.add(tx["hash"])
                
                # By serial number (genesis transactions)
                if tx.get("serial_number"):
                    blockchain_serials.add(tx["serial_number"])
                
                # By signature (transfer transactions)
                if tx.get("signature"):
                    blockchain_signatures.add(tx["signature"])
                
                # Also calculate hash from current data to catch inconsistencies
                try:
                    tx_copy = tx.copy()
                    if "hash" in tx_copy:
                        del tx_copy["hash"]
                    tx_string = json.dumps(tx_copy, sort_keys=True)
                    calculated_hash = hashlib.sha256(tx_string.encode()).hexdigest()
                    blockchain_hashes.add(calculated_hash)
                except:
                    pass
        
        self.logger.info(f"📊 Blockchain indexes: {len(blockchain_hashes)} hashes, {len(blockchain_serials)} serials, {len(blockchain_signatures)} signatures")
        
        # Remove any transaction that appears in blockchain by ANY identifier
        new_mempool = []
        removed_count = 0
        
        for tx in self.mempool:
            should_remove = False
            removal_reason = ""
            
            tx_hash = tx.get("hash")
            tx_serial = tx.get("serial_number")
            tx_signature = tx.get("signature")
            
            # Check by direct hash match
            if tx_hash and tx_hash in blockchain_hashes:
                should_remove = True
                removal_reason = f"hash match: {tx_hash[:16]}..."
            
            # Check by serial number (genesis transactions)
            elif tx_serial and tx_serial in blockchain_serials:
                should_remove = True
                removal_reason = f"serial match: {tx_serial}"
            
            # Check by signature (transfer transactions)
            elif tx_signature and tx_signature in blockchain_signatures:
                should_remove = True
                removal_reason = f"signature match: {tx_signature[:20]}..."
            
            # Check by recalculated hash (catches hash inconsistencies)
            else:
                try:
                    # Calculate hash from current transaction data
                    tx_copy = tx.copy()
                    if "hash" in tx_copy:
                        del tx_copy["hash"]
                    tx_string = json.dumps(tx_copy, sort_keys=True)
                    calculated_hash = hashlib.sha256(tx_string.encode()).hexdigest()
                    
                    if calculated_hash in blockchain_hashes:
                        should_remove = True
                        removal_reason = f"recalculated hash match: {calculated_hash[:16]}..."
                except Exception as e:
                    self.logger.warning(f"Could not recalculate hash for transaction: {e}")
            
            if should_remove:
                self.logger.info(f"🧹 Removing: {removal_reason}")
                removed_count += 1
            else:
                new_mempool.append(tx)
        
        # Update mempool
        self.mempool = new_mempool
        
        if removed_count > 0:
            self.save_mempool()
            self.logger.info(f"✅ COMPREHENSIVE CLEANUP: Removed {removed_count}/{initial_count} transactions")
            self.logger.info(f"📊 Remaining in mempool: {len(self.mempool)}")
        else:
            self.logger.info("✅ COMPREHENSIVE CLEANUP: No transactions to remove")
        
        return removed_count

    def debug_stuck_transactions(self):
        """Debug why transactions are stuck in mempool"""
        self.logger.info("=== STUCK TRANSACTIONS DEBUG ===")
        
        if not self.mempool:
            self.logger.info("Mempool is empty")
            return
        
        self.logger.info(f"Mempool has {len(self.mempool)} transactions")
        
        stuck_info = {
            "total": len(self.mempool),
            "by_type": {},
            "stuck_reasons": []
        }
        
        for tx in self.mempool:
            tx_type = tx.get("type", "unknown")
            tx_hash = tx.get("hash", "no-hash")
            
            # Count by type
            if tx_type not in stuck_info["by_type"]:
                stuck_info["by_type"][tx_type] = 0
            stuck_info["by_type"][tx_type] += 1
            
            # Check if actually mined
            is_mined = self.is_transaction_mined(tx)
            
            if is_mined:
                self.logger.info(f"❌ MINED BUT STUCK: {tx_type} - {tx_hash[:16]}...")
                stuck_info["stuck_reasons"].append(f"{tx_type} - {tx_hash[:16]}...: MINED BUT STUCK")
            
            # Check validation
            is_valid = self.validate_transaction(tx, skip_mined_check=True)
            if not is_valid:
                self.logger.info(f"❌ INVALID: {tx_type} - {tx_hash[:16]}...")
                stuck_info["stuck_reasons"].append(f"{tx_type} - {tx_hash[:16]}...: VALIDATION FAILED")
        
        # Log summary
        self.logger.info("=== STUCK TRANSACTIONS SUMMARY ===")
        for tx_type, count in stuck_info["by_type"].items():
            self.logger.info(f"  {tx_type}: {count}")
        
        for reason in stuck_info["stuck_reasons"][:10]:  # Show first 10 reasons
            self.logger.info(f"  {reason}")
        
        if len(stuck_info["stuck_reasons"]) > 10:
            self.logger.info(f"  ... and {len(stuck_info['stuck_reasons']) - 10} more")
        
        return stuck_info
    def debug_mining_issues(self):
        """Debug why transactions aren't being mined"""
        self.logger.info("=== MINING ISSUES DEBUG ===")
        
        # Check what's available for mining
        available_bills = self.get_available_bills_to_mine()
        available_rewards = self.get_available_rewards_to_mine()
        available_transfers = self.get_available_transfers_to_mine()
        
        self.logger.info(f"Available for mining:")
        self.logger.info(f"  Bills: {len(available_bills)}")
        self.logger.info(f"  Rewards: {len(available_rewards)}")
        self.logger.info(f"  Transfers: {len(available_transfers)}")
        self.logger.info(f"  Total: {len(available_bills) + len(available_rewards) + len(available_transfers)}")
        
        # Check blockchain state
        self.logger.info(f"Blockchain: {len(self.blockchain)} blocks")
        
        # Check if we have a genesis block
        if self.blockchain:
            genesis_txs = self.blockchain[0].get("transactions", [])
            self.logger.info(f"Genesis block has {len(genesis_txs)} transactions")
        
        # Check if mining is working
        if len(self.blockchain) > 1:
            recent_blocks = self.blockchain[-5:]  # Last 5 blocks
            total_tx_in_recent_blocks = sum(len(block.get("transactions", [])) for block in recent_blocks)
            self.logger.info(f"Recent blocks (last 5): {total_tx_in_recent_blocks} transactions")
        else:
            self.logger.info("Only genesis block exists")
        
        # Check if there are any genesis transactions at all
        all_genesis = []
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("type") in ["GTX_Genesis", "genesis"]:
                    all_genesis.append(tx)
        
        self.logger.info(f"Total genesis transactions in blockchain: {len(all_genesis)}")
        
        return {
            "available_for_mining": len(available_bills) + len(available_rewards) + len(available_transfers),
            "blockchain_blocks": len(self.blockchain),
            "genesis_in_blockchain": len(all_genesis),
            "mempool_size": len(self.mempool)
        }
    def force_clear_mempool(self):
        """Nuclear option: completely clear and rebuild mempool - FIXED RETURN TYPE"""
        self.logger.info("💥 FORCE CLEARING MEMPOOL")
        
        # Backup current mempool
        backup_file = f"mempool.backup.{int(time.time())}.json"
        try:
            with open(backup_file, 'w') as f:
                json.dump(self.mempool, f, indent=2)
            self.logger.info(f"📁 Mempool backed up to: {backup_file}")
        except Exception as e:
            self.logger.error(f"Failed to backup mempool: {e}")
        
        # Run comprehensive cleanup first
        cleaned = self.comprehensive_mempool_cleanup()
        
        # If still transactions remain, check if they're valid
        remaining = len(self.mempool)
        if remaining > 0:
            self.logger.info(f"🔍 {remaining} transactions still in mempool after cleanup")
            
            # Validate each remaining transaction
            valid_transactions = []
            invalid_count = 0
            for tx in self.mempool:
                if self.validate_transaction(tx, skip_mined_check=True):
                    valid_transactions.append(tx)
                else:
                    invalid_count += 1
                    self.logger.warning(f"Removing invalid transaction: {tx.get('type')} - {tx.get('hash', 'no-hash')[:16]}...")
            
            # Keep only valid transactions
            self.mempool = valid_transactions
            self.save_mempool_thread_safe()  # Use thread-safe version
            
            self.logger.info(f"✅ Force clear completed: {len(valid_transactions)} valid transactions remain")
            
            # RETURN DICT instead of int
            return {
                "basic_removed": cleaned,
                "enhanced_removed": 0,
                "comprehensive_removed": 0,
                "still_stuck": len(valid_transactions),
                "invalid_removed": invalid_count
            }
        else:
            self.logger.info("✅ Mempool completely cleared")
            return {
                "basic_removed": cleaned,
                "enhanced_removed": 0,
                "comprehensive_removed": 0,
                "still_stuck": 0,
                "invalid_removed": 0
            }
    def stop_daemon(self):
        """Stop the background daemon"""
        self.is_running = False
        self.logger.info("Stopping blockchain daemon...")
        if hasattr(self, 'daemon_thread'):
            self.daemon_thread.join(timeout=5)
        self.logger.info("Blockchain daemon stopped")

