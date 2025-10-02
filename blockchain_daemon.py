# blockchain_daemon.py
import json
import time
import hashlib
import threading
import os
from typing import List, Dict, Set
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, Future

class BlockchainDaemon:
    def __init__(self, blockchain_file="blockchain.json", mempool_file="mempool.json"):
        self.blockchain_file = blockchain_file
        self.mempool_file = mempool_file
        self.blockchain = []
        self.mempool = []
        self.mined_serials: Set[str] = set()
        self.is_running = False
        self.sync_interval = 10
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create initial files
        self.create_initial_files()
        
        # Load data
        self.load_data()

    def create_initial_files(self):
        """Create initial JSON files if they don't exist"""
        if not os.path.exists(self.blockchain_file) or os.path.getsize(self.blockchain_file) == 0:
            self.blockchain = []
            self.save_blockchain()
        
        if not os.path.exists(self.mempool_file) or os.path.getsize(self.mempool_file) == 0:
            self.mempool = []
            self.save_mempool()

    def load_data(self):
        """Load blockchain and mempool from files"""
        try:
            # Load blockchain
            if os.path.exists(self.blockchain_file):
                with open(self.blockchain_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.blockchain = json.loads(content)
            
            # Load mempool
            if os.path.exists(self.mempool_file):
                with open(self.mempool_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.mempool = json.loads(content)
            
            # Build mined indexes
            self.build_mined_indexes()
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.blockchain = []
            self.mempool = []

    def build_mined_indexes(self):
        """Build indexes of all mined serial numbers"""
        self.mined_serials.clear()
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("serial_number"):
                    self.mined_serials.add(tx["serial_number"])

    def save_blockchain(self):
        """Save blockchain to file"""
        try:
            with open(self.blockchain_file, 'w', encoding='utf-8') as f:
                json.dump(self.blockchain, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving blockchain: {e}")

    def save_mempool(self):
        """Save mempool to file"""
        try:
            with open(self.mempool_file, 'w', encoding='utf-8') as f:
                json.dump(self.mempool, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving mempool: {e}")

    def add_transaction(self, transaction: Dict) -> bool:
        """Add a transaction to mempool"""
        try:
            # Validate transaction structure
            if not self.validate_transaction_structure(transaction):
                return False
            
            # Calculate hash if not present
            if not transaction.get("hash"):
                tx_string = json.dumps(transaction, sort_keys=True)
                transaction["hash"] = hashlib.sha256(tx_string.encode()).hexdigest()
            
            # Check for duplicates
            tx_hash = transaction["hash"]
            for existing_tx in self.mempool:
                if existing_tx.get("hash") == tx_hash:
                    return False
            
            # Check if already mined
            if self.is_transaction_mined(transaction):
                return False
            
            # Add to mempool
            self.mempool.append(transaction)
            self.save_mempool()
            
            self.logger.info(f"Added {transaction.get('type')} transaction to mempool: {tx_hash}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding transaction: {e}")
            return False

    def validate_transaction_structure(self, transaction: Dict) -> bool:
        """Validate transaction structure"""
        tx_type = transaction.get("type")
        
        if tx_type == "transfer":
            required_fields = ["from", "to", "amount", "timestamp", "signature"]
        elif tx_type in ["GTX_Genesis", "genesis"]:
            required_fields = ["serial_number", "denomination", "issued_to", "timestamp"]
        elif tx_type == "reward":
            required_fields = ["to", "amount", "timestamp", "block_height"]
        else:
            return False
        
        return all(field in transaction for field in required_fields)

    def is_transaction_mined(self, transaction: Dict) -> bool:
        """Check if transaction has been mined"""
        tx_hash = transaction.get("hash")
        if not tx_hash:
            return False
        
        # Check all blocks for this transaction
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("hash") == tx_hash:
                    return True
        return False

    def get_mempool_status(self) -> Dict:
        """Get mempool status and statistics"""
        bills = [tx for tx in self.mempool if tx.get("type") in ["GTX_Genesis", "genesis"]]
        transfers = [tx for tx in self.mempool if tx.get("type") == "transfer"]
        rewards = [tx for tx in self.mempool if tx.get("type") == "reward"]
        
        return {
            "total": len(self.mempool),
            "bills": len(bills),
            "transfers": len(transfers),
            "rewards": len(rewards),
            "transactions": self.mempool
        }

    def get_blockchain_status(self) -> Dict:
        """Get blockchain status and statistics"""
        total_transactions = 0
        genesis_count = 0
        transfer_count = 0
        reward_count = 0
        
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                total_transactions += 1
                tx_type = tx.get("type")
                if tx_type in ["GTX_Genesis", "genesis"]:
                    genesis_count += 1
                elif tx_type == "transfer":
                    transfer_count += 1
                elif tx_type == "reward":
                    reward_count += 1
        
        return {
            "blocks": len(self.blockchain),
            "total_transactions": total_transactions,
            "genesis_transactions": genesis_count,
            "transfer_transactions": transfer_count,
            "reward_transactions": reward_count,
            "mined_serials": len(self.mined_serials)
        }

    # In blockchain_daemon.py, modify validate_block:
    def validate_block(self, block: Dict) -> bool:
        """Validate a mined block from external miner"""
        try:
            # Check block structure
            required_fields = ["index", "timestamp", "transactions", "previous_hash", "nonce", "hash", "miner"]
            missing_fields = [field for field in required_fields if field not in block]
            if missing_fields:
                self.logger.error(f"Missing required fields: {missing_fields}")
                return False
            
            # Verify block hash
            calculated_hash = self.calculate_block_hash(
                block["index"],
                block["previous_hash"],
                block["timestamp"],
                block["transactions"],
                block["nonce"]
            )
            
            if block["hash"] != calculated_hash:
                self.logger.error(f"Hash mismatch: {block['hash']} vs {calculated_hash}")
                return False
        
            
            # Verify chain continuity
            if block["index"] > 0:
                previous_block = self.blockchain[-1]
                if block["previous_hash"] != previous_block["hash"]:
                    return False
            
            # Verify all transactions in block
            for tx in block.get("transactions", []):
                if not self.validate_transaction_for_block(tx):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Block validation error: {e}")
            return False

    def validate_transaction_for_block(self, transaction: Dict) -> bool:
        """Validate transaction for inclusion in block"""
        # Skip mempool check for reward transactions (they're created during mining)
        if transaction.get("type") == "reward":
            return self.validate_transaction_structure(transaction)
        
        # For other transactions, check mempool
        tx_hash = transaction.get("hash")
        in_mempool = any(tx.get("hash") == tx_hash for tx in self.mempool)
        
        if not in_mempool:
            self.logger.warning(f"Transaction not in mempool: {tx_hash}")
            return False
        
        # Validate transaction structure
        if not self.validate_transaction_structure(transaction):
            return False
        
        # Check for double-spending (for genesis transactions)
        if transaction.get("type") in ["GTX_Genesis", "genesis"]:
            serial = transaction.get("serial_number")
            if serial and serial in self.mined_serials:
                return False
        
        return True

    def add_validated_block(self, block: Dict) -> bool:
        """Add a validated block to the blockchain"""
        try:
            if not self.validate_block(block):
                return False
            
            # Add to blockchain
            self.blockchain.append(block)
            
            # Remove mined transactions from mempool
            self.remove_mined_transactions(block.get("transactions", []))
            
            # Update mined indexes
            self.update_mined_indexes(block)
            
            # Save changes
            self.save_blockchain()
            self.save_mempool()
            
            self.logger.info(f"Added block #{block['index']} with {len(block['transactions'])} transactions")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding block: {e}")
            return False

    def remove_mined_transactions(self, mined_transactions: List[Dict]) -> int:
        """Remove mined transactions from mempool"""
        initial_count = len(self.mempool)
        mined_hashes = {tx.get("hash") for tx in mined_transactions if tx.get("hash")}
        
        self.mempool = [tx for tx in self.mempool if tx.get("hash") not in mined_hashes]
        
        removed_count = initial_count - len(self.mempool)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} mined transactions from mempool")
        
        return removed_count

    def update_mined_indexes(self, block: Dict):
        """Update mined indexes with transactions from new block"""
        for tx in block.get("transactions", []):
            if tx.get("serial_number"):
                self.mined_serials.add(tx["serial_number"])

    def calculate_block_hash(self, index, previous_hash, timestamp, transactions, nonce):
        """Calculate SHA-256 hash of a block"""
        if isinstance(timestamp, float):
            timestamp = int(timestamp)
        
        block_data = {
            'index': int(index),
            'previous_hash': previous_hash,
            'timestamp': timestamp,
            'transactions': json.dumps(transactions, sort_keys=True),
            'nonce': int(nonce)
        }
        
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def get_transaction_status(self, tx_hash: str) -> Dict:
        """Get transaction status and confirmations"""
        # Check mempool
        for tx in self.mempool:
            if tx.get("hash") == tx_hash:
                return {"status": "pending", "confirmations": 0}
        
        # Check blockchain
        for i, block in enumerate(self.blockchain):
            for tx in block.get("transactions", []):
                if tx.get("hash") == tx_hash:
                    confirmations = len(self.blockchain) - i - 1
                    return {
                        "status": "confirmed",
                        "confirmations": confirmations,
                        "block_height": i,
                        "block_hash": block.get("hash")
                    }
        
        return {"status": "not found"}

    def start_daemon(self):
        """Start the background daemon"""
        self.is_running = True
        
        def daemon_loop():
            while self.is_running:
                try:
                    # Periodic cleanup of mined transactions
                    self.cleanup_mined_transactions()
                    time.sleep(self.sync_interval)
                except Exception as e:
                    self.logger.error(f"Error in daemon loop: {e}")
                    time.sleep(self.sync_interval)
        
        self.daemon_thread = threading.Thread(target=daemon_loop, daemon=True)
        self.daemon_thread.start()
        self.logger.info("Blockchain daemon started")

    def cleanup_mined_transactions(self):
        """Remove mined transactions from mempool"""
        initial_count = len(self.mempool)
        
        # Get all mined transaction hashes
        mined_hashes = set()
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("hash"):
                    mined_hashes.add(tx["hash"])
        
        # Remove mined transactions
        self.mempool = [tx for tx in self.mempool if tx.get("hash") not in mined_hashes]
        
        removed_count = initial_count - len(self.mempool)
        if removed_count > 0:
            self.save_mempool()
            self.logger.info(f"Cleaned up {removed_count} mined transactions")

    def stop_daemon(self):
        """Stop the background daemon"""
        self.is_running = False
        self.logger.info("Blockchain daemon stopped")