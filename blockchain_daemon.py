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
import secrets

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
                if tx.get("type") == "GTX_Genesis" and tx.get("serial_number"):
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

    def generate_transaction_hash(self, transaction_data):
        """Generate a unique hash for any transaction"""
        # Create a string representation of the transaction data
        hash_data = {
            'type': transaction_data.get('type'),
            'to': transaction_data.get('to'),
            'amount': transaction_data.get('amount'),
            'timestamp': transaction_data.get('timestamp'),
            'block_height': transaction_data.get('block_height'),
            'miner': transaction_data.get('miner'),
            'nonce': secrets.randbelow(1000000)  # Add some randomness
        }
        
        # Convert to string and hash
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def create_reward_transaction(self, miner_address, from_address, amount, block_height, transaction_count=0):
        """Create a properly hashed reward transaction"""
        reward_tx = {
            "type": "reward",
            "to": miner_address,
            "from": from_address,
            "amount": float(amount),
            "timestamp": time.time(),
            "block_height": int(block_height),
            "description": f"Mining reward for block {block_height}",
            "miner": miner_address,
            "transactions_included": int(transaction_count),
            "fee_reward": float(transaction_count * 0.1)  # Adjust based on your fee structure
        }
        
        # Generate and add the hash
        reward_tx["hash"] = self.generate_transaction_hash(reward_tx)
        
        return reward_tx

    def add_genesis_transaction(self, serial_number: str, denomination: float, issued_to: str) -> bool:
        """
        Add a genesis transaction for a banknote to the mempool with real cryptographic signatures
        
        Args:
            serial_number: The unique serial number of the banknote
            denomination: The denomination value
            issued_to: The name of the person the banknote is issued to
        
        Returns:
            bool: True if successfully added to mempool, False otherwise
        """
        try:
            # Import the signature functions
            try:
                from signatures import DigitalSignatureManager, generate_key_pair
                HAS_SIGNATURES = True
            except ImportError:
                self.logger.warning("Signatures module not available, using fallback")
                HAS_SIGNATURES = False
            
            timestamp = int(time.time())
            
            # Generate cryptographic key pair for this banknote
            if HAS_SIGNATURES:
                try:
                    private_key, public_key = generate_key_pair()
                    
                    # Create transaction data to sign
                    transaction_data = {
                        "type": "GTX_Genesis",
                        "serial_number": serial_number,
                        "denomination": denomination,
                        "issued_to": issued_to,
                        "timestamp": timestamp,
                        "public_key": public_key
                    }
                    
                    # Create digital signature
                    signature_manager = DigitalSignatureManager()
                    signature = signature_manager.create_transaction_signature(transaction_data, private_key)
                    
                except Exception as crypto_error:
                    self.logger.warning(f"Cryptographic signature failed, using fallback: {crypto_error}")
                    HAS_SIGNATURES = False
            
            # Fallback if cryptography fails or isn't available
            if not HAS_SIGNATURES:
                # Generate deterministic keys based on serial number
                private_key_seed = f"genesis_private_{serial_number}_{timestamp}"
                public_key_seed = f"genesis_public_{serial_number}_{timestamp}"
                
                # Create hash-based "keys"
                private_key = hashlib.sha256(private_key_seed.encode()).hexdigest()
                public_key = hashlib.sha256(public_key_seed.encode()).hexdigest()
                
                # Create hash-based signature
                signature_data = f"{serial_number}{denomination}{issued_to}{timestamp}{public_key}"
                signature = hashlib.sha256(signature_data.encode()).hexdigest()
            
            # Create genesis transaction for banknote
            genesis_transaction = {
                "type": "GTX_Genesis",  # Genesis Transaction for Banknote
                "serial_number": serial_number,
                "denomination": denomination,
                "issued_to": issued_to,
                "timestamp": timestamp,
                "signature": signature,
                "public_key": public_key,
                "hash": ""  # Will be calculated by add_transaction
            }
            
            # Calculate transaction hash (this should match what add_transaction expects)
            tx_string = json.dumps(genesis_transaction, sort_keys=True)
            genesis_transaction["hash"] = hashlib.sha256(tx_string.encode()).hexdigest()
            
            # Add to mempool
            success = self.add_transaction(genesis_transaction)
            
            if success:
                self.logger.info(f"✓ Added genesis transaction for serial: {serial_number}")
                self.logger.info(f"  Denomination: {denomination}, Issued to: {issued_to}")
                self.logger.info(f"  Signature: {signature[:16]}...")
                self.logger.info(f"  Public key: {public_key[:16]}...")
                self.logger.info(f"  Transaction hash: {genesis_transaction['hash'][:16]}...")
            else:
                self.logger.warning(f"✗ Failed to add genesis transaction for serial: {serial_number}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating genesis transaction: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def add_transaction(self, transaction: Dict) -> bool:
        """Add a transaction to mempool"""
        try:
            # Validate transaction structure
            if not self.validate_transaction_structure(transaction):
                self.logger.warning(f"Invalid transaction structure: {transaction.get('type')}")
                return False
            
            # Calculate hash if not present
            if not transaction.get("hash"):
                tx_string = json.dumps(transaction, sort_keys=True)
                transaction["hash"] = hashlib.sha256(tx_string.encode()).hexdigest()
            
            # Check for duplicates in mempool
            tx_hash = transaction["hash"]
            for existing_tx in self.mempool:
                if existing_tx.get("hash") == tx_hash:
                    self.logger.warning(f"Duplicate transaction in mempool: {tx_hash}")
                    return False
            
            # Check if already mined
            if self.is_transaction_mined(transaction):
                self.logger.warning(f"Transaction already mined: {tx_hash}")
                return False
            
            # Special check for genesis transaction
            if transaction.get("type") == "genesis":
                if self.blockchain and len(self.blockchain) > 0:
                    self.logger.warning("Genesis transaction only allowed in block 0")
                    return False
            
            # Add to mempool
            self.mempool.append(transaction)
            self.save_mempool()
            
            self.logger.info(f"Added {transaction.get('type')} transaction to mempool: {tx_hash[:16]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding transaction: {e}")
            return False

    def validate_transaction_structure(self, transaction: Dict) -> bool:
        """Validate transaction structure - PROPERLY DIFFERENTIATED"""
        tx_type = transaction.get("type")
        
        if tx_type == "transfer":
            required_fields = ["from", "to", "amount", "timestamp", "signature"]
        
        elif tx_type == "GTX_Genesis":
            # Bill transactions - require denomination and cryptographic proof
            required_fields = ["serial_number", "denomination", "issued_to", "timestamp", "signature", "public_key"]
        
        elif tx_type == "genesis":
            # Single genesis block transaction - completely different structure
            required_fields = ["message", "timestamp", "hash"]  # Based on actual genesis block
        
        elif tx_type == "reward":
            # Mining reward transactions - NOW WITH HASH REQUIREMENT
            required_fields = ["to", "amount", "timestamp", "block_height", "hash"]
        
        else:
            self.logger.warning(f"Unknown transaction type: {tx_type}")
            return False
        
        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in transaction]
        if missing_fields:
            self.logger.warning(f"Missing fields for {tx_type}: {missing_fields}")
            return False
            
        return True

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
        """Get mempool status and statistics - DIFFERENTIATED"""
        bills = [tx for tx in self.mempool if tx.get("type") == "GTX_Genesis"]
        genesis = [tx for tx in self.mempool if tx.get("type") == "genesis"]
        transfers = [tx for tx in self.mempool if tx.get("type") == "transfer"]
        rewards = [tx for tx in self.mempool if tx.get("type") == "reward"]
        
        return {
            "total": len(self.mempool),
            "bills": len(bills),
            "genesis": len(genesis),
            "transfers": len(transfers),
            "rewards": len(rewards),
            "transactions": self.mempool
        }

    def get_blockchain_status(self) -> Dict:
        """Get blockchain status and statistics - DIFFERENTIATED"""
        total_transactions = 0
        genesis_count = 0
        gtx_genesis_count = 0
        transfer_count = 0
        reward_count = 0
        
        for block in self.blockchain:
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
            "blocks": len(self.blockchain),
            "total_transactions": total_transactions,
            "genesis_transactions": genesis_count,
            "gtx_genesis_transactions": gtx_genesis_count,
            "transfer_transactions": transfer_count,
            "reward_transactions": reward_count,
            "mined_serials": len(self.mined_serials)
        }

    def validate_block(self, block: Dict) -> bool:
        """Validate a mined block from external miner - EXTREME VERBOSE VERSION"""
        self.logger.info("🚨" * 20)
        self.logger.info("🚨 STARTING BLOCK VALIDATION - EXTREME VERBOSE MODE 🚨")
        self.logger.info("🚨" * 20)
        
        try:
            self.logger.info(f"📦 BLOCK RECEIVED - FULL DUMP:")
            self.logger.info(f"   📋 Block keys: {list(block.keys())}")
            self.logger.info(f"   🔢 Index: {block.get('index')}")
            self.logger.info(f"   ⏰ Timestamp: {block.get('timestamp')} (type: {type(block.get('timestamp'))})")
            self.logger.info(f"   🔗 Previous Hash: {block.get('previous_hash', '')[:32]}...")
            self.logger.info(f"   💎 Nonce: {block.get('nonce')} (type: {type(block.get('nonce'))})")
            self.logger.info(f"   🆔 Block Hash: {block.get('hash', '')[:32]}...")
            self.logger.info(f"   ⛏️  Miner: {block.get('miner', 'NOT PROVIDED')}")
            self.logger.info(f"   📄 Transactions count: {len(block.get('transactions', []))}")
            
            # Check block structure - EXTREME DETAIL
            self.logger.info("")
            self.logger.info("🔍 PHASE 1: BLOCK STRUCTURE VALIDATION")
            required_fields = ["hash", "index", "nonce", "previous_hash", "timestamp", "transactions"]
            self.logger.info(f"   📋 Required fields: {required_fields}")
            
            missing_fields = [field for field in required_fields if field not in block]
            if missing_fields:
                self.logger.error(f"   ❌ MISSING FIELDS: {missing_fields}")
                self.logger.error(f"   📊 Available fields: {list(block.keys())}")
                return False
            else:
                self.logger.info("   ✅ All required fields present")
            
            # "miner" field is optional - EXTREME DETAIL
            self.logger.info("")
            self.logger.info("🔍 PHASE 2: OPTIONAL FIELD CHECK")
            if "miner" not in block:
                self.logger.warning("   ⚠️  'miner' field missing, using default 'unknown_miner'")
                block["miner"] = "unknown_miner"
            else:
                self.logger.info(f"   ✅ Miner field present: {block['miner']}")
            
            # Verify block hash - EXTREME DETAIL
            self.logger.info("")
            self.logger.info("🔍 PHASE 3: BLOCK HASH VALIDATION")
            self.logger.info(f"   🎯 Calculating hash for block #{block['index']}...")
            
            # Log exactly what we're hashing
            self.logger.info(f"   📝 HASH INPUT DATA:")
            self.logger.info(f"      🔢 Index: {block['index']} (type: {type(block['index'])})")
            self.logger.info(f"      🔗 Previous Hash: {block['previous_hash']} (length: {len(block['previous_hash'])})")
            self.logger.info(f"      ⏰ Timestamp: {block['timestamp']} (type: {type(block['timestamp'])})")
            self.logger.info(f"      📄 Transactions count: {len(block['transactions'])}")
            self.logger.info(f"      💎 Nonce: {block['nonce']} (type: {type(block['nonce'])})")
            
            # Calculate hash with detailed logging
            calculated_hash = self.calculate_block_hash(
                block["index"],
                block["previous_hash"],
                block["timestamp"],
                block["transactions"],
                block["nonce"]
            )
            
            self.logger.info(f"   🎯 HASH COMPARISON:")
            self.logger.info(f"      📤 Submitted hash: {block['hash']}")
            self.logger.info(f"      🖥️  Calculated hash: {calculated_hash}")
            self.logger.info(f"      🔍 First 16 chars - Submitted: {block['hash'][:16]}, Calculated: {calculated_hash[:16]}")
            
            if block["hash"] != calculated_hash:
                self.logger.error("   ❌ HASH MISMATCH - BLOCK INVALID")
                self.logger.error(f"      📊 Full hash comparison:")
                self.logger.error(f"          Submitted: {block['hash']}")
                self.logger.error(f"          Calculated: {calculated_hash}")
                
                # Check if it's a leading zeros issue
                submitted_zeros = len(block['hash']) - len(block['hash'].lstrip('0'))
                calculated_zeros = len(calculated_hash) - len(calculated_hash.lstrip('0'))
                self.logger.error(f"      🔢 Leading zeros - Submitted: {submitted_zeros}, Calculated: {calculated_zeros}")
                
                return False
            else:
                self.logger.info("   ✅ HASH VALIDATION PASSED")
            
            # Verify chain continuity - EXTREME DETAIL
            self.logger.info("")
            self.logger.info("🔍 PHASE 4: CHAIN CONTINUITY VALIDATION")
            current_height = len(self.blockchain)
            expected_index = current_height
            
            self.logger.info(f"   📊 Chain state:")
            self.logger.info(f"      Current blockchain height: {current_height}")
            self.logger.info(f"      Expected block index: {expected_index}")
            self.logger.info(f"      Submitted block index: {block['index']}")
            
            if block["index"] != expected_index:
                self.logger.error(f"   ❌ BLOCK INDEX MISMATCH")
                self.logger.error(f"      Expected: {expected_index}, Got: {block['index']}")
                self.logger.error(f"      This block would create a fork or gap in the chain")
                return False
            else:
                self.logger.info("   ✅ BLOCK INDEX CORRECT")
            
            # Verify previous hash matches - EXTREME DETAIL
            self.logger.info("")
            self.logger.info("🔍 PHASE 5: PREVIOUS HASH VALIDATION")
            
            if current_height > 0:
                previous_block = self.blockchain[-1]
                self.logger.info(f"   📊 Previous block info:")
                self.logger.info(f"      Previous block index: {previous_block['index']}")
                self.logger.info(f"      Previous block hash: {previous_block['hash'][:32]}...")
                self.logger.info(f"      Submitted previous hash: {block['previous_hash'][:32]}...")
                
                if block["previous_hash"] != previous_block["hash"]:
                    self.logger.error("   ❌ PREVIOUS HASH MISMATCH")
                    self.logger.error(f"      Expected: {previous_block['hash']}")
                    self.logger.error(f"      Got: {block['previous_hash']}")
                    return False
                else:
                    self.logger.info("   ✅ PREVIOUS HASH MATCHES")
            else:
                self.logger.info("   🔰 GENESIS BLOCK CASE")
                self.logger.info(f"   📊 Genesis block validation:")
                self.logger.info(f"      Submitted previous hash: {block['previous_hash']}")
                self.logger.info(f"      Allowed values: '0' or '{'0' * 64}'")
                
                # Genesis block case - allow both "0" and 64 zeros
                if block["previous_hash"] != "0" * 64 and block["previous_hash"] != "0":
                    self.logger.error("   ❌ GENESIS BLOCK INVALID PREVIOUS HASH")
                    self.logger.error(f"      Genesis block must have '0' or 64 zeros as previous hash")
                    self.logger.error(f"      Got: {block['previous_hash']}")
                    return False
                else:
                    self.logger.info("   ✅ GENESIS BLOCK PREVIOUS HASH VALID")
            
            # Verify all transactions in block - EXTREME DETAIL
            self.logger.info("")
            self.logger.info("🔍 PHASE 6: TRANSACTION VALIDATION")
            transactions = block.get("transactions", [])
            self.logger.info(f"   📊 Transaction overview:")
            self.logger.info(f"      Total transactions: {len(transactions)}")
            
            # Count transaction types
            tx_types = {}
            for tx in transactions:
                tx_type = tx.get('type', 'UNKNOWN')
                tx_types[tx_type] = tx_types.get(tx_type, 0) + 1
            
            self.logger.info(f"      Transaction types: {tx_types}")
            
            valid_transactions = 0
            invalid_transactions = 0
            transaction_details = []
            
            for i, tx in enumerate(transactions):
                self.logger.info(f"   🔍 Validating transaction {i+1}/{len(transactions)}:")
                self.logger.info(f"      Type: {tx.get('type', 'MISSING TYPE')}")
                self.logger.info(f"      Keys: {list(tx.keys())}")
                
                is_valid = self.validate_transaction_for_block(tx, block["index"])
                if is_valid:
                    valid_transactions += 1
                    self.logger.info(f"      ✅ Transaction {i+1} VALID")
                else:
                    invalid_transactions += 1
                    self.logger.warning(f"      ❌ Transaction {i+1} INVALID")
                
                transaction_details.append({
                    'index': i,
                    'type': tx.get('type'),
                    'valid': is_valid,
                    'hash': tx.get('hash', 'NO HASH')[:16] + '...'
                })
            
            self.logger.info(f"   📊 Transaction validation summary:")
            self.logger.info(f"      Valid: {valid_transactions}")
            self.logger.info(f"      Invalid: {invalid_transactions}")
            self.logger.info(f"      Total: {len(transactions)}")
            
            # Log each transaction result
            for detail in transaction_details:
                status = "✅ VALID" if detail['valid'] else "❌ INVALID"
                self.logger.info(f"      TX {detail['index']+1}: {detail['type']} - {detail['hash']} - {status}")
            
            # Allow blocks with at least one valid transaction
            if valid_transactions == 0:
                self.logger.error("   ❌ NO VALID TRANSACTIONS IN BLOCK")
                self.logger.error(f"      Block must contain at least one valid transaction")
                self.logger.error(f"      Found {invalid_transactions} invalid transactions")
                return False
            else:
                self.logger.info(f"   ✅ TRANSACTION VALIDATION PASSED")
                if invalid_transactions > 0:
                    self.logger.warning(f"   ⚠️  Block contains {invalid_transactions} invalid transactions but has {valid_transactions} valid ones")
            
            # FINAL VALIDATION RESULT
            self.logger.info("")
            self.logger.info("🎉" * 20)
            self.logger.info("🎉 BLOCK VALIDATION SUCCESSFUL! 🎉")
            self.logger.info("🎉" * 20)
            self.logger.info(f"📊 FINAL BLOCK STATS:")
            self.logger.info(f"   🔢 Block #{block['index']} ACCEPTED")
            self.logger.info(f"   💎 Nonce: {block['nonce']}")
            self.logger.info(f"   ⛏️  Miner: {block['miner']}")
            self.logger.info(f"   📄 Transactions: {valid_transactions}/{len(transactions)} valid")
            self.logger.info(f"   🔗 Previous Hash: {block['previous_hash'][:16]}...")
            self.logger.info(f"   🆔 Block Hash: {block['hash'][:16]}...")
            
            return True
            
        except Exception as e:
            self.logger.error("💥" * 20)
            self.logger.error("💥 BLOCK VALIDATION CRASHED! 💥")
            self.logger.error("💥" * 20)
            self.logger.error(f"❌ Exception type: {type(e).__name__}")
            self.logger.error(f"❌ Exception message: {str(e)}")
            self.logger.error("📋 Full traceback:")
            import traceback
            tb_lines = traceback.format_exc().splitlines()
            for line in tb_lines:
                self.logger.error(f"   {line}")
            return False

    def calculate_block_hash(self, index, previous_hash, timestamp, transactions, nonce):
        """Calculate SHA-256 hash of a block - SINGLE CONSISTENT METHOD"""
        try:
            # Ensure proper types
            index = int(index)
            nonce = int(nonce)
            
            # Allow both float and integer timestamps
            if isinstance(timestamp, float):
                timestamp = timestamp
            else:
                timestamp = float(timestamp)
            
            # Use consistent JSON serialization
            block_data = {
                'index': index,
                'previous_hash': previous_hash,
                'timestamp': timestamp,
                'transactions': transactions,
                'nonce': nonce
            }
            
            # Consistent serialization method
            block_string = json.dumps(block_data, sort_keys=True, separators=(',', ':'))
            calculated_hash = hashlib.sha256(block_string.encode()).hexdigest()
            
            return calculated_hash
            
        except Exception as e:
            self.logger.error(f"Hash calculation error: {e}")
            return "0" * 64

    def validate_transaction_for_block(self, transaction: Dict, block_index: int) -> bool:
        """Validate transaction for inclusion in block - UPDATED REWARD LOGIC"""
        try:
            tx_type = transaction.get("type")
            
            # Reward transactions (created during mining) - UPDATED
            if tx_type == "reward":
                required = ["to", "amount", "timestamp", "block_height", "hash"]  # Now requires hash
                if not all(field in transaction for field in required):
                    self.logger.warning(f"Missing required fields for reward transaction")
                    return False
                
                # Check for duplicate hash
                tx_hash = transaction.get("hash")
                if tx_hash and self.is_transaction_mined(transaction):
                    self.logger.warning(f"Duplicate reward transaction hash: {tx_hash}")
                    return False
                
                # Allow reward transactions that match current OR previous block
                # (miners might be one block behind in their calculation)
                tx_block_height = transaction.get("block_height")
                if tx_block_height not in [block_index, block_index - 1, block_index + 1]:
                    self.logger.warning(f"Reward transaction block_height mismatch: {tx_block_height} vs current {block_index}")
                    return False
                
                # Verify reasonable reward amount
                reward_amount = transaction.get("amount", 0)
                if reward_amount <= 0 or reward_amount > 1000:  # Adjust max as needed
                    self.logger.warning(f"Invalid reward amount: {reward_amount}")
                    return False
                    
                self.logger.info(f"✅ Reward transaction validated: {reward_amount} to {transaction.get('to')}")
                return True
            
            # Genesis transaction (only in block 0)
            elif tx_type == "genesis":
                if block_index != 0:
                    self.logger.warning("Genesis transaction only allowed in block 0")
                    return False
                required = ["message", "timestamp", "hash"]
                if not all(field in transaction for field in required):
                    return False
                # Only one genesis transaction allowed
                for block in self.blockchain:
                    for tx in block.get("transactions", []):
                        if tx.get("type") == "genesis":
                            self.logger.warning("Genesis transaction already exists in blockchain")
                            return False
                return True
            
            # GTX_Genesis transactions (bills)
            elif tx_type == "GTX_Genesis":
                required = ["serial_number", "denomination", "issued_to", "timestamp", "signature", "public_key"]
                if not all(field in transaction for field in required):
                    return False
                
                # Check for double-spending of serial numbers
                serial = transaction.get("serial_number")
                if serial and serial in self.mined_serials:
                    self.logger.warning(f"Serial number already mined: {serial}")
                    return False
                
                return True
            
            # Transfer transactions
            elif tx_type == "transfer":
                required = ["from", "to", "amount", "timestamp", "signature"]
                if not all(field in transaction for field in required):
                    return False
                
                # Check if in mempool
                tx_hash = transaction.get("hash")
                if not tx_hash:
                    return False
                    
                in_mempool = any(tx.get("hash") == tx_hash for tx in self.mempool)
                if not in_mempool:
                    self.logger.warning(f"Transfer transaction not in mempool: {tx_hash[:16]}...")
                    return False
                return True
            
            # Unknown transaction type
            else:
                self.logger.warning(f"Unknown transaction type: {tx_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Transaction validation error: {e}")
            return False

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
            if tx.get("type") == "GTX_Genesis" and tx.get("serial_number"):
                self.mined_serials.add(tx["serial_number"])

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