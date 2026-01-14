# blockchain_daemon_modules/daemon.py
"""
Main BlockchainDaemon class that coordinates all modules
"""

import json
import time
import hashlib
import threading
import os
import logging
import traceback
from typing import List, Dict, Set, Optional

# Import lunalib components with fallback for different versions
logger = logging.getLogger(__name__)

# Try to import lunalib 1.6.9 components
LunalibDaemon = None
BlockchainManager = None
MempoolManager = None
GTXGenesis = None
DigitalBill = None
BillRegistry = None

try:
    # Try lunalib 1.6.9 API
    try:
        from lunalib.core.daemon import BlockchainDaemon as LunalibDaemon
        from lunalib.core.blockchain import BlockchainManager
        from lunalib.core.mempool import MempoolManager
        logger.info("✅ Loaded lunalib from lunalib.core.*")
    except ImportError:
        # Try alternative API structure
        try:
            from lunalib.blockchain import BlockchainManager
            from lunalib.mempool import MempoolManager
            logger.info("✅ Loaded lunalib from lunalib.*")
        except ImportError:
            logger.warning("⚠️ Could not import BlockchainManager/MempoolManager")
    
    # Try GTX components
    try:
        from lunalib.gtx.genesis import GTXGenesis
        from lunalib.gtx.digital_bill import DigitalBill
        from lunalib.gtx.bill_registry import BillRegistry
        logger.info("✅ Loaded GTX components from lunalib.gtx.*")
    except ImportError:
        try:
            from lunalib.genesis import GTXGenesis
            from lunalib.digital_bill import DigitalBill
            from lunalib.bill_registry import BillRegistry
            logger.info("✅ Loaded GTX components from lunalib.*")
        except ImportError:
            logger.warning("⚠️ Could not import GTX components")
except Exception as e:
    logger.warning(f"⚠️ Lunalib import error: {e}")

# Import our refactored modules
from . import validators
from . import persistence
from . import network
from . import blocks
from . import transactions


class BlockchainDaemon:
    """Main blockchain daemon class"""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 blockchain_file="blockchain_data/blockchain.json", 
                 mempool_file="mempool_data/mempool.json",
                 endpoint_url="https://bank.linglin.art"):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.blockchain_file = blockchain_file
        self.mempool_file = mempool_file
        self.endpoint_url = endpoint_url
        
        # Initialize lunalib managers
        self.blockchain_mgr = BlockchainManager(endpoint_url=endpoint_url)
        self.mempool_mgr = MempoolManager(network_endpoints=[endpoint_url])
        self.gtx_genesis = GTXGenesis()
        self.bill_registry = BillRegistry()
        
        # Local state
        self.blockchain = []
        self.mempool = []
        self.mined_serials: Set[str] = set()
        self.peers: Set[str] = set()  # Known peer nodes
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
    
    # ========== INITIALIZATION METHODS ==========
    
    def create_initial_files(self):
        """Create initial JSON files if they don't exist"""
        persistence.create_initial_files(self.blockchain_file, self.mempool_file)
    
    def load_data(self):
        """Load blockchain and mempool from files, create genesis if needed"""
        try:
            # Create directories if they don't exist
            blockchain_dir = os.path.dirname(self.blockchain_file)
            mempool_dir = os.path.dirname(self.mempool_file)
            
            if blockchain_dir and not os.path.exists(blockchain_dir):
                os.makedirs(blockchain_dir, exist_ok=True)
                self.logger.info(f"Created blockchain directory: {blockchain_dir}")
            
            if mempool_dir and not os.path.exists(mempool_dir):
                os.makedirs(mempool_dir, exist_ok=True)
                self.logger.info(f"Created mempool directory: {mempool_dir}")
            
            # Load blockchain
            self.blockchain = persistence.load_blockchain(self.blockchain_file)
            
            # Load mempool
            self.mempool = persistence.load_mempool(self.mempool_file)
            
            # Build mined indexes
            self.build_mined_indexes()
            self.logger.info(f"Built indexes for {len(self.mined_serials)} mined serials")
            
            # Check if we need to create genesis block
            if len(self.blockchain) == 0:
                self.logger.info("Blockchain is empty, checking if we should create genesis block...")
                self._create_and_add_genesis_block()
            else:
                self.logger.info(f"Blockchain already has {len(self.blockchain)} blocks")
            
            self.logger.info("Using local blockchain data on startup")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.error(traceback.format_exc())
            if not hasattr(self, 'blockchain'):
                self.blockchain = []
            if not hasattr(self, 'mempool'):
                self.mempool = []
    
    def _create_and_add_genesis_block(self):
        """Create and add genesis block if needed"""
        try:
            if self.blockchain:
                self.logger.info("Blockchain already exists, skipping genesis creation")
                return
            
            self.logger.info("Creating genesis block...")
            genesis_block = self._create_genesis_block()
            
            if genesis_block:
                self.blockchain.append(genesis_block)
                self.save_blockchain()
                self.logger.info(f"✅ Created genesis block with {len(genesis_block['transactions'])} transactions")
            else:
                self.logger.error("Failed to create genesis block")
                
        except Exception as e:
            self.logger.error(f"Error creating genesis block: {e}")
            self.logger.error(traceback.format_exc())
    
    def _create_genesis_block(self) -> Dict:
        """Create genesis block from sacred text and GTX_Genesis transactions"""
        sacred_text = "伊森林爱灵林。灵林爱伊森林。"
        
        # Create a secure genesis hash from sacred text
        genesis_data = {
            'sacred_text': sacred_text,
            'timestamp': time.time(),
            'version': '1.0.0',
            'message': 'Genesis Block - 伊森林爱灵林。灵林爱伊森林。'
        }
        
        # Calculate genesis hash using double SHA-256
        data_string = json.dumps(genesis_data, sort_keys=True)
        first_hash = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
        genesis_hash = hashlib.sha256(first_hash.encode('utf-8')).hexdigest()
        
        # Collect all GTX_Genesis transactions from mempool
        genesis_txs = [tx for tx in self.mempool if tx.get("type") == "GTX_Genesis"]
        
        # Add a special genesis transaction with the sacred text
        sacred_tx = {
            "type": "genesis",
            "from": "system",
            "to": "林灵森林",
            "amount": 0,
            "timestamp": time.time(),
            "block_height": 0,
            "hash": f"genesis_{genesis_hash[:32]}",
            "message": sacred_text,
            "description": "Genesis sacred text: Forest loves Spirit Forest. Spirit Forest loves Forest.",
            "signature": "GENESIS_SIGNATURE_伊森林爱灵林"
        }
        
        # Combine transactions (sacred text first, then GTX_Genesis)
        all_transactions = [sacred_tx] + genesis_txs
        
        # Create the genesis block
        genesis_block = {
            'index': 0,
            'previous_hash': '0' * 64,
            'timestamp': time.time(),
            'transactions': all_transactions,
            'miner': 'system',
            'difficulty': 1,
            'nonce': 0,
            'reward': 0,
            'merkleroot': self._calculate_merkle_root(all_transactions)
        }
        
        # Calculate block hash
        genesis_block['hash'] = self.calculate_block_hash(
            genesis_block['index'],
            genesis_block['previous_hash'],
            genesis_block['timestamp'],
            genesis_block['transactions'],
            genesis_block['nonce']
        )
        
        return genesis_block
    
    def _calculate_merkle_root(self, txs: List[Dict]) -> str:
        """Calculate Merkle root for transactions"""
        if not txs:
            return "0" * 64
        
        # Get transaction hashes
        tx_hashes = []
        for tx in txs:
            if 'hash' in tx:
                tx_hashes.append(tx['hash'])
            else:
                # If transaction doesn't have hash, create one
                tx_string = json.dumps(tx, sort_keys=True)
                tx_hashes.append(hashlib.sha256(tx_string.encode()).hexdigest())
        
        # Simple Merkle root calculation (pairwise hashing)
        while len(tx_hashes) > 1:
            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                if i + 1 < len(tx_hashes):
                    combined = tx_hashes[i] + tx_hashes[i + 1]
                else:
                    combined = tx_hashes[i] + tx_hashes[i]  # Duplicate last if odd
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_hashes.append(new_hash)
            tx_hashes = new_hashes
        
        return tx_hashes[0] if tx_hashes else "0" * 64
    
    # ========== PERSISTENCE METHODS ==========
    
    def save_blockchain(self):
        """Save blockchain to file"""
        return persistence.save_blockchain(self.blockchain, self.blockchain_file)
    
    def save_mempool(self):
        """Save mempool to file"""
        return persistence.save_mempool(self.mempool, self.mempool_file)
    
    # ========== NETWORK METHODS ==========
    
    def sync_with_network(self):
        """Sync with network blockchain"""
        return network.sync_with_network(
            self.blockchain, self.blockchain_mgr, self.mempool_mgr,
            self.validate_block, self.build_mined_indexes,
            self.save_blockchain, self.save_mempool
        )
    
    def get_network_blockchain_height(self) -> int:
        """Get the current blockchain height from network"""
        return network.get_network_blockchain_height(self.blockchain_mgr)
    
    def get_last_network_block_hash(self) -> str:
        """Get the hash of the last block from network"""
        return network.get_last_network_block_hash(self.blockchain_mgr)
    
    # ========== TRANSACTION METHODS ==========
    
    def add_transaction(self, transaction: Dict, broadcast: bool = True) -> bool:
        """Add a transaction to mempool"""
        result = transactions.add_transaction(
            transaction, self.mempool, self.validate_transaction_structure,
            self.is_transaction_mined, self.blockchain,
            self.mempool_mgr, self.save_mempool
        )
        
        # Only broadcast if it's a new transaction (not a duplicate)
        if result == True and broadcast:
            # Try to broadcast, but don't fail if network already has it
            self.logger.info(f"🔍 Broadcasting transaction to network...")
            broadcast_success = network.broadcast_transaction_to_network(transaction, self.mempool_mgr)
            if not broadcast_success:
                self.logger.warning(f"⚠️ Network broadcast failed (transaction may already be in network mempool)")
                # Don't fail the operation - transaction is in local mempool which is what matters
        elif result == "duplicate":
            self.logger.info(f"ℹ️ Transaction already in mempool, skipping broadcast")
        
        # Return True for both new additions and duplicates (both are success states)
        return result == True or result == "duplicate"
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Get transaction details by hash"""
        return transactions.get_transaction(
            self.mempool, self.blockchain, tx_hash,
            self.blockchain_mgr, self.mempool_mgr
        )
    
    def get_transactions_by_block(self, block_height: int) -> List[Dict]:
        """Get all transactions from a specific block"""
        return transactions.get_transactions_by_block(self.blockchain, block_height)
    
    def is_transaction_mined(self, transaction: Dict) -> bool:
        """Check if transaction has been mined"""
        return transactions.is_transaction_mined(transaction, self.blockchain)
    
    def get_mempool_status(self) -> Dict:
        """Get mempool status and statistics"""
        return transactions.get_mempool_status(self.mempool)
    
    def get_mempool_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Get a specific transaction from the mempool by hash"""
        try:
            for tx in self.mempool:
                if tx.get('hash') == tx_hash:
                    return {'found': True, 'transaction': tx, 'status': 'pending'}
            return {'found': False, 'status': 'not_in_mempool'}
        except Exception as e:
            logger.error(f"Error finding mempool transaction {tx_hash}: {e}")
            return {'found': False, 'status': 'error', 'error': str(e)}
    
    def get_blockchain_status(self) -> Dict:
        """Get blockchain status and statistics"""
        return transactions.get_blockchain_status(self.blockchain, self.mined_serials)
    
    def remove_mined_transactions(self, mined_transactions: List[Dict]) -> int:
        """Remove mined transactions from the mempool"""
        return transactions.remove_mined_transactions(
            self.mempool, mined_transactions, self.save_mempool
        )
    
    def build_mined_indexes(self):
        """Build indexes of all mined serial numbers"""
        transactions.build_mined_indexes(self.blockchain, self.mined_serials)
    
    def update_mined_indexes(self, block: Dict):
        """Update mined indexes with transactions from a new block"""
        transactions.update_mined_indexes(block, self.mined_serials)
    
    def add_genesis_transaction(self, serial_number: str, denomination: float, issued_to: str) -> bool:
        """Add a genesis transaction for a banknote"""
        return transactions.add_genesis_transaction(
            serial_number, denomination, issued_to,
            self.gtx_genesis, self.mempool, self.add_transaction
        )
    
    def generate_transaction_hash(self, transaction_data: Dict) -> str:
        """Generate a unique hash for any transaction"""
        return transactions.generate_transaction_hash(transaction_data)
    
    # NOTE: create_reward_transaction has been removed from daemon
    # Use mining.create_reward_transaction() instead for correct exponential rewards:
    #   from blockchain_daemon_modules import mining
    #   reward_tx = mining.create_reward_transaction(miner_address, block_height, difficulty, tx_count)
    
    def mark_reward_transactions_mined(self, reward_transactions, block_index=None):
        """Mark reward transactions as mined in the system"""
        return transactions.mark_reward_transactions_mined(
            reward_transactions, block_index, self.mempool_mgr
        )
    
    # ========== VALIDATION METHODS ==========
    
    def validate_transaction_structure(self, transaction: Dict) -> bool:
        """Validate transaction structure"""
        return validators.validate_transaction_structure(transaction)
    
    def validate_transaction_for_block(self, transaction: Dict, block_index: int) -> bool:
        """Validate transaction for inclusion in block"""
        return validators.validate_transaction_for_block(
            transaction, block_index, self.mempool, self.mined_serials,
            self.gtx_genesis, self.is_transaction_mined
        )
    
    def validate_block(self, block: Dict) -> bool:
        """Validate a mined block"""
        return validators.validate_block(
            block, self.blockchain, 
            self.blockchain_mgr._validate_block_structure,
            self.validate_transaction_for_block
        )
    
    def validate_block_for_submission(self, block: Dict) -> Dict:
        """Comprehensive validation for block submission"""
        return validators.validate_block_for_submission(
            block, self.get_network_blockchain_height,
            self.get_last_network_block_hash, self.calculate_block_hash
        )
    
    def validate_regular_transactions(self, txs: List[Dict]) -> Dict:
        """Validate regular (non-reward) transactions"""
        return validators.validate_regular_transactions(txs)
    
    def validate_reward_transactions(self, reward_txs: List[Dict], block_index: int, 
                                    block_data: Dict, previous_block_hash: str) -> Dict:
        """Validate reward transactions with mining proof"""
        return validators.validate_reward_transactions(
            reward_txs, block_index, block_data, previous_block_hash,
            self.is_transaction_mined, self.mempool_mgr
        )
    
    def validate_mining_proof(self, block_data: Dict, previous_block_hash: str) -> Dict:
        """Public wrapper for mining proof validation"""
        block_hash = block_data.get('hash', '')
        difficulty = block_data.get('difficulty', 0)
        all_transactions = block_data.get('transactions', [])
        non_reward_txs = [tx for tx in all_transactions if tx.get('type') != 'reward']
        
        result = validators.validate_mining_proof_internal(
            block_hash, difficulty, block_data, previous_block_hash, non_reward_txs
        )
        
        # Format for compatibility
        if result['valid']:
            return {
                'valid': True,
                'difficulty': difficulty,
                'block_hash': block_hash,
                'validation_method': result.get('method', 'unknown')
            }
        else:
            return {
                'valid': False,
                'error': result.get('error', 'Unknown validation error')
            }
    
    def calculate_block_hash(self, index: int, previous_hash: str, timestamp: float,
                            txs: List[Dict], nonce: int) -> str:
        """Calculate SHA-256 hash of a block"""
        return validators.calculate_block_hash(index, previous_hash, timestamp, txs, nonce)
    
    # ========== BLOCK METHODS ==========
    
    def is_block_already_in_chain(self, block_or_hash, *args, **kwargs):
        """Check if a block with the given hash already exists in the blockchain"""
        # Handle both block object or hash value
        if isinstance(block_or_hash, dict):
            block_hash = block_or_hash.get('hash')
        else:
            block_hash = block_or_hash
        
        # Ignore extra arguments for compatibility
        for block in self.blockchain:
            if block.get('hash') == block_hash:
                return True
        return False
    
    def is_correct_block_sequence(self, block: Dict) -> Dict:
        """Check if the block is in correct sequence with the blockchain - returns detailed result"""
        result = {
            "valid": True,
            "errors": [],
            "expected_index": None,
            "expected_previous_hash": None,
            "network_height": None,
            "local_height": None
        }
        
        try:
            # Get current blockchain height
            local_height = len(self.blockchain)
            result["local_height"] = local_height
            
            # Expected index is local height (0-indexed)
            expected_index = local_height
            result["expected_index"] = expected_index
            
            # Get expected previous hash
            if local_height > 0:
                expected_previous_hash = self.blockchain[-1].get('hash')
            else:
                expected_previous_hash = '0' * 64
            result["expected_previous_hash"] = expected_previous_hash
            
            # Check block index
            block_index = block.get('index')
            if block_index != expected_index:
                result["valid"] = False
                result["errors"].append(f"Block index mismatch: expected {expected_index}, got {block_index}")
            
            # Check previous hash
            block_previous_hash = block.get('previous_hash')
            if block_previous_hash != expected_previous_hash:
                result["valid"] = False
                result["errors"].append(f"Previous hash mismatch: expected {expected_previous_hash[:16]}..., got {block_previous_hash[:16]}...")
            
            # Try to get network height
            try:
                network_height = self.get_network_blockchain_height()
                result["network_height"] = network_height
                
                # Warn if we're ahead of network
                if local_height > network_height:
                    result["errors"].append(f"Local blockchain ahead of network: local={local_height}, network={network_height}")
            except:
                pass
            
            return result
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Error checking sequence: {e}")
            return result
    
    def get_block(self, block_identifier) -> Optional[Dict]:
        """Get block by hash, height, or 'latest'/'genesis'"""
        return blocks.get_block(
            self.blockchain, block_identifier, self.mempool_mgr,
            lambda h: network.get_block_from_network(h, self.blockchain_mgr, self.validate_block)
        )
    
    def get_block_range(self, start: int, end: int) -> List[Dict]:
        """Get a range of blocks"""
        return blocks.get_block_range(self.blockchain, start, end)
    
    def get_latest_blocks(self, count: int = 10) -> List[Dict]:
        """Get the latest N blocks"""
        return blocks.get_latest_blocks(self.blockchain, count)
    
    def get_block_by_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Find block containing a specific transaction"""
        return blocks.get_block_by_transaction(
            self.blockchain, tx_hash, self.mempool_mgr, self.blockchain_mgr
        )
    
    def get_blockchain_stats(self) -> Dict:
        """Get blockchain statistics"""
        return blocks.get_blockchain_stats(self.blockchain)
    
    def get_previous_hash(self) -> str:
        """Retrieve the hash of the last block"""
        return blocks.get_previous_hash(self.blockchain)
    
    # ========== BLOCK SUBMISSION METHODS ==========
    
    def submit_block(self, block: Dict) -> bool:
        """Submit a mined block to the network"""
        previous_hash = self.get_previous_hash()
        block['previous_hash'] = previous_hash
        self.logger.info(f"Submitting block #{block['index']} with previous hash: {previous_hash[:16]}...")
        return network.submit_block_to_network(block, self.blockchain_mgr)
    
    def submit_block_with_validation(self, block: Dict) -> bool:
        """Submit a block with comprehensive validation"""
        try:
            # 1. Comprehensive validation
            validation = self.validate_block_for_submission(block)
            
            if not validation["valid"]:
                self.logger.error("Block validation failed:")
                for error in validation["errors"]:
                    self.logger.error(f"  - {error}")
                return False
            
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    self.logger.warning(f"  - {warning}")
            
            # 2. Show validation details
            self.logger.info(f"✅ Block #{block.get('index')} validation passed")
            self.logger.info(f"   Hash: {block.get('hash', '')[:16]}...")
            self.logger.info(f"   Previous hash: {block.get('previous_hash', '')[:16]}...")
            self.logger.info(f"   Difficulty: {block.get('difficulty', 1)}")
            self.logger.info(f"   Transactions: {len(block.get('transactions', []))}")
            
            # 3. Get latest network state (confirm again)
            self.sync_with_network()
            network_height = self.get_network_blockchain_height()
            self.logger.info(f"   Network height before submission: {network_height}")
            
            # 4. Submit to network
            if self.mempool_mgr.test_connection():
                submission_result = network.submit_block_to_network(block, self.blockchain_mgr)
                
                if submission_result:
                    self.logger.info(f"✅ Successfully submitted block #{block.get('index')}")
                    # Add to local blockchain
                    self.blockchain.append(block)
                    self.update_mined_indexes(block)
                    self.save_blockchain()
                    return True
                else:
                    self.logger.error(f"❌ Failed to submit block #{block.get('index')}")
                    return False
            else:
                self.logger.error("Network connection unavailable")
                return False
                
        except Exception as e:
            self.logger.error(f"Error submitting block: {e}")
            self.logger.error(traceback.format_exc())
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
            
            # Submit to network if available
            if self.mempool_mgr.test_connection():
                network.submit_block_to_network(block, self.blockchain_mgr)
            
            self.logger.info(f"Added block #{block['index']} with {len(block['transactions'])} transactions")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding block: {e}")
            return False
    
    # ========== DAEMON METHODS ==========
    
    def start_daemon(self):
        """Start the blockchain daemon"""
        self.logger.info("🚀 Starting Blockchain Daemon...")
        self.is_running = True
        threading.Thread(target=self.daemon_loop, daemon=True).start()
        self.logger.info("✅ Blockchain Daemon is running")
    
    def daemon_loop(self):
        """Main loop for the blockchain daemon"""
        while self.is_running:
            time.sleep(self.sync_interval)
    
    def sync_all(self):
        """Synchronize all blockchain data"""
        self.logger.info("🔄 Synchronizing blockchain data...")
        self.sync_with_network()
        self.logger.info("✅ Synchronization complete.")
    
    # ========== PEER MANAGEMENT METHODS ==========
    
    def register_peer(self, peer_info: Dict) -> Dict:
        """Register a new peer node in the network"""
        try:
            if not peer_info or 'peer_url' not in peer_info:
                return {
                    'success': False,
                    'error': 'Missing peer_url in request'
                }
            
            peer_url = peer_info['peer_url'].strip()
            
            # Validate URL format
            if not peer_url.startswith(('http://', 'https://')):
                return {
                    'success': False,
                    'error': 'Invalid peer URL format (must start with http:// or https://)'
                }
            
            # Remove trailing slash
            if peer_url.endswith('/'):
                peer_url = peer_url[:-1]
            
            # Check if already exists
            if peer_url in self.peers:
                return {
                    'success': False,
                    'error': 'Peer already registered'
                }
            
            # Add to set
            self.peers.add(peer_url)
            self.logger.info(f"✅ Registered peer: {peer_url}")
            
            return {
                'success': True,
                'message': 'Peer registered successfully',
                'peer_url': peer_url
            }
            
        except Exception as e:
            self.logger.error(f"Error registering peer: {e}")
            return {
                'success': False,
                'error': f'Error registering peer: {str(e)}'
            }
    
    def get_peers_info(self) -> Dict:
        """Get list of all registered peers"""
        try:
            return {
                'success': True,
                'peers': list(self.peers),
                'count': len(self.peers)
            }
        except Exception as e:
            self.logger.error(f"Error getting peers: {e}")
            return {
                'success': False,
                'error': f'Error getting peers: {str(e)}'
            }
    
    def remove_peer_by_url(self, peer_url: str) -> Dict:
        """Remove a peer from the network"""
        try:
            peer_url = peer_url.strip()
            
            # Remove trailing slash
            if peer_url.endswith('/'):
                peer_url = peer_url[:-1]
            
            if peer_url in self.peers:
                self.peers.remove(peer_url)
                self.logger.info(f"✅ Removed peer: {peer_url}")
                return {
                    'success': True,
                    'message': 'Peer removed successfully',
                    'peer_url': peer_url
                }
            else:
                self.logger.warning(f"Peer not found: {peer_url}")
                return {
                    'success': False,
                    'error': 'Peer not found'
                }
                
        except Exception as e:
            self.logger.error(f"Error removing peer: {e}")
            return {
                'success': False,
                'error': f'Error removing peer: {str(e)}'
            }
    
    def get_peers(self) -> List[str]:
        """Get list of all registered peers (simple list)"""
        return list(self.peers)
    
    def clear_peers(self):
        """Clear all registered peers"""
        self.peers.clear()
        self.logger.info("Cleared all peers")
