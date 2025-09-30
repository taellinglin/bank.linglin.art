# Add to your app.py
import hashlib
import time
import json
from typing import List, Dict
import os

class BlockchainManager:
    def __init__(self, blockchain_file="blockchain.json", mempool_file="mempool.json"):
        self.blockchain_file = blockchain_file
        self.mempool_file = mempool_file
        self.blockchain = self.load_blockchain()
        self.mempool = self.load_mempool()
    
    def load_blockchain(self):
        """Load blockchain or create genesis block"""
        try:
            if os.path.exists(self.blockchain_file):
                with open(self.blockchain_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        
        # Create genesis block
        genesis = self.create_genesis_block()
        blockchain = [genesis]
        self.save_blockchain(blockchain)
        return blockchain
    
    def load_mempool(self):
        """Load mempool of available bills to mine"""
        try:
            if os.path.exists(self.mempool_file):
                with open(self.mempool_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def create_genesis_block(self):
        return {
            "index": 0,
            "timestamp": time.time(),
            "transactions": [{
                "type": "genesis",
                "message": "Luna Coin Genesis Block",
                "timestamp": time.time()
            }],
            "previous_hash": "0",
            "nonce": 0,
            "hash": self.calculate_block_hash(0, "0", time.time(), [], 0)
        }
    
    def calculate_block_hash(self, index, previous_hash, timestamp, transactions, nonce):
        block_string = f"{index}{previous_hash}{timestamp}{json.dumps(transactions, sort_keys=True)}{nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def get_available_bills_to_mine(self):
        """Get bills from mempool that haven't been mined yet"""
        mined_serials = self.get_mined_serial_numbers()
        available_bills = []
        
        for tx in self.mempool:
            if tx.get("type") == "GTX_Genesis":
                serial = tx.get("serial_number")
                if serial and serial not in mined_serials:
                    available_bills.append(tx)
        
        return available_bills
    
    def get_mined_serial_numbers(self):
        """Get set of all serial numbers that have been mined"""
        mined_serials = set()
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                if tx.get("type") in ["GTX_Genesis", "reward"]:
                    serial = tx.get("serial_number")
                    if serial:
                        mined_serials.add(serial)
        return mined_serials
    
    def add_new_transaction(self, transaction: Dict):
        """Add a new transaction to mempool (new bill to mine)"""
        # Validate transaction
        if not self.validate_transaction(transaction):
            return False
        
        # Check if already in mempool
        tx_id = transaction.get("signature") or transaction.get("serial_number")
        for existing_tx in self.mempool:
            existing_id = existing_tx.get("signature") or existing_tx.get("serial_number")
            if existing_id == tx_id:
                return False  # Already exists
        
        self.mempool.append(transaction)
        self.save_mempool()
        return True
    
    def validate_transaction(self, transaction: Dict):
        """Validate a new transaction"""
        if not transaction.get("type"):
            return False
        
        if transaction.get("type") == "GTX_Genesis":
            return all(key in transaction for key in ["serial_number", "denomination", "issued_to"])
        
        elif transaction.get("type") == "transfer":
            return all(key in transaction for key in ["from", "to", "amount", "signature"])
        
        return False
    
    def mine_block(self, miner_address: str, difficulty: int = 4):
        """Mine a new block with available transactions"""
        available_bills = self.get_available_bills_to_mine()
        if not available_bills:
            return None  # Nothing to mine
        
        previous_block = self.blockchain[-1]
        new_index = previous_block["index"] + 1
        
        # Select transactions to include (you can add limits based on block size)
        transactions_to_mine = available_bills[:10]  # Mine up to 10 bills at once
        
        # Create mining reward transaction
        reward_tx = self.create_reward_transaction(miner_address, len(transactions_to_mine))
        
        # All transactions for this block
        block_transactions = transactions_to_mine + [reward_tx]
        
        # Create new block (simplified - real mining would involve PoW)
        new_block = {
            "index": new_index,
            "timestamp": time.time(),
            "transactions": block_transactions,
            "previous_hash": previous_block["hash"],
            "nonce": 0,  # In real mining, this would be found through PoW
            "miner": miner_address,
            "difficulty": difficulty
        }
        
        # Calculate block hash
        new_block["hash"] = self.calculate_block_hash(
            new_block["index"],
            new_block["previous_hash"],
            new_block["timestamp"],
            new_block["transactions"],
            new_block["nonce"]
        )
        
        # Add to blockchain
        self.blockchain.append(new_block)
        
        # Remove mined transactions from mempool
        self.remove_mined_transactions(transactions_to_mine)
        
        self.save_blockchain(self.blockchain)
        self.save_mempool(self.mempool)
        
        print(f"✅ Mined block #{new_index} with {len(transactions_to_mine)} bills")
        return new_block
    
    def create_reward_transaction(self, miner_address: str, bill_count: int):
        """Create mining reward transaction"""
        base_reward = 50  # Base reward per block
        total_reward = base_reward * bill_count  # Reward based on bills mined
        
        return {
            "type": "reward",
            "to": miner_address,
            "amount": total_reward,
            "timestamp": time.time(),
            "block_height": len(self.blockchain) + 1,
            "description": f"Mining reward for {bill_count} bills"
        }
    
    def remove_mined_transactions(self, mined_transactions: List[Dict]):
        """Remove mined transactions from mempool"""
        mined_serials = set()
        for tx in mined_transactions:
            serial = tx.get("serial_number")
            if serial:
                mined_serials.add(serial)
        
        self.mempool = [tx for tx in self.mempool 
                       if tx.get("serial_number") not in mined_serials]
    
    def save_blockchain(self, blockchain):
        with open(self.blockchain_file, 'w') as f:
            json.dump(blockchain, f, indent=2)
    
    def save_mempool(self, mempool=None):
        if mempool is None:
            mempool = self.mempool
        with open(self.mempool_file, 'w') as f:
            json.dump(mempool, f, indent=2)

