#!/usr/bin/env python3
"""
block.py - Block class for LinKoin blockchain
"""
import json
import hashlib
import time
from typing import List, Dict, Set
class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, transactions: List[Dict], nonce: int = 0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = nonce
        self.hash = self.calculate_hash()
        self.mining_time = 0
        self.hash_rate = 0
        self.current_mining_hashes = 0
        self.current_hash_rate = 0
        self.current_hash = ""

    def calculate_hash(self) -> str:
        block_data = f"{self.index}{self.previous_hash}{self.timestamp}{json.dumps(self.transactions)}{self.nonce}"
        return hashlib.sha256(block_data.encode()).hexdigest()

    def mine_block(self, difficulty: int, progress_callback=None):
        """Mine the block with real-time progress tracking"""
        target = "0" * difficulty
        start_time = time.time()
        hashes_tried = 0
        last_update = start_time
        
        print(f"🎯 Mining Target: {target}")
        print("⛏️  Starting mining operation...")
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            hashes_tried += 1
            self.hash = self.calculate_hash()
            
            current_time = time.time()
            if current_time - last_update >= 1.0:  # Update every second
                elapsed = current_time - start_time
                self.hash_rate = hashes_tried / elapsed if elapsed > 0 else 0
                
                if progress_callback:
                    progress_callback({
                        'hashes': hashes_tried,
                        'hash_rate': self.hash_rate,
                        'current_hash': self.hash,
                        'elapsed_time': elapsed
                    })
                
                last_update = current_time
                hashes_tried = 0
        
        end_time = time.time()
        self.mining_time = end_time - start_time
        self.hash_rate = self.nonce / self.mining_time if self.mining_time > 0 else 0
        
        return True