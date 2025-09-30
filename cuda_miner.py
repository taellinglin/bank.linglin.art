#!/usr/bin/env python3
"""
cuda_miner.py - GPU Mining with CUDA
"""

import requests
import time
import hashlib
import json
import threading
from typing import List, Dict
import logging

try:
    import cupy as cp
    import numpy as np
    CUDA_AVAILABLE = True
    print("✅ CUDA available - using GPU mining")
except ImportError:
    CUDA_AVAILABLE = False
    print("❌ CUDA not available - falling back to CPU mining")

class CUDAMiner:
    def __init__(self, base_url: str = "https://bank.linglin.art", miner_address: str = "cuda_miner"):
        self.base_url = base_url
        self.miner_address = miner_address
        self.is_mining = False
        self.current_difficulty = 4
        self.hash_rate = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger("cuda_miner")
        
    def get_mining_candidate(self) -> Dict:
        """Get block candidate for mining"""
        try:
            # Get blockchain status to build mining candidate
            status_resp = requests.get(f"{self.base_url}/api/blockchain/status", timeout=10)
            if status_resp.status_code != 200:
                return None
            
            status_data = status_resp.json()
            blockchain_height = status_data.get("blockchain_height", 0)
            
            # Get available transactions
            mempool_resp = requests.get(f"{self.base_url}/mempool/status", timeout=10)
            if mempool_resp.status_code != 200:
                return None
                
            mempool_data = mempool_resp.json()
            active_tx = mempool_data.get("active_transactions", 0)
            
            if active_tx == 0:
                self.logger.info("⏳ No transactions available for mining")
                return None
            
            # Get the latest block for previous_hash
            blockchain_resp = requests.get(f"{self.base_url}/blockchain", timeout=10)
            if blockchain_resp.status_code != 200:
                return None
                
            blockchain = blockchain_resp.json()
            if not blockchain:
                self.logger.error("❌ Blockchain is empty - cannot mine")
                return None
            
            latest_block = blockchain[-1]
            
            # Create mining candidate
            candidate = {
                "index": blockchain_height,
                "timestamp": int(time.time()),
                "transactions": [],  # We'll get these from the mining endpoint
                "previous_hash": latest_block["hash"],
                "nonce": 0,
                "miner": self.miner_address,
                "difficulty": self.current_difficulty,
                "hash": ""
            }
            
            return candidate
            
        except Exception as e:
            self.logger.error(f"Error getting mining candidate: {e}")
            return None
    
    def cuda_mine_block(self, candidate: Dict) -> Dict:
        """Mine block using CUDA"""
        target = "0" * candidate["difficulty"]
        start_time = time.time()
        hashes_calculated = 0
        
        if CUDA_AVAILABLE:
            # GPU mining with CuPy
            return self._cuda_gpu_mining(candidate, target, start_time)
        else:
            # Fallback to CPU mining
            return self._cpu_mining(candidate, target, start_time)
    
    def _cuda_gpu_mining(self, candidate: Dict, target: str, start_time: float) -> Dict:
        """GPU mining implementation"""
        self.logger.info("🚀 Starting GPU mining with CUDA")
        
        # Prepare block data for hashing (excluding nonce and hash)
        block_data = {
            'index': candidate['index'],
            'previous_hash': candidate['previous_hash'],
            'timestamp': candidate['timestamp'],
            'transactions': json.dumps(candidate['transactions'], sort_keys=True),
            'difficulty': candidate['difficulty']
        }
        block_string = json.dumps(block_data, sort_keys=True, separators=(',', ':')).encode()
        
        batch_size = 100000  # Hashes per batch
        found_nonce = None
        
        while not self.is_mining and found_nonce is None:
            # Generate batch of nonces on GPU
            nonces = cp.arange(candidate['nonce'], candidate['nonce'] + batch_size, dtype=cp.uint32)
            
            # Prepare data for batch hashing
            # This is a simplified version - real implementation would be more complex
            hashes = []
            for i in range(batch_size):
                data = block_string + str(int(nonces[i])).encode()
                hash_val = hashlib.sha256(data).hexdigest()
                hashes.append(hash_val)
                
                if hash_val.startswith(target):
                    found_nonce = int(nonces[i])
                    break
            
            candidate['nonce'] += batch_size
            
            # Update hash rate
            elapsed = time.time() - start_time
            if elapsed > 1.0:  # Update every second
                self.hash_rate = candidate['nonce'] / elapsed
                self.logger.info(f"⛏️ Hash rate: {self.hash_rate:,.0f} H/s")
        
        if found_nonce is not None:
            candidate['nonce'] = found_nonce
            candidate['hash'] = hashes[found_nonce - (candidate['nonce'] - batch_size)]
            mining_time = time.time() - start_time
            candidate['mining_time'] = mining_time
            
            self.logger.info(f"✅ GPU mined block in {mining_time:.2f}s with nonce {found_nonce}")
            self.logger.info(f"📊 Final hash: {candidate['hash']}")
            
            return candidate
        
        return None
    
    def _cpu_mining(self, candidate: Dict, target: str, start_time: float) -> Dict:
        """CPU mining fallback"""
        self.logger.info("💻 Starting CPU mining (CUDA not available)")
        
        while not self.is_mining:
            # Calculate hash
            calculated_hash = self.calculate_block_hash(
                candidate["index"],
                candidate["previous_hash"], 
                candidate["timestamp"],
                candidate["transactions"],
                candidate["nonce"],
                candidate["difficulty"]
            )
            
            # Check if hash meets target
            if calculated_hash.startswith(target):
                candidate["hash"] = calculated_hash
                mining_time = time.time() - start_time
                candidate["mining_time"] = mining_time
                
                self.logger.info(f"✅ CPU mined block in {mining_time:.2f}s with nonce {candidate['nonce']}")
                self.logger.info(f"📊 Final hash: {calculated_hash}")
                return candidate
            
            candidate["nonce"] += 1
            
            # Update progress
            if candidate["nonce"] % 100000 == 0:
                elapsed = time.time() - start_time
                self.hash_rate = candidate["nonce"] / elapsed
                self.logger.info(f"⛏️ CPU: {candidate['nonce']:,.0f} hashes, {self.hash_rate:,.0f} H/s")
        
        return None
    
    def calculate_block_hash(self, index, previous_hash, timestamp, transactions, nonce, difficulty=None):
        """Calculate block hash (same as blockchain_daemon.py)"""
        if isinstance(timestamp, float):
            timestamp = int(timestamp)
        
        index = int(index)
        nonce = int(nonce)
        
        if transactions:
            sorted_transactions = sorted(transactions, key=lambda x: x.get('hash', ''))
            transactions_string = json.dumps(sorted_transactions, sort_keys=True, separators=(',', ':'))
        else:
            transactions_string = "[]"
        
        block_data = {
            'index': index,
            'previous_hash': previous_hash,
            'timestamp': timestamp,
            'transactions': transactions_string,
            'nonce': nonce
        }
        
        if difficulty is not None:
            block_data['difficulty'] = int(difficulty)
        
        block_string = json.dumps(block_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def submit_block(self, block: Dict) -> bool:
        """Submit mined block to the network"""
        try:
            # Use the mine endpoint to submit the block
            payload = {
                "miner_address": self.miner_address,
                "block": block
            }
            
            resp = requests.post(f"{self.base_url}/api/block/mine", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    self.logger.info(f"✅ Block #{block['index']} submitted successfully!")
                    return True
                else:
                    self.logger.error(f"❌ Block submission failed: {data.get('message')}")
            else:
                self.logger.error(f"❌ HTTP {resp.status_code}: {resp.text}")
                
        except Exception as e:
            self.logger.error(f"💥 Error submitting block: {e}")
        
        return False
    
    def start_mining(self):
        """Start the mining loop"""
        self.is_mining = True
        self.logger.info(f"🏁 Starting CUDA miner: {self.miner_address}")
        
        while self.is_mining:
            try:
                # Get mining candidate
                candidate = self.get_mining_candidate()
                if not candidate:
                    self.logger.debug("⏳ Waiting for mining candidate...")
                    time.sleep(5)
                    continue
                
                # Mine the block
                self.logger.info(f"⛏️ Mining block #{candidate['index']} with difficulty {candidate['difficulty']}")
                mined_block = self.cuda_mine_block(candidate)
                
                if mined_block:
                    # Submit the block
                    if self.submit_block(mined_block):
                        self.logger.info("💰 Mining reward earned!")
                    else:
                        self.logger.warning("⚠️ Block submission failed")
                
                # Brief pause between mining attempts
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"💥 Mining error: {e}")
                time.sleep(10)
    
    def stop_mining(self):
        """Stop mining"""
        self.is_mining = False
        self.logger.info("🛑 Stopping CUDA miner")

def main():
    """Main function to start CUDA mining"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CUDA Miner for LingCountry Treasury')
    parser.add_argument('--miner', type=str, default='cuda_miner_001', help='Miner address')
    parser.add_argument('--url', type=str, default='https://bank.linglin.art', help='Base URL')
    
    args = parser.parse_args()
    
    miner = CUDAMiner(base_url=args.url, miner_address=args.miner)
    
    try:
        miner.start_mining()
    except KeyboardInterrupt:
        miner.stop_mining()
        print("\n👋 Mining stopped by user")

if __name__ == "__main__":
    main()