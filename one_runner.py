#!/usr/bin/env python3
"""
continuous_first_miner.py
Continuously tries to mine the first real blocks - FIXED VERSION
"""

import requests
import time
import logging

BASE_URL = "https://bank.linglin.art/"

# FIXED LOGGING FORMAT - changed 'asmaize' to 'asctime'
logging.basicConfig(level=logging.INFO, format='%(asctime)s [CONTINUOUS] %(message)s')
logger = logging.getLogger("continuous_miner")

class ContinuousFirstMiner:
    def __init__(self):
        self.miner_id = 1
        self.attempts = 0
        
    def create_test_transaction(self):
        """Create a test transaction to mine"""
        tx = {
            "type": "GTX_Genesis",
            "serial_number": f"SN-CONT-MINER-{self.attempts:03d}",
            "denomination": 500.0,
            "issued_to": f"Continuous_Miner_{self.miner_id}",
            "timestamp": int(time.time()),
            "description": f"Continuous miner test #{self.attempts}"
        }
        
        try:
            resp = requests.post(f"{BASE_URL}/api/transaction/genesis", 
                               json=tx, timeout=10)
            if resp.status_code == 200:
                logger.info(f"✅ Added transaction #{self.attempts}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to add transaction: {e}")
        return False
    
    def mine_continuously(self):
        """Continuous mining attempts"""
        logger.info("🔄 Starting continuous mining...")
        
        while True:
            self.attempts += 1
            logger.info(f"🔄 Attempt #{self.attempts}")
            
            # Try to create a transaction first
            if self.attempts % 3 == 1:  # Add transaction every 3 attempts
                self.create_test_transaction()
            
            # Then try to mine
            try:
                payload = {"miner_address": f"continuous_miner_{self.miner_id}"}
                resp = requests.post(f"{BASE_URL}/api/block/mine", json=payload, timeout=30)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "success" and data.get("block"):
                        block = data["block"]
                        logger.info(f"🎉 MINED BLOCK #{block.get('index')}!")
                        logger.info(f"📦 Transactions: {len(block.get('transactions', []))}")
                        
                        # Log transaction details
                        for tx in block.get('transactions', []):
                            logger.info(f"   - {tx.get('type')}: {tx.get('serial_number', tx.get('hash', 'unknown'))[:20]}...")
                        
                        # Continue mining for more blocks
                        logger.info("🔄 Continuing to mine more blocks...")
                    else:
                        logger.info(f"⏳ {data.get('message', 'No block mined')}")
                else:
                    logger.info(f"⏳ HTTP {resp.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Mining error: {e}")
            
            # Wait before next attempt
            time.sleep(3)

if __name__ == "__main__":
    miner = ContinuousFirstMiner()
    miner.mine_continuously()