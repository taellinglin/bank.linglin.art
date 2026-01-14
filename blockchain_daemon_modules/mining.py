# blockchain_daemon_modules/mining.py
"""
Mining operations for blockchain with async support
"""

import json
import hashlib
import time
import logging
import asyncio
from typing import Dict, List, Optional, Coroutine

logger = logging.getLogger(__name__)

# Mining constants
BASE_REWARD = 1.0
MAX_MINING_DIFFICULTY = 9
MINING_TIMEOUT = 60.0  # seconds


def create_reward_transaction(miner_address: str, block_height: int, 
                              difficulty: int, transaction_count: int = 0) -> Dict:
    """
    Create a reward transaction with linear difficulty-based rewards
    
    Formula: BASE_REWARD * difficulty
    - Difficulty 1: 1.0 LKC
    - Difficulty 2: 2.0 LKC
    - Difficulty 3: 3.0 LKC
    - Difficulty 4: 4.0 LKC
    - Difficulty 5: 5.0 LKC
    - Difficulty 6: 6.0 LKC
    - Difficulty 7: 7.0 LKC
    - Difficulty 8: 8.0 LKC
    - Difficulty 9: 9.0 LKC
    """
    total_reward = BASE_REWARD * difficulty
    
    logger.info(f"💰 Creating reward: {BASE_REWARD} * {difficulty} = {total_reward} LKC")
    
    reward_tx = {
        "type": "reward",
        "from": "MINING_REWARD",
        "to": miner_address,
        "amount": total_reward,
        "timestamp": int(time.time()),
        "block_height": block_height,
        "transaction_count": transaction_count,
        "difficulty": difficulty,
        "hash": ""  # Will be calculated
    }
    
    # Calculate hash for the reward transaction
    reward_string = json.dumps(reward_tx, sort_keys=True)
    reward_tx["hash"] = hashlib.sha256(reward_string.encode()).hexdigest()
    
    logger.info(f"✅ Reward transaction created: {reward_tx['hash'][:16]}...")
    return reward_tx


def calculate_expected_reward(difficulty: int, fees: float = 0.0, 
                              empty_block: bool = False) -> float:
    """
    Calculate expected mining reward based on difficulty and fees
    
    Args:
        difficulty: Mining difficulty (1-9)
        fees: Total transaction fees in the block
        empty_block: Whether this is an empty block
    
    Returns:
        Expected reward amount in LKC
    """
    BASE_REWARD = 1.0
    
    if empty_block or fees == 0:
        # Empty block: linear base reward
        return BASE_REWARD * difficulty
    else:
        # Regular block: linear base reward + fees
        return (BASE_REWARD * difficulty) + fees


def mine_block(block: Dict, difficulty: int, max_iterations: int = 1000000) -> Optional[Dict]:
    """
    Perform proof-of-work mining on a block
    
    Args:
        block: Block data to mine (must include index, previous_hash, timestamp, transactions)
        difficulty: Number of leading zeros required in hash
        max_iterations: Maximum number of nonce attempts
    
    Returns:
        Mined block with valid hash and nonce, or None if mining failed
    """
    logger.info(f"🔨 Starting mining: difficulty {difficulty}, max iterations {max_iterations}")
    
    nonce = 0
    target = '0' * difficulty
    
    # Prepare block data
    mining_data = {
        'index': block['index'],
        'previous_hash': block['previous_hash'],
        'timestamp': block['timestamp'],
        'transactions': block.get('transactions', []),
        'nonce': 0
    }
    
    start_time = time.time()
    
    for nonce in range(max_iterations):
        mining_data['nonce'] = nonce
        
        # Calculate hash
        block_string = json.dumps(mining_data, sort_keys=True, separators=(',', ':'))
        block_hash = hashlib.sha256(block_string.encode()).hexdigest()
        
        # Check if hash meets difficulty requirement
        if block_hash.startswith(target):
            elapsed = time.time() - start_time
            logger.info(f"✅ Block mined! Nonce: {nonce}, Time: {elapsed:.2f}s")
            logger.info(f"   Hash: {block_hash}")
            
            # Return mined block
            block['nonce'] = nonce
            block['hash'] = block_hash
            block['mining_time'] = elapsed
            return block
        
        # Log progress every 100k iterations
        if nonce > 0 and nonce % 100000 == 0:
            elapsed = time.time() - start_time
            rate = nonce / elapsed if elapsed > 0 else 0
            logger.debug(f"   Mining... {nonce:,} attempts, {rate:,.0f} H/s")
    
    logger.error(f"❌ Mining failed after {max_iterations:,} attempts")
    return None


def prepare_block_for_mining(blockchain: List[Dict], transactions: List[Dict],
                             miner_address: str, difficulty: int) -> Dict:
    """
    Prepare a new block for mining
    
    Args:
        blockchain: Current blockchain
        transactions: Transactions to include in block
        miner_address: Address of the miner
        difficulty: Mining difficulty
    
    Returns:
        Block ready for mining
    """
    # Get previous block
    if blockchain:
        previous_block = blockchain[-1]
        block_index = len(blockchain)
        previous_hash = previous_block['hash']
    else:
        block_index = 0
        previous_hash = '0' * 64
    
    # Create block
    new_block = {
        'index': block_index,
        'timestamp': int(time.time()),
        'transactions': transactions,
        'previous_hash': previous_hash,
        'nonce': 0,
        'miner': miner_address,
        'difficulty': difficulty,
        'hash': ''
    }
    
    logger.info(f"📦 Prepared block #{block_index} with {len(transactions)} transactions")
    return new_block
