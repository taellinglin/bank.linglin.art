# mining_service.py
"""
Mining service with lunalib 1.6.9 async support
"""

import json
import sys
import os
import time
import asyncio
import logging
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor

# 標準出力をUTF-8に設定（Windowsのcharmapエラー回避）
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add the path to your blockchain_daemon
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the refactored modular daemon with mining support
from blockchain_daemon_modules import BlockchainDaemon, mining
from blockchain_daemon_modules.async_ops import get_mining_manager, AsyncMiningManager

# Thread pool for mining operations
_mining_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mining-service-")


async def mine_block_async(daemon: BlockchainDaemon, miner_address: str, 
                           difficulty: int = 4) -> Optional[Dict]:
    """
    Async mining operation with lunalib 1.6.9 support
    """
    try:
        loop = asyncio.get_event_loop()
        mining_mgr = get_mining_manager()
        
        # Get pending transactions
        mempool_status = daemon.get_mempool_status()
        pending_txs = daemon.mempool[:50]  # Max 50 transactions per block
        
        if not pending_txs:
            logger.warning("No transactions to mine")
            return None
        
        # Create reward transaction
        total_fees = sum(tx.get("fee", 0) for tx in pending_txs if isinstance(tx, dict))
        reward_tx = mining.create_reward_transaction(
            miner_address,
            len(daemon.blockchain),
            difficulty,
            len(pending_txs),
            fees=total_fees,
            empty_block=len(pending_txs) == 0
        )
        
        # Add reward to transactions
        transactions_to_mine = pending_txs + [reward_tx]
        
        # Prepare block
        block = mining.prepare_block_for_mining(
            daemon.blockchain,
            transactions_to_mine,
            miner_address,
            difficulty
        )
        
        # Track active miner
        miner_id = f"{miner_address}_{int(time.time())}"
        mining_mgr.track_active_miner(miner_id)
        
        try:
            # Mine the block asynchronously
            mined_block = await mining_mgr.mine_block_async(block, difficulty, max_iterations=1000000)
            return mined_block
        finally:
            mining_mgr.untrack_active_miner(miner_id)
    
    except asyncio.TimeoutError:
        logger.error("Mining operation timed out")
        return None
    except Exception as e:
        logger.error(f"Async mining error: {e}")
        return None


def mine_block_blocking(daemon: BlockchainDaemon, miner_address: str, 
                       difficulty: int = 4) -> Optional[Dict]:
    """
    Blocking mining operation (fallback)
    """
    try:
        # Get pending transactions
        mempool_status = daemon.get_mempool_status()
        pending_txs = daemon.mempool[:50]
        
        if not pending_txs:
            logger.warning("No transactions to mine")
            return None
        
        # Create reward transaction
        total_fees = sum(tx.get("fee", 0) for tx in pending_txs if isinstance(tx, dict))
        reward_tx = mining.create_reward_transaction(
            miner_address,
            len(daemon.blockchain),
            difficulty,
            len(pending_txs),
            fees=total_fees,
            empty_block=len(pending_txs) == 0
        )
        
        transactions_to_mine = pending_txs + [reward_tx]
        
        # Prepare and mine block
        block = mining.prepare_block_for_mining(
            daemon.blockchain,
            transactions_to_mine,
            miner_address,
            difficulty
        )
        
        mined_block = mining.mine_block(block, difficulty)
        return mined_block
    
    except Exception as e:
        logger.error(f"Blocking mining error: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Missing miner_address"}))
        sys.exit(1)
    
    miner_address = sys.argv[1]
    difficulty = int(sys.argv[2]) if len(sys.argv) >= 3 else 4
    
    try:
        daemon = BlockchainDaemon()
        daemon.load_data()
        
        # Try async mining first
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            mined_block = loop.run_until_complete(
                asyncio.wait_for(
                    mine_block_async(daemon, miner_address, difficulty),
                    timeout=70.0
                )
            )
            loop.close()
        except Exception as async_error:
            logger.warning(f"Async mining failed: {async_error}, falling back to blocking")
            mined_block = mine_block_blocking(daemon, miner_address, difficulty)
        
        if mined_block:
            print(json.dumps({
                "success": True,
                "block": mined_block,
                "message": f"✅ Successfully mined block #{mined_block.get('index')}"
            }))
        else:
            print(json.dumps({
                "success": False,
                "error": "Mining failed - could not find valid nonce"
            }))
            
    except Exception as e:
        logger.error(f"Mining service error: {e}")
        print(json.dumps({
            "success": False,
            "error": f"Mining error: {str(e)}"
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()