# blockchain_daemon_modules/network.py
"""
Network synchronization and communication functions
"""

import logging
import traceback
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def sync_with_network(blockchain: List[Dict], blockchain_mgr, mempool_mgr, 
                     validate_block_func, build_mined_indexes_func,
                     save_blockchain_func, save_mempool_func) -> bool:
    """Sync with network blockchain, but PRESERVE LOCAL DATA"""
    try:
        # Test network connection
        if not mempool_mgr.test_connection():
            logger.warning("Network unavailable for sync")
            return False
        
        logger.info("🔄 Syncing with network...")
        
        # Get network blockchain
        network_blockchain = blockchain_mgr.get_full_blockchain()
        
        if not network_blockchain:
            logger.warning("No blockchain data from network")
            return False
        
        # Get local height
        local_height = len(blockchain)
        network_height = len(network_blockchain)
        
        logger.info(f"Local height: {local_height}, Network height: {network_height}")
        
        # If network is ahead, sync missing blocks
        if network_height > local_height:
            logger.info(f"Network is ahead by {network_height - local_height} blocks")
            
            # Validate and add missing blocks
            for i in range(local_height, network_height):
                network_block = network_blockchain[i]
                
                # Validate block before adding
                if validate_block_func(network_block):
                    blockchain.append(network_block)
                    logger.info(f"Added network block #{i}")
                else:
                    logger.warning(f"Invalid network block #{i}, stopping sync")
                    break
            
            # Rebuild indexes
            build_mined_indexes_func()
            
            # Save updated blockchain
            save_blockchain_func()
            
            logger.info(f"✅ Synced {len(blockchain) - local_height} blocks from network")
        
        elif network_height < local_height:
            logger.warning(f"Local blockchain is ahead of network by {local_height - network_height} blocks")
            # Keep local data, don't downgrade
        
        else:
            logger.info("✅ Local and network blockchains are in sync")
        
        # Sync mempool
        sync_mempool_from_network(mempool_mgr, save_mempool_func)
        
        return True
        
    except Exception as e:
        logger.error(f"Error syncing with network: {e}")
        logger.error(traceback.format_exc())
        return False


def sync_mempool_from_network(mempool_mgr, save_mempool_func):
    """Sync mempool from network"""
    try:
        if not mempool_mgr.test_connection():
            return
        
        logger.debug("Syncing mempool from network...")
        
        # Get network mempool
        network_mempool = mempool_mgr.get_mempool()
        
        if network_mempool:
            # Here you would merge network mempool with local mempool
            # For now, just log the count
            logger.debug(f"Network mempool has {len(network_mempool)} transactions")
            
    except Exception as e:
        logger.debug(f"Error syncing mempool: {e}")


def get_network_blockchain_height(blockchain_mgr) -> int:
    """Get the current blockchain height from network"""
    try:
        network_blockchain = blockchain_mgr.get_full_blockchain()
        if network_blockchain:
            return len(network_blockchain)
        return 0
    except Exception as e:
        logger.error(f"Error getting network blockchain height: {e}")
        return 0


def get_last_network_block_hash(blockchain_mgr) -> str:
    """Get the hash of the last block from network"""
    try:
        network_blockchain = blockchain_mgr.get_full_blockchain()
        if network_blockchain and len(network_blockchain) > 0:
            return network_blockchain[-1].get('hash', '0' * 64)
        return '0' * 64
    except Exception as e:
        logger.error(f"Error getting last network block hash: {e}")
        return '0' * 64


def broadcast_transaction_to_network(transaction: Dict, mempool_mgr) -> bool:
    """Broadcast a transaction to the network"""
    try:
        if not mempool_mgr.test_connection():
            logger.debug("Network unavailable, cannot broadcast transaction")
            return False
        
        # Broadcast using mempool manager
        result = mempool_mgr.broadcast_transaction(transaction)
        
        if result:
            logger.info(f"✅ Broadcasted transaction: {transaction.get('hash', 'unknown')[:16]}...")
            return True
        else:
            logger.warning(f"Failed to broadcast transaction")
            return False
            
    except Exception as e:
        logger.error(f"Error broadcasting transaction: {e}")
        return False


def submit_block_to_network(block: Dict, blockchain_mgr) -> bool:
    """Submit a mined block to the network"""
    try:
        logger.info(f"Submitting block #{block.get('index')} to network...")
        
        result = blockchain_mgr.submit_mined_block(block)
        
        if result:
            logger.info(f"✅ Successfully submitted block #{block.get('index')}")
            return True
        else:
            logger.error(f"❌ Failed to submit block #{block.get('index')}")
            return False
            
    except Exception as e:
        logger.error(f"Error submitting block to network: {e}")
        logger.error(traceback.format_exc())
        return False


def get_block_from_network(block_height: int, blockchain_mgr, validate_block_func) -> Optional[Dict]:
    """Try to get block from network if not found locally"""
    try:
        network_block = blockchain_mgr.get_block(block_height)
        if network_block:
            # Verify the block is valid
            if validate_block_func(network_block):
                return network_block
    except Exception as e:
        logger.debug(f"Failed to get block #{block_height} from network: {e}")
    
    return None
