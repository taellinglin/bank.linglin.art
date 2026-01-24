# blockchain_daemon_modules/blocks.py
"""
Block-related operations (get_block, block enhancement, stats, etc.)
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def get_block(blockchain: List[Dict], block_identifier, mempool_mgr, 
              get_block_from_network_func) -> Optional[Dict]:
    """
    Get block by hash, height (index), or latest block.
    
    Args:
        block_identifier: Can be:
            - int: Block height/index (e.g., 701)
            - str: Block hash (e.g., "00003ac6a611a0e6c0545afc729eb01c6d0c4b5273a8af6022d8bbbf05b42490")
            - "latest": Get the latest block
            - "first": Get the first/oldest block
            - "genesis": Get the genesis block (index 0)
            
    Returns:
        dict: Block data with enhanced information or None if not found
    """
    try:
        if not blockchain:
            logger.warning("Blockchain is empty")
            return None

        if not isinstance(block_identifier, (str, int)):
            logger.warning(f"Invalid block identifier: {block_identifier}")
            return None
        
        # Handle "latest" request
        if block_identifier == "latest":
            block = blockchain[-1] if blockchain else None
            return enhance_block_data(block, blockchain) if block else None
        
        # Handle "first" request (not necessarily genesis)
        if block_identifier == "first":
            block = blockchain[0] if blockchain else None
            return enhance_block_data(block, blockchain) if block else None
        
        # Handle "genesis" request
        if block_identifier == "genesis":
            for block in blockchain:
                if block.get('index') == 0:
                    return enhance_block_data(block, blockchain)
            return None
        
        # Check if it's a block hash (string starting with 0 or has specific pattern)
        if isinstance(block_identifier, str):
            # Check if it looks like a block hash
            if len(block_identifier) >= 16 and all(c in '0123456789abcdefABCDEF' for c in block_identifier):
                # Search by hash
                for block in blockchain:
                    if block.get('hash') == block_identifier:
                        return enhance_block_data(block, blockchain)
            
            # Try to parse as int for height
            try:
                block_height = int(block_identifier)
                block_identifier = block_height
            except ValueError:
                pass
        
        # Handle block height/index
        if isinstance(block_identifier, int):
            block_height = block_identifier
            
            # Check bounds
            if block_height < 0 or block_height >= len(blockchain):
                # Try to get from network if not found locally
                network_block = get_block_from_network_func(block_height)
                if network_block:
                    logger.info(f"Found block #{block_height} from network")
                    # Add to local blockchain if it's the next block
                    if block_height == len(blockchain):
                        blockchain.append(network_block)
                    return enhance_block_data(network_block, blockchain)
                return None
            
            block = blockchain[block_height]
            return enhance_block_data(block, blockchain)
        
        logger.warning(f"Invalid block identifier: {block_identifier}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting block {block_identifier}: {e}")
        return None


def enhance_block_data(block: Dict, blockchain: List[Dict]) -> Dict:
    """
    Enhance block data with additional calculated fields.
    
    Args:
        block: Raw block data
        blockchain: Full blockchain for context
        
    Returns:
        dict: Enhanced block data
    """
    if not block:
        return None
    
    enhanced_block = block.copy()
    
    # Calculate some statistics
    transactions = block.get('transactions', [])
    
    # Count transaction types
    tx_types = {}
    for tx in transactions:
        tx_type = tx.get('type', 'unknown')
        tx_types[tx_type] = tx_types.get(tx_type, 0) + 1
    
    # Calculate total value in block
    total_value = 0
    for tx in transactions:
        if tx.get('type') == 'transfer':
            total_value += tx.get('amount', 0)
        elif tx.get('type') == 'reward':
            total_value += tx.get('amount', 0)
    
    # Calculate reward amount
    reward_amount = block.get('reward', 0)
    
    # Add enhancement fields
    enhanced_block['transaction_count'] = len(transactions)
    enhanced_block['transaction_types'] = tx_types
    enhanced_block['total_value'] = total_value
    enhanced_block['reward_amount'] = reward_amount
    
    # Add confirmation count (if block is in blockchain)
    try:
        block_index = block.get('index')
        if block_index is not None and blockchain:
            confirmations = len(blockchain) - block_index - 1
            enhanced_block['confirmations'] = max(0, confirmations)
    except:
        enhanced_block['confirmations'] = 0
    
    # Add timestamp in human-readable format
    timestamp = block.get('timestamp')
    if timestamp:
        try:
            dt = datetime.fromtimestamp(timestamp)
            enhanced_block['timestamp_formatted'] = dt.strftime('%Y-%m-%d %H:%M:%S')
            enhanced_block['timestamp_readable'] = dt.strftime('%B %d, %Y at %I:%M %p')
            enhanced_block['timestamp_relative'] = get_relative_time(dt)
        except:
            enhanced_block['timestamp_formatted'] = str(timestamp)
    
    # Add next and previous block info
    block_index = block.get('index')
    if block_index is not None and blockchain:
        if block_index > 0:
            prev_block = blockchain[block_index - 1] if block_index - 1 < len(blockchain) else None
            if prev_block:
                enhanced_block['previous_block_hash'] = prev_block.get('hash')
        else:
            enhanced_block['previous_block_hash'] = '0' * 64  # Genesis
        
        if block_index + 1 < len(blockchain):
            next_block = blockchain[block_index + 1]
            if next_block:
                enhanced_block['next_block_hash'] = next_block.get('hash')
                enhanced_block['next_block_index'] = block_index + 1
    
    # Add miner info if available
    miner = block.get('miner')
    if miner:
        enhanced_block['miner_address'] = miner
    
    # Add block size estimation
    try:
        block_size = len(json.dumps(block).encode('utf-8'))
        enhanced_block['estimated_size_bytes'] = block_size
        enhanced_block['estimated_size_kb'] = round(block_size / 1024, 2)
    except:
        enhanced_block['estimated_size_bytes'] = 0
    
    return enhanced_block


def get_relative_time(dt: datetime) -> str:
    """Get relative time string (e.g., '2 hours ago')"""
    try:
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 365:
            years = diff.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"
    except:
        return "unknown time"


def get_block_range(blockchain: List[Dict], start: int, end: int) -> List[Dict]:
    """
    Get a range of blocks.
    
    Args:
        start: Start block index (inclusive)
        end: End block index (exclusive)
        
    Returns:
        list: List of blocks in the range
    """
    try:
        if start < 0 or end > len(blockchain):
            return []
        
        blocks = blockchain[start:end]
        return [enhance_block_data(block, blockchain) for block in blocks]
    except Exception as e:
        logger.error(f"Error getting block range {start}-{end}: {e}")
        return []


def get_latest_blocks(blockchain: List[Dict], count: int = 10) -> List[Dict]:
    """
    Get the latest N blocks.
    
    Args:
        count: Number of blocks to return
        
    Returns:
        list: List of latest blocks
    """
    try:
        if not blockchain:
            return []
        
        count = min(count, len(blockchain))
        blocks = blockchain[-count:]
        return [enhance_block_data(block, blockchain) for block in reversed(blocks)]
    except Exception as e:
        logger.error(f"Error getting latest {count} blocks: {e}")
        return []


def get_block_by_transaction(blockchain: List[Dict], tx_hash: str, mempool_mgr, 
                             blockchain_mgr) -> Optional[Dict]:
    """
    Find block containing a specific transaction.
    
    Args:
        tx_hash: Transaction hash to search for
        
    Returns:
        dict: Block containing the transaction or None
    """
    try:
        for block in blockchain:
            for tx in block.get('transactions', []):
                if tx.get('hash') == tx_hash:
                    return enhance_block_data(block, blockchain)
        
        # Check network
        if mempool_mgr.test_connection():
            for height in range(len(blockchain), len(blockchain) + 100):
                network_block = blockchain_mgr.get_block(height)
                if network_block:
                    for tx in network_block.get('transactions', []):
                        if tx.get('hash') == tx_hash:
                            return enhance_block_data(network_block, blockchain)
        
        return None
    except Exception as e:
        logger.error(f"Error finding block for transaction {tx_hash}: {e}")
        return None


def get_blockchain_stats(blockchain: List[Dict]) -> Dict:
    """
    Get blockchain statistics.
    
    Returns:
        dict: Statistics about the blockchain
    """
    try:
        stats = {
            'total_blocks': len(blockchain),
            'total_transactions': 0,
            'total_value': 0,
            'total_rewards': 0,
            'genesis_block': None,
            'latest_block': None,
            'block_sizes': [],
            'transaction_types': {},
            'miners': {},
            'difficulty_distribution': {}
        }
        
        if blockchain:
            stats['genesis_block'] = blockchain[0].get('hash') if blockchain else None
            stats['latest_block'] = blockchain[-1].get('hash') if blockchain else None
            
            for block in blockchain:
                transactions = block.get('transactions', [])
                stats['total_transactions'] += len(transactions)
                
                # Count transaction types
                for tx in transactions:
                    tx_type = tx.get('type', 'unknown')
                    stats['transaction_types'][tx_type] = stats['transaction_types'].get(tx_type, 0) + 1
                    
                    # Calculate values
                    if tx_type in ['transfer', 'reward']:
                        amount = tx.get('amount', 0)
                        stats['total_value'] += amount
                        if tx_type == 'reward':
                            stats['total_rewards'] += amount
                
                # Track block sizes
                try:
                    block_size = len(json.dumps(block).encode('utf-8'))
                    stats['block_sizes'].append(block_size)
                except:
                    pass
                
                # Track miners
                miner = block.get('miner')
                if miner:
                    stats['miners'][miner] = stats['miners'].get(miner, 0) + 1
                
                # Track difficulty
                difficulty = block.get('difficulty', 1)
                stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
        
        # Calculate averages
        if stats['block_sizes']:
            stats['average_block_size'] = sum(stats['block_sizes']) / len(stats['block_sizes'])
            stats['total_blockchain_size'] = sum(stats['block_sizes'])
        
        # Add time-based stats
        if blockchain and len(blockchain) > 1:
            first_timestamp = blockchain[0].get('timestamp', 0)
            last_timestamp = blockchain[-1].get('timestamp', 0)
            time_diff = last_timestamp - first_timestamp
            
            if time_diff > 0:
                stats['blockchain_age_seconds'] = time_diff
                stats['blockchain_age_days'] = time_diff / 86400
                stats['average_block_time'] = time_diff / (len(blockchain) - 1)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting blockchain stats: {e}")
        return {}


def get_previous_hash(blockchain: List[Dict]) -> str:
    """Retrieve the hash of the last block in the blockchain."""
    if blockchain:
        return blockchain[-1]['hash']
    return '0' * 64
