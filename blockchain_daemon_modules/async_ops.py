# blockchain_daemon_modules/async_ops.py
"""
Async operations module for lunalib 1.6.9 integration
Provides non-blocking operations for blockchain operations
"""

import asyncio
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for async operations
_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="blockchain-")


async def async_blockchain_operation(operation: Callable, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """
    Execute a blockchain operation asynchronously with timeout
    
    Args:
        operation: Callable that performs the blockchain operation
        timeout: Maximum time to wait in seconds
    
    Returns:
        Result from operation or None if timeout
    """
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        
        result = await asyncio.wait_for(
            loop.run_in_executor(_thread_pool, operation),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Blockchain operation timed out after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"Async operation error: {e}")
        return None


def run_async_in_thread(coro, timeout: float = 2.0) -> Optional[Any]:
    """
    Run an async operation in a new event loop (thread-safe)
    
    Args:
        coro: Coroutine to execute
        timeout: Maximum time to wait in seconds
    
    Returns:
        Result from coroutine or None if timeout
    """
    def run_loop():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Async operation error: {e}")
            return None
        finally:
            loop.close()
    
    # Run in thread
    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    thread.join(timeout=timeout + 0.5)  # Give thread time to complete
    return None  # Thread result not easily accessible, use callback instead


class AsyncBlockchainManager:
    """Async-compatible blockchain operations manager"""
    
    def __init__(self, daemon):
        self.daemon = daemon
        self._cache = {}
        self._cache_timeout = 5.0  # Cache results for 5 seconds
        self._cache_timestamp = {}
    
    async def get_blockchain_status_async(self) -> Dict[str, Any]:
        """
        Get blockchain status asynchronously
        
        Returns:
            Dictionary with blockchain stats
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            
            status = await asyncio.wait_for(
                loop.run_in_executor(_thread_pool, self.daemon.get_blockchain_status),
                timeout=1.5
            )
            
            # Cache the result
            self._cache['status'] = status
            self._cache_timestamp['status'] = time.time()
            return status
        except asyncio.TimeoutError:
            logger.warning("get_blockchain_status timed out")
            # Return cached result if available
            if 'status' in self._cache:
                return self._cache['status']
            return {"blocks": 0, "total_transactions": 0}
        except Exception as e:
            logger.error(f"Error getting blockchain status: {e}")
            return {"blocks": 0, "total_transactions": 0}
    
    async def get_mempool_status_async(self) -> Dict[str, Any]:
        """
        Get mempool status asynchronously
        
        Returns:
            Dictionary with mempool stats
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            
            status = await asyncio.wait_for(
                loop.run_in_executor(_thread_pool, self.daemon.get_mempool_status),
                timeout=1.0
            )
            
            # Cache the result
            self._cache['mempool'] = status
            self._cache_timestamp['mempool'] = time.time()
            return status
        except asyncio.TimeoutError:
            logger.warning("get_mempool_status timed out")
            if 'mempool' in self._cache:
                return self._cache['mempool']
            return {"total": 0}
        except Exception as e:
            logger.error(f"Error getting mempool status: {e}")
            return {"total": 0}
    
    async def validate_block_async(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a block asynchronously
        
        Args:
            block: Block to validate
        
        Returns:
            Validation result dictionary
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            
            result = await asyncio.wait_for(
                loop.run_in_executor(_thread_pool, self.daemon.validate_block, block),
                timeout=2.0
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Block validation timed out")
            return {'valid': False, 'error': 'Validation timeout'}
        except Exception as e:
            logger.error(f"Block validation error: {e}")
            return {'valid': False, 'error': str(e)}
    
    def get_cached_status(self, key: str, default: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached status if still valid
        
        Args:
            key: Cache key ('status', 'mempool', etc.)
            default: Default value if cache expired
        
        Returns:
            Cached value or default
        """
        if key not in self._cache or key not in self._cache_timestamp:
            return default or {}
        
        # Check if cache is still valid
        age = time.time() - self._cache_timestamp[key]
        if age < self._cache_timeout:
            return self._cache[key]
        
        # Cache expired
        del self._cache[key]
        del self._cache_timestamp[key]
        return default or {}


class AsyncMiningManager:
    """Async-compatible mining operations manager"""
    
    def __init__(self):
        self._mining_tasks = {}
        self._active_miners = set()
    
    async def mine_block_async(self, block: Dict[str, Any], difficulty: int, max_iterations: int = 1000000) -> Optional[Dict[str, Any]]:
        """
        Mine a block asynchronously
        
        Args:
            block: Block to mine
            difficulty: Mining difficulty
            max_iterations: Maximum iterations
        
        Returns:
            Mined block or None if failed
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            
            # Import mining module
            from . import mining
            
            mined_block = await asyncio.wait_for(
                loop.run_in_executor(_thread_pool, mining.mine_block, block, difficulty, max_iterations),
                timeout=60.0  # 60 second timeout for mining
            )
            
            logger.info(f"✅ Block mined async: #{mined_block.get('index')}")
            return mined_block
        except asyncio.TimeoutError:
            logger.warning("Mining operation timed out")
            return None
        except Exception as e:
            logger.error(f"Mining error: {e}")
            return None
    
    def track_active_miner(self, miner_id: str):
        """Track an active mining operation"""
        self._active_miners.add(miner_id)
        logger.debug(f"Active miners: {len(self._active_miners)}")
    
    def untrack_active_miner(self, miner_id: str):
        """Remove a miner from active tracking"""
        self._active_miners.discard(miner_id)
        logger.debug(f"Active miners: {len(self._active_miners)}")
    
    def get_active_miners_count(self) -> int:
        """Get count of active mining operations"""
        return len(self._active_miners)


# Global managers
_blockchain_mgr = None
_mining_mgr = None


def get_blockchain_manager(daemon) -> AsyncBlockchainManager:
    """Get or create async blockchain manager"""
    global _blockchain_mgr
    if _blockchain_mgr is None:
        _blockchain_mgr = AsyncBlockchainManager(daemon)
    return _blockchain_mgr


def get_mining_manager() -> AsyncMiningManager:
    """Get or create async mining manager"""
    global _mining_mgr
    if _mining_mgr is None:
        _mining_mgr = AsyncMiningManager()
    return _mining_mgr


def cleanup_thread_pool():
    """Cleanup thread pool on shutdown"""
    _thread_pool.shutdown(wait=True)
    logger.info("Thread pool shut down")


# Register cleanup
import atexit
atexit.register(cleanup_thread_pool)
