# blockchain_daemon.py
"""
COMPATIBILITY WRAPPER with ASYNC SUPPORT

This file maintains backward compatibility with existing imports.
The actual implementation has been refactored into blockchain_daemon_modules/

To use the refactored code directly:
    from blockchain_daemon_modules import BlockchainDaemon

For backward compatibility (existing code):
    from blockchain_daemon import BlockchainDaemon  # Still works!

Async operations:
    from blockchain_daemon_modules.async_ops import get_blockchain_manager, get_mining_manager
    
    async_mgr = get_blockchain_manager(daemon)
    status = await async_mgr.get_blockchain_status_async()

Module structure:
- blockchain_daemon_modules/
  ├── __init__.py           # Package exports
  ├── daemon.py             # Main BlockchainDaemon class
  ├── validators.py         # Validation functions
  ├── persistence.py        # Save/load operations
  ├── network.py            # Network sync
  ├── blocks.py             # Block operations
  ├── transactions.py       # Transaction handling
  ├── mining.py             # Mining operations
  └── async_ops.py          # Async operations with lunalib 1.6.9 support
"""

import logging
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Import lunalib components with fallback for different versions
blockchain = None
mempool = None
daemon = None

try:
    # Try lunalib 1.6.9 API - BlockchainManager and MempoolManager
    try:
        from lunalib.core.blockchain import BlockchainManager
        from lunalib.core.mempool import MempoolManager
        logger.info("✅ Loaded BlockchainManager/MempoolManager from lunalib.core.*")
    except ImportError:
        # Try alternative API structure
        try:
            from lunalib.blockchain import BlockchainManager
            from lunalib.mempool import MempoolManager
            logger.info("✅ Loaded BlockchainManager/MempoolManager from lunalib.*")
        except ImportError:
            logger.warning("⚠️ BlockchainManager/MempoolManager not available")
            BlockchainManager = None
            MempoolManager = None
    
    # Try to initialize blockchain components if managers are available
    if BlockchainManager and MempoolManager:
        try:
            blockchain = BlockchainManager(endpoint_url="https://bank.linglin.art")
            mempool = MempoolManager(["https://bank.linglin.art"])
            logger.info("✅ Blockchain and mempool managers initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize managers: {e}")
    
    # Try LunalibDaemon if available
    try:
        from lunalib.core.daemon import BlockchainDaemon as LunalibDaemon
        if blockchain and mempool:
            daemon = LunalibDaemon(blockchain, mempool)
            daemon.start()
            logger.info("✅ Lunalib daemon started")
    except ImportError:
        logger.info("ℹ️  LunalibDaemon not available (optional)")
    except Exception as e:
        logger.warning(f"⚠️ Could not start daemon: {e}")
    
    logger.info("✅ Lunalib 1.6.9 initialized with async/threading support")
except Exception as e:
    logger.warning(f"⚠️ Lunalib initialization error: {e}")

# Import the refactored BlockchainDaemon class
from blockchain_daemon_modules import BlockchainDaemon
from blockchain_daemon_modules.async_ops import (
    get_blockchain_manager, 
    get_mining_manager,
    AsyncBlockchainManager,
    AsyncMiningManager
)

# Export for backward compatibility
__all__ = [
    'BlockchainDaemon', 
    'blockchain', 
    'mempool', 
    'daemon',
    'get_blockchain_manager',
    'get_mining_manager',
    'AsyncBlockchainManager',
    'AsyncMiningManager'
]
