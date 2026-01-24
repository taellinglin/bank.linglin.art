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
from pathlib import Path
import os
import sys

logger = logging.getLogger(__name__)


def _ensure_lunalib_root_in_path() -> None:
    """Ensure LunaLib root is on PYTHONPATH for daemon processes."""
    candidates = []
    env_path = os.environ.get("LUNALIB_ROOT") or os.environ.get("LUNALIB_PATH")
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parent
    candidates.append(repo_root.parent / "LunaLib")
    candidates.append(repo_root.parent / "lunalib")
    for candidate in candidates:
        try:
            if candidate and candidate.exists():
                path_str = str(candidate)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                break
        except Exception:
            continue


_ensure_lunalib_root_in_path()

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


def _start_lunalib_http_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start LunaLib daemon HTTP server."""
    from lunalib.core.blockchain import BlockchainManager
    from lunalib.core.mempool import MempoolManager
    from lunalib.core.daemon import BlockchainDaemon as LunalibDaemon
    from lunalib.core.daemon_server import DaemonHTTPServer

    endpoint_url = os.environ.get("LUNALIB_ENDPOINT", "https://bank.linglin.art")
    blockchain_mgr = BlockchainManager(endpoint_url=endpoint_url)
    mempool_mgr = MempoolManager([endpoint_url])
    lunalib_daemon = LunalibDaemon(blockchain_mgr, mempool_mgr)
    lunalib_daemon.start()

    server = DaemonHTTPServer(lunalib_daemon, host=host, port=port)
    if hasattr(server, "serve_forever"):
        server.serve_forever()
    elif hasattr(server, "run"):
        server.run()
    elif hasattr(server, "start"):
        server.start()
    else:
        raise RuntimeError("DaemonHTTPServer has no start method")

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


if __name__ == "__main__":
    host = os.environ.get("LUNALIB_DAEMON_HOST", "0.0.0.0")
    port = int(os.environ.get("LUNALIB_DAEMON_PORT", "8000"))
    _start_lunalib_http_server(host=host, port=port)
