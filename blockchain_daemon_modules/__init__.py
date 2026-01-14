# blockchain_daemon_modules/__init__.py
"""
Refactored blockchain daemon modules

This package provides a modular blockchain daemon implementation:
- daemon.py: Main BlockchainDaemon class
- validators.py: Block and transaction validation
- persistence.py: Data storage and loading
- network.py: Network synchronization
- blocks.py: Block operations and queries
- transactions.py: Transaction handling
- mining.py: Mining operations with exponential rewards
"""

from .daemon import BlockchainDaemon
from . import mining

# Export for backward compatibility
__all__ = ['BlockchainDaemon', 'mining']

# Version info
__version__ = '1.0.0'
