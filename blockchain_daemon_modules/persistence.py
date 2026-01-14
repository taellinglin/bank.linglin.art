# blockchain_daemon_modules/persistence.py
"""
Functions for saving and loading blockchain and mempool data
"""

import json
import os
import time
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def save_blockchain(blockchain: List[Dict], blockchain_file: str) -> bool:
    """Save blockchain to file - ATOMIC and SAFE"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(blockchain_file), exist_ok=True)
        
        # Create temp file
        temp_file = f"{blockchain_file}.tmp.{int(time.time())}"
        
        # Write to temp file
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(blockchain, f, indent=2, ensure_ascii=False)
        
        # Create backup if original exists
        if os.path.exists(blockchain_file):
            backup_file = f"{blockchain_file}.backup.{int(time.time())}"
            try:
                os.rename(blockchain_file, backup_file)
            except:
                pass
        
        # Rename temp to actual file (atomic operation on most systems)
        os.replace(temp_file, blockchain_file)
        
        logger.info(f"✅ Saved blockchain with {len(blockchain)} blocks")
        
        # Cleanup old temp and backup files
        cleanup_temp_files(os.path.dirname(blockchain_file))
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving blockchain: {e}")
        # Try to restore from backup
        restore_from_backup(blockchain_file)
        return False


def save_mempool(mempool: List[Dict], mempool_file: str) -> bool:
    """Save mempool to file - ATOMIC and SAFE"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(mempool_file), exist_ok=True)
        
        # Create temp file
        temp_file = f"{mempool_file}.tmp.{int(time.time())}"
        
        # Write to temp file
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(mempool, f, indent=2, ensure_ascii=False)
        
        # Create backup if original exists
        if os.path.exists(mempool_file):
            backup_file = f"{mempool_file}.backup.{int(time.time())}"
            try:
                os.rename(mempool_file, backup_file)
            except:
                pass
        
        # Rename temp to actual file (atomic operation on most systems)
        os.replace(temp_file, mempool_file)
        
        logger.debug(f"✅ Saved mempool with {len(mempool)} transactions")
        
        # Cleanup old temp and backup files
        cleanup_temp_files(os.path.dirname(mempool_file))
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving mempool: {e}")
        # Try to restore from backup
        restore_from_backup(mempool_file)
        return False


def load_blockchain(blockchain_file: str) -> List[Dict]:
    """Load blockchain from file"""
    try:
        if os.path.exists(blockchain_file):
            with open(blockchain_file, 'r', encoding='utf-8') as f:
                blockchain = json.load(f)
                logger.info(f"✅ Loaded blockchain with {len(blockchain)} blocks")
                return blockchain
        else:
            logger.info("No blockchain file found, starting with empty blockchain")
            return []
    except Exception as e:
        logger.error(f"Error loading blockchain: {e}")
        # Try to restore from backup
        if restore_from_backup(blockchain_file):
            return load_blockchain(blockchain_file)
        return []


def load_mempool(mempool_file: str) -> List[Dict]:
    """Load mempool from file"""
    try:
        if os.path.exists(mempool_file):
            with open(mempool_file, 'r', encoding='utf-8') as f:
                mempool = json.load(f)
                logger.info(f"✅ Loaded mempool with {len(mempool)} transactions")
                return mempool
        else:
            logger.info("No mempool file found, starting with empty mempool")
            return []
    except Exception as e:
        logger.error(f"Error loading mempool: {e}")
        # Try to restore from backup
        if restore_from_backup(mempool_file):
            return load_mempool(mempool_file)
        return []


def cleanup_temp_files(directory: str, max_age_hours: int = 24):
    """Clean up old temporary files"""
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            if '.tmp.' in filename or '.backup.' in filename:
                filepath = os.path.join(directory, filename)
                file_age = current_time - os.path.getmtime(filepath)
                
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.debug(f"Error cleaning up temp files: {e}")


def restore_from_backup(file_path: str) -> bool:
    """Try to restore from backup if save failed"""
    try:
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Find the most recent backup
        backup_files = [f for f in os.listdir(directory) if f.startswith(f"{filename}.backup.")]
        if backup_files:
            # Sort by timestamp (most recent first)
            backup_files.sort(reverse=True)
            most_recent_backup = os.path.join(directory, backup_files[0])
            
            # Restore the backup
            os.replace(most_recent_backup, file_path)
            logger.info(f"✅ Restored from backup: {backup_files[0]}")
            return True
        else:
            logger.warning(f"No backup found for {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error restoring from backup: {e}")
        return False


def create_initial_files(blockchain_file: str, mempool_file: str):
    """Create initial files if they don't exist"""
    try:
        # Create directories
        os.makedirs(os.path.dirname(blockchain_file), exist_ok=True)
        os.makedirs(os.path.dirname(mempool_file), exist_ok=True)
        
        # Create empty files if they don't exist
        if not os.path.exists(blockchain_file):
            with open(blockchain_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.info(f"Created empty blockchain file: {blockchain_file}")
        
        if not os.path.exists(mempool_file):
            with open(mempool_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.info(f"Created empty mempool file: {mempool_file}")
            
    except Exception as e:
        logger.error(f"Error creating initial files: {e}")
