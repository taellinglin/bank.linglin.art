# Clean mempool script - Remove old reward transactions
import json
import os
from datetime import datetime

def clean_mempool(filepath):
    """Remove old reward transactions from mempool"""
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return
    
    # Backup first
    backup_path = f"{filepath}.backup.{int(datetime.now().timestamp())}"
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Backed up to {backup_path}")
    
    # Filter out reward transactions
    original_count = len(data)
    reward_count = len([tx for tx in data if tx.get('type') == 'reward'])
    
    filtered = [tx for tx in data if tx.get('type') != 'reward']
    
    # Save cleaned mempool
    with open(filepath, 'w') as f:
        json.dump(filtered, f, indent=2)
    
    print(f"✅ Cleaned {filepath}")
    print(f"   Original: {original_count} transactions")
    print(f"   Removed: {reward_count} reward transactions")
    print(f"   Remaining: {len(filtered)} transactions")

# Clean both mempools
print("🧹 Cleaning mempools...\n")
clean_mempool('mempool.json')
print()
clean_mempool('blockchain_daemon/mempool.json')
print("\n✅ Done! You can now restart mining with the new reward calculation.")
