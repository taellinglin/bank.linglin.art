import json
import os

# Check main mempool
if os.path.exists('mempool.json'):
    with open('mempool.json') as f:
        mempool = json.load(f)
    reward_txs = [tx for tx in mempool if tx.get('type') == 'reward']
    print(f"Main mempool rewards: {len(reward_txs)}")
    for tx in reward_txs[:5]:
        print(f"  Block #{tx.get('block_height')}: {tx.get('amount')} LKC")

# Check daemon mempool
if os.path.exists('blockchain_daemon/mempool.json'):
    with open('blockchain_daemon/mempool.json') as f:
        daemon_mempool = json.load(f)
    daemon_rewards = [tx for tx in daemon_mempool if tx.get('type') == 'reward']
    print(f"\nDaemon mempool rewards: {len(daemon_rewards)}")
    for tx in daemon_rewards[:5]:
        print(f"  Block #{tx.get('block_height')}: {tx.get('amount')} LKC (difficulty should be in block)")
