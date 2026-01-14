#!/usr/bin/env python3
"""
Debug script to analyze a block submission error
"""

import json
import sys

def analyze_block(block_data):
    """Analyze block data and show detailed information"""
    
    print("=" * 80)
    print("🔍 BLOCK ANALYSIS")
    print("=" * 80)
    
    # Basic block info
    print(f"\n📦 BLOCK INFORMATION:")
    print(f"   Index: {block_data.get('index')}")
    print(f"   Hash: {block_data.get('hash', '')[:16]}...")
    print(f"   Previous hash: {block_data.get('previous_hash', '')[:16]}...")
    print(f"   Timestamp: {block_data.get('timestamp')}")
    print(f"   Nonce: {block_data.get('nonce')}")
    print(f"   Miner: {block_data.get('miner')}")
    print(f"   Difficulty: {block_data.get('difficulty')}")
    
    # Transaction analysis
    transactions = block_data.get('transactions', [])
    print(f"\n📋 TRANSACTIONS ({len(transactions)} total):")
    
    reward_txs = []
    other_txs = []
    
    for i, tx in enumerate(transactions):
        tx_type = tx.get('type')
        if tx_type == 'reward':
            reward_txs.append(tx)
        else:
            other_txs.append(tx)
    
    print(f"   Reward transactions: {len(reward_txs)}")
    print(f"   Other transactions: {len(other_txs)}")
    
    # Analyze reward transactions
    if reward_txs:
        print(f"\n💰 REWARD TRANSACTION ANALYSIS:")
        for i, reward_tx in enumerate(reward_txs):
            print(f"\n   Reward #{i+1}:")
            print(f"      From: {reward_tx.get('from', 'NOT SET')}")
            print(f"      To: {reward_tx.get('to')}")
            print(f"      Amount: {reward_tx.get('amount')} LKC")
            print(f"      Block height: {reward_tx.get('block_height')}")
            print(f"      Timestamp: {reward_tx.get('timestamp')}")
            print(f"      Hash: {reward_tx.get('hash', '')[:16]}...")
            
            if 'difficulty' in reward_tx:
                print(f"      Difficulty in TX: {reward_tx.get('difficulty')}")
            else:
                print(f"      ⚠️  Difficulty field: MISSING")
            
            # Calculate expected reward
            block_difficulty = block_data.get('difficulty', 1)
            BASE_REWARD = 1.0
            expected_reward = BASE_REWARD * (2 ** (block_difficulty - 1))
            
            print(f"\n      📊 REWARD CALCULATION:")
            print(f"         Block difficulty: {block_difficulty}")
            print(f"         Formula: {BASE_REWARD} * 2^({block_difficulty}-1)")
            print(f"         Expected reward: {expected_reward} LKC")
            print(f"         Actual reward: {reward_tx.get('amount')} LKC")
            print(f"         Match: {abs(reward_tx.get('amount', 0) - expected_reward) < 0.000001}")
            
            if abs(reward_tx.get('amount', 0) - expected_reward) > 0.000001:
                print(f"\n      ❌ PROBLEM DETECTED!")
                print(f"         The reward amount doesn't match the expected calculation.")
                print(f"         This block was likely created with old code that uses:")
                print(f"            OLD: BASE_REWARD * difficulty = {BASE_REWARD * block_difficulty} LKC")
                print(f"            NEW: BASE_REWARD * 10^(difficulty-1) = {expected_reward} LKC")
                print(f"\n      💡 SOLUTION:")
                print(f"         Restart the mining process with the updated code.")
                print(f"         Delete any cached blocks and regenerate them.")
    
    # Other transaction types
    if other_txs:
        print(f"\n📄 OTHER TRANSACTIONS:")
        type_counts = {}
        for tx in other_txs:
            tx_type = tx.get('type', 'unknown')
            type_counts[tx_type] = type_counts.get(tx_type, 0) + 1
        
        for tx_type, count in type_counts.items():
            print(f"   {tx_type}: {count}")
    
    print("\n" + "=" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_block.py <block_json_file>")
        print("   or: python debug_block.py '{json_data}'")
        sys.exit(1)
    
    block_input = sys.argv[1]
    
    try:
        # Try to parse as JSON string first
        if block_input.startswith('{'):
            block_data = json.loads(block_input)
        else:
            # Try to read from file
            with open(block_input, 'r') as f:
                block_data = json.load(f)
        
        analyze_block(block_data)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ File not found: {block_input}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
