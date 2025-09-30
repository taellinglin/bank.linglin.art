#!/usr/bin/env python3
"""
utils.py - Utility functions for LinKoin
"""
import json
import os
import socket
import time
from typing import List, Dict

def load_mempool(mempool_file: str = "mempool.json") -> List[Dict]:
    transactions = []
    try:
        if os.path.exists(mempool_file):
            with open(mempool_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        transaction = json.loads(line)
                        transactions.append(transaction)
            print(f"✅ Loaded {len(transactions)} transactions from mempool")
        else:
            print("⚠️  Mempool file not found")
    except Exception as e:
        print(f"❌ Error loading mempool: {e}")
    return transactions

def show_mining_animation(blockchain):
    """Show a simple mining animation when mining"""
    animations = ["⛏️ ", "🔨", "⚒️ ", "🛠️ "]
    while blockchain.is_mining:
        for anim in animations:
            if not blockchain.is_mining:
                break
            print(f"\r{anim} Mining...", end="", flush=True)
            time.sleep(0.3)
    print("\r" + " " * 20 + "\r", end="", flush=True)

def show_help():
    print("\n🎮 Luna Coin Node Commands:")
    print("  mine      - Mine pending transactions")
    print("  load      - Load transactions from mempool")
    print("  status    - Show blockchain status")
    print("  stats     - Show mining statistics")
    print("  validate  - Validate blockchain integrity")
    print("  clear     - Clear mempool file")
    print("  difficulty <n> - Set mining difficulty (1-8)")
    print("  bills     - Show available bills for current difficulty")
    print("  help      - Show this help")
    print("  exit      - Exit the node")

def get_miner_address_from_wallet():
    """Get miner address directly from wallet"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect(('127.0.0.1', 9334))  # Wallet RPC port
            s.sendall(json.dumps({
                "action": "get_mining_address"
            }).encode())
            response = s.recv(1024)
            result = json.loads(response.decode())
            if result.get("status") == "success":
                address = result.get("address")
                print(f"✅ Using mining address from wallet: {address}")
                return address
    except Exception as e:
        print(f"⚠️  Could not connect to wallet: {e}")
        print("💡 Make sure the wallet is running with: python wallet.py server")
    
    # Fallback: try to read from config file
    try:
        if os.path.exists("wallet_config.json"):
            with open("wallet_config.json", "r") as f:
                config = json.load(f)
                mining_address = config.get("mining", {}).get("mining_reward_address", "")
                if mining_address:
                    print(f"✅ Using mining address from config: {mining_address}")
                    return mining_address
    except Exception as e:
        print(f"⚠️  Could not read config: {e}")
    
    # Final fallback: ask user
    return input("Enter miner address for rewards: ") or "miner_default_address"

def list_recent_bills(blockchain):
    """List recent bills mined with their verification links"""
    print("\n💰 RECENTLY MINED BILLS:")
    print("="*80)
    
    # Get the last 10 blocks
    recent_blocks = blockchain.chain[-10:] if len(blockchain.chain) > 10 else blockchain.chain
    
    bills_found = []
    for block in recent_blocks:
        for tx in block.transactions:
            if tx.get("type") in ["reward", "genesis"] and tx.get("serial_number"):
                bills_found.append({
                    "block": block.index,
                    "denomination": f"${tx.get('denomination', 'N/A')}",
                    "serial": tx.get("serial_number", "N/A"),
                    "verification_url": tx.get("verification_url", "N/A"),
                    "miner": tx.get("to", "Unknown")
                })
    
    if not bills_found:
        print("No bills found in recent blocks")
        return
    
    for i, bill in enumerate(bills_found, 1):
        print(f"{i:2d}. Block {bill['block']:4d} | {bill['denomination']:>8} | {bill['serial']}")
        print(f"    🔗 {bill['verification_url']}")
        print(f"    ⛏️  Miner: {bill['miner']}")
        print()