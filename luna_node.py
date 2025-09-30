#!/usr/bin/env python3
"""
luna_node.py - The Luna Coin blockchain node and miner with P2P networking.
Elaborate mining system with denomination-based rewards, real-time statistics, and peer-to-peer networking.
"""
import os
import sys
import json
import time
import threading
import socket
import atexit
from typing import List, Dict
from blockchain import Blockchain
from datamanager import DataManager
def configure_ssl_for_frozen_app():
    """Configure SSL for PyInstaller frozen applications"""
    
    # Remove the DLL copying code entirely and rely on proper PyInstaller configuration
    if getattr(sys, 'frozen', False):
        # PyInstaller should handle this automatically if configured correctly
        print("🔐 Running as frozen application - SSL should be bundled")
        
        # Verify SSL is working
        try:
            import ssl
            context = ssl.create_default_context()
            print("✅ SSL is working correctly")
        except Exception as e:
            print(f"❌ SSL error: {e}")
            # Fallback: disable SSL verification for internal APIs only
            import warnings
            warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Now set PATH to include current directory
os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']


def load_mempool() -> List[Dict]:
    """Load mempool from storage"""
    transactions = DataManager.load_json("mempool.json", [])
    print(f"✅ Loaded {len(transactions)} transactions from mempool")
    return transactions

def show_mining_animation():
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
    print("\n🎮 Luna Node Commands:")
    print("  mine      - Mine pending transactions")
    print("  load      - Load transactions from mempool")
    print("  status    - Show blockchain status")
    print("  stats     - Show mining statistics")
    print("  validate  - Validate blockchain integrity")
    print("  clear     - Clear mempool file")
    print("  difficulty <n> - Set mining difficulty (1-8)")
    print("  bills     - Show available bills for current difficulty")
    print("  peers     - Show connected peers")
    print("  addpeer <addr> - Add a peer (format: ip:port)")
    print("  discover  - Discover new peers from known peers")
    print("  download blockchain - Download blockchain from peers")
    print("  download mempool - Download mempool from peers")
    print("  limits    - Show denomination limits")
    print("  help      - Show this help")
    print("  exit      - Exit the node")

def get_miner_address_from_wallet():
    """Get miner address directly from wallet"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect(('127.0.0.1', 9334))
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
    
    try:
        config_data = DataManager.load_json("wallet_config.json", {})
        mining_address = config_data.get("mining", {}).get("mining_reward_address", "")
        if mining_address:
            print(f"✅ Using mining address from config: {mining_address}")
            return mining_address
    except Exception as e:
        print(f"⚠️  Could not read config: {e}")
    
    return input("Enter miner address for rewards: ") or "miner_default_address"

def list_recent_bills():
    """List recent bills mined with their verification links"""
    print("\n💰 RECENTLY MINED BILLS:")
    print("="*80)
    
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

# Global blockchain instance
blockchain = Blockchain()
blockchain.node_host = "127.0.0.1"
blockchain.node_port = 9335

def cleanup():
    """Cleanup function for graceful shutdown"""
    print("\n💾 Performing cleanup...")
    blockchain.stop_wallet_server()
    blockchain.save_chain()
    blockchain.save_known_peers()
    print("✅ Cleanup completed")

# Register cleanup function
atexit.register(cleanup)

def main():
    ssl_context = configure_ssl_for_frozen_app()
    data_dir = DataManager.get_data_dir()
    print(f"📁 Data storage location: {data_dir}")
    
    blockchain.start_wallet_server()
    blockchain.start_peer_discovery()
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║                 LUNA BLOCKCHAIN NODE                 ║
    ║           Multi-Difficulty Mining Edition            ║
    ║              with Anti-Spam Protection               ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    print(f"📦 Genesis Block: {blockchain.chain[0].hash[:12]}...")
    print(f"⛓️  Chain Height: {len(blockchain.chain)}")
    print(f"⚙️  Difficulty: {blockchain.difficulty}")
    print(f"💰 Base Mining Reward: {blockchain.base_mining_reward} LC")
    print(f"🌐 Node Address: {blockchain.node_address}")
    print(f"🔗 Known Peers: {len(blockchain.peers)}")
    print(f"💾 Data Location: {data_dir}")
    print("🛡️  Anti-spam protection: ENABLED")
    print("Type 'help' for commands\n")
    
    while True:
        try:
            command = input("\n⛓️  node> ").strip().lower()
            
            if command == "mine":
                anim_thread = threading.Thread(target=show_mining_animation, daemon=True)
                anim_thread.start()
                
                miner_address = get_miner_address_from_wallet()
                
                if miner_address:
                    success = blockchain.mine_pending_transactions(miner_address)
                    if success:
                        print("✅ Mining completed successfully")
                else:
                    print("❌ Could not get valid miner address")
                    
            elif command == "load":
                txs = load_mempool()
                loaded_count = 0
                for tx in txs:
                    if blockchain.add_transaction(tx):
                        loaded_count += 1
                print(f"📥 Added {loaded_count}/{len(txs)} transactions to pending pool")
                
            elif command == "status":
                latest = blockchain.get_latest_block()
                stats = blockchain.get_mining_stats()
                print(f"📊 Blockchain Status:")
                print(f"   Height: {len(blockchain.chain)} blocks")
                print(f"   Pending TXs: {len(blockchain.pending_transactions)}")
                print(f"   Latest Block: {latest.hash[:16]}...")
                print(f"   Total Blocks Mined: {stats['total_blocks']}")
                print(f"   Total Rewards Minted: {stats['total_rewards']} LC")
                print(f"   Avg Mining Time: {stats['avg_time']:.2f}s")
                print(f"   Known Peers: {len(blockchain.peers)}")
                print(f"   Node Address: {blockchain.node_address}")
                print(f"   Data Location: {DataManager.get_data_dir()}")
                
            elif command == "stats":
                stats = blockchain.get_mining_stats()
                print(f"📈 Mining Statistics:")
                print(f"   Total Blocks: {stats['total_blocks']}")
                print(f"   Total Mining Time: {stats['total_time']:.2f}s")
                print(f"   Total Rewards: {stats['total_rewards']} LC")
                print(f"   Average Time/Block: {stats['avg_time']:.2f}s")
                print(f"   Current Difficulty: {blockchain.difficulty}")
                print(f"   Base Reward: {blockchain.base_mining_reward} LC")
                
            elif command == "limits":
                print("🛡️  Denomination Limits (per user):")
                for denom, limit in blockchain.denomination_limits.items():
                    print(f"   ${denom}: {limit} bills")
                    
            elif command == "check_rewards":
                print("🔍 Checking for reward transactions in node's blockchain...")
                total_rewards = 0
                
                for i, block in enumerate(blockchain.chain):
                    reward_txs = [tx for tx in block.transactions if tx.get("type") == "reward"]
                    if reward_txs:
                        print(f"Block {i}: Found {len(reward_txs)} reward transactions")
                        for tx in reward_txs:
                            print(f"  💰 Reward: {tx.get('amount', 0)} LC → {tx.get('to', 'N/A')}")
                            total_rewards += 1
                    else:
                        print(f"Block {i}: No reward transactions")
                
                print(f"📊 Total reward transactions found: {total_rewards}")
                
            elif command.startswith("difficulty"):
                parts = command.split()
                if len(parts) == 2:
                    try:
                        new_diff = int(parts[1])
                        if 1 <= new_diff <= 8:
                            blockchain.difficulty = new_diff
                            print(f"✅ Difficulty set to {new_diff}")
                            bills = blockchain.difficulty_denominations.get(new_diff, {})
                            print(f"   Available Bills: {', '.join([f'${bill}' for bill in bills.keys()])}")
                        else:
                            print("❌ Difficulty must be between 1 and 8")
                    except ValueError:
                        print("❌ Please provide a valid number")
                else:
                    print(f"Current difficulty: {blockchain.difficulty}")
                    bills = blockchain.difficulty_denominations.get(blockchain.difficulty, {})
                    print(f"Available Bills: {', '.join([f'${bill}' for bill in bills.keys()])}")
                    
            elif command == "bills":
                bills = blockchain.difficulty_denominations.get(blockchain.difficulty, {})
                print(f"💰 Available Bills for Difficulty {blockchain.difficulty}:")
                for bill, multiplier in bills.items():
                    rarity = blockchain.bill_rarity.get(bill, 1)
                    reward = blockchain.base_mining_reward * multiplier
                    requirements = blockchain.difficulty_requirements.get(bill, {})
                    print(f"   ${bill} Bill: x{multiplier} → {reward} LC")
                    print(f"      Rarity: {rarity}/100 | Min Time: {requirements.get('min_time', 0)}s")
                    print(f"      Max Attempts: {requirements.get('max_attempts', 10)}")
                    
            elif command == "validate":
                print("🔍 Validating blockchain...")
                if blockchain.is_chain_valid():
                    print("✅ Blockchain is valid and intact!")
                else:
                    print("❌ Blockchain validation failed!")
                    
            elif command == "clear":
                blockchain.clear_mempool()
                
            elif command == "bills_list" or command == "list_bills":
                list_recent_bills()
                
            elif command == "peers":
                print(f"🌐 Known Peers ({len(blockchain.peers)}):")
                for i, peer in enumerate(blockchain.peers, 1):
                    print(f"   {i:2d}. {peer}")
                    
            elif command.startswith("addpeer"):
                parts = command.split()
                if len(parts) == 2:
                    peer_address = parts[1]
                    if ":" in peer_address and peer_address.count(":") == 1:
                        if blockchain.add_peer(peer_address):
                            print(f"✅ Added peer: {peer_address}")
                        else:
                            print(f"❌ Failed to add peer: {peer_address}")
                    else:
                        print("❌ Peer address must be in format: ip:port")
                else:
                    print("❌ Usage: addpeer <ip:port>")
                    
            elif command == "discover":
                print("🔍 Discovering peers...")
                blockchain.discover_peers()
                print(f"✅ Known peers: {len(blockchain.peers)}")
                
            elif command == "download blockchain":
                print("📥 Downloading blockchain from peers...")
                if blockchain.download_blockchain():
                    print("✅ Blockchain downloaded successfully")
                else:
                    print("❌ Failed to download blockchain")
                    
            elif command == "download mempool":
                print("📥 Downloading mempool from peers...")
                if blockchain.download_mempool():
                    print("✅ Mempool downloaded successfully")
                else:
                    print("❌ Failed to download mempool")
                
            elif command == "help":
                show_help()
                print("  bills_list - Show recently mined bills with verification links")
                print("  limits     - Show denomination limits for spam protection")
                print("  check_rewards - Check reward transactions in blockchain")
                
            elif command in ["exit", "quit"]:
                print("💾 Saving blockchain and exiting...")
                break
                
            else:
                print("❌ Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n💾 Saving blockchain and exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        cleanup()
    finally:
        cleanup()