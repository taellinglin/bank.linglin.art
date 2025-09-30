#!/usr/bin/env python3
"""
wallet_server.py - Wallet server for handling wallet connections
"""
import json
import socket
import threading
import time
from typing import Dict

class WalletServer:
    def __init__(self, blockchain, host='0.0.0.0', port=9335):
        self.blockchain = blockchain
        self.host = host
        self.port = port
        self.stop_event = threading.Event()
        self.wallet_server_running = False

    def handle_wallet_connection(self, data: Dict):
        """Handle incoming wallet requests"""
        action = data.get("action")
        MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB limit

        try:
            if action == "ping":
                return {"status": "success", "message": "pong", "timestamp": time.time()}
            elif action == "get_mining_progress":
                if self.blockchain.is_mining:
                    return {
                        "status": "success", 
                        "mining": True,
                        "progress": {
                            "hashes": self.blockchain.current_mining_hashes,
                            "hash_rate": self.blockchain.current_hash_rate,
                            "current_hash": self.blockchain.current_hash
                        }
                    }
                else:
                    return {"status": "success", "mining": False}
            elif action == "get_mining_stats":
                stats = self.blockchain.get_mining_stats()
                return {"status": "success", "stats": stats}
            elif action == "get_mining_address":
                return {"status": "success", "address": "miner_default_address"}
            
            elif action == "get_balance":
                address = data.get("address")
                balance = self.blockchain.calculate_balance(address)
                return {"status": "success", "balance": balance}
            
            elif action == "add_transaction":
                transaction = data.get("transaction")
                self.blockchain.add_transaction(transaction)
                return {"status": "success", "message": "Transaction added to mempool"}
            
            elif action == "get_blockchain_info":
                return {
                    "status": "success",
                    "blockchain_height": len(self.blockchain.chain),
                    "latest_block_hash": self.blockchain.chain[-1].hash if self.blockchain.chain else None,
                    "pending_transactions": len(self.blockchain.pending_transactions),
                    "difficulty": self.blockchain.difficulty,
                    "base_reward": self.blockchain.base_mining_reward
                }
            elif action == "start_mining":
                miner_address = data.get("miner_address", "miner_default_address")
                
                def mining_thread():
                    try:
                        success = self.blockchain.mine_pending_transactions(miner_address)
                        if success:
                            self.blockchain.clear_mempool()
                    except Exception as e:
                        print(f"Mining error: {e}")
                
                thread = threading.Thread(target=mining_thread, daemon=True)
                thread.start()
                
                return {"status": "success", "message": "Mining started in background"}
            
            elif action == "get_pending_transactions":
                return {"status": "success", "transactions": self.blockchain.pending_transactions}
            elif action == "get_blockchain":
                estimated_size = len(json.dumps([block.__dict__ for block in self.blockchain.chain]))
                if estimated_size > MAX_RESPONSE_SIZE:
                    return {
                        "status": "error", 
                        "message": f"Blockchain too large ({estimated_size/1024/1024:.1f}MB). Use get_blockchain_info instead."
                    }
                # Return only the last 10 blocks
                max_blocks = 10
                chain_to_send = self.blockchain.chain[-max_blocks:] if len(self.blockchain.chain) > max_blocks else self.blockchain.chain
                
                chain_data = []
                for block in chain_to_send:
                    chain_data.append({
                        "index": block.index,
                        "previous_hash": block.previous_hash,
                        "timestamp": block.timestamp,
                        "transactions": block.transactions[:5],
                        "nonce": block.nonce,
                        "hash": block.hash,
                        "mining_time": block.mining_time
                    })
                return {"status": "success", "blockchain": chain_data, "total_blocks": len(self.blockchain.chain)}
            
            elif action == "get_difficulty_info":
                return {
                    "status": "success",
                    "difficulty": self.blockchain.difficulty,
                    "available_bills": self.blockchain.difficulty_denominations.get(self.blockchain.difficulty, {}),
                    "base_reward": self.blockchain.base_mining_reward
                }
            elif action == "load":
                from utils import load_mempool
                txs = load_mempool()
                for tx in txs:
                    self.blockchain.add_transaction(tx)
                return {"status": "success", "message": f"Loaded {len(txs)} transactions"}
            
            elif action == "clear_mempool":
                success = self.blockchain.clear_mempool()
                return {"status": "success" if success else "error", "message": "Mempool cleared" if success else "Failed to clear mempool"}
            
            return {"status": "error", "message": "Unknown action"}
        except Exception as e:
            return {"status": "error", "message": f"Server error: {str(e)}"}

    def wallet_server_thread(self):
        """Wallet server thread function"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            server_socket.settimeout(1.0)
            
            print(f"👛 Node Wallet server listening on port {self.port}")
            self.wallet_server_running = True
            
            while not self.stop_event.is_set():
                try:
                    client_socket, addr = server_socket.accept()
                    client_socket.settimeout(5.0)
                    
                    # Read the message length first (4 bytes)
                    try:
                        length_bytes = client_socket.recv(4)
                        if not length_bytes:
                            client_socket.close()
                            continue
                        
                        message_length = int.from_bytes(length_bytes, 'big')
                        
                        # Read the actual message
                        request_data = b""
                        while len(request_data) < message_length:
                            chunk = client_socket.recv(min(4096, message_length - len(request_data)))
                            if not chunk:
                                break
                            request_data += chunk
                        
                        if len(request_data) == message_length:
                            try:
                                message = json.loads(request_data.decode())
                                response = self.handle_wallet_connection(message)
                                
                                # Send response with length prefix
                                response_json = json.dumps(response)
                                response_length = len(response_json.encode())
                                client_socket.sendall(response_length.to_bytes(4, 'big'))
                                client_socket.sendall(response_json.encode())
                                
                            except json.JSONDecodeError as e:
                                error_response = {"status": "error", "message": f"Invalid JSON: {e}"}
                                error_json = json.dumps(error_response)
                                error_length = len(error_json.encode())
                                client_socket.sendall(error_length.to_bytes(4, 'big'))
                                client_socket.sendall(error_json.encode())
                        
                    except Exception as e:
                        print(f"Error handling client: {e}")
                    
                    client_socket.close()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"Wallet server client error: {e}")
        except Exception as e:
            if not self.stop_event.is_set():
                print(f"❌ Failed to start wallet server: {e}")
        finally:
            server_socket.close()
            self.wallet_server_running = False
            print("👛 Wallet server stopped")

    def start(self):
        """Start the wallet server"""
        self.wallet_thread = threading.Thread(target=self.wallet_server_thread)
        self.wallet_thread.daemon = False
        self.wallet_thread.start()
        print("✅ Node wallet server thread started")

    def stop(self):
        """Stop the wallet server gracefully"""
        print("🛑 Stopping wallet server...")
        self.stop_event.set()
        if hasattr(self, 'wallet_thread') and self.wallet_thread.is_alive():
            self.wallet_thread.join(timeout=3.0)
        print("✅ Wallet server stopped")