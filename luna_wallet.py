#!/usr/bin/env python3
"""
wallet.py - LinKoin wallet with config file and P2P functionality
"""
import json
import hashlib
import secrets
import time
import os
import sys
import socket
import threading
import base64
import binascii
import select
from pathlib import Path

class DataManager:
    """Manages data storage with EXE directory fallback to ProgramData"""
    
    _data_dir = None  # Cache the determined data directory
    
    @staticmethod
    def get_data_dir():
        """Get the best data directory (EXE dir first, then ProgramData fallback)"""
        if DataManager._data_dir is not None:
            return DataManager._data_dir
            
        # First try: Same directory as the executable
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            exe_dir = os.path.dirname(sys.executable)
            print("🔍 Running as compiled executable")
        else:
            # Running as script
            exe_dir = os.path.dirname(os.path.abspath(__file__))
            print("🔍 Running as Python script")
        
        # Test if we can write to the EXE directory
        exe_data_dir = os.path.join(exe_dir, 'data')
        print(f"🔍 Testing EXE directory: {exe_data_dir}")
        
        try:
            # Create data subdirectory and test write permissions
            os.makedirs(exe_data_dir, exist_ok=True)
            test_file = os.path.join(exe_data_dir, 'write_test.tmp')
            
            # Test write permission
            with open(test_file, 'w') as f:
                f.write('permission_test')
            
            # Test read permission
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Cleanup test file
            os.remove(test_file)
            
            print("✅ EXE directory is writable, using it for data storage")
            DataManager._data_dir = exe_data_dir
            return exe_data_dir
            
        except (PermissionError, OSError) as e:
            print(f"⚠️  Cannot write to EXE directory: {e}")
            print("🔄 Falling back to ProgramData directory...")
            
            # Fallback: ProgramData directory
            if os.name == 'nt':  # Windows
                programdata = os.environ.get('PROGRAMDATA', 'C:\\ProgramData')
                programdata_dir = os.path.join(programdata, 'Luna Suite')
                print(f"🔍 Using Windows ProgramData: {programdata_dir}")
            else:  # Linux/Mac
                programdata_dir = '/var/lib/luna-suite'
                print(f"🔍 Using Unix data directory: {programdata_dir}")
            
            # Create ProgramData directory
            try:
                os.makedirs(programdata_dir, exist_ok=True)
                
                # Test ProgramData directory too
                test_file = os.path.join(programdata_dir, 'write_test.tmp')
                with open(test_file, 'w') as f:
                    f.write('permission_test')
                os.remove(test_file)
                
                print("✅ ProgramData directory is writable, using it for data storage")
                DataManager._data_dir = programdata_dir
                return programdata_dir
                
            except Exception as prog_e:
                print(f"❌ Cannot write to ProgramData either: {prog_e}")
                print("⚠️  Falling back to temporary directory...")
                
                # Final fallback: temp directory
                import tempfile
                temp_dir = os.path.join(tempfile.gettempdir(), 'LunaSuite')
                os.makedirs(temp_dir, exist_ok=True)
                print(f"⚠️  Using temporary directory: {temp_dir}")
                DataManager._data_dir = temp_dir
                return temp_dir
    
    @staticmethod
    def get_file_path(filename):
        """Get full path for a data file with fallback support"""
        data_dir = DataManager.get_data_dir()
        return os.path.join(data_dir, filename)
    
    @staticmethod
    def save_json(filename, data):
        """Save data to JSON file with fallback support"""
        file_path = DataManager.get_file_path(filename)
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved {filename} to {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving {filename} to {file_path}: {e}")
            
            # Try emergency save to current directory
            try:
                emergency_path = filename
                with open(emergency_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"⚠️  Emergency save to: {emergency_path}")
                return True
            except Exception as emergency_e:
                print(f"💥 Critical: Could not save {filename} anywhere: {emergency_e}")
                return False
    
    @staticmethod
    def load_json(filename, default=None):
        """Load data from JSON file with fallback support"""
        if default is None:
            default = []
        
        file_path = DataManager.get_file_path(filename)
        
        # First try: Main data directory
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ Loaded {filename} from {file_path}")
                return data
        except Exception as e:
            print(f"⚠️  Error loading {filename} from {file_path}: {e}")
        
        # Fallback: Check current directory
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ Loaded {filename} from current directory")
                return data
        except Exception as e:
            print(f"⚠️  Error loading {filename} from current directory: {e}")
        
        print(f"⚠️  {filename} not found anywhere, using default")
        return default
    
    @staticmethod
    def get_data_location_info():
        """Get information about where data is being stored"""
        data_dir = DataManager.get_data_dir()
        location_type = "Unknown"
        
        if "ProgramData" in data_dir or "programdata" in data_dir.lower():
            location_type = "ProgramData"
        elif "temp" in data_dir.lower() or "tmp" in data_dir.lower():
            location_type = "Temporary Directory"
        elif "data" in data_dir.lower():
            location_type = "EXE Directory"
        else:
            location_type = "Custom Location"
        
        return location_type, data_dir
    
    @staticmethod
    def list_data_files():
        """List all data files in the data directory"""
        data_dir = DataManager.get_data_dir()
        try:
            files = os.listdir(data_dir)
            data_files = [f for f in files if f.endswith(('.json', '.dat'))]
            print(f"📁 Data files in {data_dir}:")
            for file in data_files:
                file_path = os.path.join(data_dir, file)
                size = os.path.getsize(file_path)
                print(f"   {file} ({size} bytes)")
            return data_files
        except Exception as e:
            print(f"❌ Error listing data files: {e}")
            return []

class SimpleEncryptor:
    """Simple encryption using XOR for basic obfuscation"""
    def __init__(self, key=None):
        self.key = key or "lincoin_default_key_12345"
        # Ensure key is long enough and consistent
        while len(self.key) < 32:
            self.key += self.key
        # Use only the first 32 characters for consistency
        self.key = self.key[:32]
    
    def encrypt(self, data):
        """Simple XOR encryption"""
        if not data:
            return data
            
        encrypted = bytearray()
        key_bytes = self.key.encode('utf-8')
        data_bytes = data.encode('utf-8')
        
        for i, char in enumerate(data_bytes):
            encrypted.append(char ^ key_bytes[i % len(key_bytes)])
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data):
        """Simple XOR decryption"""
        if not encrypted_data:
            return encrypted_data
            
        try:
            # First, check if it's already decrypted (JSON)
            if encrypted_data.strip().startswith('{') or encrypted_data.strip().startswith('['):
                return encrypted_data
                
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = bytearray()
            key_bytes = self.key.encode('utf-8')
            
            for i, char in enumerate(encrypted_bytes):
                decrypted.append(char ^ key_bytes[i % len(key_bytes)])
            
            return decrypted.decode('utf-8')
        except (binascii.Error, UnicodeDecodeError):
            # If decryption fails, return as-is (might already be decrypted)
            return encrypted_data
        except Exception as e:
            print(f"⚠️  Decryption error: {e}")
            return encrypted_data

class WalletConfig:
    def __init__(self, config_file="wallet_config.json"):
        self.config_file = config_file
        self.config = self.load_config() or self.default_config()

    def get_ip_from_domain(self, domain_name):
        """
        Get IP address from a domain name.
        
        Args:
            domain_name (str): The domain name to resolve (e.g., 'example.com')
        
        Returns:
            str: The IP address, or None if resolution fails
        """
        try:
            # Remove http:// or https:// if present
            if domain_name.startswith(('http://', 'https://')):
                domain_name = domain_name.split('://')[1]
            
            # Remove trailing slashes and port numbers if present
            domain_name = domain_name.split('/')[0].split(':')[0]
            
            # Get IP address
            ip_address = socket.gethostbyname(domain_name)
            return ip_address
            
        except socket.gaierror:
            print(f"❌ Could not resolve domain: {domain_name}")
            return None
        except Exception as e:
            print(f"❌ Error resolving domain {domain_name}: {e}")
            return None

    def default_config(self):
        default_peer = self.get_ip_from_domain("https://linglin.art") + ":9333"
        return {
            "network": {
                "port": 9333,
                "peers": ["63.41.180.121:9333", default_peer],
                "discovery_enabled": True,
                "max_peers": 10
            },
            "mining": {
                "mining_reward_address": "LKC_5db88dcb4e3df2cc443585234793af1d_6492dd04",
                "auto_mine": False,
                "mining_fee": 0.001
            },
            "security": {
                "encrypt_wallet": True,
                "auto_backup": True,
                "backup_interval": 3600
            },
            "rpc": {
                "enabled": True,
                "port": 9334,
                "allow_remote": False
            },
            "node": {
                "host": "127.0.0.1",
                "port": 9335
            }
        }
    
    def load_config(self):
        """Load config from DataManager with fallback"""
        return DataManager.load_json(self.config_file)
    
    def save_config(self):
        """Save config using DataManager"""
        DataManager.save_json(self.config_file, self.config)
    
    def get_peers(self):
        return self.config["network"]["peers"]
    
    def add_peer(self, peer_address):
        if peer_address not in self.config["network"]["peers"]:
            self.config["network"]["peers"].append(peer_address)
            self.save_config()
    
    def remove_peer(self, peer_address):
        if peer_address in self.config["network"]["peers"]:
            self.config["network"]["peers"].remove(peer_address)
            self.save_config()

class Wallet:
    def __init__(self, wallet_file="wallet.dat", config_file="wallet_config.json"):
        # Show data directory info on startup
        location_type, data_dir = DataManager.get_data_location_info()
        print(f"📁 Wallet data storage: {location_type} - {data_dir}")
        
        self.wallet_file = wallet_file
        self.config = WalletConfig(config_file)
        
        # Initialize encryption FIRST
        self.cipher = SimpleEncryptor()
        
        # Then load wallet
        self.addresses = self.load_wallet() or []
        self.peer_nodes = set(self.config.get_peers())
        self.running = False
        self.server_socket = None
        self.blockchain = []
        self.mempool = []
        
        # Automatically sync blockchain on startup
        print("🔄 Auto-syncing blockchain on startup...")
        self.load_blockchain()
        
        # Start P2P and RPC servers
        self.start_p2p_server()
        self.start_rpc_server()
        
        # Discover peers
        self.discover_peers()
        self.start_periodic_sync(300)
   
    def get_effective_balance(self, address):
        """Get balance including both confirmed and pending transactions"""
        confirmed_balance = 0
        pending_debits = 0
        
        # Find the wallet for this address
        wallet_addr = None
        for addr in self.addresses:
            if addr["address"] == address:
                wallet_addr = addr
                break
        
        if not wallet_addr:
            return 0
        
        # Start with confirmed balance
        confirmed_balance = wallet_addr["balance"]
        
        # Subtract pending outgoing transactions
        for tx in wallet_addr["transactions"]:
            if tx.get("status") == "pending" and tx.get("type") == "outgoing":
                pending_debits += tx.get("amount", 0)
        
        # Also check mempool for pending transactions that aren't in wallet history yet
        for tx in self.mempool:
            if tx.get("from") == address:
                # Check if this transaction is already in wallet history
                tx_exists = False
                for wtx in wallet_addr["transactions"]:
                    if (wtx.get("to") == tx.get("to") and 
                        wtx.get("amount") == tx.get("amount") and 
                        wtx.get("status") == "pending"):
                        tx_exists = True
                        break
                
                if not tx_exists:
                    pending_debits += tx.get("amount", 0)
        
        effective_balance = confirmed_balance - pending_debits
        return max(0, effective_balance)

    def check_transaction_status(self, tx_signature):
        """Check if a transaction has been confirmed"""
        # Check if transaction is in any block
        for block_index, block in enumerate(self.blockchain):
            for tx in block.get("transactions", []):
                if tx.get("signature") == tx_signature:
                    return {
                        "status": "confirmed",
                        "block_height": block_index + 1,
                        "confirmations": len(self.blockchain) - block_index
                    }
        
        # Check if transaction is still in mempool
        for tx in self.mempool:
            if tx.get("signature") == tx_signature:
                return {"status": "pending", "confirmations": 0}
        
        return {"status": "unknown", "confirmations": 0}

    def trigger_mining(self):
        """Ask the node to mine a block with pending transactions - FIXED"""
        try:
            # Use the correct action that the node understands
            response = self.send_to_node({"action": "start_mining", "miner_address": self.get_default_address()})
            if response and response.get("status") == "success":
                print("✅ Mining triggered successfully")
                # Wait a bit for mining to complete
                time.sleep(5)
                # Sync to get the new block
                self.load_blockchain()
                return True
            else:
                error_msg = response.get("message", "Unknown error") if response else "No response"
                print(f"❌ Failed to trigger mining: {error_msg}")
                return False
        except Exception as e:
            print(f"❌ Error triggering mining: {e}")
            return False
    # In wallet.py, when creating a spend transaction:
    def create_spend_transaction(self, from_address, to_address, amount, spent_transaction_signature):
        """Create a spend transaction that references the specific UTXO being spent"""
        transaction = {
            "type": "spend",
            "from": from_address,
            "to": to_address,
            "amount": amount,
            "timestamp": time.time(),
            "signature": f"spend_{int(time.time())}_{hashlib.sha256(f'{from_address}{to_address}{amount}'.encode()).hexdigest()[:16]}",
            "spent_transaction": spent_transaction_signature,  # Reference to the UTXO being spent
            "spent_output_index": 0  # Which output of the previous transaction is being spent
        }
        return transaction
    def check_node_status(self):
        """Check if the node is running and mining"""
        try:
            response = self.send_to_node({"action": "get_status"})
            if response and response.get("status") == "success":
                status = response.get("node_status", {})
                print(f"🟢 Node is running and responsive")
                print(f"   📊 Blockchain Height: {status.get('blockchain_height', 0)}")
                print(f"   📋 Mempool Size: {status.get('mempool_size', 0)}")
                print(f"   ⚙️  Difficulty: {status.get('difficulty', 0)}")
                print(f"   🌐 Connected Peers: {status.get('peers_count', 0)}")
                print(f"   ⛏️  Currently Mining: {'Yes' if status.get('is_mining') else 'No'}")
                print(f"   🔗 Node Address: {status.get('node_address', 'Unknown')}")
                return True
            else:
                error_msg = response.get("message", "Unknown error") if response else "No response"
                print(f"🔴 Node responded with error: {error_msg}")
                return False
        except Exception as e:
            print(f"🔴 Cannot connect to node: {e}")
            print("💡 Make sure the node is running with: python luna_node.py")
            return False

    def start_rpc_server(self):
        """Start RPC server to handle requests from nodes"""
        def rpc_server_thread(wallet_instance):
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                server_socket.bind(('0.0.0.0', wallet_instance.config.config["rpc"]["port"]))
                server_socket.listen(5)
                
                print(f"📡 RPC Server listening on port {wallet_instance.config.config['rpc']['port']}")
                
                while wallet_instance.running:
                    try:
                        client_socket, addr = server_socket.accept()
                        data = client_socket.recv(4096)
                        if data:
                            try:
                                message = json.loads(data.decode())
                                response = wallet_instance.handle_rpc_request(message)
                                client_socket.sendall(json.dumps(response).encode())
                            except json.JSONDecodeError:
                                client_socket.sendall(json.dumps({"status": "error", "message": "Invalid JSON"}).encode())
                        client_socket.close()
                    except Exception as e:
                        if wallet_instance.running:  # Only print if we're still running
                            print(f"RPC server client error: {e}")
            except Exception as e:
                print(f"❌ Failed to start RPC server: {e}")
            finally:
                server_socket.close()
        
        # Pass self as argument to the thread function
        threading.Thread(target=rpc_server_thread, args=(self,), daemon=True).start()

    def show_wallet_info():
        """Global function to show wallet info"""
        if wallet:
            wallet.show_wallet_info()
        else:
            print("❌ Wallet not initialized")

    def handle_rpc_request(self, message):
        """Handle RPC requests from nodes"""
        action = message.get("action")
        
        if action == "ping":
            return {"status": "success", "message": "pong", "timestamp": time.time()}
        elif action == "clear_mempool":
            block_height = message.get("block_height")
            block_hash = message.get("block_hash")
            cleared_count = message.get("cleared_transactions", 0)
            
            print(f"🧹 Mempool cleared by node (block {block_height}, cleared {cleared_count} transactions)")
            self.mempool = []  # Clear wallet's mempool
            DataManager.save_json("mempool.json", self.mempool)
            return {"status": "success", "message": "Mempool cleared"}
        elif action == "get_mining_address":
            return self.handle_mining_address_request()
        
        elif action == "get_balance":
            address = message.get("address")
            balance = self.get_balance(address)
            return {"status": "success", "balance": balance}
        
        elif action == "create_transaction":
            try:
                to_address = message.get("to_address")
                amount = message.get("amount")
                memo = message.get("memo", "")
                transaction = self.create_transaction(self.get_default_address(), to_address, amount, memo)
                return {"status": "success", "transaction": transaction}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        elif action == "get_addresses":
            return {"status": "success", "addresses": self.get_addresses()}
        
        elif action == "get_mining_reward_address":
            # Return the configured mining reward address
            mining_address = self.config.config["mining"]["mining_reward_address"]
            if not mining_address and self.addresses:
                # If no mining address is configured, use the first address
                mining_address = self.addresses[0]["address"]
                self.config.config["mining"]["mining_reward_address"] = mining_address
                self.config.save_config()
            
            return {"status": "success", "address": mining_address}
        
        return {"status": "error", "message": "Unknown action"}

    def get_balance(self, address):
        """Get balance for a specific address"""
        for wallet in self.addresses:
            if wallet["address"] == address:
                return wallet["balance"]
        return 0

    def encrypt_data(self, data):
        """Encrypt sensitive wallet data"""
        if self.config.config["security"]["encrypt_wallet"]:
            return self.cipher.encrypt(data)
        return data
    
    def decrypt_data(self, encrypted_data):
        """Decrypt wallet data"""
        if self.config.config["security"]["encrypt_wallet"]:
            return self.cipher.decrypt(encrypted_data)
        return encrypted_data
    
    def generate_address(self, label=""):
        """Generate a new cryptocurrency address with enhanced security"""
        # Generate cryptographically secure keys
        private_key = secrets.token_hex(32)  # 256-bit private key
        public_key = hashlib.sha256(private_key.encode()).hexdigest()
        
        # Create address using double hashing for security
        address_hash = hashlib.sha256(public_key.encode()).hexdigest()
        checksum = hashlib.sha256(address_hash.encode()).hexdigest()[:8]
        address = f"LKC_{address_hash[:32]}_{checksum}"
        
        wallet_data = {
            "private_key": self.encrypt_data(private_key),
            "public_key": self.encrypt_data(public_key),
            "address": address,
            "label": label,
            "created": time.time(),
            "balance": 0,
            "transactions": []
        }
        
        self.addresses.append(wallet_data)
        self.save_wallet()
        
        # Set as mining reward address if first address
        if len(self.addresses) == 1:
            self.config.config["mining"]["mining_reward_address"] = address
            self.config.save_config()
        
        return address
    
    def get_addresses(self):
        return [addr["address"] for addr in self.addresses]
    
    def get_default_address(self):
        if self.addresses:
            return self.addresses[0]["address"]
        return self.generate_address("Default Wallet")
    
    def get_address_info(self, address):
        for wallet in self.addresses:
            if wallet["address"] == address:
                return {
                    "address": wallet["address"],
                    "label": wallet["label"],
                    "balance": wallet["balance"],
                    "created": time.strftime("%Y-%m-%d %H:%M:%S", 
                              time.localtime(wallet["created"])),
                    "transaction_count": len(wallet["transactions"])
                }
        return None
    
    def create_transaction(self, from_address: str, to_address: str, amount: float, memo: str = ""):
        """Create a signed transaction ready for broadcasting"""
        for wallet in self.addresses:
            if wallet["address"] == from_address:
                # Use effective balance instead of confirmed balance
                effective_balance = self.get_effective_balance(from_address)
                
                if effective_balance < amount:
                    raise ValueError(f"Insufficient balance. Available: {effective_balance} LC, Attempted: {amount} LC")
                
                # Decrypt private key for signing
                private_key = self.decrypt_data(wallet["private_key"])
                
                transaction = {
                    "version": "1.0",
                    "type": "transfer",
                    "from": from_address,
                    "to": to_address,
                    "amount": amount,
                    "fee": self.config.config["mining"]["mining_fee"],
                    "timestamp": time.time(),
                    "memo": memo,
                    "nonce": secrets.token_hex(8),
                    "signature": self.sign_transaction(private_key, from_address, to_address, amount)
                }
                
                # Add to transaction history with pending status
                wallet["transactions"].append({
                    "type": "outgoing",
                    "to": to_address,
                    "amount": amount,
                    "timestamp": transaction["timestamp"],
                    "status": "pending",
                    "signature": transaction["signature"]
                })
                
                self.save_wallet()
                return transaction
        raise ValueError("Address not found in wallet")
    
    def sign_transaction(self, private_key: str, from_addr: str, to_addr: str, amount: float):
        """Create cryptographic signature for transaction"""
        sign_data = f"{private_key}{from_addr}{to_addr}{amount}{time.time()}"
        return hashlib.sha256(sign_data.encode()).hexdigest()
    
    def verify_transaction(self, transaction):
        """Verify transaction signature"""
        # Simple verification for demo purposes
        # In a real implementation, you'd use proper cryptographic verification
        expected_data = f"{transaction['from']}{transaction['to']}{transaction['amount']}{transaction['timestamp']}"
        expected_hash = hashlib.sha256(expected_data.encode()).hexdigest()
        return transaction["signature"].startswith(expected_hash[:16])
    
    def broadcast_transaction(self, transaction):
        """Broadcast transaction to P2P network"""
        print(f"📡 Broadcasting transaction to {len(self.peer_nodes)} peers...")
        
        success_count = 0
        for peer in list(self.peer_nodes):
            try:
                if self.send_to_peer(peer, {
                    "type": "transaction",
                    "data": transaction
                }):
                    success_count += 1
            except:
                print(f"❌ Failed to broadcast to {peer}")
                self.peer_nodes.discard(peer)
        
        # Also send to node directly
        try:
            node_host = self.config.config["node"]["host"]
            node_port = self.config.config["node"]["port"]
            response = self.send_to_node({
                "action": "add_transaction",
                "transaction": transaction
            }, node_host, node_port)
            if response and response.get("status") == "success":
                success_count += 1
                print("✅ Transaction sent to node successfully")
        except Exception as e:
            print(f"❌ Failed to send to node: {e}")
        
        print(f"✅ Transaction broadcast to {success_count} destinations")
        return success_count > 0
    
    def send_to_peer(self, peer_address, message):
        """Send message to a specific peer"""
        try:
            host, port = peer_address.split(":")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((host, int(port)))
                s.sendall(json.dumps(message).encode())
                response = s.recv(1024)
                return json.loads(response.decode())
        except:
            return False
            
    def send_to_node(self, data, host=None, port=None, timeout=10):
        """Send data to the blockchain node with better error handling"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if host is None:
                    host = self.config.config["node"]["host"]
                if port is None:
                    port = self.config.config["node"]["port"]
                    
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout)
                    s.connect((host, int(port)))
                    
                    # Send message with length prefix
                    message_json = json.dumps(data)
                    message_data = message_json.encode()
                    message_length = len(message_data)
                    
                    s.sendall(message_length.to_bytes(4, 'big'))
                    s.sendall(message_data)
                    
                    # Receive response
                    length_bytes = s.recv(4)
                    if not length_bytes or len(length_bytes) != 4:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        return {"status": "error", "message": "Invalid response length"}
                    
                    response_length = int.from_bytes(length_bytes, 'big')
                    response_data = b""
                    bytes_received = 0
                    
                    while bytes_received < response_length:
                        chunk = s.recv(min(4096, response_length - bytes_received))
                        if not chunk:
                            break
                        response_data += chunk
                        bytes_received += len(chunk)
                    
                    if bytes_received == response_length:
                        try:
                            return json.loads(response_data.decode())
                        except json.JSONDecodeError:
                            return {"status": "error", "message": "Invalid JSON response"}
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        return {"status": "error", "message": "Incomplete response"}
                        
            except socket.timeout:
                if attempt < max_retries - 1:
                    print(f"⏳ Timeout, retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                return {"status": "error", "message": "Node communication timeout"}
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    print(f"🔌 Connection refused, retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                return {"status": "error", "message": "Connection refused - node not running"}
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"🔄 Error, retrying... ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    continue
                return {"status": "error", "message": f"Node communication error: {str(e)}"}
        
        return {"status": "error", "message": "Max retries exceeded"}
    
    def start_p2p_server(self):
        """Start P2P server to receive transactions and blocks"""
        def server_thread():
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                self.server_socket.bind(('0.0.0.0', self.config.config["network"]["port"]))
                self.server_socket.listen(5)
                self.running = True
                
                print(f"🌐 P2P Server listening on port {self.config.config['network']['port']}")
                
                while self.running:
                    try:
                        client_socket, addr = self.server_socket.accept()
                        threading.Thread(target=self.handle_client, 
                                       args=(client_socket, addr)).start()
                    except:
                        if self.running:
                            break
            except Exception as e:
                print(f"❌ Failed to start P2P server: {e}")
        
        threading.Thread(target=server_thread, daemon=True).start()
    
    def handle_client(self, client_socket, addr):
        """Handle incoming P2P connections"""
        try:
            data = client_socket.recv(4096)
            if data:
                message = json.loads(data.decode())
                response = self.process_message(message, f"{addr[0]}:{addr[1]}")
                
                # Send response if any
                if response:
                    client_socket.sendall(json.dumps(response).encode())
                else:
                    client_socket.sendall(json.dumps({"status": "ok"}).encode())
        except:
            pass
        finally:
            client_socket.close()

    def debug_blockchain_content(self):
        """Debug what's actually in the blockchain"""
        print("\n🔍 Blockchain Content Debug:")
        print("=" * 60)
        print(f"Total blocks: {len(self.blockchain)}")
        
        for i, block in enumerate(self.blockchain):
            print(f"\nBlock {i}:")
            print(f"  Hash: {block.get('hash', 'N/A')[:20]}...")
            print(f"  Transactions: {len(block.get('transactions', []))}")
            
            # Show all transactions in this block
            for j, tx in enumerate(block.get("transactions", [])):
                tx_type = tx.get("type", "unknown")
                amount = tx.get("amount", 0)
                to_addr = tx.get("to", "N/A")[:20] + "..." if tx.get("to") else "N/A"
                from_addr = tx.get("from", "N/A")[:20] + "..." if tx.get("from") else "N/A"
                
                print(f"    TX {j}: {tx_type} - {amount} LC")
                print(f"      From: {from_addr}")
                print(f"      To: {to_addr}")
                
                # Show reward-specific info
                if tx_type == "reward":
                    print(f"      Denomination: {tx.get('denomination', 'N/A')}")
                    print(f"      Serial: {tx.get('serial_number', 'N/A')}")

    def process_message(self, message, peer_address):
        """Process incoming P2P messages"""
        msg_type = message.get("type")
        
        if msg_type == "transaction":
            transaction = message.get("data")
            if self.verify_transaction(transaction):
                print(f"📥 Received valid transaction from {peer_address}")
                # Add to local mempool
                self.add_to_mempool(transaction)
                
                # Relay to other peers
                self.relay_message(message, peer_address)
        
        elif msg_type == "block":
            block = message.get("data")
            print(f"📦 Received new block from {peer_address}")
            self.process_new_block(block)
                
        elif msg_type == "peer_list":
            new_peers = message.get("data", [])
            for peer in new_peers:
                if peer not in self.peer_nodes and peer != f"127.0.0.1:{self.config.config['network']['port']}":
                    self.peer_nodes.add(peer)
                    self.config.add_peer(peer)
        
        elif msg_type == "ping":
            print(f"🏓 Ping from {peer_address}")
            return {"type": "pong"}
            
        elif msg_type == "get_mining_address":
            return self.handle_mining_address_request()
        
        return None
    
    def handle_mining_address_request(self):
        """Return the mining reward address"""
        # First check if we have a configured mining reward address
        mining_address = self.config.config["mining"]["mining_reward_address"]
        
        if mining_address:
            # Verify the address exists in our wallet
            for wallet in self.addresses:
                if wallet["address"] == mining_address:
                    return {
                        "status": "success", 
                        "address": mining_address,
                        "source": "configured"
                    }
        
        # If no configured address or it doesn't exist, use the first address
        if self.addresses:
            mining_address = self.addresses[0]["address"]
            # Update config to use this address
            self.config.config["mining"]["mining_reward_address"] = mining_address
            self.config.save_config()
            
            return {
                "status": "success", 
                "address": mining_address,
                "source": "first_wallet"
            }
        else:
            # Generate a new address if none exists
            new_address = self.generate_address("Mining Reward")
            self.config.config["mining"]["mining_reward_address"] = new_address
            self.config.save_config()
            
            return {
                "status": "success", 
                "address": new_address,
                "source": "newly_created"
            }

    def debug_balances(self):
        """Debug method to see transaction processing"""
        print("\n🔍 Balance Debug:")
        print("=" * 50)
        
        for wallet in self.addresses:
            print(f"Address: {wallet['address']}")
            print(f"Balance: {wallet['balance']} LC")
            print(f"Transactions: {len(wallet['transactions'])}")
            
            # Show recent transactions
            for tx in wallet['transactions'][-5:]:  # Last 5 transactions
                tx_type = tx.get('type', 'unknown')
                amount = tx.get('amount', 0)
                status = tx.get('status', 'unknown')
                print(f"  {tx_type}: {amount} LC ({status})")
            print()

    def process_new_block(self, block):
        """Process a new block received from the network"""
        # Add to blockchain
        self.blockchain.append(block)
        
        # Update balances based on ALL transactions in the block
        self.update_balances(block.get("transactions", []))
        
        # Remove transactions from mempool that are in this block
        # Use signature or unique identifier to match transactions
        block_tx_signatures = [tx.get("signature") for tx in block.get("transactions", []) if tx.get("signature")]
        self.mempool = [tx for tx in self.mempool 
                    if tx.get("signature") not in block_tx_signatures]
        
        print(f"✅ New block added to local chain (height: {len(self.blockchain)})")
        print(f"   Contains {len(block.get('transactions', []))} transactions")
    
    def update_balances(self, transactions):
        """Update wallet balances based on transactions - reset balances first to prevent double-counting"""
        # First reset all balances to zero to prevent double-counting
        for wallet in self.addresses:
            wallet["balance"] = 0
        
        # Process all transactions in the blockchain to calculate correct balances
        for block in self.blockchain:
            for tx in block.get("transactions", []):
                # Handle mining reward transactions (they have 'to' but no 'from')
                if tx.get("type") == "reward":
                    for wallet in self.addresses:
                        if wallet["address"] == tx.get("to"):
                            wallet["balance"] += tx.get("amount", 0)
                            # Add to transaction history if not already there
                            if not any(t.get("signature") == tx.get("signature") for t in wallet["transactions"]):
                                wallet["transactions"].append({
                                    "type": "reward",
                                    "from": "mining_reward",
                                    "amount": tx.get("amount"),
                                    "timestamp": tx.get("timestamp"),
                                    "status": "confirmed",
                                    "block_height": self.blockchain.index(block) + 1
                                })
                    continue  # Skip regular transaction processing for rewards
                
                # Handle regular outgoing transactions
                if "from" in tx:
                    for wallet in self.addresses:
                        if wallet["address"] == tx.get("from"):
                            wallet["balance"] -= tx.get("amount", 0)
                            # Update transaction status
                            for wtx in wallet["transactions"]:
                                if (wtx.get("to") == tx.get("to") and 
                                    wtx.get("amount") == tx.get("amount") and 
                                    wtx.get("status") == "pending"):
                                    wtx["status"] = "confirmed"
                                    wtx["block_height"] = self.blockchain.index(block) + 1
                
                # Handle incoming transactions
                if "to" in tx:
                    for wallet in self.addresses:
                        if wallet["address"] == tx.get("to"):
                            wallet["balance"] += tx.get("amount", 0)
                            # Add to transaction history if not already there
                            if not any(t.get("signature") == tx.get("signature") for t in wallet["transactions"]):
                                wallet["transactions"].append({
                                    "type": "incoming",
                                    "from": tx.get("from", "unknown"),
                                    "amount": tx.get("amount"),
                                    "timestamp": tx.get("timestamp"),
                                    "status": "confirmed",
                                    "block_height": self.blockchain.index(block) + 1
                                })
        
        self.save_wallet()
    
    def add_to_mempool(self, transaction):
        """Add transaction to local mempool"""
        # Check if transaction already exists
        if not any(tx.get("signature") == transaction.get("signature") for tx in self.mempool):
            self.mempool.append(transaction)
            
            # Save to file using DataManager
            DataManager.save_json("mempool.json", self.mempool)

    def debug_rewards(self):
        """Debug reward transactions in the blockchain"""
        print("\n🔍 Reward Transaction Debug:")
        print("=" * 60)
        
        total_rewards_found = 0
        your_rewards = 0
        your_address = self.get_default_address()
        
        for block_index, block in enumerate(self.blockchain):
            print(f"\nBlock {block_index}:")
            transactions = block.get("transactions", [])
            
            for tx_index, tx in enumerate(transactions):
                tx_type = tx.get("type")
                amount = tx.get("amount", 0)
                to_address = tx.get("to", "")
                
                print(f"  TX {tx_index}: type={tx_type}, amount={amount}, to={to_address}")
                
                if tx_type == "reward":
                    total_rewards_found += 1
                    if to_address == your_address:
                        your_rewards += amount
                        print(f"    🎯 THIS IS YOUR REWARD! +{amount} LC")
        
        print(f"\n📊 Summary:")
        print(f"Total reward transactions found: {total_rewards_found}")
        print(f"Your rewards: {your_rewards} LC")
        print(f"Your address: {your_address}")
        
        # Also check what the wallet thinks your balance is
        for wallet in self.addresses:
            if wallet["address"] == your_address:
                print(f"Wallet balance: {wallet['balance']} LC")
                print(f"Wallet transactions: {len(wallet['transactions'])}")
                break

    def relay_message(self, message, source_peer):
        """Relay message to other peers (flooding)"""
        for peer in list(self.peer_nodes):
            if peer != source_peer:
                try:
                    self.send_to_peer(peer, message)
                except:
                    self.peer_nodes.discard(peer)
    
    def discover_peers(self):
        """Discover new peers in the network"""
        if not self.config.config["network"]["discovery_enabled"]:
            return
        
        print("🔍 Discovering peers...")
        initial_peers = list(self.peer_nodes)
        for peer in initial_peers:
            try:
                response = self.send_to_peer(peer, {
                    "type": "peer_list_request"
                })
                if response and response.get("type") == "peer_list":
                    new_peers = response.get("data", [])
                    for new_peer in new_peers:
                        if new_peer not in self.peer_nodes and new_peer != f"127.0.0.1:{self.config.config['network']['port']}":
                            self.peer_nodes.add(new_peer)
                            self.config.add_peer(new_peer)
            except:
                self.peer_nodes.discard(peer)

    def save_blockchain_to_file(self):
        """Save blockchain to local file using DataManager"""
        DataManager.save_json("blockchain.json", self.blockchain)
        print(f"💾 Blockchain saved to file ({len(self.blockchain)} blocks)")

    def sync_mempool_with_node(self):
        """Synchronize mempool with the node"""
        try:
            print("🔄 Syncing mempool with node...")
            response = self.send_to_node({"action": "get_pending_transactions"})
            
            if response and response.get("status") == "success":
                node_mempool = response.get("transactions", [])
                
                # Replace wallet's mempool with node's mempool
                self.mempool = node_mempool
                DataManager.save_json("mempool_wallet.json", self.mempool)
                
                print(f"✅ Mempool synced: {len(self.mempool)} transactions")
                return True
            else:
                print("❌ Failed to sync mempool with node")
                return False
                
        except Exception as e:
            print(f"❌ Error syncing mempool: {e}")
            return False

    def load_mempool(self):
        """Load mempool from file, but prefer node's mempool if available"""
        # First try to sync with node
        if self.sync_mempool_with_node():
            return self.mempool
        
        # Fallback to local file
        self.mempool = DataManager.load_json("mempool_wallet.json", [])
        print(f"✅ Loaded mempool from file: {len(self.mempool)} transactions")
        return self.mempool

    def load_blockchain(self):
        """Load blockchain using the same method as the node - download from web first, then peers"""
        print("🔄 Syncing with blockchain node...")
        
        # First try to get blockchain info
        try:
            info_response = self.send_to_node({"action": "get_blockchain_info"})
            if info_response and info_response.get("status") == "success":
                node_height = info_response.get("blockchain_height", 0)
                pending_txs = info_response.get("pending_transactions", 0)
                print(f"📊 Node info: Height {node_height}, Pending TXs: {pending_txs}")
                
                # If node has more blocks than we do, download the blockchain
                if node_height > len(self.blockchain):
                    print(f"📥 Node has {node_height} blocks, we have {len(self.blockchain)} - downloading...")
                    
                    # Use the same method as node to download blockchain
                    success = self.download_blockchain_from_node()
                    if success:
                        print(f"✅ Blockchain synchronized to {len(self.blockchain)} blocks")
                        return True
                    else:
                        print("❌ Failed to download blockchain from node, trying local file...")
                else:
                    print("ℹ️  Blockchain is already up to date")
                    self.update_balances_from_blockchain()
                    return True
            else:
                print("❌ Could not get blockchain info from node")
                
        except Exception as e:
            print(f"❌ Error communicating with node: {e}")
        
        # Fallback to local file
        print("🔄 Falling back to local blockchain file...")
        self.load_blockchain_from_file()
        return False

    def download_blockchain_from_node(self):
        """Download blockchain from node using the same protocol as node-to-node communication"""
        try:
            print("📥 Downloading blockchain from node...")
            
            # Request the full blockchain with a longer timeout
            response = self.send_to_node({"action": "get_blockchain"}, timeout=30)
            
            if response and response.get("status") == "success":
                # Handle the response format - same as node's download_blockchain method
                chain_data = response.get("blockchain", [])
                total_blocks = response.get("total_blocks", len(chain_data))
                
                print(f"📥 Received {len(chain_data)} blocks, total: {total_blocks}")
                
                if len(chain_data) > len(self.blockchain):
                    print(f"✅ Downloaded blockchain with {total_blocks} blocks from node")
                    
                    # Replace our chain with the downloaded one (same as node does)
                    self.blockchain = chain_data
                    
                    # Update balances from the new blockchain
                    self.update_balances_from_blockchain()
                    
                    # Save to file
                    self.save_blockchain_to_file()
                    
                    return True
                else:
                    print("ℹ️  Node has same or shorter chain, keeping current blockchain")
                    return True
            else:
                error_msg = response.get("message", "Unknown error") if response else "No response"
                print(f"❌ Failed to download blockchain from node: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading blockchain from node: {e}")
            return False

    def update_balances_from_blockchain(self):
        """Update wallet balances from the entire blockchain - FIXED VERSION"""
        print("💰 Updating balances from blockchain...")
        
        # Reset all balances to zero first to prevent double-counting
        for wallet in self.addresses:
            wallet["balance"] = 0
            # Keep transaction history but mark all as needs review
            for tx in wallet["transactions"]:
                if tx.get("status") == "confirmed":
                    tx["needs_verification"] = True
        
        # Process ALL blocks and ALL transactions
        total_transactions = 0
        reward_transactions = 0
        
        print(f"🔍 Processing {len(self.blockchain)} blocks...")
        
        for block_index, block in enumerate(self.blockchain):
            transactions = block.get("transactions", [])
            total_transactions += len(transactions)
            
            print(f"  Block {block_index}: {len(transactions)} transactions")
            
            for tx in transactions:
                tx_type = tx.get("type")
                amount = tx.get("amount", 0)
                to_address = tx.get("to", "")
                
                # CRITICAL FIX: Better reward detection
                if tx_type == "reward":
                    print(f"    🎯 REWARD TX: {amount} LC to {to_address[:16]}...")
                    
                    for wallet_addr in self.addresses:
                        if wallet_addr["address"] == to_address:
                            wallet_addr["balance"] += amount
                            reward_transactions += 1
                            
                            # Create unique signature for this reward
                            tx_signature = tx.get("signature") or f"reward_{block_index}_{to_address}_{amount}"
                            
                            # Check if this reward already exists
                            reward_exists = False
                            for existing_tx in wallet_addr["transactions"]:
                                if (existing_tx.get("type") == "reward" and 
                                    existing_tx.get("block_height") == block_index + 1 and
                                    existing_tx.get("amount") == amount):
                                    reward_exists = True
                                    existing_tx["needs_verification"] = False
                                    break
                            
                            if not reward_exists:
                                wallet_addr["transactions"].append({
                                    "type": "reward",
                                    "from": "mining_reward",
                                    "to": to_address,
                                    "amount": amount,
                                    "timestamp": tx.get("timestamp", time.time()),
                                    "status": "confirmed",
                                    "block_height": block_index + 1,
                                    "signature": tx_signature,
                                    "denomination": tx.get("denomination", "unknown"),
                                    "serial_number": tx.get("serial_number", ""),
                                    "block_hash": block.get("hash", "")
                                })
                                print(f"      ✅ ADDED REWARD: +{amount} LC")
                
                # Also handle regular transactions
                elif tx_type == "transfer":
                    # Handle outgoing
                    if tx.get("from"):
                        for wallet_addr in self.addresses:
                            if wallet_addr["address"] == tx.get("from"):
                                wallet_addr["balance"] -= amount
                    
                    # Handle incoming  
                    if to_address:
                        for wallet_addr in self.addresses:
                            if wallet_addr["address"] == to_address:
                                wallet_addr["balance"] += amount
        
        # Clean up verification markers
        for wallet in self.addresses:
            wallet["transactions"] = [tx for tx in wallet["transactions"] if not tx.get("needs_verification", False)]
        
        self.save_wallet()
        print(f"✅ Balances updated: {reward_transactions} reward transactions found")

    def load_blockchain_from_file(self):
        """Load blockchain from local file using DataManager"""
        self.blockchain = DataManager.load_json("blockchain.json", [])
        print(f"✅ Loaded blockchain from file ({len(self.blockchain)} blocks)")
        
        # Update balances from blockchain
        self.update_balances_from_blockchain()

    def save_wallet(self):
        wallet_data = {
            "addresses": self.addresses,
            "version": "1.0",
            "last_backup": time.time()
        }
        
        try:
            # Use DataManager for saving
            if self.config.config["security"]["encrypt_wallet"]:
                # Convert to JSON string first, then encrypt
                json_data = json.dumps(wallet_data, indent=2)
                encrypted_data = self.encrypt_data(json_data)
                # Save encrypted data
                DataManager.save_json(self.wallet_file, {"encrypted": encrypted_data})
            else:
                # Save plain JSON
                DataManager.save_json(self.wallet_file, wallet_data)
                
            print("💾 Wallet saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving wallet: {e}")

    def load_wallet(self):
        """Load wallet using DataManager with fallback support"""
        wallet_data = DataManager.load_json(self.wallet_file)
        
        if not wallet_data:
            print("⚠️  No wallet file found, creating new wallet...")
            return None
        
        try:
            # Handle encrypted wallet
            if isinstance(wallet_data, dict) and "encrypted" in wallet_data:
                if self.config.config["security"]["encrypt_wallet"]:
                    decrypted_data = self.decrypt_data(wallet_data["encrypted"])
                    wallet_data = json.loads(decrypted_data)
                else:
                    # Wallet is encrypted but encryption is disabled - try to decrypt anyway
                    try:
                        decrypted_data = self.decrypt_data(wallet_data["encrypted"])
                        wallet_data = json.loads(decrypted_data)
                    except:
                        print("❌ Wallet is encrypted but encryption is disabled")
                        return None
            
            # Handle plain JSON wallet
            if isinstance(wallet_data, dict) and "addresses" in wallet_data:
                return wallet_data["addresses"]
            else:
                print("⚠️  Invalid wallet format, creating new wallet...")
                return None
                
        except Exception as e:
            print(f"⚠️  Error loading wallet: {e}, creating new wallet...")
            return None
    
    def backup_wallet(self):
        """Create encrypted wallet backup using DataManager"""
        if self.config.config["security"]["auto_backup"]:
            backup_file = f"wallet_backup_{int(time.time())}.dat"
            wallet_data = {
                "addresses": self.addresses,
                "version": "1.0",
                "last_backup": time.time()
            }
            
            if self.config.config["security"]["encrypt_wallet"]:
                encrypted_data = self.encrypt_data(json.dumps(wallet_data))
                DataManager.save_json(backup_file, {"encrypted": encrypted_data})
            else:
                DataManager.save_json(backup_file, wallet_data)
                
            print(f"💾 Wallet backed up to {backup_file}")
    
    def stop(self):
        """Stop P2P server and clean up"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        self.save_wallet()

    def start_periodic_sync(self, interval=300):
        """Start periodic blockchain syncing"""
        def sync_thread():
            while self.running:
                time.sleep(interval)
                print("🔄 Periodic blockchain sync...")
                self.load_blockchain()
                self.sync_mempool_with_node()  # Sync mempool
        
        threading.Thread(target=sync_thread, daemon=True).start()

    def show_balance_with_pending(self):
        """Show balance including pending transactions"""
        print("\n💰 Balance Details:")
        print("=" * 60)
        
        for wallet in self.addresses:
            confirmed = wallet["balance"]
            effective = self.get_effective_balance(wallet["address"])
            pending_count = sum(1 for tx in wallet["transactions"] if tx.get("status") == "pending")
            
            print(f"Address: {wallet['address']}")
            print(f"  Confirmed balance: {confirmed} LC")
            print(f"  Effective balance: {effective} LC (including pending)")
            print(f"  Pending transactions: {pending_count}")
            
            # Show pending transactions
            for tx in wallet["transactions"]:
                if tx.get("status") == "pending":
                    tx_type = tx.get("type", "outgoing")
                    amount = tx.get("amount", 0)
                    to_addr = tx.get("to", "N/A")[:20] + "..."
                    print(f"    ⏳ {tx_type}: {amount} LC to {to_addr}")

    def run_interactive_mode(self):
        """Run wallet in interactive mode"""
        print("💰 LunaCoin Wallet - Interactive Mode")
        print("Type 'help' for commands, 'exit' to quit")
        
        # Show data location info
        location_type, data_dir = DataManager.get_data_location_info()
        print(f"📁 Data storage: {location_type}")
        
        # Show sync status on startup
        print(f"⛓️  Blockchain Height: {len(self.blockchain)}")
        print(f"📋 Mempool Size: {len(self.mempool)}")
        
        while True:
            try:
                command = input("\nwallet> ").strip().lower()
                
                if command == "exit" or command == "quit":
                    break
                elif command == "help":
                    self.show_help()
                elif command == "mempool":
                    print(f"📋 Mempool: {len(self.mempool)} transactions")
                    for i, tx in enumerate(self.mempool):
                        tx_type = tx.get("type", "transfer")
                        amount = tx.get("amount", 0)
                        from_addr = tx.get("from", "N/A")[:16] + "..."
                        to_addr = tx.get("to", "N/A")[:16] + "..."
                        print(f"  {i+1}. {tx_type}: {amount} LC {from_addr} → {to_addr}")
                elif command == "clearmempool":
                    confirm = input("Are you sure you want to clear the mempool? (y/n): ")
                    if confirm.lower() == 'y':
                        self.mempool = []
                        DataManager.save_json("mempool.json", self.mempool)
                        print("✅ Mempool cleared")
                elif command == "setminingaddress":
                    if self.addresses:
                        print("Select mining reward address:")
                        for i, addr in enumerate(self.addresses):
                            print(f"{i+1}. {addr['address']} - {addr.get('label', 'No label')}")
                        
                        try:
                            choice = int(input("Enter number: ")) - 1
                            if 0 <= choice < len(self.addresses):
                                mining_address = self.addresses[choice]["address"]
                                self.config.config["mining"]["mining_reward_address"] = mining_address
                                self.config.save_config()
                                print(f"✅ Mining reward address set to: {mining_address}")
                            else:
                                print("❌ Invalid selection")
                        except ValueError:
                            print("❌ Please enter a valid number")
                    else:
                        print("❌ No addresses in wallet")
                elif command == "status":
                    show_wallet_info()
                elif command == "sync":
                    print("🔄 Syncing with blockchain node...")
                    response = self.send_to_node({"action": "get_blockchain_info"})
                    if response and response.get("status") == "success":
                        print(f"📊 Node info: Height {response.get('blockchain_height', 0)}, "
                            f"Pending TXs: {response.get('pending_transactions', 0)}")
                    
                    # Sync both blockchain and mempool
                    self.load_blockchain()
                    self.sync_mempool_with_node()
                    print("✅ Complete synchronization finished")
                elif command == "new":
                    label = input("Enter wallet label: ") or ""
                    new_addr = self.generate_address(label)
                    print(f"✅ New address: {new_addr}")
                elif command == "balance":
                    self.show_balance_with_pending()
                elif command.startswith("addpeer"):
                    parts = command.split()
                    if len(parts) == 2:
                        peer_address = parts[1]
                        if ":" in peer_address:
                            self.config.add_peer(peer_address)
                            self.peer_nodes.add(peer_address)
                            print(f"✅ Added peer: {peer_address}")
                        else:
                            print("❌ Peer address must be in format: ip:port")
                    else:
                        print("❌ Usage: addpeer <ip:port>")
                elif command == "peers":
                    print("🌐 Connected Peers:")
                    for peer in self.peer_nodes:
                        print(f"  {peer}")
                elif command == "discover":
                    self.discover_peers()
                    print(f"🔍 Found {len(self.peer_nodes)} peers")
                elif command == "backup":
                    self.backup_wallet()
                elif command == "debugblockchain":
                    self.debug_blockchain_content()
                elif command == "debugrewards":
                    self.debug_rewards()
                elif command == "txstatus":
                    if self.addresses:
                        # Show status of recent transactions
                        for wallet in self.addresses:
                            print(f"\n📊 Transaction status for {wallet['address'][:20]}...:")
                            for tx in wallet["transactions"][-5:]:  # Last 5 transactions
                                tx_sig = tx.get("signature", "unknown")[:16] + "..."
                                status = self.check_transaction_status(tx.get("signature"))
                                print(f"  {tx_sig}: {status['status']} ({status['confirmations']} confirmations)")
                elif command == "mine":
                    print("⛏️  Triggering mining on node...")
                    if self.trigger_mining():
                        print("✅ Mining completed, syncing blockchain...")
                        self.load_blockchain()
                    else:
                        print("❌ Mining failed")
                elif command == "nodestatus":
                    self.check_node_status()
                elif command.startswith("send"):
                    parts = command.split()
                    if len(parts) >= 3:
                        to_address = parts[1]
                        try:
                            amount = float(parts[2])
                            memo = " ".join(parts[3:]) if len(parts) > 3 else ""
                            tx = create_transaction(to_address, amount, memo)
                            if tx:
                                print(f"📤 Transaction ID: {tx['signature'][:16]}...")
                        except ValueError as e:
                            print(f"❌ Error: {e}")
                    else:
                        print("❌ Usage: send <address> <amount> [memo]")
                elif command == "datafiles":
                    DataManager.list_data_files()
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n🛑 Shutting down wallet...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        print("\n💡 Available Commands:")
        print("  status     - Show wallet status")
        print("  new        - Generate new address")
        print("  balance    - Show balances (including pending)")
        print("  send       - Send coins (send <address> <amount> [memo])")
        print("  txstatus   - Check transaction status")
        print("  mine       - Trigger mining to confirm transactions")
        print("  nodestatus - Check node status")
        print("  peers      - Show connected peers")
        print("  addpeer    - Add peer (addpeer <ip:port>)")
        print("  discover   - Discover new peers")
        print("  sync       - Sync blockchain and mempool")
        print("  mempool    - Show pending transactions")
        print("  clearmempool - Clear local mempool")
        print("  backup     - Backup wallet")
        print("  datafiles  - List data files and locations")
        print("  exit       - Exit wallet")



# Global wallet instance
wallet = None

def init_wallet():
    global wallet
    wallet = Wallet()
    return wallet

def get_default_address():
    return wallet.get_default_address()

def create_transaction(to_address: str, amount: float, memo: str = ""):
    from_address = wallet.get_default_address()
    transaction = wallet.create_transaction(from_address, to_address, amount, memo)
    
    if wallet.broadcast_transaction(transaction):
        print("✅ Transaction created and broadcast successfully!")
        return transaction
    else:
        print("❌ Transaction failed to broadcast")
        return None

def show_wallet_info():
    print("\n💼 Your Wallet:")
    print("=" * 50)
    
    for i, addr_info in enumerate(wallet.addresses):
        decrypted_key = wallet.decrypt_data(addr_info["private_key"])[:16] + "..."
        print(f"{i+1}. {addr_info['address']}")
        print(f"   Label: {addr_info.get('label', 'No label')}")
        print(f"   Balance: {addr_info['balance']} LC")
        print(f"   Transactions: {len(addr_info['transactions'])}")
        print(f"   Public Key: {wallet.decrypt_data(addr_info['public_key'])[:16]}...")
        print()
    
    print(f"🌐 Connected Peers: {len(wallet.peer_nodes)}")
    for peer in list(wallet.peer_nodes)[:5]:  # Show first 5 peers
        print(f"   {peer}")
    if len(wallet.peer_nodes) > 5:
        print(f"   ... and {len(wallet.peer_nodes) - 5} more")
        
    print(f"⛓️  Blockchain Height: {len(wallet.blockchain)}")
    print(f"📋 Mempool Size: {len(wallet.mempool)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LinKoin Wallet with P2P")
    parser.add_argument("command", nargs="?", help="Command to execute")
    parser.add_argument("--to", help="Recipient address for send command")
    parser.add_argument("--amount", type=float, help="Amount to send")
    parser.add_argument("--memo", help="Transaction memo")
    parser.add_argument("--peer", help="Peer address to add")
    parser.add_argument("--label", help="Label for new address")
    
    args = parser.parse_args()
    
    # Initialize wallet
    wallet = init_wallet()
    
    try:
        if args.command == "new":
            label = args.label or input("Enter wallet label: ") or ""
            new_addr = wallet.generate_address(label)
            print(f"✅ New address: {new_addr}")
            
        elif args.command == "send" and args.to and args.amount:
            tx = create_transaction(args.to, args.amount, args.memo or "")
            if tx:
                print(f"📤 Transaction ID: {tx['signature'][:16]}...")
        
        elif args.command == "peers":
            if args.peer:
                wallet.config.add_peer(args.peer)
                print(f"✅ Added peer: {args.peer}")
            else:
                print("🌐 Connected Peers:")
                for peer in wallet.peer_nodes:
                    print(f"  {peer}")
                    
        elif args.command == "discover":
            wallet.discover_peers()
            print(f"🔍 Found {len(wallet.peer_nodes)} peers")
            
        elif args.command == "backup":
            wallet.backup_wallet()
            
        elif args.command == "balance":
            wallet.show_balance_with_pending()
                
        elif args.command == "sync":
            wallet.load_blockchain()
            print("✅ Blockchain synchronized")
            
        elif args.command == "interactive" or args.command is None:
            # Run in interactive mode if no command specified
            wallet.run_interactive_mode()
        
        elif args.command == "server":
            # Start in server mode (for node communication)
            print("🚀 Starting wallet server mode...")
            print("💰 Wallet server is running. Press Ctrl+C to stop.")
            print(f"📡 P2P port: {wallet.config.config['network']['port']}")
            print(f"🔌 RPC port: {wallet.config.config['rpc']['port']}")
            
            # Keep the server running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down wallet server...")
        elif args.command == "debugrewards":
            wallet.debug_rewards()
        elif args.command == "txstatus":
            if wallet.addresses:
                for wallet_addr in wallet.addresses:
                    print(f"\n📊 Transaction status for {wallet_addr['address'][:20]}...:")
                    for tx in wallet_addr["transactions"][-5:]:
                        tx_sig = tx.get("signature", "unknown")[:16] + "..."
                        status = wallet.check_transaction_status(tx.get("signature"))
                        print(f"  {tx_sig}: {status['status']} ({status['confirmations']} confirmations)")
        elif args.command == "mine":
            print("⛏️  Triggering mining on node...")
            if wallet.trigger_mining():
                print("✅ Mining completed, syncing blockchain...")
                wallet.load_blockchain()
            else:
                print("❌ Mining failed")
        elif args.command == "nodestatus":
            wallet.check_node_status()
            
        else:
            show_wallet_info()
            print("\n💡 Usage:")
            print("  python luna_wallet.py [command] [options]")
            print("  python luna_wallet.py interactive - Run in interactive mode")
            print("  python luna_wallet.py new [--label LABEL]")
            print("  python luna_wallet.py send --to ADDRESS --amount AMOUNT [--memo MEMO]")
            print("  python luna_wallet.py peers [--peer ADDRESS:PORT]")
            print("  python luna_wallet.py discover")
            print("  python luna_wallet.py backup")
            print("  python luna_wallet.py balance")
            print("  python luna_wallet.py sync")
            print("  python luna_wallet.py txstatus")
            print("  python luna_wallet.py mine")
            print("  python luna_wallet.py nodestatus")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down wallet...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        wallet.stop()