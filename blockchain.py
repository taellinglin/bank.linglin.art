from typing import List, Dict, Set
import threading
import netifaces
import socket
from block import Block
import hashlib
import time
from datamanager import DataManager
import urllib3
import json
from urllib.parse import urlparse
import uuid
from httpclient import SimpleHTTPClient
# Try to import netifaces, but provide fallback if not available
try:
    import netifaces
    NETIFACES_AVAILABLE = True
except ImportError:
    print("⚠️ netifaces not available, using basic network detection")
    NETIFACES_AVAILABLE = False
    
class Blockchain:
    def __init__(self, node_address=None):
        # Use ProgramData for all file storage
        self.chain_file = "blockchain.json"
        self.verification_base_url = "https://bank.linglin.art/verify/"
        self.chain = self.load_chain() or [self.create_genesis_block()]
        
        # Multi-difficulty mining configuration
        self.difficulty = 6
        self.pending_transactions = []
        self.base_mining_reward = 50
        self.total_blocks_mined = 0
        self.total_mining_time = 0
        self.is_mining = False
        self.wallet_server_running = False
        self.stop_event = threading.Event()
        
        # Anti-spam measures
        self.user_limits = {}  # Track bills per user
        self.max_mining_attempts = 3  # Maximum attempts per bill
        
        # P2P Networking
        self.node_address = node_address or self.generate_node_address()
        self.peers: Set[str] = set()
        self.peer_discovery_running = False
        self.peer_discovery_interval = 300
        self.known_peers_file = "known_peers.json"
        self.load_known_peers()
        
        # Mining progress tracking
        self.current_mining_hashes = 0
        self.current_hash_rate = 0
        self.current_hash = ""
        
        # Multi-difficulty mining configuration
        self.difficulty_requirements = {
            "1": {"difficulty": 0, "min_time": 0.1, "max_attempts": 10},
            "10": {"difficulty": 1, "min_time": 1, "max_attempts": 8},
            "100": {"difficulty": 2, "min_time": 5, "max_attempts": 6},
            "1000": {"difficulty": 3, "min_time": 15, "max_attempts": 4},
            "10000": {"difficulty": 4, "min_time": 30, "max_attempts": 3},
            "100000": {"difficulty": 5, "min_time": 60, "max_attempts": 2},
            "1000000": {"difficulty": 6, "min_time": 300, "max_attempts": 2},
            "10000000": {"difficulty": 7, "min_time": 900, "max_attempts": 1},
            "100000000": {"difficulty": 8, "min_time": 3600, "max_attempts": 1}
        }
        
        # Bill denominations and rarity
        self.difficulty_denominations = {
            1: {"1": 1, "10": 2},
            2: {"1": 1, "10": 2, "100": 3},
            3: {"10": 2, "100": 3, "1000": 4},
            4: {"100": 3, "1000": 4, "10000": 5},
            5: {"1000": 4, "10000": 5, "100000": 6},
            6: {"10000": 5, "100000": 6, "1000000": 7},
            7: {"100000": 6, "1000000": 7, "10000000": 8},
            8: {"1000000": 7, "10000000": 8, "100000000": 9}
        }

        self.bill_rarity = {
            "1": 100, "10": 90, "100": 80, "1000": 70,
            "10000": 60, "100000": 50, "1000000": 40,
            "10000000": 30, "100000000": 20
        }
        
        # User limits per denomination
        self.denomination_limits = {
            "1": 100,      # Can create many 1's
            "10": 50,      # Fewer 10's
            "100": 25,     # Even fewer 100's
            "1000": 10,
            "10000": 5,
            "100000": 3,
            "1000000": 2,
            "10000000": 1,
            "100000000": 1
        }
        
        self.sync_all()

    def save_chain(self):
        """Save blockchain to storage - FIXED WITH DEBUG"""
        try:
            chain_data = []
            for block in self.chain:
                chain_data.append({
                    "index": block.index,
                    "previous_hash": block.previous_hash,
                    "timestamp": block.timestamp,
                    "transactions": block.transactions,  # ✅ Make sure this includes ALL transactions
                    "nonce": block.nonce,
                    "hash": block.hash,
                    "mining_time": block.mining_time
                })
            
            print(f"💾 Attempting to save {len(chain_data)} blocks to {self.chain_file}")
            
            success = DataManager.save_json(self.chain_file, chain_data)
            if success:
                print(f"✅ Blockchain saved successfully with {len(chain_data)} blocks")
            else:
                print("❌ Blockchain save failed!")
                
            return success
        except Exception as e:
            print(f"💥 CRITICAL: Error saving blockchain: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_node_address(self):
        """Generate a unique node address"""
        return f"node_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]}"

    def load_known_peers(self):
        """Load known peers from storage"""
        try:
            peers_data = DataManager.load_json(self.known_peers_file, [])
            self.peers = set(peers_data)
            print(f"✅ Loaded {len(self.peers)} known peers")
        except Exception as e:
            print(f"⚠️  Could not load known peers: {e}")
            self.peers = set()

    def save_known_peers(self):
        """Save known peers to storage"""
        try:
            DataManager.save_json(self.known_peers_file, list(self.peers))
        except Exception as e:
            print(f"❌ Error saving known peers: {e}")

    def add_peer(self, peer_address: str):
        """Add a peer if it's not ourselves"""
        if "://" in peer_address:
            url = urlparse(peer_address)
            peer_address = f"{url.hostname}:{url.port or 9335}"
        
        if not peer_address or ":" not in peer_address:
            print(f"❌ Invalid peer address format: {peer_address}")
            return False
        
        # Skip self-connections
        local_ips = ["127.0.0.1", "localhost", "0.0.0.0", "::1"]
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            local_ips.extend([local_ip, f"{local_ip}:9335"])
            
            if NETIFACES_AVAILABLE:
                for interface in netifaces.interfaces():
                    addrs = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addrs:
                        for addr in addrs[netifaces.AF_INET]:
                            ip = addr['addr']
                            local_ips.extend([ip, f"{ip}:9335"])
        except:
            pass
        
        if peer_address in local_ips:
            print(f"⏭️  Skipping self as peer: {peer_address}")
            return False
        
        if peer_address not in self.peers:
            self.peers.add(peer_address)
            self.save_known_peers()
            print(f"➕ Added peer: {peer_address}")
            return True
        else:
            print(f"ℹ️  Peer already known: {peer_address}")
        
        return False

    def remove_peer(self, peer_address: str):
        """Remove a peer from the known peers list"""
        if peer_address in self.peers:
            self.peers.remove(peer_address)
            self.save_known_peers()
            print(f"✅ Removed peer: {peer_address}")
            return True
        print(f"ℹ️  Peer not found: {peer_address}")
        return False

    def discover_peers(self):
        """Discover new peers from existing peers"""
        if not self.peers:
            print("⚠️  No known peers to discover from")
            return
        
        print(f"🔍 Discovering peers from {len(self.peers)} known peers...")
        new_peers_count = 0
        
        for peer in list(self.peers):
            try:
                print(f"   Asking {peer} for peers...")
                response = self.send_to_peer(peer, {"action": "get_peers"})
                if response and response.get("status") == "success":
                    new_peers = response.get("peers", [])
                    for new_peer in new_peers:
                        if self.add_peer(new_peer):
                            new_peers_count += 1
                    print(f"   ✅ Discovered {len(new_peers)} peers from {peer}")
                else:
                    print(f"   ❌ No response from {peer}")
            except Exception as e:
                print(f"   ❌ Error discovering peers from {peer}: {e}")
        
        print(f"✅ Discovered {new_peers_count} new peers total")

    def peer_discovery_loop(self):
        """Continuous peer discovery loop"""
        self.peer_discovery_running = True
        discovery_count = 0
        
        while not self.stop_event.is_set():
            try:
                discovery_count += 1
                print(f"🔄 Peer discovery round {discovery_count}...")
                self.discover_peers()
                
                # Wait for the next discovery interval
                for i in range(self.peer_discovery_interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Error in peer discovery: {e}")
                time.sleep(60)
        
        self.peer_discovery_running = False

    def test_peer_connectivity(self, peer_address: str = None):
        """Test connectivity to peers"""
        peers_to_test = [peer_address] if peer_address else list(self.peers)
        
        if not peers_to_test:
            print("❌ No peers to test")
            return
        
        print(f"🔌 Testing connectivity to {len(peers_to_test)} peers...")
        
        for peer in peers_to_test:
            try:
                print(f"   Testing {peer}...")
                response = self.send_to_peer(peer, {"action": "ping"}, timeout=5)
                if response and response.get("status") == "success":
                    print(f"   ✅ {peer} - Reachable")
                else:
                    print(f"   ❌ {peer} - Unreachable")
            except Exception as e:
                print(f"   ❌ {peer} - Error: {e}")

    def start_peer_discovery(self):
        """Start the peer discovery service"""
        if not self.peer_discovery_running:
            discovery_thread = threading.Thread(target=self.peer_discovery_loop, daemon=True)
            discovery_thread.start()
            print("✅ Peer discovery service started")

    def list_peers(self):
        """List all known peers"""
        print("📋 Known peers:")
        for peer in self.peers:
            print(f"   - {peer}")
        print(f"   Total: {len(self.peers)} peers")

    def send_to_peer(self, peer_address: str, data: Dict, timeout=10):
        """Send data to a peer node using direct socket connection"""
        try:
            if "://" in peer_address:
                url = urlparse(peer_address)
                host = url.hostname
                port = url.port or 9335
            else:
                if ":" in peer_address:
                    host, port_str = peer_address.split(":", 1)
                    port = int(port_str)
                else:
                    host = peer_address
                    port = 9335
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((host, port))
                
                data_json = json.dumps(data)
                data_length = len(data_json.encode())
                s.sendall(data_length.to_bytes(4, 'big'))
                s.sendall(data_json.encode())
                
                length_bytes = s.recv(4)
                if not length_bytes:
                    return None
                
                response_length = int.from_bytes(length_bytes, 'big')
                response_data = b""
                while len(response_data) < response_length:
                    chunk = s.recv(min(4096, response_length - len(response_data)))
                    if not chunk:
                        break
                    response_data += chunk
                
                if len(response_data) == response_length:
                    return json.loads(response_data.decode())
                else:
                    print(f"❌ Incomplete response from {peer_address}")
                    
        except socket.timeout:
            print(f"❌ Timeout communicating with {peer_address}")
        except ConnectionRefusedError:
            print(f"❌ Connection refused by {peer_address}")
        except Exception as e:
            print(f"❌ Error communicating with peer {peer_address}: {e}")
        
        return None

    def broadcast(self, data: Dict, exclude_peers=None):
        """Broadcast data to all known peers"""
        exclude_peers = exclude_peers or []
        successful_broadcasts = 0
        
        for peer in list(self.peers):
            if peer in exclude_peers:
                continue
                
            response = self.send_to_peer(peer, data)
            if response and response.get("status") == "success":
                successful_broadcasts += 1
        
        return successful_broadcasts

    def download_from_web(self, resource_type="blockchain"):
        """Download blockchain or mempool from web server using simple HTTP client"""
        base_url = "https://bank.linglin.art/"
        url = f"{base_url}{resource_type}"
        
        try:
            print(f"🌐 Downloading {resource_type} from {url}...")
            
            # First, let's try to get the raw response to see what we're getting
            response_text = SimpleHTTPClient.get(url)
            if response_text:
                print(f"📥 Raw response length: {len(response_text)} characters")
                
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    return False
            else:
                print(f"❌ No response from {url}")
                return False
            
            if data is not None:
                if resource_type == "blockchain":
                    if resource_type == "blockchain":
                        new_chain = []
                    for block_data in data:
                        block = Block(
                            block_data["index"],
                            block_data["previous_hash"],
                            block_data["timestamp"],
                            block_data["transactions"],
                            block_data.get("nonce", 0)
                        )
                        block.hash = block_data["hash"]
                        block.mining_time = block_data.get("mining_time", 0)
                        new_chain.append(block)
                    
                    if self.validate_chain(new_chain):
                        self.chain = new_chain
                        # Check if save_chain method exists before calling it
                        if hasattr(self, 'save_chain') and callable(getattr(self, 'save_chain')):
                            self.save_chain()
                        else:
                            # Fallback: save directly
                            chain_data = []
                            for block in self.chain:
                                chain_data.append({
                                    "index": block.index,
                                    "previous_hash": block.previous_hash,
                                    "timestamp": block.timestamp,
                                    "transactions": block.transactions,
                                    "nonce": block.nonce,
                                    "hash": block.hash,
                                    "mining_time": block.mining_time
                                })
                            DataManager.save_json(self.chain_file, chain_data)
                        print(f"✅ Downloaded and validated blockchain with {len(data)} blocks")
                        return True
                    else:
                        print("❌ Downloaded blockchain failed validation")
                    
                elif resource_type == "mempool":
                    print(f"🔍 Mempool data type: {type(data)}")
                    
                    if isinstance(data, list):
                        self.pending_transactions = data
                        self.save_mempool()
                        print(f"✅ Downloaded mempool with {len(data)} transactions")
                        return True
                    elif isinstance(data, dict):
                        # Maybe the mempool is wrapped in a dictionary
                        if 'transactions' in data:
                            self.pending_transactions = data['transactions']
                            self.save_mempool()
                            print(f"✅ Downloaded mempool with {len(data['transactions'])} transactions")
                            return True
                        else:
                            print(f"❌ Mempool data dict doesn't contain 'transactions' key: {data.keys()}")
                            return False
                    else:
                        print(f"❌ Mempool data is unexpected type: {type(data)}")
                        return False
            else:
                print(f"❌ No data received from {url}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {resource_type}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def sync_all(self):
        """One command to sync everything"""
        print("🔄 Starting complete network synchronization...")
        
        # Try web sync first
        print("🌐 Trying to sync from linglin.art...")
        web_success = self.download_from_web("blockchain")
        
        if not web_success:
            print("❌ Could not sync blockchain from web")
        else:
            print("✅ Successfully synced from web")
        
        # Sync mempool
        print("🌐 Trying to sync mempool from linglin.art...")
        web_mempool_success = self.download_from_web("mempool")
        
        if not web_mempool_success:
            print("❌ Could not sync mempool from web")
        else:
            print("✅ Successfully synced mempool from web")
        
        # Discover peers
        self.discover_peers()
        
        print("✅ Complete synchronization finished")

    def download_blockchain(self, peer_address: str = None):
        """Download blockchain from a peer"""
        peers_to_try = [peer_address] if peer_address else list(self.peers)
        
        print(f"🔍 Current peers: {list(self.peers)}")
        print(f"🔍 Peers to try: {peers_to_try}")
        
        if not peers_to_try:
            print("❌ No peers available to download blockchain from")
            return False
        
        for peer in peers_to_try:
            try:
                print(f"📥 Attempting to download blockchain from {peer}...")
                
                ping_response = self.send_to_peer(peer, {"action": "ping"}, timeout=5)
                if not ping_response or ping_response.get("status") != "success":
                    print(f"❌ Peer {peer} is not responding to ping")
                    continue
                    
                print(f"✅ Peer {peer} is reachable, requesting blockchain...")
                
                response = self.send_to_peer(peer, {"action": "get_blockchain"}, timeout=30)
                
                if not response:
                    print(f"❌ No response from {peer}")
                    continue
                    
                if response.get("status") == "success":
                    chain_data = response.get("blockchain", [])
                    total_blocks = response.get("total_blocks", 0)
                    
                    print(f"🔍 Received {len(chain_data)} blocks, I have {len(self.chain)} blocks")
                    
                    if len(chain_data) > len(self.chain):
                        print(f"✅ Downloaded blockchain with {total_blocks} blocks from {peer}")
                        
                        new_chain = []
                        for block_data in chain_data:
                            block = Block(
                                block_data["index"],
                                block_data["previous_hash"],
                                block_data["timestamp"],
                                block_data["transactions"],
                                block_data.get("nonce", 0)
                            )
                            block.hash = block_data["hash"]
                            block.mining_time = block_data.get("mining_time", 0)
                            new_chain.append(block)
                        
                        if self.validate_chain(new_chain):
                            self.chain = new_chain
                            self.save_chain()
                            print(f"✅ Blockchain updated to {len(self.chain)} blocks")
                            return True
                        else:
                            print("❌ Downloaded blockchain failed validation")
                    else:
                        print("ℹ️  Peer has same or shorter chain, keeping current blockchain")
                else:
                    error_msg = response.get("message", "Unknown error")
                    print(f"❌ Failed to download blockchain from {peer}: {error_msg}")
                    
            except socket.timeout:
                print(f"❌ Timeout connecting to {peer}")
            except ConnectionRefusedError:
                print(f"❌ Connection refused by {peer}")
            except Exception as e:
                print(f"❌ Error downloading blockchain from {peer}: {e}")
        
        print("❌ Failed to download blockchain from all peers")
        return False

    def save_mempool(self):
        """Save pending transactions to storage"""
        try:
            DataManager.save_json("mempool.json", self.pending_transactions)
        except Exception as e:
            print(f"❌ Error saving mempool: {e}")

    def download_mempool(self, peer_address: str = None):
        """Download mempool transactions from a peer, with web fallback"""
        peers_to_try = [peer_address] if peer_address else list(self.peers)
        new_transactions = 0
        
        # First try peers
        if peers_to_try:
            print(f"🔍 Trying {len(peers_to_try)} peer(s) for mempool download...")
            
            for peer in peers_to_try:
                try:
                    print(f"📥 Downloading mempool from {peer}...")
                    response = self.send_to_peer(peer, {"action": "get_pending_transactions"})
                    
                    if response and response.get("status") == "success":
                        transactions = response.get("transactions", [])
                        
                        for tx in transactions:
                            if not any(t.get("signature") == tx.get("signature") for t in self.pending_transactions):
                                self.pending_transactions.append(tx)
                                new_transactions += 1
                        
                        self.save_mempool()
                        
                        print(f"✅ Added {new_transactions} new transactions from {peer}'s mempool")
                        return True
                    else:
                        print(f"❌ Failed to download mempool from {peer}")
                except Exception as e:
                    print(f"❌ Error downloading mempool from {peer}: {e}")
        
        # If no peers available or all peers failed, try web fallback
        print("🌐 No peers available or all failed, trying web fallback...")
        return self.download_mempool_from_web()

    def download_mempool_from_web(self):
        """Download mempool from https://bank.linglin.art/mempool"""
        web_url = "https://bank.linglin.art/mempool"
        new_transactions = 0
        
        try:
            print(f"📥 Downloading mempool from {web_url}...")
            
            # Use the existing SimpleHTTPClient
            response_text = SimpleHTTPClient.get(web_url)
            
            if response_text:
                try:
                    data = json.loads(response_text)
                    
                    # Handle different response formats
                    if isinstance(data, list):
                        transactions = data
                    elif isinstance(data, dict) and 'transactions' in data:
                        transactions = data['transactions']
                    else:
                        print(f"❌ Unexpected mempool format from web: {type(data)}")
                        return False
                    
                    for tx in transactions:
                        # Check if transaction already exists
                        if not any(t.get("signature") == tx.get("signature") for t in self.pending_transactions):
                            # Also check if it's already mined in blockchain
                            if not self.is_transaction_mined(tx):
                                self.pending_transactions.append(tx)
                                new_transactions += 1
                    
                    self.save_mempool()
                    
                    if new_transactions > 0:
                        print(f"✅ Added {new_transactions} new transactions from web mempool")
                    else:
                        print("ℹ️  No new transactions found in web mempool")
                    
                    return new_transactions > 0
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error from web mempool: {e}")
                    return False
            else:
                print("❌ No response from web mempool")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading mempool from web: {e}")
            return False

    def validate_chain(self, chain: List[Block]) -> bool:
        """Validate a blockchain"""
        for i in range(1, len(chain)):
            current_block = chain[i]
            previous_block = chain[i-1]
            
            if current_block.hash != current_block.calculate_hash():
                print(f"❌ Block {i} hash is invalid!")
                return False
            
            if current_block.previous_hash != previous_block.hash:
                print(f"❌ Block {i} previous hash doesn't match!")
                return False
        
        return True

    def generate_serial_number(self):
        """Generate a unique serial number in SN-###-###-#### format"""
        serial_id = str(uuid.uuid4().int)[:10]
        formatted_serial = f"SN-{serial_id[:3]}-{serial_id[3:6]}-{serial_id[6:10]}"
        return formatted_serial

    def clear_mempool(self):
        """Clear the mempool"""
        try:
            self.pending_transactions = []
            DataManager.save_json("mempool.json", [])
            print("✅ Mempool cleared")
            return True
        except Exception as e:
            print(f"❌ Error clearing mempool: {e}")
            return False

    def create_genesis_block(self) -> Block:
        genesis_serial = self.generate_serial_number()
        return Block(0, "0", time.time(), [{
            "type": "genesis", 
            "message": "Luna Genesis Block",
            "timestamp": time.time(),
            "reward": 10000,
            "denomination": "10000",
            "difficulty": 1,
            "serial_number": genesis_serial,
            "verification_url": f"{self.verification_base_url}{genesis_serial}"
        }])

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, new_block: Block):
        new_block.previous_hash = self.get_latest_block().hash
        self.chain.append(new_block)
        self.total_blocks_mined += 1
        self.total_mining_time += new_block.mining_time
        self.save_chain()
        
        block_data = {
            "action": "new_block",
            "block": {
                "index": new_block.index,
                "previous_hash": new_block.previous_hash,
                "timestamp": new_block.timestamp,
                "transactions": new_block.transactions,
                "nonce": new_block.nonce,
                "hash": new_block.hash,
                "mining_time": new_block.mining_time
            }
        }
        self.broadcast(block_data)

    def check_user_limits(self, user_id, denomination):
        """Check if user has reached their limit for a denomination"""
        if user_id not in self.user_limits:
            self.user_limits[user_id] = {}
        
        user_denom_count = self.user_limits[user_id].get(denomination, 0)
        max_allowed = self.denomination_limits.get(denomination, 1)
        
        return user_denom_count < max_allowed

    def increment_user_limit(self, user_id, denomination):
        """Increment user's count for a denomination"""
        if user_id not in self.user_limits:
            self.user_limits[user_id] = {}
        
        self.user_limits[user_id][denomination] = self.user_limits[user_id].get(denomination, 0) + 1

    def add_transaction(self, transaction: Dict):
        """Add transaction to mempool with anti-spam checks"""
        
        # Extract user_id and denomination
        user_id = transaction.get("user_id", "unknown")
        denomination = transaction.get("denomination", "1")
        
        # Anti-spam: Check user limits
        if not self.check_user_limits(user_id, denomination):
            print(f"❌ User {user_id} exceeded limit for denomination {denomination}")
            return False
        
        # Check if this transaction has already been mined in the blockchain
        if self.is_transaction_mined(transaction):
            print(f"⏭️  Transaction already mined, skipping: {transaction.get('signature', 'unknown')[:16]}...")
            return False
        
        # Check if transaction already exists in mempool
        tx_signature = transaction.get("signature")
        if any(t.get("signature") == tx_signature for t in self.pending_transactions):
            print(f"⏭️  Transaction already in mempool: {tx_signature[:16]}...")
            return False
        
        # Add mining requirements based on denomination
        requirements = self.difficulty_requirements.get(denomination, {"difficulty": 0, "min_time": 0.1, "max_attempts": 10})
        transaction["mining_requirements"] = {
            "required_difficulty": requirements["difficulty"],
            "minimum_time_seconds": requirements["min_time"],
            "max_mining_attempts": requirements["max_attempts"],
            "mining_attempts": 0
        }
        
        self.pending_transactions.append(transaction)
        self.save_mempool()
        
        # Increment user limit
        self.increment_user_limit(user_id, denomination)
        
        # Broadcast the new transaction to peers
        tx_data = {
            "action": "new_transaction",
            "transaction": transaction
        }
        self.broadcast(tx_data)
        
        print(f"✅ Transaction added to mempool: {tx_signature[:16]}...")
        print(f"   💰 Denomination: ${denomination}")
        print(f"   ⛏️  Required Difficulty: {requirements['difficulty']}")
        print(f"   ⏱️  Minimum Time: {requirements['min_time']}s")
        return True

    def is_transaction_mined(self, transaction: Dict) -> bool:
        """Check if a transaction has already been mined in the blockchain"""
        tx_signature = transaction.get("signature")
        if not tx_signature:
            return False
        
        # Check all blocks in the blockchain
        for block in self.chain:
            for tx in block.transactions:
                if tx.get("signature") == tx_signature:
                    return True
                # Also check by content for GTX_Genesis transactions
                if (tx.get("type") == "GTX_Genesis" and 
                    transaction.get("type") == "GTX_Genesis" and
                    tx.get("serial_number") == transaction.get("serial_number")):
                    return True
        return False

    def determine_bill_denomination(self, block_hash):
        """Determine which bill denomination was found based on block hash and difficulty"""
        current_difficulty = self.difficulty
        available_bills = self.difficulty_denominations.get(current_difficulty, {"1": 1})
        
        hash_int = int(block_hash[:8], 16)
        
        weighted_choices = []
        for bill, multiplier in available_bills.items():
            weight = self.bill_rarity.get(bill, 1)
            weighted_choices.extend([bill] * weight)
        
        selected_bill = weighted_choices[hash_int % len(weighted_choices)]
        multiplier = available_bills[selected_bill]
        actual_reward = self.base_mining_reward * multiplier
        
        return selected_bill, multiplier, actual_reward

    def mine_pending_transactions(self, mining_reward_address: str):
        if not self.pending_transactions:
            print("❌ No transactions to mine!")
            return False
        
        print(f"📦 Preparing block with {len(self.pending_transactions)} transactions...")
        
        # Filter transactions by difficulty requirements and attempt limits
        mineable_transactions = []
        for tx in self.pending_transactions:
            requirements = tx.get("mining_requirements", {})
            attempts = requirements.get("mining_attempts", 0)
            max_attempts = requirements.get("max_mining_attempts", self.max_mining_attempts)
            
            if attempts < max_attempts:
                mineable_transactions.append(tx)
            else:
                print(f"⏭️  Skipping transaction {tx.get('signature', 'unknown')[:16]}... - exceeded max attempts")
        
        if not mineable_transactions:
            print("❌ No mineable transactions (all exceeded attempt limits)")
            return False
        
        def mining_progress(stats):
            hashes_per_sec = stats['hash_rate']
            if hashes_per_sec > 1_000_000:
                hash_rate_str = f"{hashes_per_sec/1_000_000:.2f} MH/s"
            elif hashes_per_sec > 1_000:
                hash_rate_str = f"{hashes_per_sec/1_000:.2f} KH/s"
            else:
                hash_rate_str = f"{hashes_per_sec:.2f} H/s"
            
            self.current_mining_hashes = stats['hashes']
            self.current_hash_rate = stats['hash_rate']
            self.current_hash = stats['current_hash']
            
            print(f"\r⛏️  Mining: {stats['hashes']:,.0f} hashes | {hash_rate_str} | Current: {stats['current_hash'][:self.difficulty+2]}...", 
                end="", flush=True)

        # Create the block with mineable transactions
        block = Block(len(self.chain), self.get_latest_block().hash, time.time(), mineable_transactions.copy())
        
        self.is_mining = True
        print("\n" + "="*60)
        print("🚀 STARTING MINING OPERATION")
        print("="*60)
        print(f"⚙️  Difficulty Level: {self.difficulty}")
        print(f"💵 Available Bills: {', '.join(self.difficulty_denominations[self.difficulty].keys())}")
        print(f"📊 Mineable Transactions: {len(mineable_transactions)}/{len(self.pending_transactions)}")
        print("="*60)
        
        start_time = time.time()
        block.mine_block(self.difficulty, mining_progress)
        end_time = time.time()
        
        self.is_mining = False
        
        self.current_mining_hashes = 0
        self.current_hash_rate = 0
        self.current_hash = ""
        
        # Determine bill denomination and create reward transaction
        bill_denomination, multiplier, actual_reward = self.determine_bill_denomination(block.hash)
        serial_number = self.generate_serial_number()
        verification_url = f"{self.verification_base_url}{serial_number}"
        
        # Create reward transaction
        reward_tx = {
            "type": "reward",
            "to": mining_reward_address,
            "amount": actual_reward,
            "base_reward": self.base_mining_reward,
            "denomination": bill_denomination,
            "multiplier": multiplier,
            "timestamp": time.time(),
            "block_height": len(self.chain),  # This will be the new block's height
            "signature": f"reward_{len(self.chain)}_{hashlib.sha256(mining_reward_address.encode()).hexdigest()[:16]}",
            "block_hash": block.hash,
            "difficulty": self.difficulty,
            "serial_number": serial_number,
            "verification_url": verification_url
        }
        
        # ✅ CRITICAL FIX: Add the reward transaction to the block's transactions
        block.transactions.append(reward_tx)
        
        print("\n\n" + "="*60)
        print("✅ BLOCK MINED SUCCESSFULLY!")
        print("="*60)
        
        # Add the block to the blockchain (this includes the reward transaction)
        self.add_block(block)
        
        # Clear only the mined transactions from mempool
        mined_signatures = [tx.get("signature") for tx in mineable_transactions]
        self.pending_transactions = [tx for tx in self.pending_transactions if tx.get("signature") not in mined_signatures]
        self.save_mempool()
        
        # ✅ Broadcast mempool clearance to all peers (including wallet)
        clearance_data = {
            "action": "clear_mempool",
            "block_height": len(self.chain) - 1,
            "block_hash": block.hash,
            "cleared_transactions": len(mineable_transactions)
        }
        self.broadcast(clearance_data)
        
        mining_time = end_time - start_time
        print(f"📊 Mining Statistics:")
        print(f"   Block Height: {block.index}")
        print(f"   Hash: {block.hash[:16]}...")
        print(f"   Nonce: {block.nonce:,}")
        print(f"   Mining Time: {mining_time:.2f} seconds")
        print(f"   Hash Rate: {block.hash_rate:,.2f} H/s")
        print(f"   Difficulty: {self.difficulty}")
        print(f"   Transactions: {len(block.transactions)}")  # This now includes the reward
        print(f"   💰 Bill Found: ${bill_denomination} Bill")
        print(f"   📈 Multiplier: x{multiplier}")
        print(f"   ⛏️  Base Reward: {self.base_mining_reward} LC")
        print(f"   🎯 Total Reward: {actual_reward} LC → {mining_reward_address}")
        print(f"   🔢 Serial Number: {serial_number}")
        print(f"   🔗 Verification: {verification_url}")
        
        return True

    def get_mining_stats(self):
        if self.total_blocks_mined == 0:
            return {
                'total_blocks': 0,
                'avg_time': 0,
                'total_time': 0,
                'total_rewards': 0
            }
        
        total_rewards = 0
        for block in self.chain:
            for tx in block.transactions:
                if tx.get("type") == "reward" or tx.get("type") == "genesis":
                    total_rewards += tx.get("amount", 0)
        
        return {
            'total_blocks': self.total_blocks_mined,
            'avg_time': self.total_mining_time / self.total_blocks_mined,
            'total_time': self.total_mining_time,
            'total_rewards': total_rewards
        }

    def is_chain_valid(self) -> bool:
        return self.validate_chain(self.chain)

    def load_chain(self):
        """Load blockchain from storage - FIXED"""
        try:
            chain_data = DataManager.load_json(self.chain_file)
            if chain_data:
                chain = []
                for block_data in chain_data:
                    block = Block(
                        block_data["index"],
                        block_data["previous_hash"],
                        block_data["timestamp"],
                        block_data["transactions"],  # ✅ Critical: Load ALL transactions
                        block_data.get("nonce", 0)
                    )
                    block.hash = block_data["hash"]
                    block.mining_time = block_data.get("mining_time", 0)
                    chain.append(block)
                
                print(f"✅ Loaded blockchain with {len(chain)} blocks")
                
                # DEBUG: Check if rewards are in loaded blocks
                reward_count = 0
                for block in chain:
                    for tx in block.transactions:
                        if tx.get("type") == "reward":
                            reward_count += 1
                print(f"🔍 Found {reward_count} reward transactions in loaded chain")
                
                return chain
        except Exception as e:
            print(f"❌ Error loading blockchain: {e}")
            import traceback
            traceback.print_exc()
        return None
    def is_transaction_mined(self, transaction: Dict) -> bool:
        """Check if a transaction has already been mined in the blockchain"""
        tx_signature = transaction.get("signature")
        if not tx_signature:
            return False
        
        # Check all blocks in the blockchain
        for block in self.chain:
            for tx in block.transactions:
                if tx.get("signature") == tx_signature:
                    return True
                # Also check by content for GTX_Genesis transactions
                if (tx.get("type") == "GTX_Genesis" and 
                    transaction.get("type") == "GTX_Genesis" and
                    tx.get("serial_number") == transaction.get("serial_number")):
                    return True
        return False
    def handle_wallet_connection(self, data):
        """Handle incoming wallet requests"""
        action = data.get("action")
        MAX_RESPONSE_SIZE = 10 * 1024 * 1024

        try:
            if action == "ping":
                return {"status": "success", "message": "pong", "timestamp": time.time()}
            elif action == "get_status":  # ADD THIS NEW ACTION
                return {
                    "status": "success",
                    "node_status": {
                        "mining_enabled": True,  # Mining is always available
                        "auto_mine": False,      # No auto-mining in this version
                        "mempool_size": len(self.pending_transactions),
                        "blockchain_height": len(self.chain),
                        "difficulty": self.difficulty,
                        "peers_count": len(self.peers),
                        "is_mining": self.is_mining,
                        "node_address": self.node_address,
                        "wallet_connected": True
                    }
                }
            elif action == "add_transaction":
                transaction = data.get("transaction")
                if self.add_transaction(transaction):
                    return {"status": "success", "message": "Transaction added to mempool"}
                else:
                    return {"status": "error", "message": "Failed to add transaction (validation failed)"}
            elif action == "get_mining_progress":
                if self.is_mining:
                    return {
                        "status": "success", 
                        "mining": True,
                        "progress": {
                            "hashes": self.current_mining_hashes,
                            "hash_rate": self.current_hash_rate,
                            "current_hash": self.current_hash
                        }
                    }
                else:
                    return {"status": "success", "mining": False}
            elif action == "get_mining_stats":
                stats = self.get_mining_stats()
                return {"status": "success", "stats": stats}
            elif action == "get_mining_address":
                return {"status": "success", "address": "miner_default_address"}
            
            elif action == "get_balance":
                address = data.get("address")
                balance = self.calculate_balance(address)
                return {"status": "success", "balance": balance}
            
            elif action == "add_transaction":
                transaction = data.get("transaction")
                if self.add_transaction(transaction):
                    return {"status": "success", "message": "Transaction added to mempool"}
                else:
                    return {"status": "error", "message": "Failed to add transaction (spam protection or duplicate)"}
            
            elif action == "get_blockchain_info":
                return {
                    "status": "success",
                    "blockchain_height": len(self.chain),
                    "latest_block_hash": self.chain[-1].hash if self.chain else None,
                    "pending_transactions": len(self.pending_transactions),
                    "difficulty": self.difficulty,
                    "base_reward": self.base_mining_reward,
                    "node_address": self.node_address,
                    "peers_count": len(self.peers),
                    "user_limits": self.denomination_limits
                }
            elif action == "start_mining":
                miner_address = data.get("miner_address", "miner_default_address")
                
                def mining_thread():
                    try:
                        success = self.mine_pending_transactions(miner_address)
                        if success:
                            print("✅ Mining completed successfully")
                    except Exception as e:
                        print(f"Mining error: {e}")
                
                thread = threading.Thread(target=mining_thread, daemon=True)
                thread.start()
                
                return {"status": "success", "message": "Mining started in background"}
            
            elif action == "get_pending_transactions":
                return {"status": "success", "transactions": self.pending_transactions}
            elif action == "get_blockchain":
                # For wallet synchronization, return ALL transactions without limits
                chain_data = []
                for block in self.chain:
                    chain_data.append({
                        "index": block.index,
                        "previous_hash": block.previous_hash,
                        "timestamp": block.timestamp,
                        "transactions": block.transactions,  # ✅ ALL transactions
                        "nonce": block.nonce,
                        "hash": block.hash,
                        "mining_time": block.mining_time
                    })
                
                # Check size and warn if large, but still send it
                estimated_size = len(json.dumps(chain_data))
                if estimated_size > MAX_RESPONSE_SIZE:
                    print(f"⚠️  Large blockchain response: {estimated_size/1024/1024:.1f}MB")
                
                return {"status": "success", "blockchain": chain_data, "total_blocks": len(self.chain)}
            
            elif action == "get_difficulty_info":
                return {
                    "status": "success",
                    "difficulty": self.difficulty,
                    "available_bills": self.difficulty_denominations.get(self.difficulty, {}),
                    "base_reward": self.base_mining_reward,
                    "difficulty_requirements": self.difficulty_requirements,
                    "denomination_limits": self.denomination_limits
                }
            
            elif action == "get_peers":
                return {"status": "success", "peers": list(self.peers)}
            
            elif action == "add_peer":
                peer_address = data.get("peer_address")
                if peer_address and self.add_peer(peer_address):
                    return {"status": "success", "message": f"Peer {peer_address} added"}
                else:
                    return {"status": "error", "message": "Failed to add peer"}
            
            elif action == "new_block":
                block_data = data.get("block")
                if block_data:
                    if block_data["index"] > len(self.chain) - 1:
                        print(f"📥 Received new block #{block_data['index']} from peer")
                        block = Block(
                            block_data["index"],
                            block_data["previous_hash"],
                            block_data["timestamp"],
                            block_data["transactions"],
                            block_data.get("nonce", 0)
                        )
                        block.mining_time = block_data.get("mining_time", 0)
                        block.hash = block_data.get("hash", block.calculate_hash())
                        
                        if block.hash == block.calculate_hash() and block.previous_hash == self.get_latest_block().hash:
                            self.chain.append(block)
                            self.save_chain()
                            print(f"✅ Added new block #{block.index} from peer")
                            return {"status": "success", "message": "Block added"}
                    
                    return {"status": "success", "message": "Block already exists or invalid"}
                else:
                    return {"status": "error", "message": "No block data provided"}
            
            elif action == "new_transaction":
                transaction = data.get("transaction")
                if transaction:
                    if not any(t.get("signature") == transaction.get("signature") for t in self.pending_transactions):
                        self.pending_transactions.append(transaction)
                        self.save_mempool()
                        print(f"📥 Received new transaction from peer")
                        return {"status": "success", "message": "Transaction added to mempool"}
                    else:
                        return {"status": "success", "message": "Transaction already in mempool"}
                else:
                    return {"status": "error", "message": "No transaction data provided"}
            
            return {"status": "error", "message": "Unknown action"}
        except Exception as e:
            return {"status": "error", "message": f"Server error: {str(e)}"}

    def calculate_balance(self, address):
        """Calculate balance for a specific address"""
        balance = 0
        for block in self.chain:
            for tx in block.transactions:
                if tx.get("to") == address:
                    balance += tx.get("amount", 0)
                if tx.get("from") == address:
                    balance -= tx.get("amount", 0)
        
        for tx in self.pending_transactions:
            if tx.get("type") != "reward":
                if tx.get("to") == address:
                    balance += tx.get("amount", 0)
                if tx.get("from") == address:
                    balance -= tx.get("amount", 0)
        
        return balance

    def wallet_server_thread(self):
        """Wallet server thread function"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind(('0.0.0.0', 9335))
            server_socket.listen(5)
            server_socket.settimeout(1.0)
            
            print(f"👛 Node Wallet server listening on port 9335")
            self.wallet_server_running = True
            
            while not self.stop_event.is_set():
                try:
                    client_socket, addr = server_socket.accept()
                    client_socket.settimeout(5.0)
                    
                    try:
                        length_bytes = client_socket.recv(4)
                        if not length_bytes:
                            client_socket.close()
                            continue
                        
                        message_length = int.from_bytes(length_bytes, 'big')
                        
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

    def start_wallet_server(self):
        """Start the wallet server to handle wallet connections"""
        self.wallet_thread = threading.Thread(target=self.wallet_server_thread)
        self.wallet_thread.daemon = False
        self.wallet_thread.start()
        print("✅ Node wallet server thread started")

    def stop_wallet_server(self):
        """Stop the wallet server gracefully"""
        print("🛑 Stopping wallet server...")
        self.stop_event.set()
        if hasattr(self, 'wallet_thread') and self.wallet_thread.is_alive():
            self.wallet_thread.join(timeout=3.0)
        print("✅ Wallet server stopped")