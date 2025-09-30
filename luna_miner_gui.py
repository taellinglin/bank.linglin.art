#!/usr/bin/env python3
"""
lunacoin_miner_kivy.py - Kivy GUI for the Luna Coin miner with red and black theme.
"""
import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, DictProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
import threading
import time
import json
import socket
import os
import queue

# Set window size and background color
Window.clearcolor = (0.05, 0.05, 0.05, 1)  # Dark gray background
Window.size = (900, 700)

class MinerGUI(BoxLayout):
    status_text = StringProperty("Ready")
    height_text = StringProperty("0")
    total_blocks_text = StringProperty("0")
    total_rewards_text = StringProperty("0")
    avg_time_text = StringProperty("0")
    pending_txs_text = StringProperty("0")
    progress_text = StringProperty("Not mining")
    is_mining = BooleanProperty(False)
    mined_serials = DictProperty({})
    difficulty = NumericProperty(4)
    
    def __init__(self, **kwargs):
        super(MinerGUI, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 5
        self.spacing = 5
        
        # Initialize mining state
        self.mining_thread = None
        self.log_queue = queue.Queue()
        self.auto_refresh_active = True
        
        # Start update threads
        Clock.schedule_interval(self.update_stats, 2)  # More frequent updates
        Clock.schedule_interval(self.process_log_queue, 0.1)
        # Start automatic transaction refresh
        Clock.schedule_once(self.load_transactions_on_startup, 1)
        Clock.schedule_interval(self.auto_refresh_transactions, 10)  # Refresh every 10 seconds
        
    def load_transactions_on_startup(self, dt):
        """Load transactions automatically when the GUI starts"""
        self.safe_log("Loading transactions on startup...")
        self.load_transactions(None)  # None because we don't have a button instance
        
    def auto_refresh_transactions(self, dt):
        """Automatically refresh transactions periodically"""
        if self.auto_refresh_active and not self.is_mining:
            self.safe_log("Auto-refreshing transactions...")
            self.load_transactions(None)
        
    def build_ui(self):
        # Header with smaller height and left-aligned text
        header = Label(
            text="🌜 Luna Miner 🌛", 
            font_size='16sp',
            bold=True,
            color=(1, 0.2, 0.2, 1),  # Red text
            size_hint_y=None,
            height=40
        )
        self.add_widget(header)
        
        # Row of 3 buttons with smaller padding
        buttons_layout = BoxLayout(
            size_hint_y=None, 
            height=35, 
            spacing=5,
            padding=2
        )
        
        # Mine button
        self.mine_btn = ToggleButton(
            text="Start Mining", 
            on_press=self.toggle_mining,
            background_color=(0.5, 0, 0, 1),  # Dark red
            color=(1, 1, 1, 1),  # White text
            size_hint_y=None,
            height=30,
            font_size='12sp'
        )
        buttons_layout.add_widget(self.mine_btn)
        
        # Load transactions button - USE THE EXPOSED LOAD FUNCTION
        load_btn = Button(
            text="Load Transactions", 
            on_press=self.load_transactions_from_node,
            background_color=(0.5, 0, 0, 1),  # Dark red
            color=(1, 1, 1, 1),  # White text
            size_hint_y=None,
            height=30,
            font_size='12sp'
        )
        buttons_layout.add_widget(load_btn)
        
        # Clear mempool button
        clear_btn = Button(
            text="Clear Mempool", 
            on_press=self.clear_mempool,
            background_color=(0.5, 0, 0, 1),  # Dark red
            color=(1, 1, 1, 1),  # White text
            size_hint_y=None,
            height=30,
            font_size='12sp'
        )
        buttons_layout.add_widget(clear_btn)
        
        self.add_widget(buttons_layout)
        
        # Auto-refresh toggle
        refresh_layout = BoxLayout(
            size_hint_y=None, 
            height=30, 
            spacing=5,
            padding=2
        )
        
        refresh_layout.add_widget(Label(text="Auto-refresh:", color=(1, 1, 1, 1), font_size='12sp'))
        
        self.auto_refresh_btn = ToggleButton(
            text="ON", 
            state='down',
            on_press=self.toggle_auto_refresh,
            background_color=(0.3, 0.5, 0.3, 1),  # Green when on
            color=(1, 1, 1, 1),
            size_hint=(None, None),
            size=(60, 25),
            font_size='11sp'
        )
        refresh_layout.add_widget(self.auto_refresh_btn)
        
        refresh_layout.add_widget(Label(text="(Every 10 sec)", color=(0.7, 0.7, 0.7, 1), font_size='11sp'))
        
        self.add_widget(refresh_layout)
        
        # Stats panel - at the bottom of the buttons
        stats_layout = GridLayout(
            cols=2, 
            size_hint_y=None, 
            height=180,  # Increased height for additional stats
            spacing=3, 
            padding=3
        )
        
        stats_data = [
            ("Status:", "status_text", False),
            ("Block Height:", "height_text", False),
            ("Total Blocks Mined:", "total_blocks_text", False),
            ("Total Rewards:", "total_rewards_text", False),
            ("Average Time/Block:", "avg_time_text", False),
            ("Pending Transactions:", "pending_txs_text", False),
            ("Hash Rate:", "hash_rate_text", True),  # New hash rate display
            ("Current Hash:", "current_hash_text", True),  # New current hash display
            ("Mining Progress:", "progress_text", True)  # Red text for progress
        ]
        
        for label, prop, is_red in stats_data:
            stats_layout.add_widget(Label(text=label, color=(1, 1, 1, 1), font_size='12sp'))
            value_label = Label(text=getattr(self, prop) if hasattr(self, prop) else "", 
                               color=(1, 0.2, 0.2, 1) if is_red else (1, 1, 1, 1), 
                               font_size='12sp')
            setattr(self, f"{prop}_label", value_label)
            stats_layout.add_widget(value_label)
        
        self.add_widget(stats_layout)
        
        # Progress bar with red color
        self.progress_bar = ProgressBar(max=100, value=0)
        self.add_widget(self.progress_bar)
        
        # Tabs with red headers
        self.tabs = TabbedPanel(
            do_default_tab=False,
            tab_width=100,
            background_color=(0.1, 0.1, 0.1, 1)
        )
        
        # Customize tab appearance
        self.tabs.background_color = (0.1, 0.1, 0.1, 1)
        
        # Blockchain tab
        blockchain_tab = TabbedPanelItem(text='Blockchain')
        blockchain_tab.background_color = (0.2, 0, 0, 1)  # Red tab
        blockchain_scroll = ScrollView()
        self.blockchain_text = TextInput(
            text='Blockchain data will appear here...',
            readonly=True,
            background_color=(0.1, 0.1, 0.1, 1),
            foreground_color=(1, 1, 1, 1),  # White text
            size_hint_y=None,
            font_size='11sp'
        )
        self.blockchain_text.bind(minimum_height=self.blockchain_text.setter('height'))
        blockchain_scroll.add_widget(self.blockchain_text)
        blockchain_tab.add_widget(blockchain_scroll)
        self.tabs.add_widget(blockchain_tab)
        
        # Transactions tab
        transactions_tab = TabbedPanelItem(text='Transactions')
        transactions_tab.background_color = (0.2, 0, 0, 1)  # Red tab
        transactions_scroll = ScrollView()
        self.transactions_text = TextInput(
            text='Transaction data will appear here...',
            readonly=True,
            background_color=(0.1, 0.1, 0.1, 1),
            foreground_color=(1, 1, 1, 1),  # White text
            size_hint_y=None,
            font_size='11sp'
        )
        self.transactions_text.bind(minimum_height=self.transactions_text.setter('height'))
        transactions_scroll.add_widget(self.transactions_text)
        transactions_tab.add_widget(transactions_scroll)
        self.tabs.add_widget(transactions_tab)
        
        # Bills tab
        bills_tab = TabbedPanelItem(text='Bills')
        bills_tab.background_color = (0.2, 0, 0, 1)  # Red tab
        bills_scroll = ScrollView()
        self.bills_text = TextInput(
            text='Bill data will appear here...',
            readonly=True,
            background_color=(0.1, 0.1, 0.1, 1),
            foreground_color=(1, 1, 1, 1),  # White text
            size_hint_y=None,
            font_size='11sp'
        )
        self.bills_text.bind(minimum_height=self.bills_text.setter('height'))
        bills_scroll.add_widget(self.bills_text)
        bills_tab.add_widget(bills_scroll)
        self.tabs.add_widget(bills_tab)
        
        # Log tab
        log_tab = TabbedPanelItem(text='Log')
        log_tab.background_color = (0.2, 0, 0, 1)  # Red tab
        log_scroll = ScrollView()
        self.log_text = TextInput(
            text='Log messages will appear here...',
            readonly=True,
            background_color=(0.1, 0.1, 0.1, 1),
            foreground_color=(1, 1, 1, 1),  # White text
            size_hint_y=None,
            font_size='11sp'
        )
        self.log_text.bind(minimum_height=self.log_text.setter('height'))
        log_scroll.add_widget(self.log_text)
        log_tab.add_widget(log_scroll)
        self.tabs.add_widget(log_tab)
        
        self.add_widget(self.tabs)
        
        # Difficulty controls at the bottom
        difficulty_layout = BoxLayout(
            size_hint_y=None, 
            height=35, 
            spacing=5,
            padding=2
        )
        
        difficulty_layout.add_widget(Label(text="Difficulty:", color=(1, 1, 1, 1), font_size='12sp'))
        
        self.diff_spinner = Spinner(
            text=str(self.difficulty),
            values=[str(i) for i in range(1, 9)],
            background_color=(0.5, 0, 0, 1),  # Dark red
            color=(1, 1, 1, 1),  # White text
            size_hint=(None, None),
            size=(60, 30),
            font_size='12sp'
        )
        difficulty_layout.add_widget(self.diff_spinner)
        
        set_btn = Button(
            text="Set Difficulty", 
            on_press=self.set_difficulty,
            background_color=(0.5, 0, 0, 1),  # Dark red
            color=(1, 1, 1, 1),  # White text
            size_hint_y=None,
            height=30,
            font_size='12sp'
        )
        difficulty_layout.add_widget(set_btn)
        
        self.add_widget(difficulty_layout)
        
        # Add initial content
        self.safe_log("Miner GUI initialized")
        Clock.schedule_once(lambda dt: self.update_blockchain_info(), 0.1)
        Clock.schedule_once(lambda dt: self.update_bills_info(), 0.1)
        
        # Initialize new properties
        self.hash_rate_text = "0 H/s"
        self.current_hash_text = "N/A"
    
    def toggle_auto_refresh(self, instance):
        """Toggle automatic transaction refresh"""
        self.auto_refresh_active = (instance.state == 'down')
        if self.auto_refresh_active:
            instance.text = "ON"
            instance.background_color = (0.3, 0.5, 0.3, 1)  # Green
            self.safe_log("Auto-refresh enabled")
        else:
            instance.text = "OFF"
            instance.background_color = (0.5, 0.3, 0.3, 1)  # Red
            self.safe_log("Auto-refresh disabled")
    
    def safe_log(self, message):
        """Add message to log queue (thread-safe)"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}\n")
    
    def process_log_queue(self, dt):
        """Process log messages from queue in main thread"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.text += message
                # Auto-scroll to bottom
                self.log_text.cursor = (0, len(self.log_text.text))
        except queue.Empty:
            pass
        
    def send_to_node(self, data):
        """Send command to node and get response with proper framing"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect(('127.0.0.1', 9335))
                
                # Send message with length prefix
                message_json = json.dumps(data)
                message_data = message_json.encode()
                message_length = len(message_data)
                
                # Send length first
                s.sendall(message_length.to_bytes(4, 'big'))
                # Send message
                s.sendall(message_data)
                
                # Receive response length
                length_bytes = s.recv(4)
                if not length_bytes:
                    return {"status": "error", "message": "No response from node"}
                
                response_length = int.from_bytes(length_bytes, 'big')
                
                # Receive response data
                response_data = b""
                while len(response_data) < response_length:
                    chunk = s.recv(min(65536, response_length - len(response_data)))
                    if not chunk:
                        break
                    response_data += chunk
                
                if len(response_data) == response_length:
                    return json.loads(response_data.decode())
                else:
                    return {"status": "error", "message": "Incomplete response from node"}
                    
        except Exception as e:
            self.safe_log(f"Error connecting to node: {e}")
            return {"status": "error", "message": str(e)}
    
    def toggle_mining(self, instance):
        """Start or stop mining"""
        if instance.state == 'down':  # Start mining
            self.is_mining = True
            self.mine_btn.text = "Stop Mining"
            self.status_text = "Mining..."
            self.progress_text = "Mining in progress..."
            
            # Start mining in a separate thread
            self.mining_thread = threading.Thread(target=self.mine, daemon=True)
            self.mining_thread.start()
            
            # Start progress bar animation
            Clock.schedule_interval(self.update_progress_bar, 0.1)
        else:  # Stop mining
            self.is_mining = False
            self.mine_btn.text = "Start Mining"
            self.status_text = "Mining stopped"
            self.progress_text = "Mining stopped"
            
            # Stop progress bar animation
            Clock.unschedule(self.update_progress_bar)
            self.progress_bar.value = 0
    
    def update_progress_bar(self, dt):
        """Animate progress bar while mining"""
        if self.progress_bar.value >= 100:
            self.progress_bar.value = 0
        else:
            self.progress_bar.value += 5
    
    def mine(self):
        """Mine blocks"""
        try:
            # Get mining address
            response = self.send_to_node({"action": "get_mining_address"})
            if response.get("status") != "success":
                self.safe_log("Error getting mining address")
                return
                
            mining_address = response.get("address", "miner_default_address")
            
            # Start actual mining
            response = self.send_to_node({
                "action": "start_mining",
                "miner_address": mining_address
            })
            
            if response.get("status") == "success":
                self.safe_log("Mining started successfully")
                # Update stats periodically while mining
                while self.is_mining:
                    time.sleep(2)
                    # Schedule UI updates in main thread
                    Clock.schedule_once(lambda dt: self.update_stats())
                    Clock.schedule_once(lambda dt: self.update_blockchain_info())
                    Clock.schedule_once(lambda dt: self.update_transactions_info())
                    Clock.schedule_once(lambda dt: self.update_bills_info())
            else:
                self.safe_log(f"Mining failed: {response.get('message')}")
                
        except Exception as e:
            self.safe_log(f"Error in mining: {e}")
        finally:
            self.is_mining = False
            # Schedule UI updates in main thread
            Clock.schedule_once(lambda dt: setattr(self.mine_btn, 'state', 'normal'))
            Clock.schedule_once(lambda dt: setattr(self.mine_btn, 'text', 'Start Mining'))
            Clock.schedule_once(lambda dt: setattr(self, 'status_text', 'Mining stopped'))
            Clock.schedule_once(lambda dt: setattr(self, 'progress_text', 'Mining stopped'))
            Clock.schedule_once(lambda dt: setattr(self.progress_bar, 'value', 0))
            
    def load_transactions_from_node(self, instance):
        """Use the exposed load function to load transactions"""
        self.safe_log("Using exposed load function to load transactions...")
        response = self.send_to_node({"action": "load"})
        if response.get("status") == "success":
            self.safe_log("Transactions loaded successfully using exposed load function")
            self.update_transactions_info()
        else:
            self.safe_log(f"Failed to load transactions: {response.get('message')}")
    
    def load_transactions(self, instance):
        """Load transactions from mempool - for auto-refresh"""
        response = self.send_to_node({"action": "get_pending_transactions"})
        if response.get("status") == "success":
            transactions = response.get("transactions", [])
            self.safe_log(f"Loaded {len(transactions)} transactions from mempool")
            self.update_transactions_info()
        else:
            self.safe_log("Failed to load transactions from node")
    
    def clear_mempool(self, instance):
        """Clear the mempool"""
        response = self.send_to_node({"action": "clear_mempool"})
        if response.get("status") == "success":
            self.safe_log("Mempool cleared successfully")
            self.update_transactions_info()
        else:
            self.safe_log("Failed to clear mempool")

    def set_difficulty(self, instance):
        """Set mining difficulty"""
        try:
            self.difficulty = int(self.diff_spinner.text)
            self.safe_log(f"Setting difficulty to {self.difficulty} (handled by node console)")
            self.update_bills_info()
        except ValueError:
            self.safe_log("Invalid difficulty value")
    
    def update_stats(self, dt=None):
        """Update statistics display with mining progress"""
        # Get blockchain info
        response = self.send_to_node({"action": "get_blockchain_info"})
        if response.get("status") == "success":
            self.height_text = str(response.get("blockchain_height", 0))
            self.pending_txs_text = str(response.get("pending_transactions", 0))
        
        # Get mining stats
        stats_response = self.send_to_node({"action": "get_mining_stats"})
        if stats_response.get("status") == "success":
            stats = stats_response.get("stats", {})
            self.total_blocks_text = str(stats.get('total_blocks', 0))
            self.total_rewards_text = f"{stats.get('total_rewards', 0):.2f} LC"
            self.avg_time_text = f"{stats.get('avg_time', 0):.2f} s"
            
            # Update status to "Started" when mining
            if self.is_mining:
                self.status_text = "Started"
        
        # Get mining progress if mining
        if self.is_mining:
            progress_response = self.send_to_node({"action": "get_mining_progress"})
            if progress_response.get("status") == "success" and progress_response.get("mining"):
                progress = progress_response.get("progress", {})
                hashes = progress.get("hashes", 0)
                hash_rate = progress.get("hash_rate", 0)
                current_hash = progress.get("current_hash", "N/A")
                
                # Format hash rate
                if hash_rate > 1_000_000:
                    hash_rate_str = f"{hash_rate/1_000_000:.2f} MH/s"
                elif hash_rate > 1_000:
                    hash_rate_str = f"{hash_rate/1_000:.2f} KH/s"
                else:
                    hash_rate_str = f"{hash_rate:.2f} H/s"
                
                self.hash_rate_text = hash_rate_str
                self.current_hash_text = current_hash[:16] + "..." if len(current_hash) > 16 else current_hash
                self.progress_text = f"{hashes:,} hashes, {hash_rate_str}"
    
    def update_blockchain_info(self, dt=None):
        """Update blockchain information display"""
        response = self.send_to_node({"action": "get_blockchain"})
        if response.get("status") == "success":
            blockchain = response.get("blockchain", [])
            
            # Clear and update text area
            blockchain_text = ""
            for block in blockchain:
                blockchain_text += f"Block {block.get('index', 0)}:\n"
                blockchain_text += f"  Hash: {block.get('hash', '')[:16]}...\n"
                blockchain_text += f"  Previous: {block.get('previous_hash', '')[:16]}...\n"
                blockchain_text += f"  Transactions: {len(block.get('transactions', []))}\n"
                blockchain_text += f"  Nonce: {block.get('nonce', 0)}\n"
                blockchain_text += "-" * 40 + "\n"
            
            self.blockchain_text.text = blockchain_text
        
        # Update transactions
        self.update_transactions_info()
    
    def update_transactions_info(self, dt=None):
        """Update transactions information display"""
        response = self.send_to_node({"action": "get_pending_transactions"})
        if response.get("status") == "success":
            transactions = response.get("transactions", [])
            
            # Clear and update text area
            transactions_text = ""
            for i, tx in enumerate(transactions):
                transactions_text += f"Transaction {i+1}:\n"
                if tx.get('type') == 'reward':
                    transactions_text += f"  Type: Mining Reward\n"
                    transactions_text += f"  To: {tx.get('to', 'Unknown')}\n"
                    transactions_text += f"  Amount: {tx.get('amount', 0)} LC\n"
                    if 'denomination' in tx:
                        transactions_text += f"  Bill: ${tx.get('denomination')}\n"
                        transactions_text += f"  Multiplier: x{tx.get('multiplier', 1)}\n"
                        # Track serial numbers for mined bills
                        if 'serial' in tx:
                            denom = tx.get('denomination')
                            serial = tx.get('serial')
                            if denom not in self.mined_serials:
                                self.mined_serials[denom] = []
                            if serial not in self.mined_serials[denom]:
                                self.mined_serials[denom].append(serial)
                                self.safe_log(f"Mined ${denom} bill with serial: {serial}")
                else:
                    transactions_text += f"  From: {tx.get('from', 'Unknown')}\n"
                    transactions_text += f"  To: {tx.get('to', 'Unknown')}\n"
                    transactions_text += f"  Amount: {tx.get('amount', 0)} LC\n"
                transactions_text += "-" * 30 + "\n"
            
            self.transactions_text.text = transactions_text
    
    def update_bills_info(self, dt=None):
        """Update bills information display with serial numbers"""
        response = self.send_to_node({"action": "get_difficulty_info"})
        if response.get("status") == "success":
            difficulty = response.get("difficulty", 4)
            available_bills = response.get("available_bills", {})
            base_reward = response.get("base_reward", 50)
            
            # Get mined serials from blockchain
            mined_serials_response = self.send_to_node({"action": "get_mined_serials"})
            if mined_serials_response.get("status") == "success":
                self.mined_serials = mined_serials_response.get("mined_serials", {})
            
            # Build bills text
            bills_text = f"Current Difficulty: {difficulty}\n"
            bills_text += f"Base Reward: {base_reward} LC\n\n"
            bills_text += "Available Bills:\n"
            
            for bill, multiplier in available_bills.items():
                reward = base_reward * multiplier
                bills_text += f"  ${bill} Bill: x{multiplier} → {reward} LC\n"
                
                # Show serial numbers for mined bills of this denomination
                if bill in self.mined_serials and self.mined_serials[bill]:
                    bills_text += f"    Mined Serials:\n"
                    for serial in self.mined_serials[bill]:
                        bills_text += f"      {serial}\n"
            
            bills_text += "\nHigher difficulty levels offer:\n"
            bills_text += "- Higher value bills\n"
            bills_text += "- Higher multipliers\n"
            bills_text += "- Greater mining challenges\n"
            
            self.bills_text.text = bills_text

class LunaMinerApp(App):
    def build(self):
        self.title = "Luna Miner"
        self.gui = MinerGUI()
        self.gui.build_ui()
        return self.gui

if __name__ == "__main__":
    LunaMinerApp().run()