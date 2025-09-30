#!/usr/bin/env python3
"""
luna_node_gui.py - Dark Theme Kivy GUI for Luna Coin blockchain node and miner
"""

import os
import sys
import threading
import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.config import Config
from kivy.graphics import Color, Rectangle
from kivy.core.text import Label as CoreLabel

# Set window size and dark theme - MUCH SMALLER
Config.set('graphics', 'width', '600')
Config.set('graphics', 'height', '400')
Config.set('graphics', 'minimum_width', '600')
Config.set('graphics', 'minimum_height', '400')

# Dark theme colors
DARK_BG = (0.1, 0.1, 0.1, 1)  # Almost black
DARK_PANEL = (0.15, 0.15, 0.15, 1)
DARK_CARD = (0.2, 0.2, 0.2, 1)
ACCENT_RED = (1, 0.2, 0.2, 1)  # Bright red
ACCENT_WHITE = (0.9, 0.9, 0.9, 1)  # Off-white
TEXT_COLOR = (0.8, 0.8, 0.8, 1)

# Import your existing node functionality
from luna_node import Blockchain, DataManager, configure_ssl_for_frozen_app

class DarkLabel(Label):
    """Dark theme label with custom styling"""
    def __init__(self, **kwargs):
        kwargs.setdefault('color', TEXT_COLOR)
        kwargs.setdefault('size_hint_y', None)
        kwargs.setdefault('height', 25)  # Smaller
        super().__init__(**kwargs)

class DarkButton(Button):
    """Dark theme button with red accent"""
    def __init__(self, **kwargs):
        kwargs.setdefault('background_color', DARK_CARD)
        kwargs.setdefault('color', ACCENT_WHITE)
        kwargs.setdefault('size_hint_y', None)
        kwargs.setdefault('height', 25)  # Smaller
        kwargs.setdefault('background_normal', '')
        kwargs.setdefault('background_down', '')
        super().__init__(**kwargs)
        
        with self.canvas.before:
            Color(*ACCENT_RED)
            self.border_rect = Rectangle(pos=self.pos, size=self.size)
        
        self.bind(pos=self.update_border, size=self.update_border)
    
    def update_border(self, *args):
        self.border_rect.pos = (self.pos[0] - 1, self.pos[1] - 1)
        self.border_rect.size = (self.size[0] + 2, self.size[1] + 2)

class DarkTextInput(TextInput):
    """Dark theme text input"""
    def __init__(self, **kwargs):
        kwargs.setdefault('background_color', DARK_CARD)
        kwargs.setdefault('foreground_color', ACCENT_WHITE)
        kwargs.setdefault('size_hint_y', None)
        kwargs.setdefault('height', 25)  # Smaller
        kwargs.setdefault('background_normal', '')
        kwargs.setdefault('background_active', '')
        kwargs.setdefault('cursor_color', ACCENT_RED)
        kwargs.setdefault('selection_color', ACCENT_RED)
        super().__init__(**kwargs)

class DarkSpinner(Spinner):
    """Dark theme spinner"""
    def __init__(self, **kwargs):
        kwargs.setdefault('background_color', DARK_CARD)
        kwargs.setdefault('color', ACCENT_WHITE)
        kwargs.setdefault('size_hint_y', None)
        kwargs.setdefault('height', 25)  # Smaller
        super().__init__(**kwargs)

class ConsoleOutput(ScrollView):
    """Console output display area with dark theme"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = DARK_BG
        
        self.layout = GridLayout(cols=1, size_hint_y=None, spacing=1, padding=2)  # Tighter spacing
        self.layout.bind(minimum_height=self.layout.setter('height'))
        self.add_widget(self.layout)
        
    def add_message(self, message, message_type="info"):
        """Add a message to the console"""
        timestamp = time.strftime("%H:%M:%S")
        
        if message_type == "error":
            prefix = "[ERROR]"
        elif message_type == "success":
            prefix = "[OK]"
        elif message_type == "warning":
            prefix = "[WARN]"
        else:
            prefix = "[INFO]"
        
        message_text = f"{prefix} [{timestamp}] {message}"
        label = DarkLabel(
            text=message_text, 
            height=20,  # Smaller
            text_size=(self.width - 10, None),
            halign='left',
            valign='middle'
        )
        self.layout.add_widget(label)
        Clock.schedule_once(lambda dt: self.scroll_to(label), 0.1)

class ControlPanel(GridLayout):
    """Main control panel with compact card layout"""
    
    def __init__(self, main_app, **kwargs):
        super().__init__(**kwargs)
        self.main_app = main_app
        self.cols = 2
        self.spacing = 3
        self.padding = 3
        self.size_hint_y = None
        self.height = 150
        
        self.create_controls()
    
    def create_controls(self):
        # Mining Controls
        mine_label = DarkLabel(text='Mining:', size_hint_x=None, width=80, height=20)
        self.add_widget(mine_label)
        
        self.mining_address = DarkTextInput(
            text='miner_default', 
            hint_text='Address',
            size_hint_x=None,
            width=150,
            height=20
        )
        self.add_widget(self.mining_address)
        
        mine_btn = DarkButton(text='Mine', size_hint_x=None, width=60, height=20)
        mine_btn.bind(on_press=self.start_mining)
        self.add_widget(mine_btn)
        
        load_btn = DarkButton(text='Load Tx', size_hint_x=None, width=60, height=20)
        load_btn.bind(on_press=self.load_transactions)
        self.add_widget(load_btn)
        
        # Difficulty Controls
        diff_label = DarkLabel(text='Difficulty:', size_hint_x=None, width=80, height=20)
        self.add_widget(diff_label)
        
        self.difficulty_spinner = DarkSpinner(
            text='6', 
            values=[str(i) for i in range(1, 9)],
            size_hint_x=None,
            width=50,
            height=20
        )
        self.add_widget(self.difficulty_spinner)
        
        set_diff_btn = DarkButton(text='Set', size_hint_x=None, width=40, height=20)
        set_diff_btn.bind(on_press=self.set_difficulty)
        self.add_widget(set_diff_btn)
        
        bills_btn = DarkButton(text='Bills', size_hint_x=None, width=40, height=20)
        bills_btn.bind(on_press=self.show_available_bills)
        self.add_widget(bills_btn)
        
        # Network Controls
        peer_label = DarkLabel(text='Peer:', size_hint_x=None, width=80, height=20)
        self.add_widget(peer_label)
        
        self.peer_address = DarkTextInput(
            text='', 
            hint_text='ip:port',
            size_hint_x=None,
            width=150,
            height=20
        )
        self.add_widget(self.peer_address)
        
        add_peer_btn = DarkButton(text='Add', size_hint_x=None, width=40, height=20)
        add_peer_btn.bind(on_press=self.add_peer)
        self.add_widget(add_peer_btn)
        
        discover_btn = DarkButton(text='Discover', size_hint_x=None, width=60, height=20)
        discover_btn.bind(on_press=self.discover_peers)
        self.add_widget(discover_btn)
        
        # Blockchain Controls
        chain_label = DarkLabel(text='Chain:', size_hint_x=None, width=80, height=20)
        self.add_widget(chain_label)
        
        status_btn = DarkButton(text='Status', size_hint_x=None, width=50, height=20)
        status_btn.bind(on_press=self.show_status)
        self.add_widget(status_btn)
        
        stats_btn = DarkButton(text='Stats', size_hint_x=None, width=40, height=20)
        stats_btn.bind(on_press=self.show_stats)
        self.add_widget(stats_btn)
        
        validate_btn = DarkButton(text='Valid', size_hint_x=None, width=40, height=20)
        validate_btn.bind(on_press=self.validate_chain)
        self.add_widget(validate_btn)
        
        sync_btn = DarkButton(text='Sync', size_hint_x=None, width=40, height=20)
        sync_btn.bind(on_press=self.sync_all)
        self.add_widget(sync_btn)

    # ADD BACK ALL THE MISSING METHODS:
    def start_mining(self, instance):
        address = self.mining_address.text.strip()
        if not address:
            self.main_app.log("Please enter a mining address", "error")
            return
        
        self.main_app.log(f"Starting mining with address: {address}", "info")
        
        def mining_thread():
            try:
                success = self.main_app.blockchain.mine_pending_transactions(address)
                if success:
                    self.main_app.log("Mining completed successfully!", "success")
                else:
                    self.main_app.log("Mining failed or no transactions to mine", "warning")
            except Exception as e:
                self.main_app.log(f"Mining error: {e}", "error")
        
        thread = threading.Thread(target=mining_thread, daemon=True)
        thread.start()
    
    def load_transactions(self, instance):
        self.main_app.log("Loading transactions from mempool...", "info")
        
        def load_thread():
            try:
                from luna_node import load_mempool
                txs = load_mempool()
                loaded_count = 0
                for tx in txs:
                    if self.main_app.blockchain.add_transaction(tx):
                        loaded_count += 1
                self.main_app.log(f"Added {loaded_count}/{len(txs)} transactions", "success")
            except Exception as e:
                self.main_app.log(f"Error loading transactions: {e}", "error")
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def set_difficulty(self, instance):
        try:
            new_diff = int(self.difficulty_spinner.text)
            if 1 <= new_diff <= 8:
                self.main_app.blockchain.difficulty = new_diff
                self.main_app.log(f"Difficulty set to {new_diff}", "success")
                bills = self.main_app.blockchain.difficulty_denominations.get(new_diff, {})
                bill_list = ', '.join([f'${bill}' for bill in bills.keys()])
                self.main_app.log(f"Available Bills: {bill_list}", "info")
            else:
                self.main_app.log("Difficulty must be between 1 and 8", "error")
        except ValueError:
            self.main_app.log("Invalid difficulty value", "error")
    
    def show_available_bills(self, instance):
        current_diff = self.main_app.blockchain.difficulty
        bills = self.main_app.blockchain.difficulty_denominations.get(current_diff, {})
        
        self.main_app.log(f"Bills for Difficulty {current_diff}:", "info")
        for bill, multiplier in bills.items():
            reward = self.main_app.blockchain.base_mining_reward * multiplier
            self.main_app.log(f"  ${bill}: x{multiplier} → {reward} LC", "info")
    
    def add_peer(self, instance):
        peer_address = self.peer_address.text.strip()
        if not peer_address or ":" not in peer_address:
            self.main_app.log("Please enter valid peer address (ip:port)", "error")
            return
        
        self.main_app.log(f"Adding peer: {peer_address}", "info")
        if self.main_app.blockchain.add_peer(peer_address):
            self.main_app.log(f"Peer {peer_address} added", "success")
            self.peer_address.text = ""
        else:
            self.main_app.log(f"Failed to add peer: {peer_address}", "error")
    
    def discover_peers(self, instance):
        self.main_app.log("Discovering peers...", "info")
        
        def discover_thread():
            try:
                self.main_app.blockchain.discover_peers()
                peer_count = len(self.main_app.blockchain.peers)
                self.main_app.log(f"Discovery complete. Peers: {peer_count}", "success")
            except Exception as e:
                self.main_app.log(f"Error discovering peers: {e}", "error")
        
        thread = threading.Thread(target=discover_thread, daemon=True)
        thread.start()
    
    def sync_all(self, instance):
        self.main_app.log("Starting network synchronization...", "info")
        
        def sync_thread():
            try:
                self.main_app.blockchain.sync_all()
                self.main_app.log("Synchronization completed", "success")
            except Exception as e:
                self.main_app.log(f"Sync error: {e}", "error")
        
        thread = threading.Thread(target=sync_thread, daemon=True)
        thread.start()
    
    def show_status(self, instance):
        latest = self.main_app.blockchain.get_latest_block()
        stats = self.main_app.blockchain.get_mining_stats()
        
        self.main_app.log("Blockchain Status:", "info")
        self.main_app.log(f"  Height: {len(self.main_app.blockchain.chain)} blocks", "info")
        self.main_app.log(f"  Pending TXs: {len(self.main_app.blockchain.pending_transactions)}", "info")
        self.main_app.log(f"  Latest Block: {latest.hash[:16]}...", "info")
        self.main_app.log(f"  Total Blocks Mined: {stats['total_blocks']}", "info")
        self.main_app.log(f"  Known Peers: {len(self.main_app.blockchain.peers)}", "info")
    
    def show_stats(self, instance):
        stats = self.main_app.blockchain.get_mining_stats()
        
        self.main_app.log("Mining Statistics:", "info")
        self.main_app.log(f"  Total Blocks: {stats['total_blocks']}", "info")
        self.main_app.log(f"  Total Rewards: {stats['total_rewards']} LC", "info")
        self.main_app.log(f"  Avg Time/Block: {stats['avg_time']:.2f}s", "info")
        self.main_app.log(f"  Difficulty: {self.main_app.blockchain.difficulty}", "info")
    
    def validate_chain(self, instance):
        self.main_app.log("Validating blockchain...", "info")
        
        def validate_thread():
            try:
                if self.main_app.blockchain.is_chain_valid():
                    self.main_app.log("Blockchain is valid!", "success")
                else:
                    self.main_app.log("Blockchain validation failed!", "error")
            except Exception as e:
                self.main_app.log(f"Validation error: {e}", "error")
        
        thread = threading.Thread(target=validate_thread, daemon=True)
        thread.start()

class StatusBar(BoxLayout):
    """Compact status bar at the bottom"""
    
    def __init__(self, main_app, **kwargs):
        super().__init__(**kwargs)
        self.main_app = main_app
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = 25  # More compact
        self.spacing = 5
        self.padding = 3
        
        with self.canvas.before:
            Color(*DARK_PANEL)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        
        self.bind(pos=self.update_rect, size=self.update_rect)
        
        # Status indicators - more compact
        self.chain_label = DarkLabel(text="⛓️0", size_hint_x=None, width=50, height=20)
        self.add_widget(self.chain_label)
        
        self.tx_label = DarkLabel(text="📝0", size_hint_x=None, width=50, height=20)
        self.add_widget(self.tx_label)
        
        self.peers_label = DarkLabel(text="🌐0", size_hint_x=None, width=50, height=20)
        self.add_widget(self.peers_label)
        
        self.diff_label = DarkLabel(text="⚡0", size_hint_x=None, width=50, height=20)
        self.add_widget(self.diff_label)
        
        # Spacer
        self.add_widget(Label(size_hint_x=1))
        
        self.node_label = DarkLabel(text="Node...", size_hint_x=None, width=100, height=20)
        self.add_widget(self.node_label)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def update_status(self):
        chain_height = len(self.main_app.blockchain.chain)
        pending_txs = len(self.main_app.blockchain.pending_transactions)
        peers_count = len(self.main_app.blockchain.peers)
        difficulty = self.main_app.blockchain.difficulty
        
        self.chain_label.text = f"⛓️{chain_height}"
        self.tx_label.text = f"📝{pending_txs}"
        self.peers_label.text = f"🌐{peers_count}"
        self.diff_label.text = f"⚡{difficulty}"
        self.node_label.text = f"{self.main_app.blockchain.node_address[:10]}..."

class LunaNodeGUI(TabbedPanel):
    """Main GUI application with dark theme"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_default_tab = False
        self.background_color = DARK_BG
        self.border = [0, 0, 0, 0]
        self.tab_width = 100  # Smaller tabs
        
        # Initialize blockchain
        configure_ssl_for_frozen_app()
        self.blockchain = Blockchain()
        self.blockchain.start_wallet_server()
        self.blockchain.start_peer_discovery()
        
        self.setup_gui()
        
        # Start status update timer
        Clock.schedule_interval(self.update_status, 1.0)
    
    def setup_gui(self):
        # Controls Tab
        controls_tab = TabbedPanelItem(text='Controls')
        controls_layout = BoxLayout(orientation='vertical', spacing=0, padding=0)
        
        # Control panel
        self.control_panel = ControlPanel(self)
        controls_layout.add_widget(self.control_panel)
        
        # Status bar at bottom
        self.status_bar = StatusBar(self)
        controls_layout.add_widget(self.status_bar)
        
        controls_tab.add_widget(controls_layout)
        self.add_widget(controls_tab)
        
        # Console Tab
        console_tab = TabbedPanelItem(text='Console')
        console_layout = BoxLayout(orientation='vertical', spacing=0, padding=0)
        
        # REMOVED the console background rectangle
        self.console_output = ConsoleOutput()
        console_layout.add_widget(self.console_output)
        
        console_tab.add_widget(console_layout)
        self.add_widget(console_tab)
        
        # Network Tab - Simplified
        network_tab = TabbedPanelItem(text='Network')
        network_layout = BoxLayout(orientation='vertical', padding=5, spacing=5)
        
        peer_scroll = ScrollView()
        self.peer_list = GridLayout(cols=1, spacing=2, size_hint_y=None)
        self.peer_list.bind(minimum_height=self.peer_list.setter('height'))
        peer_scroll.add_widget(self.peer_list)
        network_layout.add_widget(peer_scroll)
        
        network_tab.add_widget(network_layout)
        self.add_widget(network_tab)
        
        # Blockchain Tab - Simplified
        chain_tab = TabbedPanelItem(text='Chain')
        chain_layout = BoxLayout(orientation='vertical', padding=5, spacing=5)
        
        info_grid = GridLayout(cols=1, spacing=5, size_hint_y=None, height=80)
        
        self.chain_info_label = DarkLabel(text="Loading...", height=40)
        info_grid.add_widget(self.chain_info_label)
        
        self.mempool_info_label = DarkLabel(text="Loading...", height=40)
        info_grid.add_widget(self.mempool_info_label)
        
        chain_layout.add_widget(info_grid)
        chain_tab.add_widget(chain_layout)
        self.add_widget(chain_tab)

    # Remove the update_console_bg method entirely
    
    def log(self, message, message_type="info"):
        """Add a message to the console output"""
        Clock.schedule_once(lambda dt: self.console_output.add_message(message, message_type), 0)
    
    def update_status(self, dt):
        """Update all status displays"""
        self.status_bar.update_status()
        self.update_peer_list()
        self.update_blockchain_info()
    
    def update_peer_list(self):
        """Update the peer list display"""
        self.peer_list.clear_widgets()
        
        if not self.blockchain.peers:
            no_peers_label = DarkLabel(text="No peers", height=25)
            self.peer_list.add_widget(no_peers_label)
            return
        
        for peer in self.blockchain.peers:
            peer_label = DarkLabel(text=peer, height=25)
            self.peer_list.add_widget(peer_label)
    
    def update_blockchain_info(self):
        """Update blockchain information display"""
        chain_height = len(self.blockchain.chain)
        pending_txs = len(self.blockchain.pending_transactions)
        
        latest_block = self.blockchain.get_latest_block()
        latest_hash = latest_block.hash[:12] + "..." if latest_block else "N/A"
        
        self.chain_info_label.text = f"Height: {chain_height}\nHash: {latest_hash}"
        self.mempool_info_label.text = f"Pending: {pending_txs}\nDiff: {self.blockchain.difficulty}"
    
    def on_stop(self):
        self.blockchain.stop_wallet_server()
        self.blockchain.save_chain()
        self.blockchain.save_known_peers()
        self.log("Shutdown complete", "info")

class LunaNodeApp(App):
    """Kivy App wrapper"""
    
    def build(self):
        self.title = "Luna Node"
        Window.clearcolor = DARK_BG
        return LunaNodeGUI()
    
    def on_stop(self):
        if hasattr(self.root, 'on_stop'):
            self.root.on_stop()

if __name__ == '__main__':
    try:
        LunaNodeApp().run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()