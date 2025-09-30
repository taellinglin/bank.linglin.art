# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Common data files and hidden imports for all executables
common_datas = [
    ('templates/*', 'templates'),
    ('static/*', 'static'),
    ('fonts/*', 'fonts'),
    ('portraits/*', 'portraits'),
    ('master.txt', '.'),
    ('wallet_config.json', '.'),
    ('blockchain.json', '.'),
    ('mempool.json', '.')
]

common_hiddenimports = [
    'flask',
    'sqlalchemy',
    'flask_sqlalchemy',
    'jinja2',
    'PIL',
    'cairosvg',
    'reportlab',
    'svglib',
    'requests',
    'sqlalchemy.ext',
    'sqlalchemy.dialects.sqlite',
    'threading',
    'datetime',
    'json',
    'os',
    'sys',
    'base64',
    'hashlib',
    'secrets',
    'time',
    'socket',
    'subprocess',
    'glob',
    're',
    'shutil',
    'argparse',
    'concurrent.futures',
    'io',
    'random'
]

# Analysis for each executable
a_main = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=common_datas,
    hiddenimports=common_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_app = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=common_datas,
    hiddenimports=common_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_wallet = Analysis(
    ['lunacoin_wallet.py'],
    pathex=[],
    binaries=[],
    datas=common_datas,
    hiddenimports=common_hiddenimports + ['tkinter', 'select'],  # Add GUI-specific imports
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_node = Analysis(
    ['lunacoin_node.py'],
    pathex=[],
    binaries=[],
    datas=common_datas,
    hiddenimports=common_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_miner_gui = Analysis(
    ['luna_miner_gui.py'],
    pathex=[],
    binaries=[],
    datas=common_datas,
    hiddenimports=common_hiddenimports + ['tkinter'],  # Add GUI-specific imports
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Create PYZ for each
pyz_main = PYZ(a_main.pure, a_main.zipped_data, cipher=block_cipher)
pyz_app = PYZ(a_app.pure, a_app.zipped_data, cipher=block_cipher)
pyz_wallet = PYZ(a_wallet.pure, a_wallet.zipped_data, cipher=block_cipher)
pyz_node = PYZ(a_node.pure, a_node.zipped_data, cipher=block_cipher)
pyz_miner_gui = PYZ(a_miner_gui.pure, a_miner_gui.zipped_data, cipher=block_cipher)

# Create executables with different icons and names
exe_main = EXE(
    pyz_main,
    a_main.scripts,
    a_main.binaries,
    a_main.zipfiles,
    a_main.datas,
    [],
    name='LingBanknotes',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Show console for main.py
    icon='bills_icon.ico',
)

exe_app = EXE(
    pyz_app,
    a_app.scripts,
    a_app.binaries,
    a_app.zipfiles,
    a_app.datas,
    [],
    name='LingWebServer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console for web server
    icon='web_icon.ico',
)

exe_wallet = EXE(
    pyz_wallet,
    a_wallet.scripts,
    a_wallet.binaries,
    a_wallet.zipfiles,
    a_wallet.datas,
    [],
    name='LunaWallet',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI application
    icon='wallet_icon.ico',
)

exe_node = EXE(
    pyz_node,
    a_node.scripts,
    a_node.binaries,
    a_node.zipfiles,
    a_node.datas,
    [],
    name='LunaNode',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Show console for node
    icon='node_icon.ico',
)

exe_miner_gui = EXE(
    pyz_miner_gui,
    a_miner_gui.scripts,
    a_miner_gui.binaries,
    a_miner_gui.zipfiles,
    a_miner_gui.datas,
    [],
    name='LunaMiner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI application
    icon='miner_icon.ico',
)

# Collect all executables
coll = COLLECT(
    exe_main,
    exe_app,
    exe_wallet,
    exe_node,
    exe_miner_gui,
    name='LunaCoin_Suite'
)