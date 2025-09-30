# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Collect SSL certificates and required modules for wallet
hiddenimports = [
    'urllib3',
    'certifi',
    'chardet',
    'idna',
    'requests',
    'http',
    'http.client',
    'ssl',
    'socket',
    'json',
    'hashlib',
    'secrets',
    'base64',
    'binascii',
    'threading',
    'netifaces'
]

# Collect certificate files
certifi_data = collect_data_files('certifi')

a = Analysis(
    ['luna_wallet.py'],
    pathex=[],
    binaries=[
        # Add SSL and crypto libraries
        (r"C:\Users\User\miniconda3\DLLs\_ssl.pyd", "."),
        (r"C:\Users\User\miniconda3\DLLs\_hashlib.pyd", "."),
        (r"C:\Users\User\miniconda3\Library\bin\libssl-1_1-x64.dll", "."),
        (r"C:\Users\User\miniconda3\Library\bin\libcrypto-1_1-x64.dll", "."),
        (r"C:\Users\User\miniconda3\DLLs\_ctypes.pyd", "."),
        (r"C:\Users\User\miniconda3\DLLs\_bz2.pyd", "."),
    ],
    datas=certifi_data + [
        # Add certificate bundle
        (r"C:\Users\User\miniconda3\Lib\site-packages\certifi\cacert.pem", "certifi"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_ssl.py'],  # 👈 Add SSL runtime hook
    excludes=[
        'tkinter',  # Reduce size by excluding GUI modules
        'unittest',
        'email',
        'pydoc',
        'pystray',  # Remove if you're not using system tray
        'PIL',      # Remove if you're not using images
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LunaWallet',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for wallet (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='wallet_icon.ico',
)