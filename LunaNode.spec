# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os
import glob

def find_venv_dlls(venv_path='win-venv'):
    """Auto-detect DLLs in the venv"""
    dlls_path = os.path.join(venv_path, 'DLLs')
    binaries = []
    datas = []
    
    if os.path.exists(dlls_path):
        # Look for SSL-related files
        ssl_patterns = [
            '_ssl.pyd',
            '_hashlib.pyd', 
            '_socket.pyd',
            'libssl*',
            'libcrypto*',
        ]
        
        for pattern in ssl_patterns:
            for dll_path in glob.glob(os.path.join(dlls_path, pattern)):
                if os.path.isfile(dll_path):
                    if dll_path.endswith('.pyd'):
                        binaries.append((dll_path, '.'))
                    else:
                        datas.append((dll_path, '.'))
                    print(f"📦 Found: {os.path.basename(dll_path)}")
    
    return binaries, datas

# Auto-detect DLLs from win-venv
venv_binaries, venv_datas = find_venv_dlls()

a = Analysis(
    ['luna_node.py'],
    pathex=[],
    binaries=venv_binaries,
    datas=venv_datas,
    hiddenimports=[
        '_ssl',
        '_hashlib', 
        '_socket',
        'ssl',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='LunaNode',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='node_icon.ico',
)