# -*- mode: python ; coding: utf-8 -*-
import os

a = Analysis(
    ['luna_node.py'],
    pathex=[],
    binaries=[
        (r"C:\Users\User\miniconda3\DLLs\_ctypes.pyd", "."),
        (r"C:\Users\User\miniconda3\DLLs\_bz2.pyd", "."),
        (r"C:\Users\User\miniconda3\Library\bin\ffi.dll", "."),
        (r"C:\Users\User\miniconda3\Library\bin\LIBBZ2.dll", "."),
    ],
    datas=[],
    hiddenimports=['_ctypes'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

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
    upx=False,  # 👈 leave UPX off
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
