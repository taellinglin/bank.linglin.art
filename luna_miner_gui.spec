# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['luna_miner_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('fonts/*', 'fonts')],
    hiddenimports=[
        'kivy',
        'kivy.core.window.window_sdl2',
        'kivy.core.text._text_sdl2',
        'kivy.core.window.window_egl_rpi',
        'ctypes',
        '_ctypes',
    ],
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
    [],
    exclude_binaries=True,
    name='LunaMiner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LunaMiner',
)