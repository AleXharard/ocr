# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — Key Auto, un singur EXE, fără consolă, admin."""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

spec_dir = os.path.dirname(os.path.abspath(SPEC))

block_cipher = None

datas = [
    (os.path.join(spec_dir, "glyph_bank.npz"), "."),
    (os.path.join(spec_dir, "poze mina", "piatra.png"), os.path.join("poze mina")),
]
datas += collect_data_files("customtkinter")
datas += collect_data_files("rapidocr_onnxruntime")
datas += collect_data_files("onnxruntime")

binaries = collect_dynamic_libs("cv2")

hiddenimports = [
    "PIL._tkinter_finder",
    "rapidocr_onnxruntime",
    "onnxruntime",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "keyboard",
    "pydirectinput",
    "mss",
    "dxcam",
    "busteni",
    "mina",
    "game_input",
    "logger",
    "vision",
    "ui_widgets",
    "app_paths",
    "comtypes",
    "comtypes.client",
]

a = Analysis(
    [os.path.join(spec_dir, "main.py")],
    pathex=[spec_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="KeyAuto",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
    icon=None,
)
