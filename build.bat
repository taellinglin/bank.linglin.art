@echo off
echo Building LunaNode and LunaWallet...
:: Clean previous builds
if exist "dist\" rmdir /s /q "dist"
if exist "build\" rmdir /s /q "build"

:: Build LunaNode
echo Building LunaNode...
pyinstaller LunaNode.spec
if errorlevel 1 (
    echo Error building LunaNode
    pause
    exit /b 1
)

:: Build LunaWallet
echo Building LunaWallet...
pyinstaller LunaWallet.spec
if errorlevel 1 (
    echo Error building LunaWallet
    pause
    exit /b 1
)

:: Create installer
echo Creating installer...
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" Luna.iss

if errorlevel 1 (
    echo Error creating installer
    pause
    exit /b 1
)

echo Build complete! Installer: Output\Luna_Setup.exe
pause