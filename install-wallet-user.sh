#!/bin/bash
# LinKoin Wallet Installer - User Mode
# Installs everything in the user's home directory

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Luna Wallet Installer (User Mode)${NC}"
echo "=========================================="

# Check if wallet files exist
if [ ! -f "luna_wallet.py" ]; then
    echo -e "${YELLOW}Warning: wallet.py not found in current directory.${NC}"
    echo "Please run this script from the directory containing your wallet files."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set user-specific installation paths
BIN_DIR="${HOME}/.local/bin"
APPLICATIONS_DIR="${HOME}/.local/share/applications"
ICONS_DIR="${HOME}/.local/share/icons"
DATA_DIR="${HOME}/.luna-wallet"

echo "Installation paths:"
echo "  Binaries: ${BIN_DIR}"
echo "  Data: ${DATA_DIR}"
echo "  Applications: ${APPLICATIONS_DIR}"

# Create directories if they don't exist
mkdir -p "${BIN_DIR}"
mkdir -p "${APPLICATIONS_DIR}"
mkdir -p "${ICONS_DIR}"
mkdir -p "${DATA_DIR}"

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
    echo -e "${YELLOW}Warning: ${BIN_DIR} is not in your PATH${NC}"
    echo "You may need to add this to your shell profile:"
    echo "  export PATH=\"\${HOME}/.local/bin:\$PATH\""
    echo "Add it to ~/.bashrc, ~/.zshrc, or your shell's configuration file."
    echo
fi

# Copy wrapper script
echo "Installing wrapper script..."
cp ./ "${BIN_DIR}/"
chmod +x "${BIN_DIR}/luna-wallet"

# Copy wallet files to data directory if they exist
if [ -f "luna_wallet.py" ]; then
    echo "Copying wallet files to data directory..."
    cp luna_wallet.py "${DATA_DIR}/"
    chmod +x "${DATA_DIR}/luna_wallet.py"
fi

if [ -f "wallet_config.json" ]; then
    cp wallet_config.json "${DATA_DIR}/"
fi

# Create desktop entry
echo "Creating desktop entry..."
cat > "${APPLICATIONS_DIR}/luna-wallet.desktop" << EOF
[Desktop Entry]
Name=Luna Coin Wallet
Comment=Luna Coin Cryptocurrency Wallet
Exec=${BIN_DIR}/luna-wallet interactive
Icon=network-wireless
Terminal=true
Type=Application
Categories=Finance;Network;
Keywords=crypto;wallet;blockchain;
EOF

# Create a simple icon (optional)
if command -v convert &> /dev/null; then
    echo "Creating application icon..."
    convert -size 64x64 xc:blue -pointsize 12 -fill white -gravity center -annotate +0+0 "LKC" "${ICONS_DIR}/linkoin-wallet.png" 2>/dev/null || true
fi

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "You can now run the wallet using:"
echo "  luna-wallet interactive   # Interactive mode"
echo "  luna-wallet server        # Server mode"
echo "  luna-wallet --help        # Show help"
echo ""
echo "The wallet data is stored in: ${DATA_DIR}/"
echo ""
echo -e "${YELLOW}If you can't run 'linkoin-wallet' directly, you may need to:${NC}"
echo "  source ~/.bashrc  # Or restart your terminal"
echo "  Or run: ${BIN_DIR}/luna-wallet"
