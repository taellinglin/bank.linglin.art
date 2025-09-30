#!/bin/bash
# LinKoin Wallet Installer

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}LinKoin Wallet Installer${NC}"
echo "============================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this script as root."
    exit 1
fi

# Check if wallet files exist
if [ ! -f "luna_wallet.py" ] || [ ! -f "wallet_config.json" ]; then
    echo "Error: Required wallet files not found in current directory."
    echo "Please run this script from the directory containing wallet.py and wallet_config.json"
    exit 1
fi

# Copy wrapper script
echo "Installing wrapper script..."
sudo cp linkoin-wallet /usr/local/bin/linkoin-wallet
sudo chmod +x /usr/local/bin/linkoin-wallet

# Create desktop entry (optional)
if [ -d "${HOME}/.local/share/applications" ]; then
    echo "Creating desktop entry..."
    cat > /tmp/linkoin-wallet.desktop << EOF
[Desktop Entry]
Name=LinKoin Wallet
Comment=LinKoin Cryptocurrency Wallet
Exec=linkoin-wallet interactive
Icon=network-wireless
Terminal=true
Type=Application
Categories=Finance;Network;
Keywords=crypto;wallet;blockchain;
EOF

    mv /tmp/linkoin-wallet.desktop ${HOME}/.local/share/applications/
fi

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "You can now run the wallet using:"
echo "  linkoin-wallet interactive   # Interactive mode"
echo "  linkoin-wallet server        # Server mode"
echo "  linkoin-wallet --help        # Show help"
echo ""
echo "The wallet data will be stored in: ${HOME}/.linkoin-wallet/"