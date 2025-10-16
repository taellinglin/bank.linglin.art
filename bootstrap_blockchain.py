import json
import hashlib

genesis_data = {
    "type": "genesis",
    "message": "Welcome to LingBanknotes Blockchain - The First Decentralized Digital Currency Platform for Personalized Banknotes",
    "timestamp": 1735689600,
    "creator": "LingBanknotes System",
    "version": "1.0.0",
    "hash": ""  # This will be calculated
}

# Calculate the hash (same method as in your BlockchainDaemon)
genesis_string = json.dumps(genesis_data, sort_keys=True)
calculated_hash = hashlib.sha256(genesis_string.encode()).hexdigest()

# Update with the correct hash
genesis_data["hash"] = calculated_hash

print("Genesis transaction with correct hash:")
print(json.dumps(genesis_data, indent=2))