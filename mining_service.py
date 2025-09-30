# mining_service.py
import json
import sys
import os
import time

# Add the path to your blockchain_daemon
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blockchain_daemon import BlockchainDaemon

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Missing miner_address"}))
        sys.exit(1)
    
    miner_address = sys.argv[1]
    difficulty = int(sys.argv[2]) if len(sys.argv) >= 3 else 4
    
    try:
        daemon = BlockchainDaemon()
        daemon.load_data()
        
        # Mine using the synchronous version
        result = daemon._mine_pending_transactions_sync(miner_address, difficulty)
        
        if result:
            print(json.dumps({
                "success": True, 
                "block": result,
                "message": f"Successfully mined block #{result.get('index')}"
            }))
        else:
            print(json.dumps({
                "success": False, 
                "error": "No block mined - no valid transactions or mining failed"
            }))
            
    except Exception as e:
        print(json.dumps({
            "success": False, 
            "error": f"Mining error: {str(e)}"
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()