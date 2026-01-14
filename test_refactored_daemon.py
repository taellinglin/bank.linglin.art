# test_refactored_daemon.py
"""Test script to verify the refactored blockchain daemon works correctly"""

print("Testing refactored blockchain daemon...")
print("=" * 60)

# Test 1: Import the daemon
print("\n1. Testing import...")
try:
    from blockchain_daemon import BlockchainDaemon
    print("   ✅ Import successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    exit(1)

# Test 2: Create daemon instance
print("\n2. Testing daemon instantiation...")
try:
    daemon = BlockchainDaemon(
        blockchain_file="test_blockchain.json",
        mempool_file="test_mempool.json"
    )
    print("   ✅ Daemon created successfully")
except Exception as e:
    print(f"   ❌ Daemon creation failed: {e}")
    exit(1)

# Test 3: Check key methods exist
print("\n3. Testing key methods exist...")
required_methods = [
    'add_transaction',
    'get_transaction',
    'validate_transaction_structure',
    'validate_block',
    'save_blockchain',
    'save_mempool',
    'sync_with_network',
    'get_block',
    'get_mempool_status',
    'get_blockchain_status',
    'calculate_block_hash',
    'get_previous_hash',
    'submit_block',
    'add_genesis_transaction'
]

all_methods_present = True
for method_name in required_methods:
    if hasattr(daemon, method_name):
        print(f"   ✅ {method_name}")
    else:
        print(f"   ❌ {method_name} - MISSING!")
        all_methods_present = False

if not all_methods_present:
    print("\n   ⚠️ Some methods are missing!")
    exit(1)

# Test 4: Test basic operations
print("\n4. Testing basic operations...")

try:
    # Test get_previous_hash
    prev_hash = daemon.get_previous_hash()
    print(f"   ✅ get_previous_hash() works: {prev_hash[:16]}...")
    
    # Test get_mempool_status
    mempool_status = daemon.get_mempool_status()
    print(f"   ✅ get_mempool_status() works: {mempool_status['total']} transactions")
    
    # Test get_blockchain_status
    blockchain_status = daemon.get_blockchain_status()
    print(f"   ✅ get_blockchain_status() works: {blockchain_status['blocks']} blocks")
    
    # Test calculate_block_hash
    test_hash = daemon.calculate_block_hash(0, "0"*64, 1234567890.0, [], 0)
    print(f"   ✅ calculate_block_hash() works: {test_hash[:16]}...")
    
except Exception as e:
    print(f"   ❌ Basic operations failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Verify module structure
print("\n5. Verifying module structure...")
try:
    from blockchain_daemon_modules import validators, persistence, network, blocks, transactions
    print("   ✅ validators module")
    print("   ✅ persistence module")
    print("   ✅ network module")
    print("   ✅ blocks module")
    print("   ✅ transactions module")
except Exception as e:
    print(f"   ❌ Module structure verification failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe refactored blockchain daemon is working correctly.")
print("Your app.py can continue using 'from blockchain_daemon import BlockchainDaemon'")
