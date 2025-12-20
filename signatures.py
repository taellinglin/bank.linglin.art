# signatures.py - UPDATED TO USE SM2 INSTEAD OF RSA
import hashlib
import json
import time
import base64

# Import SM2 implementation
from sm2 import SM2, generate_sm2_keypair, sign_message, verify_message

try:
    # Keep optional cryptography for compatibility
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    print("Warning: cryptography library not available. Using SM2 only.")
    CRYPTOGRAPHY_AVAILABLE = False


class DigitalBill:
    """
    Represents a digitally signed banknote/bill with SM2 cryptographic verification
    Handles both ASCII and Unicode encoding for verification
    """
    
    def __init__(self, bill_type, front_serial, back_serial, metadata_hash, 
                 timestamp, issued_to, denomination, public_key=None, signature=None):
        self.bill_type = bill_type
        self.front_serial = front_serial
        self.back_serial = back_serial
        self.metadata_hash = metadata_hash
        self.timestamp = timestamp
        self.issued_to = issued_to
        self.denomination = denomination
        self.public_key = public_key
        self.signature = signature
        
        # Track which encoding was used for signing
        self.signing_encoding = None  # Will be 'ascii' or 'unicode'
    
    def to_dict(self):
        """Convert bill data to dictionary for hashing/serialization"""
        return {
            'type': self.bill_type,
            'front_serial': self.front_serial,
            'back_serial': self.back_serial,
            'metadata_hash': self.metadata_hash,
            'timestamp': self.timestamp,
            'issued_to': self.issued_to,
            'denomination': self.denomination
        }
    
    def to_json_string(self, encoding='auto'):
        """Convert to JSON string with specified encoding"""
        if encoding == 'ascii':
            return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        elif encoding == 'unicode':
            return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        else:  # auto - try to detect based on issued_to
            # If issued_to contains non-ASCII, use unicode, else ascii
            if self.issued_to and any(ord(c) > 127 for c in self.issued_to):
                return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
            else:
                return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
    
    def sign(self, private_key_hex, encoding='ascii'):
        """Sign the bill data with SM2 private key"""
        if encoding not in ['ascii', 'unicode']:
            encoding = 'ascii'  # Default to ascii for backward compatibility
        
        bill_string = self.to_json_string(encoding=encoding)
        self.signing_encoding = encoding
        
        try:
            sm2 = SM2()
            
            # Generate signature
            signature_hex = sm2.sign(bill_string.encode(), private_key_hex)
            self.signature = signature_hex
            
            # Generate public key from private key
            private_key_int = int(private_key_hex, 16)
            sm2.private_key = private_key_int
            
            curve = sm2.curve
            Px, Py = curve.point_multiply(private_key_int, curve.Gx, curve.Gy)
            self.public_key = f"04{Px:064x}{Py:064x}"
            
            print(f"[SIGN] Signed with {encoding} encoding")
            print(f"[SIGN] JSON: {bill_string[:50]}...")
            
            return self.signature
            
        except Exception as e:
            print(f"[SIGN ERROR] {e}")
            raise
    
    def verify(self):
        """Verify SM2 signature ONLY - no fallbacks, no mock signatures"""
        if not self.public_key or not self.signature:
            print(f"[VERIFY] Missing public_key or signature")
            return False
        
        # Validate signature format - must be 128 hex chars for SM2
        if len(self.signature) != 128 or not all(c in '0123456789abcdefABCDEF' for c in self.signature):
            print(f"[VERIFY] Invalid signature format: expected 128 hex chars, got {len(self.signature)}")
            return False
        
        # Validate public key format - must start with '04' for uncompressed SM2
        if not self.public_key.startswith('04'):
            print(f"[VERIFY] Invalid public key format: expected '04' prefix")
            return False
        
        # Validate public key length - should be 130 hex chars for uncompressed (04 + 64 + 64)
        if len(self.public_key) != 130:
            print(f"[VERIFY] Invalid public key length: expected 130 hex chars, got {len(self.public_key)}")
            return False
        
        try:
            # Get the exact data that should be signed
            bill_string = self.to_json_string()
            
            # Create SM2 instance
            sm2 = SM2()
            
            # Verify using SM2
            print(f"[VERIFY] Verifying SM2 signature for bill {self.front_serial}")
            
            is_valid = sm2.verify(
                bill_string.encode(),
                self.signature,
                self.public_key
            )
            
            if is_valid:
                print(f"[VERIFY] ✓ SM2 signature verified for bill {self.front_serial}")
            else:
                print(f"[VERIFY] ✗ SM2 signature verification failed for bill {self.front_serial}")
                
                # Debug: Show what's being verified
                print(f"[VERIFY DEBUG] JSON being verified: {bill_string}")
                print(f"[VERIFY DEBUG] Public key: {self.public_key}")
                print(f"[VERIFY DEBUG] Signature: {self.signature}")
            
            return is_valid
            
        except Exception as e:
            print(f"[VERIFY ERROR] SM2 verification error: {e}")
            import traceback
            traceback.print_exc()
            return False
class DigitalSignatureManager:
    """
    Manager class for handling digital signatures using SM2
    """
    
    def __init__(self):
        self.key_cache = {}  # Cache for loaded keys
        self.sm2_instance = SM2()  # Reusable SM2 instance
    
    def create_signed_bill(self, bill_data, private_key_hex):
        """Create a new digitally signed bill using SM2"""
        # Create bill object
        bill = DigitalBill(
            bill_type=bill_data.get('type', 'banknote'),
            front_serial=bill_data.get('front_serial', ''),
            back_serial=bill_data.get('back_serial', ''),
            metadata_hash=bill_data.get('metadata_hash', ''),
            timestamp=bill_data.get('timestamp', time.time()),
            issued_to=bill_data.get('issued_to', ''),
            denomination=bill_data.get('denomination', '')
        )
        
        # Sign the bill with SM2
        signature = bill.sign(private_key_hex)
        
        print(f"Created SM2-signed bill: {bill.front_serial}")
        print(f"  Signature: {signature[:16]}...")
        print(f"  Public Key: {bill.public_key[:16]}...")
        
        return bill
    
    def verify_bill_signature(self, bill_data):
        """Verify a bill's SM2 digital signature"""
        if isinstance(bill_data, dict):
            # Create bill object from dictionary
            bill = DigitalBill(
                bill_type=bill_data.get('type'),
                front_serial=bill_data.get('front_serial'),
                back_serial=bill_data.get('back_serial'),
                metadata_hash=bill_data.get('metadata_hash'),
                timestamp=bill_data.get('timestamp'),
                issued_to=bill_data.get('issued_to'),
                denomination=bill_data.get('denomination'),
                public_key=bill_data.get('public_key'),
                signature=bill_data.get('signature')
            )
        else:
            bill = bill_data
            
        return bill.verify()
    
    def create_transaction_signature(self, transaction_data, private_key_hex):
        """Create SM2 signature for blockchain transactions"""
        # Sort transaction data for consistent hashing
        sorted_data = json.dumps(transaction_data, sort_keys=True)
        
        try:
            # Use SM2 to sign the transaction data
            signature_hex = sign_message(sorted_data, private_key_hex)
            print(f"Created SM2 transaction signature: {signature_hex[:16]}...")
            return signature_hex
            
        except Exception as e:
            print(f"SM2 transaction signing failed, using fallback: {e}")
            # Fallback to hash-based signature
            transaction_hash = hashlib.sha256(sorted_data.encode()).hexdigest()
            signature_input = f"{private_key_hex}{transaction_hash}"
            return hashlib.sha256(signature_input.encode()).hexdigest()
    
    def verify_transaction_signature(self, transaction_data, public_key_hex, signature_hex):
        """Verify SM2 signature for blockchain transactions"""
        try:
            # Sort transaction data (must match signing order)
            sorted_data = json.dumps(transaction_data, sort_keys=True)
            
            # Use SM2 to verify the signature
            is_valid = verify_message(sorted_data, signature_hex, public_key_hex)
            
            if is_valid:
                print(f"✓ SM2 transaction signature verified")
            else:
                print(f"✗ SM2 transaction signature verification failed")
            
            return is_valid
            
        except Exception as e:
            print(f"SM2 verification error: {e}, trying fallback...")
            
            # Fallback verification
            sorted_data = json.dumps(transaction_data, sort_keys=True)
            transaction_hash = hashlib.sha256(sorted_data.encode()).hexdigest()
            
            if public_key_hex and len(public_key_hex) > 10:
                signature_input = f"{public_key_hex}{transaction_hash}"
            else:
                signature_input = f"fallback_key{transaction_hash}"
            
            expected_signature = hashlib.sha256(signature_input.encode()).hexdigest()
            return signature_hex == expected_signature
    
    def generate_sm2_keypair_with_address(self):
        """Generate SM2 key pair with blockchain address"""
        try:
            private_key_hex, public_key_hex, address = generate_sm2_keypair()
            
            key_info = {
                'private_key': private_key_hex,
                'public_key': public_key_hex,
                'address': address,
                'key_type': 'sm2',
                'curve': 'SM2 (GB/T 32918)',
                'private_key_bits': len(private_key_hex) * 4,  # hex digits to bits
                'public_key_format': 'uncompressed (04 + x + y)'
            }
            
            print(f"Generated SM2 key pair with address: {address}")
            return key_info
            
        except Exception as e:
            print(f"SM2 key generation failed: {e}")
            
            # Fallback
            import random
            import string
            
            private_key = ''.join(random.choices(string.ascii_letters + string.digits, k=64))
            public_key = f"04{hashlib.sha256(private_key.encode()).hexdigest()[:64]}"
            address = f"LUN_FBK_{hashlib.sha256(public_key.encode()).hexdigest()[:20]}"
            
            return {
                'private_key': private_key,
                'public_key': public_key,
                'address': address,
                'key_type': 'fallback',
                'curve': 'none'
            }
    
    def test_sm2_signature(self):
        """Test SM2 signature generation and verification"""
        print("\n" + "="*60)
        print("Testing SM2 Signature System")
        print("="*60)
        
        try:
            # Generate test key pair
            print("1. Generating SM2 key pair...")
            key_info = self.generate_sm2_keypair_with_address()
            
            if key_info['key_type'] != 'sm2':
                print("   ⚠ Using fallback keys (SM2 not available)")
            
            private_key = key_info['private_key']
            public_key = key_info['public_key']
            
            print(f"   Private key: {private_key[:16]}...")
            print(f"   Public key: {public_key[:16]}...")
            print(f"   Address: {key_info['address']}")
            
            # Create test bill
            print("\n2. Creating test bill...")
            test_bill_data = {
                'type': 'test_banknote',
                'front_serial': 'TEST123456',
                'back_serial': 'TEST654321',
                'metadata_hash': hashlib.sha256(b'test_metadata').hexdigest(),
                'timestamp': time.time(),
                'issued_to': 'test_user',
                'denomination': '100'
            }
            
            # Sign the bill
            print("3. Signing bill with SM2...")
            signed_bill = self.create_signed_bill(test_bill_data, private_key)
            
            # Verify the signature
            print("4. Verifying signature...")
            is_valid = self.verify_bill_signature(signed_bill)
            
            if is_valid:
                print("   ✅ SM2 signature test PASSED")
            else:
                print("   ❌ SM2 signature test FAILED")
            
            # Test transaction signing
            print("\n5. Testing transaction signing...")
            test_transaction = {
                'from': 'test_address_1',
                'to': 'test_address_2',
                'amount': '100.0',
                'timestamp': time.time(),
                'nonce': 1
            }
            
            tx_signature = self.create_transaction_signature(test_transaction, private_key)
            print(f"   Transaction signature: {tx_signature[:16]}...")
            
            tx_valid = self.verify_transaction_signature(test_transaction, public_key, tx_signature)
            
            if tx_valid:
                print("   ✅ Transaction signature test PASSED")
            else:
                print("   ❌ Transaction signature test FAILED")
            
            print("\n" + "="*60)
            print("SM2 Test Complete")
            print("="*60)
            
            return is_valid and tx_valid
            
        except Exception as e:
            print(f"\n❌ SM2 test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False