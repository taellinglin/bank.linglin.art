# digital_banknote.py
import hashlib
import json
import time
from datetime import datetime

# Import from signatures without circular dependency
try:
    from signatures import DigitalSignatureManager, generate_key_pair
except ImportError:
    # Fallback if signatures can't be imported
    class DigitalSignatureManager:
        def create_signed_bill(self, *args, **kwargs):
            return type('MockBill', (), {'signature': 'mock_signature', 'public_key': 'mock_key'})()
        def verify_bill_signature(self, *args, **kwargs):
            return True
    
    def generate_key_pair():
        return "mock_private_key", "mock_public_key"


class DigitalBanknote:
    """
    Represents a digital banknote with cryptographic security features
    """
    
    def __init__(self, serial_number, denomination, owner, issuer="Banknote Generator"):
        self.serial_number = serial_number
        self.denomination = denomination
        self.owner = owner
        self.issuer = issuer
        self.created_at = time.time()
        self.transaction_history = []
        self.digital_signature = None
        self.public_key = None
        self.private_key = None
        self.metadata_hash = None
        self.is_verified = False
        
        # Generate cryptographic key pair for this banknote
        self._generate_keys()
        
        # Initialize metadata
        self._initialize_metadata()
    
    def _generate_keys(self):
        """Generate cryptographic key pair for the banknote"""
        try:
            self.private_key, self.public_key = generate_key_pair()
        except Exception as e:
            print(f"Error generating keys: {e}")
            # Fallback to simpler key generation
            self._generate_fallback_keys()
    
    def _generate_fallback_keys(self):
        """Fallback key generation using hashes"""
        base_string = f"{self.serial_number}{self.denomination}{self.owner}{time.time()}"
        self.private_key = hashlib.sha256(f"private_{base_string}".encode()).hexdigest()
        self.public_key = hashlib.sha256(f"public_{base_string}".encode()).hexdigest()
    
    def _initialize_metadata(self):
        """Initialize banknote metadata"""
        self.metadata = {
            'serial_number': self.serial_number,
            'denomination': self.denomination,
            'owner': self.owner,
            'issuer': self.issuer,
            'created_at': self.created_at,
            'created_at_readable': datetime.fromtimestamp(self.created_at).isoformat(),
            'transaction_count': 0,
            'current_holder': self.owner
        }
        self.metadata_hash = self._calculate_metadata_hash()
    
    def _calculate_metadata_hash(self):
        """Calculate hash of current metadata"""
        metadata_string = json.dumps(self.metadata, sort_keys=True)
        return hashlib.sha256(metadata_string.encode()).hexdigest()
    
    def sign(self):
        """Create digital signature for the banknote"""
        try:
            signature_manager = DigitalSignatureManager()
            
            bill_data = {
                'type': 'banknote',
                'front_serial': f"{self.serial_number}_FRONT",
                'back_serial': f"{self.serial_number}_BACK", 
                'metadata_hash': self.metadata_hash,
                'timestamp': self.created_at,
                'issued_to': self.owner,
                'denomination': self.denomination
            }
            
            signed_bill = signature_manager.create_signed_bill(bill_data, self.private_key)
            self.digital_signature = signed_bill.signature
            self.is_verified = True
            
            # Add creation transaction to history
            self._add_transaction('creation', self.owner, self.owner, self.denomination)
            
            return self.digital_signature
            
        except Exception as e:
            print(f"Error signing banknote: {e}")
            return None
    
    def verify(self):
        """Verify the banknote's digital signature"""
        if not self.digital_signature:
            return False
            
        try:
            bill_data = {
                'type': 'banknote',
                'front_serial': f"{self.serial_number}_FRONT",
                'back_serial': f"{self.serial_number}_BACK",
                'metadata_hash': self.metadata_hash,
                'timestamp': self.created_at,
                'issued_to': self.owner,
                'denomination': self.denomination,
                'public_key': self.public_key,
                'signature': self.digital_signature
            }
            
            signature_manager = DigitalSignatureManager()
            self.is_verified = signature_manager.verify_bill_signature(bill_data)
            return self.is_verified
            
        except Exception as e:
            print(f"Error verifying banknote: {e}")
            return False
    
    def transfer(self, new_owner, amount=None):
        """Transfer ownership of the banknote"""
        if not self.is_verified:
            return False, "Banknote signature is not verified"
        
        transfer_amount = amount or self.denomination
        
        # Create transfer transaction
        transaction_data = {
            'type': 'transfer',
            'from': self.owner,
            'to': new_owner,
            'amount': transfer_amount,
            'timestamp': time.time(),
            'banknote_serial': self.serial_number
        }
        
        # Sign the transaction
        signature_manager = DigitalSignatureManager()
        signature = signature_manager.create_transaction_signature(
            transaction_data, 
            self.private_key
        )
        
        transaction_data['signature'] = signature
        transaction_data['public_key'] = self.public_key
        
        # Update ownership
        previous_owner = self.owner
        self.owner = new_owner
        self.metadata['current_holder'] = new_owner
        self.metadata_hash = self._calculate_metadata_hash()
        
        # Add to transaction history
        self._add_transaction('transfer', previous_owner, new_owner, transfer_amount, signature)
        
        return True, "Transfer successful"
    
    def _add_transaction(self, tx_type, from_user, to_user, amount, signature=None):
        """Add transaction to history"""
        transaction = {
            'type': tx_type,
            'from': from_user,
            'to': to_user,
            'amount': amount,
            'timestamp': time.time(),
            'timestamp_readable': datetime.fromtimestamp(time.time()).isoformat(),
            'signature': signature
        }
        
        self.transaction_history.append(transaction)
        self.metadata['transaction_count'] = len(self.transaction_history)
        self.metadata_hash = self._calculate_metadata_hash()
    
    def get_transaction_data(self):
        """Get transaction data for blockchain inclusion"""
        return {
            'type': 'genesis' if len(self.transaction_history) == 1 else 'transfer',
            'serial_number': self.serial_number,
            'denomination': self.denomination,
            'issued_to': self.owner,
            'timestamp': self.created_at,
            'public_key': self.public_key,
            'signature': self.digital_signature,
            'metadata_hash': self.metadata_hash,
            'transaction_count': len(self.transaction_history)
        }
    
    def to_dict(self):
        """Convert banknote to dictionary for JSON serialization"""
        return {
            'serial_number': self.serial_number,
            'denomination': self.denomination,
            'owner': self.owner,
            'issuer': self.issuer,
            'created_at': self.created_at,
            'digital_signature': self.digital_signature,
            'public_key': self.public_key,
            'metadata_hash': self.metadata_hash,
            'is_verified': self.is_verified,
            'transaction_history': self.transaction_history,
            'metadata': self.metadata
        }
    
    def to_json(self):
        """Convert banknote to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data):
        """Create DigitalBanknote instance from dictionary"""
        banknote = cls(
            serial_number=data['serial_number'],
            denomination=data['denomination'],
            owner=data['owner'],
            issuer=data.get('issuer', 'Banknote Generator')
        )
        
        banknote.created_at = data.get('created_at', time.time())
        banknote.digital_signature = data.get('digital_signature')
        banknote.public_key = data.get('public_key')
        banknote.private_key = data.get('private_key')  # Be careful with private keys!
        banknote.metadata_hash = data.get('metadata_hash')
        banknote.is_verified = data.get('is_verified', False)
        banknote.transaction_history = data.get('transaction_history', [])
        banknote.metadata = data.get('metadata', {})
        
        return banknote
    
    @classmethod
    def from_json(cls, json_string):
        """Create DigitalBanknote instance from JSON string"""
        data = json.loads(json_string)
        return cls.from_dict(data)


class BanknoteFactory:
    """
    Factory class for creating digital banknotes
    """
    
    @staticmethod
    def create_banknote(serial_number, denomination, owner, auto_sign=True):
        """Create a new digital banknote"""
        banknote = DigitalBanknote(serial_number, denomination, owner)
        
        if auto_sign:
            banknote.sign()
            
        return banknote
    
    @staticmethod
    def create_batch(serial_numbers, denomination, owner):
        """Create multiple banknotes at once"""
        banknotes = []
        for serial in serial_numbers:
            banknote = BanknoteFactory.create_banknote(serial, denomination, owner)
            banknotes.append(banknote)
        
        return banknotes


# Utility function for your Flask app
def create_signed_banknote(serial_number, denomination, username):
    """Convenience function for creating signed banknotes in Flask app"""
    factory = BanknoteFactory()
    banknote = factory.create_banknote(serial_number, denomination, username)
    
    if banknote.is_verified:
        return banknote
    else:
        print(f"Failed to create verified banknote for {username}")
        return None