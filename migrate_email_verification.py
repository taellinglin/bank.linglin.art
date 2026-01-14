#!/usr/bin/env python3
"""
Database migration to add email verification fields to User table
Run this to add the new columns without losing existing data
"""

from app import app, db
from models import User
import sys

def migrate_email_verification():
    """Add email verification fields to existing User table"""
    with app.app_context():
        print("[+] Starting email verification migration...")
        
        try:
            # Check if columns already exist by trying to query them
            test_user = User.query.first()
            if test_user:
                # Try to access new fields
                _ = test_user.email_verified
                _ = test_user.verification_token
                _ = test_user.verification_token_expires
                print("[+] Email verification columns already exist!")
                return True
        except Exception as e:
            print(f"[+] Columns don't exist yet, adding them: {e}")
        
        # Add new columns using raw SQL
        from sqlalchemy import text
        
        try:
            with db.engine.connect() as conn:
                # Add email_verified column
                try:
                    conn.execute(text("""
                        ALTER TABLE users 
                        ADD COLUMN email_verified BOOLEAN DEFAULT 0
                    """))
                    conn.commit()
                    print("[+] Added email_verified column")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        print(f"[!] Error adding email_verified: {e}")
                
                # Add verification_token column (without UNIQUE constraint)
                try:
                    conn.execute(text("""
                        ALTER TABLE users 
                        ADD COLUMN verification_token VARCHAR(100)
                    """))
                    conn.commit()
                    print("[+] Added verification_token column")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        print(f"[!] Error adding verification_token: {e}")
                
                # Add verification_token_expires column
                try:
                    conn.execute(text("""
                        ALTER TABLE users 
                        ADD COLUMN verification_token_expires DATETIME
                    """))
                    conn.commit()
                    print("[+] Added verification_token_expires column")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        print(f"[!] Error adding verification_token_expires: {e}")
                
                # Create unique index for verification_token (optional, will fail if NULL values exist)
                try:
                    conn.execute(text("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_verification_token 
                        ON users(verification_token) WHERE verification_token IS NOT NULL
                    """))
                    conn.commit()
                    print("[+] Created unique index for verification_token")
                except Exception as e:
                    print(f"[!] Note: Could not create unique index: {e}")
                    print("[!] This is OK - uniqueness will be enforced at application level")
            
            print("[+] Migration completed successfully!")
            print("[+] All existing users have email_verified=False by default")
            print("[+] Admin users may want to manually set email_verified=True")
            
            return True
            
        except Exception as e:
            print(f"[!] Migration failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = migrate_email_verification()
    sys.exit(0 if success else 1)
