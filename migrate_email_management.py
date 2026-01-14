#!/usr/bin/env python
"""
Database migration script to add email management features:
- pending_email field to users table
- email_history table for tracking email changes
"""

import os
import sys

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import User, EmailHistory
from sqlalchemy import text

def migrate_database():
    """Apply database migrations for email management"""
    with app.app_context():
        print("[MIGRATION] Starting email management migration...")
        
        try:
            # Check if pending_email column exists
            inspector = db.inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('users')]
            
            # Add pending_email column if it doesn't exist
            if 'pending_email' not in columns:
                print("[MIGRATION] Adding pending_email column to users table...")
                with db.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE users ADD COLUMN pending_email VARCHAR(120)"))
                    conn.commit()
                print("[MIGRATION] ✓ Added pending_email column")
            else:
                print("[MIGRATION] pending_email column already exists")
            
            # Create email_history table if it doesn't exist
            print("[MIGRATION] Creating email_history table...")
            db.create_all()
            print("[MIGRATION] ✓ Created email_history table")
            
            print("[MIGRATION] ✓ Migration completed successfully!")
            return True
            
        except Exception as e:
            print(f"[MIGRATION] ✗ Error during migration: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("=" * 60)
    print("Email Management Migration Script")
    print("=" * 60)
    
    success = migrate_database()
    
    if success:
        print("\n✓ Database migration completed successfully!")
        print("\nNew features added:")
        print("  • pending_email field for email change verification")
        print("  • email_history table for tracking email changes")
        print("\nYou can now:")
        print("  • Change your email from your profile page")
        print("  • View your email change history")
        print("  • Resend verification emails")
    else:
        print("\n✗ Migration failed. Please check the error messages above.")
        sys.exit(1)
