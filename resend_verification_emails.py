#!/usr/bin/env python3
"""
Resend verification emails to all unverified users
This script will:
1. Find all users with email_verified=False
2. Generate new verification tokens for them
3. Send verification emails
4. Skip users who are already verified
"""

from app import app, db
from models import User
from email_service import send_verification_email
import sys

def resend_all_verification_emails():
    """Resend verification emails to all unverified users"""
    with app.app_context():
        print("[+] Starting verification email resend process...")
        print("=" * 60)
        
        # Get all users
        all_users = User.query.all()
        print(f"[+] Found {len(all_users)} total users")
        
        # Filter unverified users
        unverified_users = [u for u in all_users if not u.email_verified]
        verified_users = [u for u in all_users if u.email_verified]
        
        print(f"[+] {len(verified_users)} users already verified (will skip)")
        print(f"[+] {len(unverified_users)} users need verification")
        print("=" * 60)
        
        if len(unverified_users) == 0:
            print("[+] All users are already verified! ✅")
            return True
        
        # Ask for confirmation
        print("\n⚠️  This will send verification emails to the following users:")
        for i, user in enumerate(unverified_users, 1):
            print(f"  {i}. {user.username} <{user.email}>")
        
        response = input("\nContinue? [y/N]: ")
        if response.lower() != 'y':
            print("[!] Cancelled by user")
            return False
        
        print("\n[+] Starting email sending process...")
        print("=" * 60)
        
        success_count = 0
        fail_count = 0
        
        app_url = "http://localhost:5000"  # Change this for production
        
        for user in unverified_users:
            try:
                # Generate new verification token
                verification_token = user.generate_verification_token()
                db.session.commit()
                
                # Send email
                send_verification_email(
                    user.email,
                    user.username,
                    verification_token,
                    app_url
                )
                
                print(f"  ✅ Sent to {user.username} <{user.email}>")
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ Failed for {user.username} <{user.email}>: {e}")
                fail_count += 1
        
        print("=" * 60)
        print(f"\n[+] Results:")
        print(f"  ✅ Success: {success_count}")
        print(f"  ❌ Failed: {fail_count}")
        print(f"  ⏭️  Skipped (already verified): {len(verified_users)}")
        
        if success_count > 0:
            print(f"\n[+] {success_count} verification emails sent successfully!")
            print("[+] Users have 24 hours to verify their email addresses")
        
        return True

def test_single_user(username):
    """Test by sending verification email to a specific user"""
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        
        if not user:
            print(f"[!] User '{username}' not found")
            return False
        
        print(f"[+] Found user: {user.username} <{user.email}>")
        print(f"[+] Email verified: {user.email_verified}")
        
        if user.email_verified:
            response = input("\n⚠️  User is already verified. Send email anyway? [y/N]: ")
            if response.lower() != 'y':
                print("[!] Cancelled")
                return False
        
        try:
            # Generate new verification token
            verification_token = user.generate_verification_token()
            db.session.commit()
            
            app_url = "http://localhost:5000"
            send_verification_email(
                user.email,
                user.username,
                verification_token,
                app_url
            )
            
            print(f"[+] ✅ Verification email sent to {user.email}")
            print(f"[+] Token: {verification_token[:20]}...")
            print(f"[+] Verification link: {app_url}/verify-email/{verification_token}")
            
            return True
            
        except Exception as e:
            print(f"[!] Failed to send email: {e}")
            import traceback
            traceback.print_exc()
            return False

def list_users_status():
    """List all users and their verification status"""
    with app.app_context():
        users = User.query.all()
        
        print("\n" + "=" * 80)
        print(f"{'Username':<20} {'Email':<30} {'Verified':<10} {'Admin':<10}")
        print("=" * 80)
        
        for user in users:
            verified_status = "✅ Yes" if user.email_verified else "❌ No"
            admin_status = "🔑 Admin" if user.is_admin else ""
            print(f"{user.username:<20} {user.email:<30} {verified_status:<10} {admin_status:<10}")
        
        print("=" * 80)
        print(f"\nTotal users: {len(users)}")
        print(f"Verified: {sum(1 for u in users if u.email_verified)}")
        print(f"Unverified: {sum(1 for u in users if not u.email_verified)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage email verification for users")
    parser.add_argument("--all", action="store_true", help="Send verification emails to all unverified users")
    parser.add_argument("--user", type=str, help="Send verification email to a specific user")
    parser.add_argument("--list", action="store_true", help="List all users and their verification status")
    
    args = parser.parse_args()
    
    if args.list:
        list_users_status()
    elif args.user:
        success = test_single_user(args.user)
        sys.exit(0 if success else 1)
    elif args.all:
        success = resend_all_verification_emails()
        sys.exit(0 if success else 1)
    else:
        # Default: show status
        list_users_status()
        print("\nUsage:")
        print("  --list        List all users")
        print("  --user NAME   Send to specific user")
        print("  --all         Send to all unverified users")
