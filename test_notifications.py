#!/usr/bin/env python3
"""
Test notification system
Manually trigger notifications for testing
"""

from app import app, db
from models import User, Settings
from notification_scheduler import NotificationScheduler
from email_service import send_generation_ready_notification, send_banknote_generation_notification
from datetime import datetime, timedelta
import sys

def test_cooldown_notification(username):
    """Test sending cooldown expired notification"""
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        
        if not user:
            print(f"[!] User '{username}' not found")
            return False
        
        print(f"[+] User: {user.username}")
        print(f"[+] Email: {user.email}")
        print(f"[+] Email verified: {user.email_verified}")
        print(f"[+] Last generation: {user.last_generation}")
        print(f"[+] Can generate: {user.can_generate_money()}")
        
        if not user.email_verified:
            print("[!] Email not verified. Send anyway? [y/N]")
            if input().lower() != 'y':
                return False
        
        try:
            send_generation_ready_notification(
                user.email,
                user.username,
                days_until_next=0
            )
            print(f"[+] ✅ Cooldown notification sent to {user.email}")
            return True
        except Exception as e:
            print(f"[!] Failed to send notification: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_generation_complete_notification(username):
    """Test sending generation complete notification"""
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        
        if not user:
            print(f"[!] User '{username}' not found")
            return False
        
        print(f"[+] User: {user.username}")
        print(f"[+] Email: {user.email}")
        print(f"[+] Email verified: {user.email_verified}")
        
        if not user.email_verified:
            print("[!] Email not verified. Send anyway? [y/N]")
            if input().lower() != 'y':
                return False
        
        # Use sample data
        try:
            send_banknote_generation_notification(
                user.email,
                user.username,
                denomination="100, 1000, 10000",
                count=27,  # 3 denominations × 9 banknotes
                serial_numbers=["SNB-ABC123", "SNB-DEF456", "SNB-GHI789"]
            )
            print(f"[+] ✅ Generation complete notification sent to {user.email}")
            return True
        except Exception as e:
            print(f"[!] Failed to send notification: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_scheduler_cycle():
    """Run one cycle of the scheduler manually"""
    print("[+] Running scheduler cycle...")
    scheduler = NotificationScheduler(check_interval=60)
    scheduler.force_check()
    print("[+] Scheduler cycle complete")

def show_eligible_users():
    """Show users eligible for cooldown notification"""
    with app.app_context():
        settings = Settings.query.first()
        cooldown_days = settings.cooldown_days if settings else 7
        
        now = datetime.utcnow()
        cutoff_time = now - timedelta(days=cooldown_days)
        
        users = User.query.filter(
            User.last_generation.isnot(None),
            User.last_generation <= cutoff_time,
            User.email_verified == True
        ).all()
        
        print("\n" + "=" * 80)
        print(f"Users eligible for cooldown notification (cooldown: {cooldown_days} days)")
        print("=" * 80)
        
        if not users:
            print("No users eligible")
            return
        
        for user in users:
            days_ago = (now - user.last_generation).days
            print(f"{user.username:<20} {user.email:<30} Last gen: {days_ago} days ago")
        
        print("=" * 80)

def set_user_last_generation(username, days_ago):
    """Set a user's last_generation to test cooldown"""
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        
        if not user:
            print(f"[!] User '{username}' not found")
            return False
        
        new_time = datetime.utcnow() - timedelta(days=days_ago)
        user.last_generation = new_time
        db.session.commit()
        
        print(f"[+] Set {user.username}'s last_generation to {days_ago} days ago")
        print(f"[+] Date: {new_time}")
        print(f"[+] Can generate now: {user.can_generate_money()}")
        
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test notification system")
    parser.add_argument("--cooldown", type=str, metavar="USERNAME", help="Test cooldown notification for user")
    parser.add_argument("--complete", type=str, metavar="USERNAME", help="Test generation complete notification for user")
    parser.add_argument("--scheduler", action="store_true", help="Run scheduler cycle manually")
    parser.add_argument("--eligible", action="store_true", help="Show users eligible for cooldown notification")
    parser.add_argument("--set-last-gen", nargs=2, metavar=("USERNAME", "DAYS_AGO"), help="Set user's last_generation for testing")
    
    args = parser.parse_args()
    
    if args.cooldown:
        success = test_cooldown_notification(args.cooldown)
        sys.exit(0 if success else 1)
    elif args.complete:
        success = test_generation_complete_notification(args.complete)
        sys.exit(0 if success else 1)
    elif args.scheduler:
        test_scheduler_cycle()
    elif args.eligible:
        show_eligible_users()
    elif args.set_last_gen:
        username, days_ago = args.set_last_gen
        success = set_user_last_generation(username, int(days_ago))
        sys.exit(0 if success else 1)
    else:
        print("Notification Test Utility")
        print("\nUsage:")
        print("  --cooldown USERNAME       Test cooldown notification")
        print("  --complete USERNAME       Test generation complete notification")
        print("  --scheduler               Run scheduler cycle manually")
        print("  --eligible                Show users eligible for notification")
        print("  --set-last-gen USER DAYS  Set user's last_generation for testing")
        print("\nExamples:")
        print("  python test_notifications.py --cooldown 'username'")
        print("  python test_notifications.py --set-last-gen 'username' 31")
        print("  python test_notifications.py --eligible")
