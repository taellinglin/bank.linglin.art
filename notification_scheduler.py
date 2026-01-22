"""
Background notification scheduler for sending emails
Checks for:
1. Users whose cooldown has expired and can generate banknotes again
2. Completed generation tasks that need notification
"""

import threading
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationScheduler:
    def __init__(self, check_interval=3600):  # Check every hour
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.notified_users = set()  # Track who we've already notified
    
    def start(self):
        """Start the background scheduler"""
        if self.running:
            logger.info("[SCHEDULER] Already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"[SCHEDULER] Started - checking every {self.check_interval}s")
    
    def stop(self):
        """Stop the background scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("[SCHEDULER] Stopped")
    
    def _run(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Import here to avoid circular import
                from app import app
                with app.app_context():
                    self._check_cooldown_expired()
                    self._check_completed_generations()
            except Exception as e:
                logger.error(f"[SCHEDULER] Error in check cycle: {e}")
            
            # Sleep in small intervals so we can stop quickly
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _check_cooldown_expired(self):
        """Check for users whose cooldown has expired"""
        try:
            # Import here to avoid circular import
            from models import User, Settings
            from email_service import send_generation_ready_notification
            
            # Get settings for cooldown period
            settings = Settings.query.first()
            cooldown_days = settings.cooldown_days if settings else 7
            
            # Find users who:
            # 1. Have generated before (last_generation is not None)
            # 2. Cooldown period has expired
            # 3. Have verified email
            # 4. Haven't been notified yet
            
            now = datetime.utcnow()
            cutoff_time = now - timedelta(days=cooldown_days)
            
            users = User.query.filter(
                User.last_generation.isnot(None),
                User.last_generation <= cutoff_time,
                User.email_verified == True
            ).all()
            
            notified_count = 0
            for user in users:
                # Check if we've already notified this user
                user_key = f"{user.id}_{user.last_generation.isoformat()}"
                if user_key in self.notified_users:
                    continue
                
                # Check if they can generate (double-check)
                if not user.can_generate_money():
                    continue
                
                try:
                    send_generation_ready_notification(
                        user.email,
                        user.username,
                        days_until_next=0
                    )
                    
                    self.notified_users.add(user_key)
                    notified_count += 1
                    logger.info(f"[SCHEDULER] Sent generation-ready notification to {user.username}")
                    
                except Exception as e:
                    logger.error(f"[SCHEDULER] Failed to send notification to {user.username}: {e}")
            
            if notified_count > 0:
                logger.info(f"[SCHEDULER] Sent {notified_count} generation-ready notifications")
                
        except Exception as e:
            logger.error(f"[SCHEDULER] Error checking cooldown: {e}")
    
    def _check_completed_generations(self):
        """Check for completed generation tasks that haven't been notified"""
        try:
            # Find recently completed tasks (within last hour) that might need notification
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            from models import User, GenerationTask, Banknote
            from email_service import send_banknote_generation_notification
            completed_tasks = GenerationTask.query.filter(
                GenerationTask.status.in_(["completed", "partial"]),
                GenerationTask.completed_at >= one_hour_ago
            ).all()
            
            # Group by user
            user_tasks = {}
            for task in completed_tasks:
                if task.user_id not in user_tasks:
                    user_tasks[task.user_id] = []
                user_tasks[task.user_id].append(task)
            
            notified_count = 0
            for user_id, tasks in user_tasks.items():
                try:
                    user = User.query.get(user_id)
                    if not user or not user.email_verified:
                        continue
                    
                    # Check if we've already notified for these tasks
                    task_ids = tuple(sorted(t.id for t in tasks))
                    notification_key = f"gen_{user_id}_{task_ids}"
                    
                    if notification_key in self.notified_users:
                        continue
                    
                    # Get banknotes created recently for this user (front side only)
                    recent_banknotes = Banknote.query.filter(
                        Banknote.user_id == user_id,
                        Banknote.created_at >= one_hour_ago,
                        Banknote.side == "front",
                    ).all()

                    total_banknotes = len(recent_banknotes)
                    denominations = {b.denomination for b in recent_banknotes if b.denomination}
                    serial_numbers = [b.serial_number for b in recent_banknotes if b.serial_number]
                    
                    if total_banknotes > 0:
                        denom_str = ", ".join(str(d) for d in sorted(denominations)) if denominations else "Multiple"
                        
                        send_banknote_generation_notification(
                            user.email,
                            user.username,
                            denomination=denom_str,
                            count=total_banknotes,
                            serial_numbers=serial_numbers[:5]  # First 5
                        )
                        
                        self.notified_users.add(notification_key)
                        notified_count += 1
                        logger.info(f"[SCHEDULER] Sent generation-complete notification to {user.username}")
                
                except Exception as user_error:
                    logger.error(f"[SCHEDULER] Error notifying user {user_id}: {user_error}")
            
            if notified_count > 0:
                logger.info(f"[SCHEDULER] Sent {notified_count} generation-complete notifications")
                
        except Exception as e:
            logger.error(f"[SCHEDULER] Error checking completed generations: {e}")
    
    def force_check(self):
        """Force an immediate check (useful for testing)"""
        from app import app
        logger.info("[SCHEDULER] Forcing immediate check...")
        with app.app_context():
            self._check_cooldown_expired()
            self._check_completed_generations()
        logger.info("[SCHEDULER] Forced check complete")

# Global scheduler instance
scheduler = None

def init_notification_scheduler(check_interval=3600):
    """Initialize and start the notification scheduler"""
    global scheduler
    if scheduler is None:
        scheduler = NotificationScheduler(check_interval=check_interval)
        scheduler.start()
    return scheduler

def get_scheduler():
    """Get the global scheduler instance"""
    return scheduler

if __name__ == "__main__":
    # Test mode
    print("[+] Starting notification scheduler in test mode...")
    print("[+]  - imports only needed here")
    from app import app
    print("[+] Will check every 60 seconds")
    
    scheduler = NotificationScheduler(check_interval=60)
    scheduler.start()
    
    try:
        print("[+] Scheduler running. Press Ctrl+C to stop.")
        print("[+] Performing initial check...")
        scheduler.force_check()
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[+] Stopping scheduler...")
        scheduler.stop()
        print("[+] Done!")
