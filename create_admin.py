import sys
from app import app, db
from models import User

def create_admin(username="Sanny Lin"):
    with app.app_context():
        # Check if user already exists
        user = User.query.filter_by(username=username).first()
        if user:
            print(f"[INFO] User '{username}' already exists. Making admin...")
            user.is_admin = True
        else:
            # Create new user
            user = User(
                username=username,
                is_admin=True,
                balance=0
            )
            db.session.add(user)
            print(f"[SUCCESS] Created admin user '{username}'")

        db.session.commit()
        print(f"[DONE] User '{username}' is now an admin.")

if __name__ == "__main__":
    # Optional: allow a custom name from CLI
    username = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Ling Lin"
    create_admin(username)
