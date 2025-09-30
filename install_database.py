# install_database.py
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your app and db from app.py
from app import app, db

def init_database():
    """Initialize the database with all required tables"""
    print("Initializing database...")
    
    with app.app_context():
        try:
            # Drop all tables first (use with caution in production!)
            db.drop_all()
            print("✓ Dropped existing tables")
            
            # Create all tables
            db.create_all()
            print("✓ Created all tables successfully")
            
            # Verify tables were created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"✓ Tables in database: {tables}")
            
            # Check if User table has the last_generation column
            if 'user' in tables:
                columns = [col['name'] for col in inspector.get_columns('user')]
                print(f"✓ User table columns: {columns}")
                if 'last_generation' in columns:
                    print("✓ last_generation column exists in User table")
                else:
                    print("✗ last_generation column missing from User table")
            
            print("\nDatabase initialization completed successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error initializing database: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    if init_database():
        print("\nYou can now run your Flask application:")
        print("python app.py")
    else:
        print("\nDatabase initialization failed!")
        sys.exit(1)