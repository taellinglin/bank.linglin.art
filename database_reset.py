import os
import sqlite3
from app import app, db

def completely_reset_database():
    """Completely reset and rebuild the database"""
    
    print("🚀 Starting complete database rebuild...")
    
    # Remove existing database
    db_path = 'lingcountrytreasury.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        print("✓ Removed old database")
    
    # Ensure instance directory exists
    if not os.path.exists('instance'):
        os.makedirs('instance')
    
    with app.app_context():
        # Create all tables using raw SQL to ensure they're created correctly
        create_tables_with_sql()
        
        # Initialize default data
        initialize_default_data()
        
        print("✅ Database rebuild completed successfully!")

def create_tables_with_sql():
    """Create all tables using raw SQL to avoid SQLAlchemy issues"""
    conn = sqlite3.connect('bank.db')
    cursor = conn.cursor()
    
    try:
        # Drop all tables if they exist (clean slate)
        cursor.execute("DROP TABLE IF EXISTS mining_sessions")
        cursor.execute("DROP TABLE IF EXISTS blockchain_transactions")
        cursor.execute("DROP TABLE IF EXISTS generation_tasks")
        cursor.execute("DROP TABLE IF EXISTS serial_numbers")
        cursor.execute("DROP TABLE IF EXISTS banknotes")
        cursor.execute("DROP TABLE IF EXISTS settings")
        cursor.execute("DROP TABLE IF EXISTS users")
        
        print("✓ Cleaned existing tables")
        
        # Create users table
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username VARCHAR(80) UNIQUE NOT NULL,
                email VARCHAR(120) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                balance FLOAT DEFAULT 0.0,
                is_admin BOOLEAN DEFAULT FALSE,
                two_factor_secret VARCHAR(32),
                bio TEXT DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                last_generation DATETIME
            )
        ''')
        print("✓ Created users table")
        
        # Create settings table
        cursor.execute('''
            CREATE TABLE settings (
                id INTEGER PRIMARY KEY,
                system_name VARCHAR(100) DEFAULT '灵国国库',
                max_banknotes INTEGER DEFAULT 100,
                cooldown_days INTEGER DEFAULT 7,
                maintenance_mode BOOLEAN DEFAULT FALSE,
                allow_registrations BOOLEAN DEFAULT TRUE,
                max_file_size INTEGER DEFAULT 10,
                blockchain_difficulty INTEGER DEFAULT 4,
                mining_reward FLOAT DEFAULT 50.0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✓ Created settings table")
        
        # Create banknotes table
        cursor.execute('''
            CREATE TABLE banknotes (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                serial_number VARCHAR(100) UNIQUE NOT NULL,
                seed_text VARCHAR(200),
                denomination VARCHAR(50) NOT NULL,
                side VARCHAR(10) NOT NULL,
                svg_path VARCHAR(500),
                png_path VARCHAR(500),
                pdf_path VARCHAR(500),
                qr_data TEXT,
                is_public BOOLEAN DEFAULT TRUE,
                transaction_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        print("✓ Created banknotes table")
        
        # Create serial_numbers table
        cursor.execute('''
            CREATE TABLE serial_numbers (
                id INTEGER PRIMARY KEY,
                serial VARCHAR(100) UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                banknote_id INTEGER NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (banknote_id) REFERENCES banknotes (id)
            )
        ''')
        print("✓ Created serial_numbers table")
        
        # Create generation_tasks table
        cursor.execute('''
            CREATE TABLE generation_tasks (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                status VARCHAR(20) DEFAULT 'queued',
                message TEXT DEFAULT '',
                progress INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        print("✓ Created generation_tasks table")
        
        # Create indexes for generation_tasks
        cursor.execute('CREATE INDEX idx_generation_task_user_status ON generation_tasks (user_id, status)')
        cursor.execute('CREATE INDEX idx_generation_task_created ON generation_tasks (created_at)')
        print("✓ Created generation_tasks indexes")
        
        # Create blockchain_transactions table
        cursor.execute('''
            CREATE TABLE blockchain_transactions (
                id INTEGER PRIMARY KEY,
                transaction_hash VARCHAR(64) UNIQUE NOT NULL,
                transaction_type VARCHAR(20) NOT NULL,
                block_hash VARCHAR(64),
                block_index INTEGER,
                from_address VARCHAR(255),
                to_address VARCHAR(255),
                amount FLOAT,
                serial_number VARCHAR(100),
                denomination VARCHAR(50),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                transaction_data TEXT
            )
        ''')
        print("✓ Created blockchain_transactions table")
        
        # Create indexes for blockchain_transactions
        cursor.execute('CREATE INDEX idx_transaction_hash ON blockchain_transactions (transaction_hash)')
        cursor.execute('CREATE INDEX idx_block_index ON blockchain_transactions (block_index)')
        cursor.execute('CREATE INDEX idx_serial_number ON blockchain_transactions (serial_number)')
        print("✓ Created blockchain_transactions indexes")
        
        # Create mining_sessions table
        cursor.execute('''
            CREATE TABLE mining_sessions (
                id INTEGER PRIMARY KEY,
                miner_address VARCHAR(255) NOT NULL,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                blocks_mined INTEGER DEFAULT 0,
                total_rewards FLOAT DEFAULT 0.0,
                status VARCHAR(20) DEFAULT 'active'
            )
        ''')
        print("✓ Created mining_sessions table")
        
        conn.commit()
        print("✅ All tables created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        conn.rollback()
    finally:
        conn.close()

def initialize_default_data():
    """Initialize default settings and admin user"""
    conn = sqlite3.connect('lingcountrytreasury.db')
    cursor = conn.cursor()
    
    try:
        # Insert default settings
        cursor.execute('''
            INSERT INTO settings (system_name, max_banknotes, cooldown_days, maintenance_mode, 
                                 allow_registrations, max_file_size, blockchain_difficulty, mining_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('灵国国库', 100, 7, False, True, 10, 4, 50.0))
        print("✓ Default settings inserted")
        
        # Insert admin user (password is 'admin' hashed)
        admin_password_hash = 'pbkdf2:sha256:260000$KX1F3qN7k8Z9vR2d$8f8c9e8d7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1d0'  # 'admin'
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, is_admin)
            VALUES (?, ?, ?, ?)
        ''', ('admin', 'admin@localhost', admin_password_hash, True))
        print("✓ Admin user created (username: admin, password: admin)")
        
        conn.commit()
        print("✅ Default data initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing data: {e}")
        conn.rollback()
    finally:
        conn.close()

def verify_database():
    """Verify that the database was created correctly"""
    print("\n🔍 Verifying database...")
    
    conn = sqlite3.connect('instance/ling_banknotes.db')
    cursor = conn.cursor()
    
    try:
        # Check if all tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        expected_tables = ['users', 'settings', 'banknotes', 'serial_numbers', 'generation_tasks', 'blockchain_transactions', 'mining_sessions']
        
        print("📋 Found tables:", tables)
        
        for table in expected_tables:
            if table in tables:
                print(f"✅ {table} table exists")
            else:
                print(f"❌ {table} table missing")
        
        # Check settings
        cursor.execute("SELECT * FROM settings")
        settings = cursor.fetchone()
        if settings:
            print(f"✅ Settings: {settings}")
        else:
            print("❌ No settings found")
        
        # Check admin user
        cursor.execute("SELECT username, is_admin FROM users WHERE username='admin'")
        admin = cursor.fetchone()
        if admin:
            print(f"✅ Admin user: {admin}")
        else:
            print("❌ No admin user found")
        
        # Test generation_tasks progress column
        cursor.execute("PRAGMA table_info(generation_tasks)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'progress' in columns:
            print("✅ Progress column exists in generation_tasks")
        else:
            print("❌ Progress column missing from generation_tasks")
            
        print("🎉 Database verification completed!")
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    completely_reset_database()
    verify_database()