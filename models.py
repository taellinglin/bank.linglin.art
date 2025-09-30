# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import secrets
import string
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    balance = db.Column(db.Float, default=0.0)
    is_admin = db.Column(db.Boolean, default=False)
    two_factor_secret = db.Column(db.String(32))
    bio = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    last_generation = db.Column(db.DateTime)
    
    # Relationships
    banknotes = db.relationship('Banknote', backref='user', lazy=True, cascade="all, delete-orphan")
    serial_numbers = db.relationship('SerialNumber', backref='user', lazy=True, cascade="all, delete-orphan")
    generation_tasks = db.relationship('GenerationTask', backref='user', lazy=True, cascade="all, delete-orphan")
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_totp_uri(self):
        import pyotp
        return pyotp.totp.TOTP(self.two_factor_secret).provisioning_uri(
            name=self.username,
            issuer_name="灵国国库"
        )
    
    def can_generate_money(self):
        """Check if user can generate money based on cooldown period"""
        if not self.last_generation:
            return True
        
        cooldown_days = 7  # Default cooldown period
        settings = Settings.query.first()
        if settings:
            cooldown_days = settings.cooldown_days
        
        next_generation_date = self.last_generation + timedelta(days=cooldown_days)
        return datetime.utcnow() >= next_generation_date
    
    def days_until_next_generation(self):
        """Calculate days until next money generation is allowed"""
        if not self.last_generation:
            return 0
        
        cooldown_days = 7  # Default cooldown period
        settings = Settings.query.first()
        if settings:
            cooldown_days = settings.cooldown_days
        
        next_generation_date = self.last_generation + timedelta(days=cooldown_days)
        days_left = (next_generation_date - datetime.utcnow()).days
        return max(0, days_left)
    
    def get_total_banknote_value(self):
        """Calculate total value of user's banknotes"""
        total = 0
        for banknote in self.banknotes:
            try:
                # Only count front side banknotes to avoid double counting
                if banknote.side.lower() == 'front':
                    total += float(banknote.denomination)
            except (ValueError, TypeError):
                continue
        return total
    
    def get_active_generation_task(self):
        """Get the user's most recent active generation task"""
        from sqlalchemy import desc
        return GenerationTask.query.filter_by(
            user_id=self.id
        ).filter(
            ~GenerationTask.status.in_(['completed', 'failed', 'cancelled'])
        ).order_by(desc(GenerationTask.created_at)).first()
    
    def get_recent_generation_tasks(self, limit=5):
        """Get user's recent generation tasks"""
        from sqlalchemy import desc
        return GenerationTask.query.filter_by(
            user_id=self.id
        ).order_by(desc(GenerationTask.created_at)).limit(limit).all()
    
    def __repr__(self):
        return f'<User {self.username}>'

class Banknote(db.Model):
    __tablename__ = 'banknotes'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    serial_number = db.Column(db.String(100), unique=True, nullable=False)
    seed_text = db.Column(db.String(200))
    denomination = db.Column(db.String(50), nullable=False)
    side = db.Column(db.String(10), nullable=False)  # 'front' or 'back'
    svg_path = db.Column(db.String(500))
    png_path = db.Column(db.String(500))
    pdf_path = db.Column(db.String(500))
    qr_data = db.Column(db.Text)
    is_public = db.Column(db.Boolean, default=True)
    transaction_data = db.Column(db.Text)  # JSON string for blockchain transactions
    digital_signature = db.Column(db.Text, nullable=True)
    public_key = db.Column(db.Text, nullable=True) 
    metadata_hash = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    serials = db.relationship('SerialNumber', backref='banknote', lazy=True, cascade="all, delete-orphan")
    
    def get_transaction_data(self):
        """Parse and return transaction data as dict"""
        if self.transaction_data:
            try:
                return json.loads(self.transaction_data)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_transaction_data(self, data):
        """Set transaction data as JSON string"""
        self.transaction_data = json.dumps(data)
    
    def get_verification_url(self):
        """Generate verification URL for this banknote"""
        from flask import url_for
        return url_for('verify_serial', serial_id=self.serial_number, _external=True)
    
    def __repr__(self):
        return f'<Banknote {self.serial_number} - {self.denomination}>'

class SerialNumber(db.Model):
    __tablename__ = 'serial_numbers'
    
    id = db.Column(db.Integer, primary_key=True)
    serial = db.Column(db.String(100), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    banknote_id = db.Column(db.Integer, db.ForeignKey('banknotes.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SerialNumber {self.serial}>'

class GenerationTask(db.Model):
    __tablename__ = 'generation_tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.String(20), default='queued')  # queued, pending, processing, completed, failed, cancelled
    message = db.Column(db.Text, default="")
    progress = db.Column(db.Integer, default=0)  # 0-100 percentage
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    # Indexes for better query performance
    __table_args__ = (
        db.Index('idx_generation_task_user_status', 'user_id', 'status'),
        db.Index('idx_generation_task_created', 'created_at'),
    )
    
    def is_completed(self):
        return self.status in ['completed', 'failed', 'cancelled']
    
    def duration(self):
        """Calculate task duration in seconds"""
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None
    
    def __repr__(self):
        return f'<GenerationTask {self.id} - {self.status}>'

class Settings(db.Model):
    __tablename__ = 'settings'
    
    id = db.Column(db.Integer, primary_key=True)
    system_name = db.Column(db.String(100), default="灵国国库")
    max_banknotes = db.Column(db.Integer, default=100)
    cooldown_days = db.Column(db.Integer, default=7)
    maintenance_mode = db.Column(db.Boolean, default=False)
    allow_registrations = db.Column(db.Boolean, default=True)
    max_file_size = db.Column(db.Integer, default=10)  # MB
    blockchain_difficulty = db.Column(db.Integer, default=4)
    mining_reward = db.Column(db.Float, default=50.0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Settings {self.system_name}>'

class BlockchainTransaction(db.Model):
    __tablename__ = 'blockchain_transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    transaction_hash = db.Column(db.String(64), unique=True, nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)  # genesis, transfer, reward
    block_hash = db.Column(db.String(64))
    block_index = db.Column(db.Integer)
    from_address = db.Column(db.String(255))
    to_address = db.Column(db.String(255))
    amount = db.Column(db.Float)
    serial_number = db.Column(db.String(100))
    denomination = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    transaction_data = db.Column(db.Text)  # Full transaction JSON
    
    # Index for faster queries
    __table_args__ = (
        db.Index('idx_transaction_hash', 'transaction_hash'),
        db.Index('idx_block_index', 'block_index'),
        db.Index('idx_serial_number', 'serial_number'),
    )
    
    def get_transaction_data(self):
        """Parse transaction data as dict"""
        if self.transaction_data:
            try:
                return json.loads(self.transaction_data)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_transaction_data(self, data):
        """Set transaction data as JSON string"""
        self.transaction_data = json.dumps(data)
    
    def __repr__(self):
        return f'<BlockchainTransaction {self.transaction_hash[:16]}...>'

class MiningSession(db.Model):
    __tablename__ = 'mining_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    miner_address = db.Column(db.String(255), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    blocks_mined = db.Column(db.Integer, default=0)
    total_rewards = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='active')  # active, completed, stopped
    
    def duration(self):
        """Calculate mining session duration"""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()
    
    def __repr__(self):
        return f'<MiningSession {self.miner_address} - {self.blocks_mined} blocks>'

def init_default_settings():
    """Initialize default settings if they don't exist"""
    if not Settings.query.first():
        default_settings = Settings()
        db.session.add(default_settings)
        db.session.commit()

def init_admin_user():
    """Create default admin user if no users exist"""
    if not User.query.first():
        admin = User(
            username="admin",
            email="admin@localhost",
            is_admin=True
        )
        admin.set_password("admin")
        db.session.add(admin)
        db.session.commit()
        print("Default admin user created: admin/admin")

# Utility functions
def generate_serial_number():
    """Generate a unique serial number"""
    while True:
        # Generate random serial components
        part1 = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        part2 = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(4))
        part3 = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        
        serial = f"SN-{part1}-{part2}-{part3}"
        
        # Check if serial already exists
        if not SerialNumber.query.filter_by(serial=serial).first():
            return serial

def get_user_stats(user_id):
    """Get comprehensive statistics for a user"""
    user = User.query.get(user_id)
    if not user:
        return None
    
    stats = {
        'total_banknotes': len(user.banknotes),
        'front_banknotes': len([b for b in user.banknotes if b.side.lower() == 'front']),
        'back_banknotes': len([b for b in user.banknotes if b.side.lower() == 'back']),
        'total_value': user.get_total_banknote_value(),
        'public_banknotes': len([b for b in user.banknotes if b.is_public]),
        'completed_tasks': len([t for t in user.generation_tasks if t.status == 'completed']),
        'pending_tasks': len([t for t in user.generation_tasks if t.status in ['queued', 'pending', 'processing']]),
        'last_generation': user.last_generation,
        'can_generate': user.can_generate_money(),
        'days_until_next': user.days_until_next_generation()
    }
    
    return stats