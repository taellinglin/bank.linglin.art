#!/usr/bin/env python3
"""
Database migration script to add new columns to settings table
"""

from app import app, db
from models import Settings

def migrate_database():
    with app.app_context():
        engine = db.engine
        with engine.connect() as conn:
            # List of columns to add
            columns = [
                ("portrait_prompt", "TEXT", "'A professional portrait of a person, high quality, detailed face, neutral background'"),
                ("background_prompt", "TEXT", "'A beautiful fantasy landscape with mountains and rivers, mystical atmosphere'"),
                ("bill_width_mm", "FLOAT", "160.0"),
                ("bill_height_mm", "FLOAT", "60.0"),
                ("bill_title", "VARCHAR(100)", "'灵国国库'"),
                ("bill_subtitle", "VARCHAR(100)", "'天圆地方'"),
                ("bill_dpi", "FLOAT", "300.0"),
                ("font_dir", "VARCHAR(255)", "'./fonts'"),
                ("bg_dir", "VARCHAR(255)", "'./backgrounds'")
            ]

            for col_name, col_type, default_val in columns:
                try:
                    sql = f"ALTER TABLE settings ADD COLUMN {col_name} {col_type} DEFAULT {default_val}"
                    conn.execute(db.text(sql))
                    print(f'✓ Added {col_name} column')
                except Exception as e:
                    print(f'⚠ {col_name} column may already exist: {str(e)[:50]}...')

        print('Database migration completed!')

if __name__ == "__main__":
    migrate_database()