import os
import sys
from datetime import datetime
from collections import defaultdict

# Add the current directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import directly from your app and models
from app import app, db
from models import Banknote

def clean_up_banknotes():
    with app.app_context():
        print("Starting banknote cleanup...")
        
        # Get all banknotes
        all_banknotes = Banknote.query.all()
        print(f"Total banknotes in database: {len(all_banknotes)}")
        
        # Count duplicates by serial_number and side
        serial_side_count = defaultdict(int)
        
        for banknote in all_banknotes:
            key = f"{banknote.serial_number}_{banknote.side}"
            serial_side_count[key] += 1
        
        # Find duplicates
        duplicates = {k: v for k, v in serial_side_count.items() if v > 1}
        print(f"Found {len(duplicates)} duplicate serial_number+side combinations")
        
        # Remove duplicates (keep the oldest one)
        removed_count = 0
        for key, count in duplicates.items():
            serial_number, side = key.rsplit('_', 1)
            
            # Get all duplicates for this serial_number+side, ordered by creation date
            duplicates_to_remove = Banknote.query.filter_by(
                serial_number=serial_number, 
                side=side
            ).order_by(Banknote.created_at.asc()).all()  # Oldest first
            
            # Keep the oldest one, remove the newer duplicates
            for duplicate in duplicates_to_remove[1:]:
                print(f"Removing duplicate: {duplicate.serial_number} ({duplicate.side}) created at {duplicate.created_at}")
                db.session.delete(duplicate)
                removed_count += 1
        
        # Commit the removal of duplicates
        if removed_count > 0:
            db.session.commit()
            print(f"Removed {removed_count} duplicate banknotes")
        else:
            print("No duplicates found to remove")
        
        # Now recount the total value - only count each denomination once
        banknotes_after_cleanup = Banknote.query.all()
        
        # Create a dictionary to track which denominations we've already counted
        counted_denominations = set()
        total_value = 0
        valid_denominations = 0
        invalid_denominations = 0
        
        for note in banknotes_after_cleanup:
            # Only count each serial number once (ignore front/back duplicates)
            if note.serial_number in counted_denominations:
                continue
                
            try:
                # Try to convert denomination to float
                value = float(note.denomination)
                total_value += value
                valid_denominations += 1
                counted_denominations.add(note.serial_number)
            except (ValueError, TypeError):
                # Handle cases where denomination can't be converted to number
                invalid_denominations += 1
                print(f"Invalid denomination '{note.denomination}' for banknote {note.id} (SN: {note.serial_number})")
        
        print(f"\nCleanup completed!")
        print(f"Total banknote records: {len(banknotes_after_cleanup)}")
        print(f"Complete banknotes (front+back pairs): {valid_denominations}")
        print(f"Banknotes with invalid denominations: {invalid_denominations}")
        print(f"Total value (counting each denomination once): ${(total_value/2):,.2f}")
        
        return total_value/2

if __name__ == "__main__":
    total_value = clean_up_banknotes()