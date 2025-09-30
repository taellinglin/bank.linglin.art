# generate_worker.py - Runs generation in separate process
import sys
import os
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_app():
    """Create Flask app instance without running it"""
    from flask import Flask
    from models import db
    
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lingcountrytreasury.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.secret_key = os.environ.get("SECRET_KEY", "ILoveYouForeverXOXO")
    
    db.init_app(app)
    return app

def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_worker.py <user_id> <username> <task_id>")
        return
    
    user_id = int(sys.argv[1])
    username = sys.argv[2]
    task_id = int(sys.argv[3])
    
    print(f"[WORKER] Starting generation for user {user_id}, task {task_id}")
    
    try:
        # Create app instance directly instead of importing from app
        app = create_app()
        
        with app.app_context():
            from models import GenerationTask, db
            from generate import generate_for_user
            from utils import mark_generation_complete
            
            # Update task status
            task = GenerationTask.query.get(task_id)
            if task:
                task.status = 'processing'
                task.message = "Generation in progress..."
                db.session.commit()
            
            denominations = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
            results = []
            total_pairs = 0
            
            # Process denominations
            for i, denom in enumerate(denominations):
                try:
                    if task:
                        task.message = f"Generating denomination {denom} ({i+1}/{len(denominations)})..."
                        db.session.commit()
                    
                    pairs_created = generate_for_user(
                        username=username,
                        user_id=user_id,
                        force_regenerate=False,
                        specific_denom=denom,
                        single_denom=True,
                        max_threads=1
                    )
                    
                    results.append({
                        'denom': denom,
                        'success': True,
                        'pairs_created': pairs_created
                    })
                    total_pairs += pairs_created
                    
                    print(f"[GENERATION] Denomination {denom}: {pairs_created} pairs created")
                    
                except Exception as e:
                    error_msg = f"Error generating {denom}: {str(e)}"
                    print(f"[GENERATION ERROR] {error_msg}")
                    results.append({
                        'denom': denom,
                        'success': False,
                        'error': error_msg
                    })
            
            # Calculate final status
            successful = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]
            
            if len(successful) == len(denominations):
                status = 'completed'
                message = f"All {len(denominations)} denominations generated! {total_pairs} pairs created."
            elif successful:
                status = 'partial'
                message = f"Partial: {len(successful)}/{len(denominations)} denominations. {total_pairs} pairs created."
            else:
                status = 'failed'
                message = "All denominations failed to generate."
            
            # Update task status directly
            if task:
                task.status = status
                task.message = message
                task.completed_at = time.time()
                db.session.commit()
            
            print(f"[WORKER] Generation completed for task {task_id}: {status}")
            
    except Exception as e:
        print(f"[WORKER ERROR] {e}")
        import traceback
        traceback.print_exc()
        
        # Mark as failed directly
        try:
            app = create_app()
            with app.app_context():
                from models import GenerationTask, db
                task = GenerationTask.query.get(task_id)
                if task:
                    task.status = 'failed'
                    task.message = f"Worker error: {str(e)}"
                    task.completed_at = time.time()
                    db.session.commit()
        except Exception as db_error:
            print(f"[CRITICAL] Could not update task status: {db_error}")

if __name__ == '__main__':
    main()