# run.py
import os

# Disable Flask debug reloader
os.environ['FLASK_DEBUG'] = '0'

# Import and run your app
from app import app

if __name__ == '__main__':
    print("Starting app without debug mode...")
    app.run(host='0.0.0.0', port=5555, debug=False)