import os
import sys
import json 
class DataManager:
    """Manages data storage with EXE directory fallback to ProgramData"""
    
    @staticmethod
    def get_data_dir():
        """Get the best data directory (EXE dir first, then ProgramData fallback)"""
        # First try: Same directory as the executable
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            exe_dir = os.path.dirname(sys.executable)
        else:
            # Running as script
            exe_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Test if we can write to the EXE directory
        exe_data_dir = os.path.join(exe_dir, 'data')
        try:
            # Create data subdirectory and test write permissions
            os.makedirs(exe_data_dir, exist_ok=True)
            test_file = os.path.join(exe_data_dir, 'write_test.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"📁 Using EXE directory for data: {exe_data_dir}")
            return exe_data_dir
        except (PermissionError, OSError):
            # Fallback: ProgramData directory
            if os.name == 'nt':  # Windows
                programdata = os.environ.get('PROGRAMDATA', 'C:\\ProgramData')
                programdata_dir = os.path.join(programdata, 'Luna Suite')
            else:  # Linux/Mac
                programdata_dir = '/var/lib/luna-suite'
            
            # Create ProgramData directory
            os.makedirs(programdata_dir, exist_ok=True)
            print(f"📁 Using ProgramData directory for data: {programdata_dir}")
            return programdata_dir
    
    @staticmethod
    def get_file_path(filename):
        """Get full path for a data file with fallback support"""
        data_dir = DataManager.get_data_dir()
        return os.path.join(data_dir, filename)
    
    @staticmethod
    def save_json(filename, data):
        """Save data to JSON file with fallback support"""
        file_path = DataManager.get_file_path(filename)
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved {filename} to {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
            return False
    
    @staticmethod
    def load_json(filename, default=None):
        """Load data from JSON file with fallback support"""
        if default is None:
            default = []
        
        file_path = DataManager.get_file_path(filename)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"⚠️  {filename} not found, using default")
                return default
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return default