#!/usr/bin/env python3
"""
Fix SSL issues in Python environment
"""
import os
import sys
import ssl

def fix_ssl_environment():
    print("🔧 Fixing SSL environment...")
    
    # Method 1: Use certifi for certificates
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        print(f"✅ Using certifi certificates: {certifi.where()}")
    except ImportError:
        print("❌ certifi not available, installing...")
        os.system(f'"{sys.executable}" -m pip install certifi')
        try:
            import certifi
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            print(f"✅ Installed and using certifi")
        except:
            print("❌ Failed to install certifi")
    
    # Method 2: Test SSL
    try:
        print(f"OpenSSL version: {ssl.OPENSSL_VERSION}")
        context = ssl.create_default_context()
        print("✅ SSL context created successfully")
    except Exception as e:
        print(f"❌ SSL context failed: {e}")
    
    # Method 3: Test HTTPS connection
    try:
        import urllib.request
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen('https://www.google.com', context=context, timeout=10) as response:
            print(f"✅ HTTPS connection test passed: {response.getcode()}")
    except Exception as e:
        print(f"❌ HTTPS test failed: {e}")

if __name__ == "__main__":
    fix_ssl_environment()