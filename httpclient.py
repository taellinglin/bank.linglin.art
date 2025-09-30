import json
import urllib
# Try to use a simpler HTTP client that works better with PyInstaller
try:
    import urllib.request
    import urllib.error
    # Create a simple SSL context that should work with PyInstaller
    # Configure SSL at application startup
    
    # Disable SSL verification for simplicity
    SSL_AVAILABLE = True
    print("🔐 SSL configured (verification disabled)")
except ImportError as e:
    print(f"⚠️ SSL not available: {e}")
    SSL_AVAILABLE = False

class SimpleHTTPClient:
    """A simple HTTP client that works with PyInstaller"""
    
    @staticmethod
    def get(url, timeout=30):
        """Simple HTTP GET request"""
        try:
            if SSL_AVAILABLE:
                # Use urllib with SSL disabled
                req = urllib.request.Request(url)
                response = urllib.request.urlopen(req, timeout=timeout)
                return response.read().decode('utf-8')
            else:
                # Fallback: try without SSL for http URLs
                if url.startswith('https://'):
                    # Try http instead
                    http_url = url.replace('https://', 'http://', 1)
                    print(f"⚠️  SSL not available, trying HTTP: {http_url}")
                    req = urllib.request.Request(http_url)
                    response = urllib.request.urlopen(req, timeout=timeout)
                    return response.read().decode('utf-8')
                else:
                    req = urllib.request.Request(url)
                    response = urllib.request.urlopen(req, timeout=timeout)
                    return response.read().decode('utf-8')
        except Exception as e:
            print(f"❌ HTTP request failed for {url}: {e}")
            return None
    
    @staticmethod
    def get_json(url, timeout=30):
        """Get JSON data from URL"""
        response_text = SimpleHTTPClient.get(url, timeout)
        if response_text:
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error for {url}: {e}")
        return None