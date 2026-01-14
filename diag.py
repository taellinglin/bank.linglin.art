import socket
import psutil

def check_port(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('127.0.0.1', port))
        print(f"Port {port} is available")
        s.close()
        return True
    except OSError as e:
        print(f"Port {port} is in use or blocked: {e}")
        return False

# Test common ports
for port in [5000, 5555, 5001, 8080, 8000]:
    check_port(port)