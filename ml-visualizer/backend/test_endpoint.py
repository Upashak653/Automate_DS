"""
Quick test to verify AI agent endpoint is registered
"""

import requests

# Test if endpoint exists
try:
    response = requests.post('http://localhost:5000/api/ai-agent/analyze-overfitting', 
                            data={'api_key': 'test'})
    print(f"✅ Endpoint exists! Status: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.ConnectionError:
    print("❌ Backend not running! Start with: python app.py")
except Exception as e:
    print(f"Error: {e}")

# List all routes
try:
    response = requests.get('http://localhost:5000/api/health')
    if response.status_code == 200:
        print("\n✅ Backend is running")
    else:
        print(f"\n⚠️ Backend returned: {response.status_code}")
except:
    print("\n❌ Cannot connect to backend")
