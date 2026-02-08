"""
Test if your OpenAI API key works
"""

from openai import OpenAI
import sys

# Get API key from user
if len(sys.argv) > 1:
    api_key = sys.argv[1]
else:
    api_key = input("Enter your OpenAI API key: ")

print(f"\n🔑 Testing API key...")
print(f"   Length: {len(api_key)}")
print(f"   Starts with: {api_key[:10]}...")

try:
    # Test with OpenAI client
    client = OpenAI(api_key=api_key)
    
    print(f"\n✅ Client initialized successfully")
    print(f"🧪 Testing API call...")
    
    # Make a simple test call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'API key works!'"}],
        max_tokens=10
    )
    
    print(f"✅ API call successful!")
    print(f"📝 Response: {response.choices[0].message.content}")
    print(f"\n🎉 Your API key is VALID and working!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\n💡 Possible issues:")
    print(f"   1. API key is invalid or expired")
    print(f"   2. No credits in your OpenAI account")
    print(f"   3. API key doesn't have access to gpt-4o-mini")
    print(f"   4. Network/firewall blocking OpenAI")
