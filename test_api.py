import json
import os
import requests

# Get API key from environment
api_key = os.getenv('OPENAI_API_KEY', 'sk-5aa5ea0263c942f896210d035529b47b')

# Test the API
def test_interact_with_model(chosen_model, my_query):
    url = "https://openwebui.crc.nd.edu/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": chosen_model,
        "messages": [{"role": "user", "content": my_query}],
    }
    
    print(f"Testing model: {chosen_model}")
    print(f"Query: {my_query[:100]}...")
    print("-" * 80)
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        print("✓ API call successful!")
        print("\nFull Response:")
        print(json.dumps(result, indent=2))
        print("\n" + "=" * 80)
        
        # Try to extract key information
        print("\nExtracted Information:")
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0].get('message', {}).get('content', '')
            print(f"Content: {content[:200]}...")
        
        if 'usage' in result:
            print(f"\nToken Usage: {result['usage']}")
        else:
            print("\n⚠ Warning: No usage information in response")
            
        return result
    else:
        print(f"✗ Failed to interact with the model")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

# Test 1: Simple query
print("=" * 80)
print("TEST 1: Simple Query")
print("=" * 80)
result1 = test_interact_with_model("gpt-oss:120b", "What is the capital of France?")

print("\n\n")

# Test 2: JSON output request
print("=" * 80)
print("TEST 2: JSON Output Request")
print("=" * 80)
json_query = """Please provide a JSON response with the following structure:
{
  "topics": ["topic1", "topic2", "topic3"],
  "sentiment": "positive"
}

Analyze this text and return JSON: "I love using naloxone, it saved my friend's life!"
"""
result2 = test_interact_with_model("gpt-oss:120b", json_query)

print("\n\n")

# Test 3: Check if JSON mode parameter works
print("=" * 80)
print("TEST 3: Testing JSON Mode Parameter (if supported)")
print("=" * 80)

def test_json_mode():
    url = "https://openwebui.crc.nd.edu/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-oss:120b",
        "messages": [{"role": "user", "content": "Return a JSON object with keys 'name' and 'age'"}],
        "response_format": {"type": "json_object"}  # OpenAI-style JSON mode
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        print("✓ JSON mode parameter accepted!")
        print(json.dumps(result, indent=2))
        return True
    else:
        print(f"✗ JSON mode parameter not supported or failed")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        return False

test_json_mode()

print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Please review the responses above to determine:")
print("1. Does the API return 'usage' with token counts?")
print("2. What is the exact structure of the response?")
print("3. Does 'response_format' parameter work for JSON mode?")
print("4. How should we extract the content from the response?")
