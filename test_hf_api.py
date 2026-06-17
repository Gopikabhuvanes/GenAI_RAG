#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY", "")

if not HF_API_KEY:
    print("❌ HF_API_KEY not found in .env file")
    exit(1)

print(f"✓ API Key found: {HF_API_KEY[:20]}...")

# Try Inference Endpoints with serverless API
models_to_test = [
    ("gpt2", "https://api-inference.huggingface.co/models/gpt2"),
    ("distilgpt2", "https://api-inference.huggingface.co/models/distilgpt2"),
    ("EleutherAI/gpt-neo-125m", "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-125m"),
]

headers = {"Authorization": f"Bearer {HF_API_KEY}"}
test_prompt = "What is AI?"

for model_name, url in models_to_test:
    print(f"\nTesting {model_name}...")
    print(f"   URL: {url}")
    
    payload = {"inputs": test_prompt}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   ✓ SUCCESS! Model works!")
                print(f"   Response: {result}")
                print(f"\n✅ Use model: {model_name}")
                break
            except json.JSONDecodeError:
                print(f"   Response: {response.text[:100]}")
        else:
            error_msg = response.text[:150]
            print(f"   Error ({response.status_code}): {error_msg}")
    except Exception as e:
        print(f"   Exception: {str(e)[:100]}")


