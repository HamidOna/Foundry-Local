#!/usr/bin/env python3
"""
Quick Tool Calling Test - Minimal Example
==========================================
Fast test to verify Mode A works after upgrade.
"""

import json
from openai import OpenAI
from foundry_local import FoundryLocalManager

def get_weather(location: str) -> dict:
    """Mock weather function."""
    print(f"   🌤️  Getting weather for {location}...")
    return {"location": location, "temperature": 22, "condition": "Sunny"}

# Setup
MODEL = "phi-4-cuda-gpu"
manager = FoundryLocalManager(MODEL)
client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
model_id = manager.get_model_info(MODEL).id

# Tool schema
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

print("\n" + "="*60)
print("🧪 QUICK TOOL CALLING TEST")
print("="*60)

# Test query
response = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=TOOLS,
    tool_choice="auto",
    temperature=0.0
)

msg = response.choices[0].message

print(f"\n📦 Response:")
print(f"   Content: {msg.content}")
print(f"   tool_calls: {msg.tool_calls}")

if msg.tool_calls:
    print("\n✅ MODE A WORKING! tool_calls attribute is populated")

    for tc in msg.tool_calls:
        print(f"\n🔧 Tool Call:")
        print(f"   ID: {tc.id}")
        print(f"   Function: {tc.function.name}")
        print(f"   Arguments: {tc.function.arguments}")

        # Execute
        args = json.loads(tc.function.arguments)
        result = get_weather(**args)
        print(f"   Result: {result}")

    print("\n🎉 Tool calling is working correctly with Mode A!")

else:
    print("\n❌ MODE A NOT WORKING")
    print("   tool_calls is None")
    print(f"\n   Raw content: {msg.content}")

    if "functools" in (msg.content or ""):
        print("\n   ℹ️  Response contains 'functools' - server not converting to Mode A")
        print("   This means you need Mode B parser or server needs configuration")
    else:
        print("\n   ℹ️  No tool calls detected at all")

print("\n" + "="*60 + "\n")
