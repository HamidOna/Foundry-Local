#!/usr/bin/env python3
"""
Simple Mode A (Automatic Tool Calling) Test
===========================================
Tests OpenAI's tool_calls API with Foundry Local
"""

import json
from openai import OpenAI
from foundry_local import FoundryLocalManager

# =============================================================================
# 1. DEFINE YOUR TOOLS (Python Functions)
# =============================================================================

def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get current weather for a location (mock data)."""
    print(f"   🌤️  Calling get_weather({location}, {unit})")
    # In real app, call weather API here
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "Sunny"
    }

def calculate(operation: str, a: float, b: float) -> dict:
    """Perform basic math operations."""
    print(f"   🧮 Calling calculate({operation}, {a}, {b})")

    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }

    return {
        "operation": operation,
        "result": operations.get(operation, "Unknown operation")
    }

# Tool registry
TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate
}

# =============================================================================
# 2. DEFINE TOOL SCHEMAS (OpenAI Format)
# =============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'Paris' or 'New York'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic math calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "Math operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }
    }
]

# =============================================================================
# 3. TEST MODE A
# =============================================================================

def test_mode_a(model_alias: str, query: str):
    """Test automatic tool calling (Mode A)."""

    print("="*70)
    print("🧪 TESTING MODE A: Automatic Tool Calling")
    print("="*70)

    # Connect to Foundry Local
    print(f"\n📡 Connecting to model: {model_alias}")
    manager = FoundryLocalManager(model_alias)
    model_info = manager.get_model_info(model_alias)

    print(f"✅ Connected: {model_info.id}")
    print(f"   Supports Tool Calling: {model_info.supports_tool_calling}")

    if not model_info.supports_tool_calling:
        print("⚠️  WARNING: Model may not support tool calling!")

    client = OpenAI(
        base_url=manager.endpoint,
        api_key=manager.api_key
    )

    # Initial conversation
    messages = [{"role": "user", "content": query}]

    print(f"\n💬 User Query: {query}")
    print("\n" + "-"*70)

    # Agentic loop (allow multiple tool calls)
    max_iterations = 5

    for iteration in range(max_iterations):
        print(f"\n🔄 Iteration {iteration + 1}")

        # Call model with tools
        response = client.chat.completions.create(
            model=model_info.id,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",  # ← KEY: Enable automatic tool calling
            temperature=0.1,
            max_tokens=2048
        )

        message = response.choices[0].message

        # Check if model returned text
        if message.content:
            print(f"   💬 Assistant: {message.content}")

        # Check for tool calls (Mode A indicator)
        if not message.tool_calls:
            print("   ✅ No tool calls - task complete")
            break

        # Add assistant message to history
        messages.append(message)

        # Execute each tool call
        print(f"\n   🔧 Tool Calls Found: {len(message.tool_calls)}")

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n   ┌─ Tool Call ID: {tool_call.id}")
            print(f"   │  Function: {function_name}")
            print(f"   │  Arguments: {json.dumps(function_args, indent=6)}")

            # Execute the function
            if function_name in TOOLS:
                result = TOOLS[function_name](**function_args)
                print(f"   └─ Result: {json.dumps(result, indent=6)}")
            else:
                result = {"error": f"Unknown function: {function_name}"}
                print(f"   └─ ❌ Error: Unknown function")

            # Add tool result back to conversation (REQUIRED for Mode A)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,  # ← MUST include this
                "name": function_name,
                "content": json.dumps(result)
            })

    print("\n" + "="*70)
    print("✅ Test Complete!")
    print("="*70)

# =============================================================================
# 4. RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Configuration
    MODEL_ALIAS = "phi-4-mini"  # Change to your model

    print("\n🚀 Mode A Tool Calling Test Suite\n")

    # Test 1: Single tool call
    print("\n📋 TEST 1: Single Tool Call")
    test_mode_a(MODEL_ALIAS, "What's the weather in Paris?")

    # Test 2: Multiple tool calls
    print("\n\n📋 TEST 2: Multiple Tool Calls")
    test_mode_a(MODEL_ALIAS, "What's the weather in Tokyo, and what's 25 + 17?")

    # Test 3: Complex calculation
    print("\n\n📋 TEST 3: Math Operation")
    test_mode_a(MODEL_ALIAS, "Multiply 12.5 by 8")

    print("\n✨ All tests complete!\n")
