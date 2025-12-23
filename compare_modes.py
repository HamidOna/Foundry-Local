#!/usr/bin/env python3
"""
Mode A vs Mode B Comparison
============================
Shows the difference between:
- Mode A: Automatic tool_calls (OpenAI standard)
- Mode B: Functools text parsing (Phi-4 specific)
"""

import json
import re
from openai import OpenAI
from foundry_local import FoundryLocalManager

# =============================================================================
# TOOLS
# =============================================================================

def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get weather (mock)."""
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "Sunny"
    }

TOOLS = {"get_weather": get_weather}

TOOL_SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

# =============================================================================
# MODE A: Automatic tool_calls
# =============================================================================

def test_mode_a(client: OpenAI, model_id: str):
    """Test Mode A: Uses message.tool_calls attribute."""

    print("\n" + "="*70)
    print("MODE A: Automatic Tool Calls (OpenAI Standard)")
    print("="*70)

    messages = [{"role": "user", "content": "What's the weather in London?"}]

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",  # ← Mode A key parameter
        temperature=0.1,
        max_tokens=2048
    )

    message = response.choices[0].message

    print("\n📦 Response Structure:")
    print(f"   - message.content: {message.content}")
    print(f"   - message.tool_calls: {message.tool_calls}")

    # Check for tool_calls attribute
    if message.tool_calls:
        print("\n✅ MODE A DETECTED: Found tool_calls attribute")

        for tool_call in message.tool_calls:
            print(f"\n   Tool Call ID: {tool_call.id}")
            print(f"   Function Name: {tool_call.function.name}")
            print(f"   Arguments: {tool_call.function.arguments}")

            # Parse and execute
            args = json.loads(tool_call.function.arguments)
            result = TOOLS[tool_call.function.name](**args)

            print(f"   Result: {result}")

            # Return result (Mode A format)
            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,  # ← Mode A specific
                "name": tool_call.function.name,
                "content": json.dumps(result)
            })

        # Get final response
        final_response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.1,
            max_tokens=2048
        )

        print(f"\n   Final Answer: {final_response.choices[0].message.content}")

    else:
        print("\n❌ MODE A NOT DETECTED: No tool_calls attribute")
        print("   Falling back to Mode B...")
        test_mode_b_from_response(message.content)

# =============================================================================
# MODE B: Functools text parsing
# =============================================================================

def test_mode_b(client: OpenAI, model_id: str):
    """Test Mode B: Parse functools from text."""

    print("\n" + "="*70)
    print("MODE B: Functools Text Parsing (Phi-4 Specific)")
    print("="*70)

    # For Mode B, tools are passed differently (check your model config)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=TOOL_SCHEMAS,  # Still pass schemas
        temperature=0.1,
        max_tokens=4096
    )

    content = response.choices[0].message.content
    print(f"\n📦 Raw Response: {content}")

    test_mode_b_from_response(content)

def test_mode_b_from_response(content: str):
    """Parse functools format from text response."""

    # Check for functools pattern
    functools_match = re.search(r'functools\s*\[(.*?)\]', content, re.DOTALL)

    if functools_match:
        print("\n✅ MODE B DETECTED: Found functools pattern")

        try:
            # Parse the JSON inside functools[...]
            calls_json = f"[{functools_match.group(1)}]"
            tool_calls = json.loads(calls_json)

            print(f"   Parsed {len(tool_calls)} tool call(s)")

            for call in tool_calls:
                func_name = call.get("name")
                func_args = call.get("arguments", {})

                print(f"\n   Function: {func_name}")
                print(f"   Arguments: {json.dumps(func_args, indent=6)}")

                # Execute
                result = TOOLS[func_name](**func_args)
                print(f"   Result: {result}")

        except json.JSONDecodeError as e:
            print(f"   ❌ JSON Parse Error: {e}")

    else:
        print("\n❌ MODE B NOT DETECTED: No functools pattern found")
        print("   This model might not support tool calling at all")

# =============================================================================
# COMPARISON
# =============================================================================

def compare_modes(model_alias: str):
    """Run both modes and compare."""

    print("="*70)
    print("🔬 MODE A vs MODE B COMPARISON")
    print("="*70)

    # Setup
    manager = FoundryLocalManager(model_alias)
    model_info = manager.get_model_info(model_alias)

    print(f"\nModel: {model_info.id}")
    print(f"Supports Tool Calling: {model_info.supports_tool_calling}")

    client = OpenAI(
        base_url=manager.endpoint,
        api_key=manager.api_key
    )

    # Test Mode A
    print("\n\n" + "🅰️ " * 20)
    test_mode_a(client, model_info.id)

    # Test Mode B
    print("\n\n" + "🅱️ " * 20)
    test_mode_b(client, model_info.id)

    # Summary
    print("\n\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print("""
MODE A (Automatic):
  ✓ Uses message.tool_calls attribute
  ✓ Structured response format
  ✓ Returns with role="tool" and tool_call_id
  ✓ OpenAI standard
  ⚠ May not work with Phi-4 (experimental)

MODE B (Functools):
  ✓ Uses text parsing with regex
  ✓ Pattern: functools[{"name": ..., "arguments": ...}]
  ✓ Works with Phi-4-mini (production)
  ⚠ Requires custom parsing logic
    """)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    MODEL_ALIAS = "phi-4-mini"  # Change to your model

    print("\n🚀 Starting Mode Comparison Test\n")

    try:
        compare_modes(MODEL_ALIAS)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Comparison complete!\n")
