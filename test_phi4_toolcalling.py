#!/usr/bin/env python3
"""
Phi-4 Tool Calling Test Script (Mode A - OpenAI Compatible)
============================================================
Tests automatic tool calling with phi-4-cuda-gpu after 0.8.117 upgrade.
"""

import json
from openai import OpenAI
from foundry_local import FoundryLocalManager

# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

def generate_quiz(topic: str, num_questions: int = 3) -> dict:
    """Generate a quiz on a given topic."""
    print(f"\n   ⚡ Executing: generate_quiz(topic='{topic}', num_questions={num_questions})")

    questions = []
    for i in range(num_questions):
        questions.append({
            "id": i + 1,
            "question": f"What is an important concept about {topic}?",
            "options": ["A) Concept 1", "B) Concept 2", "C) Concept 3", "D) Concept 4"],
            "correct_answer": "A"
        })

    result = {
        "quiz_id": f"quiz_{topic.lower().replace(' ', '_')}",
        "topic": topic,
        "questions": questions,
        "total": len(questions)
    }

    print(f"   ✅ Generated {len(questions)} questions")
    return result


def save_quiz(quiz_id: str, quiz_data: dict) -> dict:
    """Save quiz to file (mock implementation)."""
    print(f"\n   ⚡ Executing: save_quiz(quiz_id='{quiz_id}')")

    # In real implementation, save to file
    # For testing, just return success
    result = {
        "status": "saved",
        "quiz_id": quiz_id,
        "file_path": f"./quizzes/{quiz_id}.json",
        "question_count": len(quiz_data.get("questions", []))
    }

    print(f"   ✅ Quiz saved: {result['file_path']}")
    return result


def calculate(operation: str, a: float, b: float) -> dict:
    """Perform mathematical calculations."""
    print(f"\n   ⚡ Executing: calculate(operation='{operation}', a={a}, b={b})")

    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else None
    }

    result_value = operations.get(operation)

    result = {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result_value,
        "success": result_value is not None
    }

    print(f"   ✅ Result: {result_value}")
    return result


# Tool registry
AVAILABLE_TOOLS = {
    "generate_quiz": generate_quiz,
    "save_quiz": save_quiz,
    "calculate": calculate
}

# Tool schemas (OpenAI format)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": "Generate a quiz with multiple choice questions on a given topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic for the quiz (e.g., 'Photosynthesis', 'World War II')"
                    },
                    "num_questions": {
                        "type": "integer",
                        "description": "Number of questions to generate (default: 3)"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_quiz",
            "description": "Save a generated quiz to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "quiz_id": {
                        "type": "string",
                        "description": "Unique identifier for the quiz"
                    },
                    "quiz_data": {
                        "type": "object",
                        "description": "The quiz data containing questions and metadata"
                    }
                },
                "required": ["quiz_id", "quiz_data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform"
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
# TEST FUNCTION
# =============================================================================

def test_tool_calling(client: OpenAI, model_id: str, user_query: str, test_name: str):
    """Test tool calling with a specific query."""

    print("\n" + "="*70)
    print(f"🧪 TEST: {test_name}")
    print("="*70)
    print(f"📝 Query: {user_query}\n")

    messages = [{"role": "user", "content": user_query}]
    max_iterations = 5

    for iteration in range(max_iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
        print("-" * 70)

        # Make API call
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=2048
        )

        message = response.choices[0].message

        # Debug output
        print("\n📦 Response Details:")
        print(f"   Content: {message.content}")
        print(f"   tool_calls: {message.tool_calls}")

        # Check for tool calls
        if not message.tool_calls:
            print("\n✅ No tool calls - Final response received")
            print(f"\n💬 Assistant: {message.content}")
            return message.content

        # Process tool calls (Mode A)
        print(f"\n🔧 Found {len(message.tool_calls)} tool call(s)")

        # Add assistant message to history
        messages.append(message)

        # Execute each tool call
        for idx, tool_call in enumerate(message.tool_calls, 1):
            print(f"\n   Tool Call #{idx}:")
            print(f"   ├─ ID: {tool_call.id}")
            print(f"   ├─ Function: {tool_call.function.name}")
            print(f"   └─ Arguments: {tool_call.function.arguments}")

            # Parse arguments
            try:
                func_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"   ❌ Failed to parse arguments: {e}")
                continue

            # Execute function
            func_name = tool_call.function.name
            if func_name in AVAILABLE_TOOLS:
                try:
                    result = AVAILABLE_TOOLS[func_name](**func_args)
                    print(f"   📊 Result: {json.dumps(result, indent=6)}")
                except Exception as e:
                    result = {"error": str(e)}
                    print(f"   ❌ Execution error: {e}")
            else:
                result = {"error": f"Unknown function: {func_name}"}
                print(f"   ❌ Unknown function: {func_name}")

            # Add tool result to messages (Mode A format)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(result)
            })

    print("\n⚠️  Max iterations reached")
    return None

# =============================================================================
# MAIN TEST SUITE
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 PHI-4 TOOL CALLING TEST SUITE")
    print("   Testing OpenAI-compatible Mode A after 0.8.117 upgrade")
    print("="*70)

    # Setup
    MODEL_ALIAS = "phi-4-cuda-gpu"

    try:
        print(f"\n📡 Connecting to Foundry Local...")
        manager = FoundryLocalManager(MODEL_ALIAS)
        model_info = manager.get_model_info(MODEL_ALIAS)

        print(f"   ✅ Model: {model_info.id}")
        print(f"   ✅ Supports Tool Calling: {model_info.supports_tool_calling}")

        if not model_info.supports_tool_calling:
            print("\n⚠️  WARNING: Model reports tool calling not supported!")
            print("   This test may fail. Check model configuration.\n")

        client = OpenAI(
            base_url=manager.endpoint,
            api_key=manager.api_key
        )

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return

    # Run tests
    tests = [
        {
            "name": "Single Tool Call - Quiz Generation",
            "query": "Generate a quiz about photosynthesis with 3 questions"
        },
        {
            "name": "Multiple Tool Calls - Quiz + Save",
            "query": "Create a quiz about space exploration with 2 questions and save it"
        },
        {
            "name": "Different Tool - Calculator",
            "query": "What is 156 multiplied by 23?"
        },
        {
            "name": "Parallel Tools - Quiz + Math",
            "query": "Generate a 2-question quiz about mathematics and also calculate 50 divided by 5"
        }
    ]

    results = []
    for test in tests:
        try:
            result = test_tool_calling(
                client,
                model_info.id,
                test["query"],
                test["name"]
            )
            results.append({"test": test["name"], "success": result is not None})
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            results.append({"test": test["name"], "success": False, "error": str(e)})

    # Summary
    print("\n\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)

    for idx, result in enumerate(results, 1):
        status = "✅ PASS" if result.get("success") else "❌ FAIL"
        print(f"{idx}. {status} - {result['test']}")
        if "error" in result:
            print(f"   Error: {result['error']}")

    passed = sum(1 for r in results if r.get("success"))
    total = len(results)

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Mode A is working correctly.")
    elif passed > 0:
        print("\n⚠️  Some tests failed. Check the output above.")
    else:
        print("\n❌ All tests failed. Mode A may not be working.")
        print("\nTroubleshooting:")
        print("1. Verify Foundry Local version: foundry --version")
        print("2. Check model configuration in samples/python/functioncalling/README.md")
        print("3. Ensure phi-4-cuda-gpu is properly loaded")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
