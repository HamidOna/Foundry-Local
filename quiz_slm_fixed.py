#!/usr/bin/env python3
"""
MULTI-AGENT QUIZ SYSTEM - Automatic Function Calling Demo
==========================================================
Educational demonstration of:
1. Automatic function calling with Foundry Local (functools format)
2. Multi-agent orchestration pattern

Model: Phi-4
Format: functools[{"name": "...", "arguments": {...}}, ...]

Prerequisites:
1. foundry model run phi-4-mini
2. Configure inference_model.json for function calling (see README)
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI

try:
    from foundry_local import FoundryLocalManager
    USING_FOUNDRY = True
except ImportError:
    USING_FOUNDRY = False
    print("⚠️ foundry-local-sdk not installed. Using manual endpoint config.")

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ALIAS = "phi-4"
OUTPUT_DIR = Path("quiz_data")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

def create_quiz(source_text: str, num_questions: int) -> Dict[str, Any]:
    """Generate quiz questions from source text."""
    print(f"   ⚡ [create_quiz] Generating {num_questions} questions...")

    sentences = [s.strip() for s in source_text.split('.') if len(s.strip()) > 10]
    questions = []

    for i in range(min(num_questions, max(1, len(sentences)))):
        topic = sentences[i % len(sentences)][:50] if sentences else "the content"
        questions.append({
            "id": i + 1,
            "question": f"What is the significance of: '{topic}...'?",
            "options": ["A) Key concept", "B) Not mentioned", "C) Minor detail"],
            "correct_answer": "A"
        })

    result = {"questions": questions, "count": len(questions)}
    print(f"   ✅ Created {len(questions)} questions")
    return result


def save_quiz(quiz_data: Dict[str, Any]) -> Dict[str, Any]:
    """Save quiz to disk and return quiz_id."""
    print(f"   ⚡ [save_quiz] Saving quiz...")

    if not quiz_data or "questions" not in quiz_data:
        return {"error": "Invalid quiz_data"}

    quiz_id = f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filepath = OUTPUT_DIR / f"{quiz_id}.json"

    with open(filepath, 'w') as f:
        json.dump({"quiz_id": quiz_id, "questions": quiz_data["questions"]}, f, indent=2)

    print(f"   ✅ Saved: {filepath}")
    return {"quiz_id": quiz_id, "path": str(filepath)}


def load_quiz(quiz_id: str) -> Dict[str, Any]:
    """Load quiz from disk."""
    print(f"   ⚡ [load_quiz] Loading {quiz_id}...")

    filepath = OUTPUT_DIR / f"{quiz_id}.json"
    if not filepath.exists():
        return {"error": f"Quiz not found: {quiz_id}"}

    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"   ✅ Loaded {len(data.get('questions', []))} questions")
    return data


def grade_responses(quiz_id: str, responses: List[str]) -> Dict[str, Any]:
    """Grade user responses against correct answers."""
    print(f"   ⚡ [grade_responses] Grading {quiz_id}...")

    quiz = load_quiz(quiz_id)
    if "error" in quiz:
        return quiz

    questions = quiz.get("questions", [])
    score = 0
    details = []

    for i, q in enumerate(questions):
        user_ans = responses[i].upper()[0] if i < len(responses) else "X"
        correct = q["correct_answer"][0]
        is_correct = user_ans == correct
        if is_correct:
            score += 1
        details.append({"q": i+1, "user": user_ans, "correct": correct, "result": "✓" if is_correct else "✗"})

    result = {"quiz_id": quiz_id, "score": score, "total": len(questions), "details": details}
    print(f"   ✅ Score: {score}/{len(questions)}")
    return result


def create_report(quiz_id: str, grading_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create markdown report of results."""
    print(f"   ⚡ [create_report] Creating report...")

    if not grading_results or "score" not in grading_results:
        return {"error": "Invalid grading results"}

    filepath = OUTPUT_DIR / f"{quiz_id}_report.md"
    content = f"""# Quiz Report: {quiz_id}
**Score:** {grading_results['score']}/{grading_results['total']}

## Details
| Q | Your Answer | Correct | Result |
|---|-------------|---------|--------|
"""
    for d in grading_results.get("details", []):
        content += f"| {d['q']} | {d['user']} | {d['correct']} | {d['result']} |\n"

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"   ✅ Report: {filepath}")
    return {"path": str(filepath)}


# Tool registry
TOOLS = {
    "create_quiz": create_quiz,
    "save_quiz": save_quiz,
    "load_quiz": load_quiz,
    "grade_responses": grade_responses,
    "create_report": create_report
}

# Tool schemas (CORRECTED OpenAI format)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "create_quiz",
            "description": "Generate quiz questions from source text",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_text": {
                        "type": "string",
                        "description": "The text to create questions from"
                    },
                    "num_questions": {
                        "type": "integer",
                        "description": "Number of questions to generate"
                    }
                },
                "required": ["source_text", "num_questions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_quiz",
            "description": "Save quiz data to disk and get a quiz_id",
            "parameters": {
                "type": "object",
                "properties": {
                    "quiz_data": {
                        "type": "object",
                        "description": "Quiz data with questions array from create_quiz"
                    }
                },
                "required": ["quiz_data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_quiz",
            "description": "Load a saved quiz by ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "quiz_id": {
                        "type": "string",
                        "description": "The quiz identifier"
                    }
                },
                "required": ["quiz_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grade_responses",
            "description": "Grade user responses for a quiz",
            "parameters": {
                "type": "object",
                "properties": {
                    "quiz_id": {
                        "type": "string",
                        "description": "The quiz identifier"
                    },
                    "responses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of user answers like ['A', 'B', 'C']"
                    }
                },
                "required": ["quiz_id", "responses"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_report",
            "description": "Create a markdown report of quiz results",
            "parameters": {
                "type": "object",
                "properties": {
                    "quiz_id": {
                        "type": "string",
                        "description": "The quiz identifier"
                    },
                    "grading_results": {
                        "type": "object",
                        "description": "Results from grade_responses"
                    }
                },
                "required": ["quiz_id", "grading_results"]
            }
        }
    }
]


# =============================================================================
# FUNCTOOLS PARSER & EXECUTOR
# =============================================================================

def extract_functools(response: str) -> Optional[List[Dict]]:
    """
    Extract function calls from Phi-4's functools format.
    Format: functools[{"name": "...", "arguments": {...}}, ...]
    """
    if not response:
        return None

    match = re.search(r'functools\s*\[(.*)\]', response, re.DOTALL)
    if not match:
        return None

    try:
        calls = json.loads(f"[{match.group(1)}]")
        return calls if calls else None
    except json.JSONDecodeError:
        return None


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with given arguments."""
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}

    try:
        return TOOLS[name](**arguments)
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}


# =============================================================================
# AGENT CLASS
# =============================================================================

class Agent:
    """
    Agent with automatic function calling using Phi-4's functools format.
    """

    def __init__(self, name: str, tools: List[str], client: OpenAI, model_id: str):
        self.name = name
        self.client = client
        self.model_id = model_id
        self.tool_schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in tools]

    def run(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with automatic function calling."""
        print(f"\n{'='*60}")
        print(f"🤖 [{self.name}]")
        print(f"{'='*60}")

        messages = [{"role": "user", "content": task}]
        max_iterations = 6

        for iteration in range(max_iterations):
            print(f"\n📍 Iteration {iteration + 1}")

            # Call model with tools
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=self.tool_schemas,
                temperature=0.00001,
                max_tokens=2048,
                top_p=1.0
            )

            content = response.choices[0].message.content or ""
            print(f"   Response: {content[:250]}{'...' if len(content) > 250 else ''}")

            # Extract functools calls
            func_calls = extract_functools(content)

            if not func_calls:
                print(f"   ✅ No functools - task complete")
                context["summary"] = content
                break

            messages.append({"role": "assistant", "content": content})

            # Execute each function call automatically
            for call in func_calls:
                tool_name = call.get("name")
                arguments = call.get("arguments", {})

                print(f"\n   🔧 FUNCTOOLS CALL: {tool_name}")
                print(f"      Args: {json.dumps(arguments)[:200]}")

                result = execute_tool(tool_name, arguments)

                # Update context
                if tool_name == "create_quiz" and "questions" in result:
                    context["quiz_data"] = result
                elif tool_name == "save_quiz" and "quiz_id" in result:
                    context["quiz_id"] = result["quiz_id"]
                elif tool_name == "grade_responses" and "score" in result:
                    context["grading_results"] = result
                elif tool_name == "create_report" and "path" in result:
                    context["report_path"] = result["path"]

                print(f"      ✅ Result: {json.dumps(result)[:150]}")

                # Feed result back
                messages.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' returned: {json.dumps(result)}"
                })

        return context


# =============================================================================
# COORDINATOR
# =============================================================================

class QuizCoordinator:
    """
    Multi-agent coordinator for quiz workflow.

    Agents:
        QuizMaster → create_quiz, save_quiz
        Grader → load_quiz, grade_responses, create_report
    """

    def __init__(self):
        if USING_FOUNDRY:
            print(f"🔄 Connecting to Foundry Local ({MODEL_ALIAS})...")
            manager = FoundryLocalManager(MODEL_ALIAS)
            model_info = manager.get_model_info(MODEL_ALIAS)

            self.client = OpenAI(
                base_url=manager.endpoint,
                api_key=manager.api_key
            )
            self.model_id = model_info.id
            print(f"✅ Connected: {self.model_id}")
        else:
            self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
            self.model_id = MODEL_ALIAS

        self.quiz_master = Agent("QuizMaster", ["create_quiz", "save_quiz"], self.client, self.model_id)
        self.grader = Agent("Grader", ["load_quiz", "grade_responses", "create_report"], self.client, self.model_id)

    def run(self, source_text: str, num_questions: int = 3):
        """Run complete quiz workflow."""

        print("\n" + "="*70)
        print("🎓 MULTI-AGENT QUIZ SYSTEM")
        print("   Phi-4-mini | functools[] | Automatic Function Calling")
        print("="*70)

        # Phase 1: QuizMaster
        print("\n📝 PHASE 1: Quiz Creation")
        task = f"""Create a quiz with {num_questions} questions from this text and save it.

TEXT: {source_text}

Call create_quiz first, then save_quiz with the result."""

        context = self.quiz_master.run(task, {})

        quiz_id = context.get("quiz_id")
        if not quiz_id:
            print("❌ Failed to create quiz")
            return

        # Phase 2: User takes quiz
        print("\n" + "="*70)
        print(f"📋 PHASE 2: Take Quiz ({quiz_id})")
        print("="*70)

        for q in context.get("quiz_data", {}).get("questions", []):
            print(f"\n{q['id']}. {q['question']}")
            for opt in q["options"]:
                print(f"   {opt}")

        answers = input("\n📝 Your answers (e.g., A,B,C): ").strip()
        responses = [a.strip().upper() for a in answers.split(",")]

        # Phase 3: Grader
        print("\n" + "="*70)
        print("📊 PHASE 3: Grading")
        print("="*70)

        task = f"""Grade quiz "{quiz_id}" with responses {json.dumps(responses)} and create a report.

Call grade_responses first, then create_report with the results."""

        context = self.grader.run(task, {"quiz_id": quiz_id, "responses": responses})

        # Results
        print("\n" + "="*70)
        print("🏆 RESULTS")
        print("="*70)

        results = context.get("grading_results", {})
        print(f"\n📊 Score: {results.get('score', 0)}/{results.get('total', 0)}")
        if context.get("report_path"):
            print(f"📄 Report: {context['report_path']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 FOUNDRY LOCAL - FUNCTION CALLING DEMO")
    print("   Model: Phi-4-mini")
    print("   Format: functools[{...}]")
    print("="*70)

    coordinator = QuizCoordinator()

    print("\n📖 Paste source text (Enter twice to finish):")
    print("-"*40)

    lines = []
    while True:
        try:
            line = input()
            if not line and lines:
                break
            lines.append(line)
        except EOFError:
            break

    source_text = "\n".join(lines)
    if not source_text.strip():
        print("❌ No text provided")
        return

    num_q = input("Number of questions (default 3): ").strip()
    num_questions = int(num_q) if num_q.isdigit() else 3

    coordinator.run(source_text, num_questions)
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
