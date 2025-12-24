# Testing Phi-4 Tool Calling After 0.8.117 Upgrade

This guide helps you test that tool calling works correctly after upgrading to Foundry Local 0.8.117.

## Prerequisites

1. **Upgrade to 0.8.117 or higher:**
   ```bash
   foundry --version  # Should show 0.8.117+
   ```

2. **Install dependencies:**
   ```bash
   pip install openai foundry-local-sdk
   ```

3. **Ensure Phi-4 is loaded:**
   ```bash
   foundry model ls | grep phi-4
   ```

## Quick Test (30 seconds)

Run the minimal test to verify Mode A works:

```bash
python quick_test_toolcalling.py
```

**What to look for:**
- ✅ `tool_calls: [ChatCompletionMessageToolCall(...)]` → Mode A working!
- ❌ `tool_calls: None` → Mode A not working, see troubleshooting

## Full Test Suite (2 minutes)

Run comprehensive tests with multiple scenarios:

```bash
python test_phi4_toolcalling.py
```

This tests:
1. Single tool call (quiz generation)
2. Multiple tool calls (generate + save)
3. Different tools (calculator)
4. Parallel tool execution

**Expected output:**
```
🎯 Results: 4/4 tests passed
🎉 All tests passed! Mode A is working correctly.
```

## What Changed in 0.8.117?

From the release notes (0.8.115-0.8.117):
- ✅ Fixed #346: Tool calling now returns results in streaming mode
- ✅ Fixed #341: Exception handling for network disconnections
- ✅ Fixed #335: Guidance error with `tool_choice=required`
- ✅ Fixed #336: Enforcing "required" field validation

## Understanding the Response

### Mode A Working (Expected):
```python
response.choices[0].message.tool_calls = [
    ChatCompletionMessageToolCall(
        id='call_abc123',
        function=Function(
            name='generate_quiz',
            arguments='{"topic": "space"}'
        )
    )
]
```

### Mode A Not Working:
```python
response.choices[0].message.tool_calls = None
response.choices[0].message.content = 'functools[{"name": "generate_quiz", ...}]'
```

## Troubleshooting

### Issue: `tool_calls` is None

**Check 1: Version**
```bash
foundry --version
# Must be 0.8.117 or higher
```

**Check 2: Model supports tool calling**
```python
from foundry_local import FoundryLocalManager
manager = FoundryLocalManager("phi-4-cuda-gpu")
model_info = manager.get_model_info("phi-4-cuda-gpu")
print(model_info.supports_tool_calling)  # Should be True
```

**Check 3: Tool schema format**
Ensure your tools follow this exact format:
```python
{
    "type": "function",  # ← Required wrapper
    "function": {
        "name": "function_name",
        "description": "What it does",
        "parameters": {
            "type": "object",  # ← Required
            "properties": { ... },
            "required": ["param1"]  # ← Required array
        }
    }
}
```

**Check 4: Restart service**
```bash
foundry service restart
```

### Issue: "functools" appears in response content

This means the server is NOT converting functools to tool_calls.

**Possible causes:**
1. Phi-4 prompt template not configured (see `/samples/python/functioncalling/README.md`)
2. Running an older version than 0.8.117
3. Model doesn't support automatic conversion (wait for tomorrow's release)

**Workaround:** Use Mode B parser (manual functools parsing)

### Issue: Model not found

```bash
# Load the model first
foundry model run phi-4-cuda-gpu

# Or check available models
foundry model ls
```

## Integration with Your Code

Once tests pass, your existing code should work:

```python
from openai import OpenAI
from foundry_local import FoundryLocalManager

# Your setup
manager = FoundryLocalManager("phi-4-cuda-gpu")
client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
model_id = manager.get_model_info("phi-4-cuda-gpu").id

# Your tool calling code
response = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "Generate a quiz"}],
    tools=YOUR_TOOLS,
    tool_choice="auto"
)

# Mode A: This should work now
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        # Execute tools
        ...
```

## Next Steps

1. **Run quick test** → Verify Mode A works
2. **Run full test suite** → Validate all scenarios
3. **Integrate with your app** → Use your BaseAgent code
4. **Wait for tomorrow's release** → Additional improvements coming

## Questions?

- Check logs: `foundry service logs`
- Verify endpoint: `echo $FOUNDRY_LOCAL_ENDPOINT` or check manager.endpoint
- Compare with working examples in `/samples/python/functioncalling/`

---

**Expected Timeline:**
- Now (0.8.117): Bug fixes, Mode A should work
- Tomorrow: Enhanced function calling features (per your info)

Good luck with testing! 🚀
