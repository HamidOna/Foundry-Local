# Testing Mode A (Automatic Tool Calling)

Quick guide to test Mode A tool calling with Foundry Local.

## What is Mode A?

**Mode A** = OpenAI's standard automatic tool calling:
- Uses `message.tool_calls` attribute (structured API response)
- Returns tool calls as objects, not text
- Requires `tool_choice="auto"` parameter
- Returns results with `role="tool"` and `tool_call_id`

## Prerequisites

```bash
pip install openai foundry-local-sdk
```

Make sure Foundry Local is running with a model that supports tool calling.

## Quick Start

### 1. Simple Mode A Test

```bash
python test_mode_a.py
```

This will test:
- Single tool call (weather lookup)
- Multiple parallel tool calls
- Math operations

**What to look for:**
- ✅ `message.tool_calls` is populated (Mode A working)
- ❌ `message.tool_calls` is None (Mode A not supported)

### 2. Compare Mode A vs Mode B

```bash
python compare_modes.py
```

This shows the difference between:
- **Mode A**: Structured `tool_calls` response
- **Mode B**: Text-based `functools[...]` parsing

## Expected Behavior

### If Mode A Works:
```python
response.choices[0].message.tool_calls = [
    {
        "id": "call_abc123",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "Paris"}'
        }
    }
]
```

### If Mode B (Functools):
```
functools[{"name": "get_weather", "arguments": {"location": "Paris"}}]
```

### If Neither Works:
The model doesn't support tool calling - check:
```python
model_info.supports_tool_calling  # Should be True
```

## Configuration

Edit the scripts to change the model:

```python
MODEL_ALIAS = "phi-4-mini"  # Change this
```

## Key Differences

| Feature | Mode A | Mode B |
|---------|--------|--------|
| **Format** | Structured API response | Text parsing |
| **Attribute** | `message.tool_calls` | Parse from `message.content` |
| **Tool Results** | `role="tool"` + `tool_call_id` | Custom format |
| **Standard** | OpenAI compatible | Model-specific |
| **Phi-4 Support** | ❌ Experimental | ✅ Production |

## Troubleshooting

### "tool_calls is None"
Mode A not supported by this model. Use Mode B (functools) instead.

### "No functools pattern found"
Mode B not configured. Check your model's prompt template configuration.

### "Model doesn't support tool calling"
Check `model_info.supports_tool_calling` flag. May need to:
1. Update model configuration
2. Use a different model
3. Wait for Mode A support in next release

## Next Steps

Once you determine which mode works:

- **Mode A works**: Use the `quiz_slm_working.py` pattern
- **Mode B works**: Use the `fl_tools..ipynb` pattern
- **Neither works**: Check model configuration or use a different model

## Questions?

- Check `/samples/python/functioncalling/` for working examples
- Review model configuration in Foundry Local settings
- Test with `supportsToolCalling` flag first
