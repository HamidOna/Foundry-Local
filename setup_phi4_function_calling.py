#!/usr/bin/env python3
"""
Setup script to enable function calling for Phi-4 models in Foundry Local
Configures the inference_model.json file with proper prompt template
"""

import json
from pathlib import Path
from foundry_local import FoundryLocalManager

# Configuration template for function calling
FUNCTION_CALLING_CONFIG = {
    "Name": "",  # Will be filled dynamically
    "PromptTemplate": {
        "system": "<|system|>{Content}<|tool|>{Tool}<|/tool|><|end|>",
        "user": "<|user|>{Content}<|end|>",
        "assistant": "<|assistant|>{Content}<|end|>",
        "tool": "<|tool|>{Tool}<|/tool|>",
        "prompt": "<|system|> You are a helpful assistant with these tools. If you decide to call functions:\n* prefix function calls with functools marker (no closing marker required)\n* all function calls should be generated in a single JSON list formatted as functools[{\"name\": [function name], \"arguments\": [function arguments as JSON]}, ...]\n  * follow the provided JSON schema. Do not hallucinate arguments or values. Do not blindly copy values from the provided samples\n  * respect the argument type formatting. E.g., if the type is number and format is float, write value 7 as 7.0\n  * make sure you pick the right functions that match the user intent<|end|><|user|>{Content}<|end|><|assistant|>"
    }
}


def find_model_directory(cache_path: Path, model_id: str) -> Path:
    """Find the model directory in cache."""
    # Model ID format: Phi-4-cuda-gpu:1
    # Directory might be: Phi-4-cuda-gpu_1 or similar

    # Try exact match first
    model_dir = cache_path / model_id.replace(":", "_")
    if model_dir.exists():
        return model_dir

    # Search for directories containing the model name
    model_name = model_id.split(":")[0]  # Get "Phi-4-cuda-gpu"
    for directory in cache_path.iterdir():
        if directory.is_dir() and model_name.lower() in directory.name.lower():
            return directory

    raise FileNotFoundError(f"Could not find model directory for {model_id}")


def setup_function_calling(model_alias: str = "phi-4-cuda-gpu"):
    """Setup function calling configuration for Phi-4."""
    print("="*70)
    print("🔧 PHI-4 FUNCTION CALLING SETUP")
    print("="*70)

    # Initialize Foundry Local
    print("\n1️⃣ Connecting to Foundry Local...")
    manager = FoundryLocalManager(bootstrap=False)
    manager.start_service()

    # Get cache location
    cache_location = manager.get_cache_location()
    cache_path = Path(cache_location)
    print(f"   ✅ Cache location: {cache_path}")

    # Get model info
    print(f"\n2️⃣ Finding model '{model_alias}'...")
    model_info = manager.get_model_info(model_alias)
    print(f"   ✅ Model ID: {model_info.id}")
    print(f"   ✅ Execution Provider: {model_info.execution_provider}")

    # Check if model is cached
    cached_models = manager.list_cached_models()
    if model_info not in cached_models:
        print(f"\n   ⚠️  Model not cached. Downloading...")
        manager.download_model(model_alias)

    # Find model directory
    print(f"\n3️⃣ Locating model directory...")
    try:
        model_dir = find_model_directory(cache_path, model_info.id)
        print(f"   ✅ Found: {model_dir}")
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        print(f"\n   Manual setup required:")
        print(f"   1. Go to: {cache_path}")
        print(f"   2. Find folder for: {model_info.id}")
        print(f"   3. Create/edit inference_model.json")
        return

    # Create or update inference_model.json
    config_file = model_dir / "inference_model.json"
    print(f"\n4️⃣ Configuring function calling...")

    # Set model name in config
    config = FUNCTION_CALLING_CONFIG.copy()
    config["Name"] = model_info.id

    # Backup existing config if it exists
    if config_file.exists():
        backup_file = model_dir / "inference_model.json.backup"
        print(f"   📦 Backing up existing config to: {backup_file.name}")
        with open(config_file, 'r') as f:
            backup_data = f.read()
        with open(backup_file, 'w') as f:
            f.write(backup_data)

    # Write new config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"   ✅ Created: {config_file}")

    # Instructions
    print("\n" + "="*70)
    print("✅ CONFIGURATION COMPLETE!")
    print("="*70)
    print("\n📋 NEXT STEPS:")
    print("\n1. Restart Foundry Service:")
    print("   foundry service restart")
    print("\n2. (Optional) Unload the currently loaded model:")
    print(f"   foundry model unload {model_alias}")
    print("\n3. Load the model again:")
    print(f"   foundry model run {model_alias}")
    print("\n4. Run your quiz script!")
    print("\n" + "="*70)

    # Show config preview
    print("\n📄 Configuration Preview:")
    print("-"*70)
    print(json.dumps(config, indent=2))
    print("-"*70)


if __name__ == "__main__":
    import sys

    model_alias = sys.argv[1] if len(sys.argv) > 1 else "phi-4-cuda-gpu"

    try:
        setup_function_calling(model_alias)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Make sure Foundry Local is installed")
        print(f"2. Ensure the model is downloaded: foundry model download {model_alias}")
        print(f"3. Check cache location: foundry cache location")
        import traceback
        traceback.print_exc()
