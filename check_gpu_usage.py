#!/usr/bin/env python3
"""
GPU Usage Diagnostic for Foundry Local
Check if models are actually using CUDA/GPU
"""

from foundry_local import FoundryLocalManager
import json

print("="*70)
print("🔍 FOUNDRY LOCAL GPU DIAGNOSTIC")
print("="*70)

# Initialize without loading a model
manager = FoundryLocalManager(bootstrap=False)
manager.start_service()

print("\n📊 System Information:")
print(f"   Service URI: {manager.endpoint}")
print(f"   API Key: {manager.api_key}")

print("\n📦 Available Phi-4 Models:")
print("-"*70)
all_models = manager.list_catalog()
phi4_models = [m for m in all_models if 'phi-4' in m.id.lower()]

for model in phi4_models[:15]:
    print(f"\n   Alias: {model.alias}")
    print(f"   ID: {model.id}")
    print(f"   Execution Provider: {model.execution_provider}")
    if model.ep_override:
        print(f"   EP Override: {model.ep_override}")

print("\n\n💾 Currently Cached Models:")
print("-"*70)
cached = manager.list_cached_models()
for model in cached:
    if 'phi-4' in model.id.lower() or 'phi4' in model.id.lower():
        print(f"\n   ✓ {model.alias}")
        print(f"     ID: {model.id}")
        print(f"     EP: {model.execution_provider}")

print("\n\n🔥 Currently LOADED Models:")
print("-"*70)
loaded = manager.list_loaded_models()
if not loaded:
    print("   ❌ No models currently loaded")
else:
    for model in loaded:
        print(f"\n   🎯 {model.alias}")
        print(f"      ID: {model.id}")
        print(f"      Execution Provider: {model.execution_provider}")
        print(f"      EP Override: {model.ep_override if model.ep_override else 'None'}")

print("\n\n💡 RECOMMENDATIONS:")
print("="*70)

cuda_models = [m for m in phi4_models if 'cuda' in m.id.lower()]
if cuda_models:
    print(f"\n✅ CUDA models available! Try:")
    print(f"   foundry model run {cuda_models[0].alias}")
    print(f"   or in Python:")
    print(f'   MODEL_ALIAS = "{cuda_models[0].alias}"')
else:
    print("\n⚠️ No CUDA models found. You may need to download one.")

if loaded:
    for model in loaded:
        if 'cuda' not in model.execution_provider.lower():
            print(f"\n⚠️ WARNING: Model '{model.alias}' is loaded but NOT using CUDA!")
            print(f"   Current EP: {model.execution_provider}")
            print(f"   This explains why GPU utilization is 0%")
            print(f"\n   FIX:")
            print(f"   1. Unload: foundry model unload {model.alias}")
            print(f"   2. Use CUDA variant: foundry model run phi-4-cuda-gpu")

print("\n" + "="*70)
