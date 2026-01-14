#!/usr/bin/env python3
"""
Merge CoPE-A LoRA adapter with base Gemma-2-9b model.
Creates a standalone model that vLLM can load directly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL_ID = "google/gemma-2-9b"
ADAPTER_MODEL_ID = "zentropi-ai/cope-a-9b"
OUTPUT_DIR = os.path.expanduser("~/cope_inference/cope-merged")

def main():
    print(f"Loading base model: {BASE_MODEL_ID}")
    print("This will load in FP16 for merging (uses ~18GB VRAM temporarily)")
    
    # Load base model in FP16 for clean merge
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )
    
    print(f"Loading and applying adapter: {ADAPTER_MODEL_ID}")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_MODEL_ID,
        torch_dtype=torch.float16,
    )
    
    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Done! Merged model saved.")
    print(f"\nModel location: {OUTPUT_DIR}")
    print("You can now use this with vLLM.")

if __name__ == "__main__":
    main()
