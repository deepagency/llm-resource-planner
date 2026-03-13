import os
import requests
from transformers import AutoConfig

# ✅ Use environment variables for safety
TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def get_model_config(model_name):
    """Fetch config.json directly. Much faster and uses minimal memory."""
    # We use the official HF library to fetch just the config, not the weights.
    try:
        config = AutoConfig.from_pretrained(model_name, token=TOKEN)
        return config
    except Exception as e:
        raise ValueError(f"Could not load config for {model_name}. Check the repo name and token. Error: {e}")

def calculate_params(config):
    """Estimate parameters based on config architecture."""
    # This is an estimation logic. Most modern LLMs use these keys.
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", 4096))
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "num_layers", 32))
    
    # Heuristic for parameter estimation if not provided in config
    # This is a standard approximation for Transformer-based models
    if hasattr(config, "num_attention_heads"):
        # Very rough estimate; actual param count varies by model architecture
        params = (hidden_size**2 * num_layers * 12) / 1e9 
    else:
        params = 0 # Fallback
        
    return params, num_layers, hidden_size

def estimate_memory(params, hidden_size, num_layers, dtype="float16"):
    """
    Mathematical estimation of VRAM.
    dtype_size: 2 bytes for fp16, 0.5 bytes for 4-bit quantization.
    """
    bytes_per_param = {"fp32": 4, "float16": 2, "int8": 1, "int4": 0.5}.get(dtype, 2)
    
    # 1. Weights
    weight_mem = (params * 1e9 * bytes_per_param) / (1024**3)
    
    # 2. KV Cache (Simplified approximation)
    # 2 * hidden_size * num_layers * bytes_per_param
    kv_cache = (2 * hidden_size * num_layers * bytes_per_param * 4096) / (1024**3)
    
    total = weight_mem + (kv_cache * 1.2) # 1.2 buffer for overhead
    return weight_mem, kv_cache, total

def main(model_name):
    print(f"--- Analyzing {model_name} ---")
    config = get_model_config(model_name)
    params, layers, hidden = calculate_params(config)
    
    w, kv, total = estimate_memory(params, hidden, layers)
    
    print(f"Estimated Parameters: ~{params:.2f}B")
    print(f"Memory (Weights): {w:.2f} GB")
    print(f"Memory (KV Cache @ 4k): {kv:.2f} GB")
    print(f"Total Recommended VRAM: {total:.2f} GB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF Model ID")
    args = parser.parse_args()
    main(args.model)
