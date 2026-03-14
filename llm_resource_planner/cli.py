import os
import argparse
from transformers import AutoConfig

TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


def get_model_config(model_name):
    try:
        config = AutoConfig.from_pretrained(model_name, token=TOKEN)
        return config
    except Exception as e:
        raise ValueError(
            f"Could not load config for {model_name}. Check the repo name and token. Error: {e}"
        )


def calculate_params(config):
    """
    Estimate parameter count from transformer architecture metadata.
    We rely only on hidden size and layer count because these fields
    are present across nearly all Hugging Face transformer configs.
    """

    hidden_size = (
        getattr(config, "hidden_size", None)
        or getattr(config, "d_model", None)
        or getattr(config, "n_embd", None)
        or getattr(config, "dim", None)
    )

    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
        or getattr(config, "n_layers", None)
    )

    if hidden_size is None or num_layers is None:
        raise ValueError(
            "Unsupported model architecture: missing hidden_size or layer count in config."
        )

    # Standard transformer block approximation
    params = (hidden_size**2 * num_layers * 12) / 1e9

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

    total = weight_mem + (kv_cache * 1.2)  # 1.2 buffer for overhead
    return weight_mem, kv_cache, total


def main(model_name, gpu=None):
    print(f"--- Analyzing {model_name} ---")
    config = get_model_config(model_name)
    params, layers, hidden = calculate_params(config)

    w, kv, total = estimate_memory(params, hidden, layers)

    print(f"Estimated Parameters: ~{params:.2f}B")
    print(f"Memory (Weights): {w:.2f} GB")
    print(f"Memory (KV Cache @ 4k): {kv:.2f} GB")
    print(f"Total Recommended VRAM: {total:.2f} GB")
    if gpu is not None:
        print(f"\nGPU Memory Provided: {gpu:.2f} GB")

        if total <= gpu:
            print("✔ Model should fit in available VRAM")
        else:
            print("✖ Model likely exceeds available VRAM")


def cli():
    """CLI entrypoint used by setuptools."""
    parser = argparse.ArgumentParser(
        description="Estimate LLM VRAM requirements and optionally check GPU fit"
    )
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument(
        "--gpu",
        type=float,
        help="Available GPU VRAM in GB to check if the model fits",
    )
    args = parser.parse_args()
    main(args.model, args.gpu)


if __name__ == "__main__":
    cli()
