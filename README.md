<div align="center">

# 🧠 LLM Resource Planner

**A high-performance CLI utility for calculating VRAM requirements before you deploy.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/llm-resource-planner/graphs/commit-activity)

</div>

---

### ⚡ The Problem
Deploying LLMs is a guessing game. Downloading 50GB of weights only to hit a "CUDA Out of Memory" error is a waste of bandwidth and time. **LLM Resource Planner** eliminates the guesswork by performing static analysis on model configurations.

### 🛠 How it Works
Instead of downloading model weights, this tool fetches the `config.json` via the Hugging Face API to mathematically derive:
*   **Weight Footprint:** Based on your chosen quantization (FP16, INT8, INT4).
*   **KV Cache Requirements:** Estimating overhead at specific sequence lengths.
*   **Operational Buffer:** Factoring in activation memory and system overhead.

### 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-resource-planner.git
cd llm-resource-planner

# Install dependencies
pip install -r requirements.txt

# Execute
python main.py microsoft/Phi-3.5-mini-instruct
```

### 🔐 Authentication
For gated models (e.g., Llama-3), set your environment variable:

```bash
export HUGGINGFACE_TOKEN='your_hf_token_here'
```

### 📊 Performance Output
```text
--- Analyzing microsoft/Phi-3.5-mini-instruct ---
Estimated Parameters: ~3.82B
Memory (Weights):      7.64 GB
Memory (KV Cache @ 4k): 2.00 GB
Total Recommended VRAM: 9.77 GB
```

### 🛡 Responsible Usage Policy
This tool is built for developer research and infrastructure planning. To maintain the health of the Hugging Face API:
*   **Respect Rate Limits:** This script is for on-demand use, not for high-frequency automated scraping.
*   **Compliance:** All usage must adhere to the [Hugging Face Terms of Service](https://huggingface.co/terms-of-service).
*   **Non-Abusive Intent:** Do not integrate this into services intended for mass data harvesting.

### ⚠️ Disclaimer
Estimates are theoretical. Actual VRAM utilization varies significantly based on the inference engine (e.g., `vLLM`, `Ollama`, `TensorRT-LLM`) and system-specific optimizations. We recommend a **10–20% safety margin** in your infrastructure provisioning.

### 🤝 Contributing
Found a model architecture that behaves unexpectedly? **Open an Issue.** We welcome PRs that refine our heuristic memory models.

---
*Built with ❤️ for the open-source AI community.*
