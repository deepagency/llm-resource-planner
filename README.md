<div align="center">

# 🧠 LLM Resource Planner

**Estimate GPU VRAM requirements for Hugging Face LLMs without downloading model weights.**

The **LLM Resource Planner** is a lightweight Python CLI tool that analyzes Hugging Face model configurations and estimates the GPU memory required for inference.

It enables developers to perform **AI infrastructure planning** before downloading large model checkpoints.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/deepagency/llm-resource-planner/graphs/commit-activity)

</div>

---

### 🚀 Quick Start

#### Install

```bash
pip install llm-resource-planner
```

#### Run the planner

```bash
llm-plan microsoft/Phi-3.5-mini-instruct
```

Example output:

```text
--- Analyzing microsoft/Phi-3.5-mini-instruct ---
Estimated Parameters: ~3.62B
Memory (Weights): 6.75 GB
Memory (KV Cache @ 4k): 1.50 GB
Total Recommended VRAM: 8.55 GB
```

---

### Example Models

```bash
llm-plan meta-llama/Meta-Llama-3-8B
llm-plan mistralai/Mistral-7B-Instruct
llm-plan microsoft/Phi-3.5-mini-instruct
```

---

### CLI Usage

Show command help:

```bash
llm-plan --help
```

Basic usage:

```bash
llm-plan <huggingface-model-id>
```

Example:

```bash
llm-plan meta-llama/Meta-Llama-3-8B
```

---

### What the Tool Does

The planner retrieves a model's **configuration metadata** from Hugging Face using:

```
transformers.AutoConfig
```

It extracts architectural parameters such as:

- hidden size
- number of transformer layers
- number of attention heads

Using these values, the tool estimates:

1. **Model parameter count**
2. **Memory required for model weights**
3. **Memory required for the attention KV cache**
4. **A buffered VRAM estimate for inference**

This analysis occurs **without downloading model weights**.

---

### Estimation Method

The tool uses a heuristic approximation commonly applied to transformer architectures.

#### Parameter Count Estimate

```
params ≈ hidden_size² × num_layers × 12
```

This approximates the parameter count for standard transformer blocks.

---

#### Weight Memory

```
weight_memory = params × dtype_bytes
```

Where precision is assumed to be:

| Precision | Bytes |
| --------- | ----- |
| FP32      | 4     |
| FP16      | 2     |
| INT8      | 1     |
| INT4      | 0.5   |

(Current CLI defaults to FP16.)

---

#### KV Cache Estimate

The KV cache memory is approximated as:

```
kv_cache = 2 × hidden_size × num_layers × bytes_per_param × context_length
```

The current implementation assumes:

```
context_length = 4096
```

---

#### Recommended VRAM

A safety margin is applied:

```
total_vram ≈ weight_memory + (kv_cache × 1.2)
```

This accounts for runtime memory overhead.

---

#### Authentication

Some Hugging Face models require authentication.

Set your Hugging Face token:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

The planner will automatically use the token when retrieving model metadata.

---

### Development Installation

Clone the repository:

```bash
git clone https://github.com/deepagency/llm-resource-planner.git
cd llm-resource-planner
```

Install in editable mode:

```bash
pip install -e .
```

Run the tool:

```bash
llm-plan microsoft/Phi-3.5-mini-instruct
```

---

### Assumptions and Limitations

This tool provides **heuristic estimates**.

Results may differ depending on:

- inference engine (`vLLM`, `Ollama`, `TensorRT-LLM`, etc.)
- batching strategies
- runtime graph optimizations
- GPU memory fragmentation
- custom model architectures

The estimator is primarily designed for **standard transformer architectures**.

For production deployments, maintain a **10–20% safety margin**.

---

### 🤝 Contributing

Contributions are welcome.

If you discover:

- models producing inaccurate estimates
- improved parameter estimation heuristics
- support for additional architectures

please open an **Issue** or submit a **Pull Request**.

See **CONTRIBUTING.md** for development guidelines.

---

### 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

<div align="center">

Built for the open-source AI community.

</div>
