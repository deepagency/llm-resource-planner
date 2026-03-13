# Contributing to LLM Resource Planner

First off, thank you for considering a contribution to **LLM Resource Planner**! It is people like you that make the open-source community thrive.

## 🚀 How Can I Contribute?

### Reporting Bugs
If you find a model architecture that returns unexpected values, or if the script crashes, please [open an issue](https://github.com/YOUR_USERNAME/llm-resource-planner/issues). When reporting a bug, please include:
*   The **Model ID** you were testing (e.g., `meta-llama/Llama-3.1-8B`).
*   The **expected vs. actual** output.
*   Your **Python environment** (e.g., Python 3.10, Linux/Windows).

### Improving the Heuristics
The memory estimation in this tool is based on standard transformer architecture heuristics. If you have knowledge about specific model architectures (e.g., Mixture-of-Experts or newer attention mechanisms) and want to improve the accuracy of our calculations, we welcome your Pull Requests!

### Suggesting Features
Have an idea to make this tool better? Feel free to open an issue with the "feature request" label.

## 🛠 Development Workflow

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
3.  **Make your changes** and ensure the code is clean.
4.  **Commit your changes** with a descriptive message:
    ```bash
    git commit -m "feat: add support for [model type] estimation"
    ```
5.  **Push to your branch:**
    ```bash
    git push origin feature/my-new-feature
    ```
6.  **Submit a Pull Request** against the `main` branch.

## 🤝 Code of Conduct
By participating in this project, you are expected to maintain a respectful and professional demeanor. Let's keep our community welcoming to everyone.

## 📜 License
By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).
