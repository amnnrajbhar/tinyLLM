# TinyGPT: Aman's Personal AI Portfolio

This project is a miniature Large Language Model (LLM) built entirely from scratch using PyTorch. Instead of being trained on the entire internet, it is trained exclusively on the professional portfolio, resume, and skills of Aman Rajbhar.

It uses a character-level **Transformer architecture** (the exact same underlying math powering GPT-4 and Claude) condensed into roughly 200 lines of pure Python.

## 🚀 Features

* **Zero API Calls:** This is not a wrapper. The matrix multiplications, attention mechanisms, and backpropagation run locally on your machine.
* **Custom Dataset:** The model learns grammar, syntax, and facts purely from Aman's hardcoded portfolio data.
* **Multi-Head Self-Attention:** Implements the core $Q, K, V$ architecture that allows the model to understand the context of words.
* **Temperature & Top-K Sampling:** Includes advanced generation controls to prevent the model from outputting gibberish and to fine-tune its "creativity."
* **Interactive Mode:** After training, you can chat with the model directly in your terminal and watch it auto-complete sentences about Aman.

## 🛠️ Prerequisites

You need Python installed along with PyTorch. 

```bash
pip install torch