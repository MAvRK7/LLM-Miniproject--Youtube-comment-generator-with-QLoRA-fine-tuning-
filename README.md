# 💬 LLM Mini‑Project — YouTube Comment Generator with QLoRA Fine‑Tuning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Framework](https://img.shields.io/badge/Framework-Transformers%20%7C%20PEFT-orange)]()
[![Model](https://img.shields.io/badge/Model-Mistral--7B--Instruct-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📌 Overview
This project implements a **YouTube reply generator** powered by the **Mistral‑7B‑Instruct** large language model, fine‑tuned using **QLoRA** for efficient adaptation on consumer hardware.

The model generates **contextually relevant, human‑like replies** to YouTube comments, trained on a curated dataset of real comment–reply pairs.  
For an **enhanced RAG‑based version**, see the [improved repository](https://github.com/MAvRK7/Mini-project-YouTube-comment-generator-improved-with-RAG).

---

## 🧠 Key Features
- **Base Model**: [Mistral‑7B‑Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) — instruction‑tuned LLM for high‑quality text generation.
- **Fine‑Tuning Method**: [QLoRA](https://arxiv.org/abs/2305.14314) — memory‑efficient low‑rank adaptation for large models.
- **Dataset**: [`shawhin/shawgpt-youtube-comments`](https://huggingface.co/datasets/shawhin/shawgpt-youtube-comments) from Hugging Face.
- **Libraries**: `transformers`, `peft`, `datasets`, `bitsandbytes` for 4‑bit quantization.
- **Open Source**: Fully reproducible pipeline with publicly available tools and data.

---

## 🛰 Dataset
- **Source**: Hugging Face dataset `shawhin/shawgpt-youtube-comments`
- **Content**: YouTube comments paired with human‑written replies
- **Preprocessing**:
  - Text cleaning (removing emojis, HTML tags, excessive whitespace)
  - Tokenization with model‑specific tokenizer
  - Train/validation split for fine‑tuning

---

## 🧪 Methodology

### **1. Data Preparation**
- Load dataset from Hugging Face
- Apply preprocessing and tokenization
- Format into instruction–response pairs for supervised fine‑tuning

### **2. Model Setup**
- Load **Mistral‑7B‑Instruct** in 4‑bit precision using `bitsandbytes`
- Apply **QLoRA adapters** via `peft` for parameter‑efficient fine‑tuning

### **3. Training**
- **Loss Function**: Cross‑entropy over token predictions
- **Optimizer**: AdamW with weight decay
- **Learning Rate Schedule**: Cosine decay with warmup
- **Batch Size**: Optimized for GPU memory via gradient accumulation
- **Epochs**: Tuned for convergence without overfitting

### **4. Inference**
- Input: Raw YouTube comment text
- Output: Contextually relevant, stylistically appropriate reply
- Decoding: Beam search or nucleus sampling for diversity control

---

## 📊 Example Output

**Input Comment**:  
> "This tutorial was super helpful, thanks!"

**Generated Reply**:  
> "Glad you found it useful! Let me know if you have any questions or need more examples."

---

## 🚀 Key Takeaways
- **QLoRA** enables fine‑tuning of large models like Mistral‑7B on a single GPU without sacrificing quality.
- Domain‑specific fine‑tuning yields **more relevant and engaging replies** than generic LLMs.
- The pipeline is **fully reproducible** and adaptable to other conversational domains.

---

## 📂 Repository Structure
- `train.py` — Fine‑tuning script with QLoRA integration
- `inference.py` — Script for generating replies from trained model
- `requirements.txt` — Python dependencies
- `notebooks/` — Jupyter notebooks for experimentation

---

## 💡 Skills Demonstrated
- Large Language Model fine‑tuning (QLoRA, PEFT)
- Efficient training with 4‑bit quantization
- Dataset preprocessing for conversational AI
- Hugging Face `transformers` & `datasets` integration
- Deployment‑ready inference pipeline

---

## 🛠 Installation

**Prerequisites**:
- Python 3.10+
- CUDA‑enabled GPU (for training)

---

## Workflow Diagram

<img width="210" height="630" alt="image" src="https://github.com/user-attachments/assets/a9d83be3-cad2-4d79-b690-4edc2b82940c" />

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

---

## 🙏 Acknowledgments
- Mistral AI for the base model

- Hugging Face for datasets and libraries

- shawhin/shawgpt-youtube-comments dataset



