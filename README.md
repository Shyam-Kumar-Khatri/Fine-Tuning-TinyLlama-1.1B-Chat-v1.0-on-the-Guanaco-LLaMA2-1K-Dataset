# 🦙 Fine-Tuning-TinyLlama-1.1B-Chat-v1.0-on-the-Guanaco-LLaMA2-1K-Dataset

## 📜 Overview
This project demonstrates how to fine-tune the **TinyLlama/TinyLlama-1.1B-Chat-v1.0** model on the **mlabonne/guanaco-llama2-1k** dataset using **QLoRA (Quantized Low-Rank Adaptation)** inside a Google Colab environment.  
The goal is to adapt the base TinyLlama chat model to better align with the Guanaco dataset’s style and content, resulting in a more fine-tuned conversational AI.  

**Why QLoRA?**  
QLoRA extends LoRA by using **4-bit quantization**, significantly reducing the memory requirements for loading and training large models, without sacrificing performance. This means we can fine-tune billion-parameter models on a single GPU (even on free or Colab Pro tiers).  

Instead of retraining the whole model, QLoRA adds small trainable adapter layers on top of a quantized model — making training **faster**, **cheaper**, and **more memory-efficient** while preserving output quality.  

**Final Output:**  
By the end of the notebook, you’ll have a **fine-tuned TinyLlama model** stored as LoRA adapters (trained with QLoRA), along with checkpoints and logs, ready for inference or further training.


---

## 📑 Table of Contents
1. [Overview](#-overview)  
2. [Dataset](#-dataset)  
3. [Model](#-model)  
4. [Installation](#-installation)  
5. [Training Steps](#-training-steps)  
6. [Demo](#-demo)  
7. [Results](#-results)  
8. [Checkpoints](#-checkpoints)  
9. [References](#-references)  

---

## Dataset

### What is `mlabonne/guanaco-llama2-1k`?

`mlabonne/guanaco-llama2-1k` is a **small, preformatted subset (≈1000 samples)** of the broader `openassistant-guanaco` instruction-following dataset. The maintainer processed these examples so they already match **LLaMA-2 / chat-style prompt formatting**, which makes the set ready to use for chat/instruction fine-tuning experiments without additional prompt reformatting. This is especially handy for quick experiments or demos where you don't want to spend time writing conversion scripts. :contentReference[oaicite:0]{index=0}

---

### Format and fields

- The dataset is provided in a simple JSON-like / Hugging Face `datasets` format, where each example typically includes fields you can expect in instruction datasets such as:
  - an **instruction/prompt** or a `document`/`dialogue` (depending on how it was processed),
  - an **expected target/response** (the model's desired output).
- The author processed the examples specifically to conform to **LLaMA-2’s chat/prompt format**, so entries are already wrapped or structured for models that expect `<s>[INST] ... [/INST]` style or equivalent chat-format prompts. That reduces preprocessing work. :contentReference[oaicite:1]{index=1}

> Practical note: when you load the dataset with `datasets.load_dataset("mlabonne/guanaco-llama2-1k")`, inspect one example (e.g., `dataset["train"][0]`) to confirm the exact field names used (`instruction`, `input`, `output`, `prompt`, `response`, etc.). Different forks or conversions sometimes rename fields.

---

### What kind of text does it contain?

- The samples are **instruction-response pairs** derived from the OpenAssistant / Guanaco family of datasets. They represent conversational, question-answer, and instruction-following content (short dialogues, question/answer pairs, instruction-completion examples).  
- The style is **chatty / instruction-like**, not long-form articles — so it is suitable for making a chat-style assistant or improving conversational behavior.

---

### Why this dataset is suitable for fine-tuning TinyLlama

1. **Instruction-following examples:** Guanaco-style data is explicitly designed to teach models to *follow user instructions* and produce helpful responses — exactly the behaviour you usually want from a chat model. :contentReference[oaicite:2]{index=2}  
2. **Ready-to-use formatting:** This particular subset is already reformatted to the LLaMA-2 prompt style, so you save time and avoid mistakes when preparing prompts for TinyLlama. :contentReference[oaicite:3]{index=3}  
3. **Small & fast for experimentation:** At ~1k examples it’s small enough to run end-to-end demonstrations, debug training pipelines, and test QLoRA/LoRA setups quickly before scaling to larger instruction datasets.  
4. **Proven utility:** Guanaco-like datasets have been used widely for instruction-tuning and have produced solid improvement in conversational assistants — the family of Guanaco subsets has become a de-facto quick-start for many LLaMA fine-tuning tutorials. :contentReference[oaicite:4]{index=4}

---

## Model

### TinyLlama / `TinyLlama-1.1B-Chat-v1.0` — short description

- **What it is:** `TinyLlama-1.1B-Chat-v1.0` is a compact, chat-tuned variant of the TinyLlama family — a 1.1-billion parameter model designed to follow the LLaMA architecture and tokenizer conventions while being lightweight enough for practical use on modest hardware. :contentReference[oaicite:5]{index=5}  
- **Size & compute:** ~**1.1B parameters** — much smaller than the multi-billion LLMs, which makes it quick to load, run, and fine-tune for experimentation. :contentReference[oaicite:6]{index=6}  
- **Architecture & compatibility:** TinyLlama follows the *LLaMA-style* transformer architecture and uses the same or compatible tokenizer conventions (so code that works with LLaMA-style transformers typically works with TinyLlama with minimal changes). :contentReference[oaicite:7]{index=7}  
- **Purpose:** This chat variant is already adapted toward conversational behaviour (chat prompts / assistant style). It is ideal as a starting point for building specialized assistants or instruction-following models via fine-tuning. :contentReference[oaicite:8]{index=8}

---

### Why TinyLlama is a good fit for your project

- **Small & fast:** 1.1B parameters are manageable on consumer GPUs (with quantization/QLoRA), enabling faster iteration and dramatically lower cost compared to 7B+ or 13B models. :contentReference[oaicite:9]{index=9}  
- **Good baseline for chat:** The “Chat” variant already includes chat-style conditioning, so you typically need fewer training steps to see improvements on conversational tasks. :contentReference[oaicite:10]{index=10}  
- **Ecosystem & community support:** TinyLlama has active community support, adapter/quantized files, and example notebooks (so you’ll find reference code and pretrained variants you can reuse). :contentReference[oaicite:11]{index=11}

---

### How this interacts with QLoRA / LoRA

- Because TinyLlama is relatively small, it **works extremely well** with parameter-efficient approaches like **LoRA** and **QLoRA**: you can train quality adapters quickly and keep base model files small while still achieving meaningful improvements in behavior. If you quantize the base model and train LoRA adapters on top (QLoRA), you get the best tradeoff: very low GPU memory usage, minimal disk for adapters, and fast turnaround. :contentReference[oaicite:12]{index=12}

---

## 🛠️ Installation & Requirements

This project requires **Python 3.8+** and makes heavy use of **Hugging Face Transformers**, **PEFT**, and **bitsandbytes** for **quantized (QLoRA) fine-tuning** of large language models.  
It is designed to run on **Google Colab** or a local machine with a **GPU (at least 12GB VRAM)** for efficient training.

### 🔧 **Environment Setup**

1. **Clone the repository (if hosted on GitHub):**
    ```bash
    git clone https://github.com/<your-username>/Fine-Tuning-TinyLlama-1.1B-Chat-v1.0-on-Guanaco-LLaMA2-1K.git
    cd Fine-Tuning-TinyLlama-1.1B-Chat-v1.0-on-Guanaco-LLaMA2-1K
    ```

2. **Create and activate a virtual environment (recommended for local runs):**
    ```bash
    python -m venv venv
    source venv/bin/activate       # For Linux/MacOS
    venv\Scripts\activate          # For Windows
    ```

3. **Install all dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

If you’re running in **Google Colab**, you don’t need a virtual environment — just make sure to run the installation cells in the notebook.

---

### 📦 **Required Libraries**

This project uses the following core libraries:

- **torch** – Deep learning framework for training and inference  
- **transformers** – Hugging Face library for TinyLlama model and tokenizers  
- **datasets** – For loading and managing the Guanaco dataset  
- **accelerate** – For optimized multi-GPU/TPU/CPU usage  
- **peft** – Parameter-Efficient Fine-Tuning (LoRA & QLoRA)  
- **bitsandbytes** – For 4-bit quantization to reduce GPU memory usage  
- **trl** – Training reinforcement learning models with LLMs (used for training loop utilities)  
- **pandas, numpy** – Data handling and preprocessing  
- **matplotlib, seaborn** – Visualization of training metrics  
- **google-colab (optional)** – For running the notebook online  

Alternatively, you can install manually without a `requirements.txt`:

```bash
pip install torch transformers datasets accelerate peft bitsandbytes trl pandas numpy matplotlib seaborn
```

---

## 🚀 Usage / How to Run

Follow these steps to run the notebook and fine-tune **TinyLlama/TinyLlama-1.1B-Chat-v1.0** on the **mlabonne/guanaco-llama2-1k** dataset:

1. **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/<your-username>/Fine-Tuning-TinyLlama-1.1B-Chat-v1.0-on-Guanaco-LLaMA2-1K.git
    cd Fine-Tuning-TinyLlama-1.1B-Chat-v1.0-on-Guanaco-LLaMA2-1K
    pip install -r requirements.txt
    ```

2. **Open the Jupyter Notebook or Google Colab:**
    ```bash
    jupyter notebook
    ```
    Open `Fine_Tuning_TinyLlama_1_1B_Chat_v1_0_on_the_Guanaco_LLaMA2_1K_Dataset.ipynb` in Jupyter, or upload it to **Google Colab**.

3. **Run the notebook step-by-step:**
    - Load the **Guanaco LLaMA2-1K** dataset from Hugging Face  
    - Preprocess and tokenize the dataset  
    - Set up **QLoRA** configuration for parameter-efficient fine-tuning  
    - Fine-tune the **TinyLlama model** on the dataset  
    - Save and export adapters & checkpoints  
    - Evaluate model outputs on custom prompts  

4. **(Optional) Use Colab GPU for faster training:**
    - Go to **Runtime > Change Runtime Type > GPU**  
    - Preferably select **T4** or **A100** GPU for best performance with QLoRA

---

