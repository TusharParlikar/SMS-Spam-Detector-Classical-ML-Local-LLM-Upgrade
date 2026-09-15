# 📩 SMS Spam Detector — Classical ML vs. Local LLM

A lightweight NLP project comparing **classical machine learning** with a **local Large Language Model (LLM)** for SMS spam detection.

Two approaches are evaluated on the same test data:

| Approach | Technique |
|---|---|
| **Classical ML** | Multinomial Naive Bayes + Bag-of-Words |
| **Local LLM** | Zero-shot classification via LangChain + Ollama (Llama 3.2) |

---

## 📑 Table of Contents

- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Ollama Setup](#-ollama-setup)
- [Run Locally](#️-run-locally)
- [Evaluation](#-evaluation)
- [Future Improvements](#-future-improvements)
- [Objective](#-objective)

---

## 🚀 Features

- Text preprocessing with NLTK
- Tokenization, stopword removal, and lemmatization
- Bag-of-Words feature extraction
- Multinomial Naive Bayes classifier
- Zero-shot LLM classification
- Local LLM inference using Ollama
- LangChain integration
- Model evaluation using Accuracy, Precision, Recall, and F1-score

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| NLP | NLTK |
| Data | Pandas |
| Classical ML | Scikit-learn |
| LLM Orchestration | LangChain, LangChain-Ollama |
| Local Inference | Ollama, Llama 3.2 |

---

## 📁 Project Structure

```
sms-spam-detector/
├── data/
│   ├── raw/
│   │   └── sms_spam_collection.csv
│   └── processed/
│       └── cleaned_data.csv
├── notebooks/
│   └── project1.ipynb
├── src/
│   ├── preprocessing.py        # tokenization, stopword removal, lemmatization
│   ├── naive_bayes_model.py    # BoW + Multinomial Naive Bayes
│   ├── llm_classifier.py       # LangChain + Ollama zero-shot classifier
│   └── evaluate.py             # accuracy, precision, recall, F1
├── models/
│   └── naive_bayes.pkl
├── results/
│   └── metrics_comparison.csv
├── project1.py                 # entry point (script version)
├── requirements.txt
└── README.md
```

> Adjust folder/file names to match your actual repo — this tree reflects the components the project already describes.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv nlp-env
nlp-env\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv nlp-env
source nlp-env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install nltk pandas scikit-learn langchain langchain-ollama
```

### 4. Download NLTK resources

Run once in Python:

```python
import nltk

nltk.download([
    "stopwords",
    "wordnet",
    "punkt",
    "punkt_tab"
])
```

---

## 🤖 Ollama Setup

Ollama is required for the local LLM component.

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull the model:

   ```bash
   ollama pull llama3.2
   ```

3. Start the Ollama server if it isn't already running:

   ```bash
   ollama serve
   ```

---

## ▶️ Run Locally

**Python script**

```bash
python project1.py
```

**Jupyter Notebook**

```bash
jupyter notebook
```

Open the project notebook and run the cells in order.

---

## 📊 Evaluation

Both models are scored on the same held-out test set using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

This gives a direct, apples-to-apples comparison between a lightweight classical ML model and a local zero-shot LLM.

---

## 🔮 Future Improvements

- Hybrid Naive Bayes + LLM classifier
- Confidence-based LLM routing
- Experiment with different Ollama models
- Structured LLM outputs
- Confusion matrix visualization
- Inference-time benchmarking

---

## 🎯 Objective

To explore the practical differences between traditional NLP/ML techniques and local LLM-based classification — including accuracy, speed, and computational requirements.
