# SMS Spam Detector — Classical ML + Local LLM

A lightweight NLP project that compares **classical machine learning** with a **local Large Language Model (LLM)** for SMS spam detection.

The project implements two approaches:

* **Naive Bayes** with Bag-of-Words features
* **Zero-shot LLM classification** using LangChain and Ollama

Both models are evaluated on the same test data to compare their performance.

## 🚀 Features

* Text preprocessing with NLTK
* Tokenization, stopword removal, and lemmatization
* Bag-of-Words feature extraction
* Multinomial Naive Bayes classifier
* Zero-shot LLM classification
* Local LLM inference using Ollama
* LangChain integration
* Model evaluation using Accuracy, Precision, Recall, and F1-score

## 🛠️ Tech Stack

* Python 3.9+
* NLTK
* Pandas
* Scikit-learn
* LangChain
* LangChain-Ollama
* Ollama
* Llama 3.2

## 📦 Installation

### Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### Create a virtual environment

**Windows:**

```bash
python -m venv nlp-env
nlp-env\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv nlp-env
source nlp-env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install nltk pandas scikit-learn langchain langchain-ollama
```

### Download NLTK resources

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

## 🤖 Ollama Setup

Ollama is required for the local LLM component.

Install Ollama from:

[Ollama](https://ollama.com?utm_source=chatgpt.com)

Then download the model:

```bash
ollama pull llama3.2
```

Start Ollama if required:

```bash
ollama serve
```

## ▶️ Run Locally

For a Python file:

```bash
python project1.py
```

For a Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and run the cells in order.

## 📊 Evaluation

The two approaches are compared using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

This provides a direct comparison between a lightweight **classical ML model** and a **local zero-shot LLM**.

## 🔮 Future Improvements

* Hybrid Naive Bayes + LLM classifier
* Confidence-based LLM routing
* Experiment with different Ollama models
* Structured LLM outputs
* Confusion matrix visualization
* Inference-time benchmarking

## 🎯 Objective

To explore the practical differences between **traditional NLP/ML techniques and local LLM-based classification**, including their accuracy, speed, and computational requirements.
