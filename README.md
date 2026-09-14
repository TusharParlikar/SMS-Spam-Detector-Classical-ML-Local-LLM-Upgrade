# SMS Spam Detector — Classical ML + Local LLM Upgrade

A small NLP project that builds the same spam/ham classifier two ways — a classical bag-of-words + Naive Bayes model, and a zero-shot LLM classifier running locally via **Ollama** and **LangChain** — then scores both on the same test data so you can see exactly how they compare.



## Requirements

- Python 3.9+
- pip packages: `nltk`, `pandas`, `scikit-learn`, `langchain`, `langchain-ollama`
- [Ollama](https://ollama.com) installed and running — only needed for Part B (the LLM upgrade)

## Setup

### 1. Create an environment and install packages

```bash
python -m venv nlp-env
source nlp-env/bin/activate        # Windows: nlp-env\Scripts\activate

pip install nltk pandas scikit-learn langchain langchain-ollama
```

### 2. Download NLTK's data files (one time)

```python
import nltk
nltk.download(['stopwords', 'wordnet', 'punkt', 'punkt_tab'])
```

### 3. The dataset — no download step needed

`project1.md` loads the real SMS Spam Collection dataset (5,572 messages: 4,825 ham / 747 spam) directly by URL with `pandas.read_csv(url, ...)` — nothing to manually download or save. Original source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

### 4. Set up Ollama (for Part B only)

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2      # one-time model download, a few GB
ollama serve               # starts the local server if it isn't already running
```

Check it's alive with `ollama list` in a terminal, or by visiting `http://localhost:11434` in a browser.

## How to run

Work through `project1.md` top to bottom — copy each code block into a `.ipynb` notebook or a `.py` file and run it in order:

1. **Part A** — trains and evaluates the Naive Bayes classifier.
2. **Part B** — builds the zero-shot LLM classifier (requires Ollama running).
3. **Part C** — runs both on the same test set and prints a side-by-side comparison.

## Example output

**Part A (verified — this is real output from actually running the pipeline on the live dataset)**, `test_size=0.2`, `random_state=42`:

```
              precision    recall  f1-score   support

         ham     0.9826    0.9948    0.9887       965
        spam     0.9638    0.8867    0.9236       150

    accuracy                         0.9803      1115
```

**Part B (illustrative)** — this one can't be pre-verified here since it depends on Ollama running locally with whichever model you pull, so treat the shape below as a guide, not a guarantee:

```
LLM (zero-shot, LangChain + Ollama):
              precision    recall  f1-score   support
         ham       0.98      0.97      0.98       965
        spam       0.86      0.91      0.88       150
```

## Notes

- Part B can be slow on a large test set since the LLM generates a fresh reply per message — try it on a small slice (e.g. `X_test[:50]`) first.
- Everything in Part B runs locally: no API key, no data leaves your machine.
- Swapping `llama3.2` for a different pulled model is a one-line change in `project1.md`.

## Next steps

- Try a different Ollama model (`ollama pull mistral`, `ollama pull gemma2`) and see how the LLM classifier's scores shift.
- Build a hybrid classifier: run Naive Bayes first, and only send low-confidence predictions to the LLM.
- Extend the LangChain chain with `.with_structured_output()` for stricter, schema-enforced responses instead of parsing raw text.
