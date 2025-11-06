# 🧠 Sentimental Analysis in Product Reviews: ABSA with Transformers & LLMs

**Author:** Ved Ghuikhedkar

## Overview

This repository is a full showcase of my academic-industry project on **Aspect-Based Sentiment Analysis (ABSA)**, focusing specifically on laptop reviews. With the recent evolution of natural language models, the goal was to experiment with different transformer architectures (DistilBERT, DeBERTa) and cutting-edge LLMs (LLaMA3) for more nuanced, interpretable customer insights.

I designed, fine-tuned, and benchmarked models, comparing **classic transformers** and **large language models** (LLMs) in both few-shot and zero-shot learning scenarios. Special emphasis is placed on **model interpretability** using SHAP to visualize and trust predictions.

---

## 🚀 What’s Inside?

- **Jupyter Notebooks:** Step-by-step code from data cleaning, model training, to evaluation and interpretability.
- **PDF Report:** “Decoding the Interpretation of Aspect-Based Sentiments in Laptop Reviews using Transformer Models and LLMs.”
- **Custom Scripts:** For dataset preparation, transforming and encoding aspect categories, and running benchmarks.
- **Results & Interpretability Visuals:** SHAP word attributions, comparison to LLM highlighted tokens, charts/tables.
- **Requirements:** All dependencies are listed for easy setup.

---

## 🛠️ Technologies & Libraries

- Python 3.x
- **Transformers:** DistilBERT, DeBERTa (via Hugging Face)
- **Large Language Models:** Meta LLaMA3
- SHAP (Shapley Additive Explanations), LIME
- Pandas, NumPy, scikit-learn
- Google Colab (recommended for GPU training)

---

## 💡 Problem Statement

Aspect-Based Sentiment Analysis breaks reviews into **attribute-level** opinions (e.g., battery life, display quality) instead of just general scores. This matters, because:

- **Traditional sentiment analysis misses the “why”** behind ratings—ABSA tells businesses what customers actually liked/disliked.
- Enabling **fine-grained insights** for both manufacturers and consumers.
- Evaluating the power of advanced transformer models and LLMs for multilingual, domain-adaptable sentiment prediction.

---

## ✨ My Approach

1. **Dataset:** Used ABSA16 Laptops Train/Test set (SemEval competition). Annotated laptop reviews with aspect categories and sentiment polarity.
2. **Preprocessing:** Tokenization, aspect category appending, padding/truncation, polarity encoding.
3. **Experiments:**
   - *Experiment 1*: Fine-tuning DistilBERT and DeBERTa — tested performance with/without aspect categories.
   - *Experiment 2*: Zero-shot & few-shot learning using LLaMA3 with custom prompt engineering.
   - *Experiment 3*: Interpretability using SHAP (word attributions) compared to LLaMA3 explanations.
4. **Metrics:** Accuracy, F1-score; interpretability via Jaccard similarity, Precision, Recall, F1, Coverage.

---

## 📊 Key Results

| Model         | Setup                | Accuracy (%) | F1 Score |
|---------------|----------------------|-------------|----------|
| DistilBERT    | Text only            | 77.96       | 77.30    |
| DistilBERT    | Text + Aspect        | 87.66       | 87.47    |
| DeBERTa       | Text only            | 80.84       | 80.83    |
| DeBERTa       | Text + Aspect        | 90.66       | 90.56    |
| LLaMA3        | Zero-shot            | 77.01       | 78.32    |
| LLaMA3        | Few-shot (+aspect)   | 83.47       | 83.54    |

**Interpretability (SHAP vs LLM):**
- Jaccard Similarity: 0.2258
- Precision: 0.6996
- Recall: 0.2417
- Coverage: 0.9048

---

## 🏆 Contribution Highlights

- **Benchmarked transformer models vs LLMs for ABSA.**
- **Improved accuracy by supplying aspect categories as input.**
- **Explored zero-shot/few-shot learning for real-world adaptability.**
- **Used SHAP for true visual interpretability of NLP models.**
- **Compared machine-attributed and human-logical explanations for model trust.**

---


---

## 🏁 How To Run

1. **Clone the Repository**
git clone https://github.com/vedghuikhedkar/sentimental-analysis.git
cd sentimental-analysis

2. **Install requirements**
pip install -r requirements.txt

3. **Run Notebooks:** Launch Jupyter Notebook or use Google Colab (recommended for GPU).

4. **Review the PDF:** For methodology, results, conclusions.

---

## 📣 Why This Project Stands Out

- Brings **research, code, and business impact** in one place.
- Uses **latest NLP technology—LLMs and interpretable AI.**
- Shows my **technical depth, ability to explain, and practical thinking**.
- **Relevant to real-world jobs**: Product analytics, data science, MLOps, NLP, tech consulting.

---

## 👋 Contact

For project queries, suggestions, or opportunities:  
**Ved Ghuikhedkar**  
LinkedIn: https://www.linkedin.com/in/ved-ghuikhedkar/
Email: vedghuikhedkar2@gmail.com
