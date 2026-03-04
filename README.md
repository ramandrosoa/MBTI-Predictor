# 🧠 MBTI Personality Predictor

[This notebook](MBTI_NLP.ipynb) covers the full pipeline from **exploratory analysis to prediction** — using textual data to classify Myers-Briggs Type Indicator (MBTI) personality types with classical machine learning.

🚀 **Live Demo:** https://huggingface.co/spaces/ramandrosoa/mbti-predictor

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-11557C?logo=python&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-4C72B0?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?logo=huggingface&logoColor=black)

---

## 🗂️ Overview

Rather than predicting all 16 MBTI types at once, the problem is decomposed into **four independent binary classifiers**, one per dimension:

| Dimension | Class 1 | Class 0 |
|-----------|---------|---------|
| IE | Introvert (I) | Extrovert (E) |
| NS | Intuitive (N) | Sensing (S) |
| TF | Thinking (T) | Feeling (F) |
| JP | Judging (J) | Perceiving (P) |

---

## 🔬 Pipeline

### 📊 1. Exploratory Analysis
- Examined post counts per MBTI type
- Dataset was found to be **imbalanced** across types

### 🧹 2. Preprocessing
- **Tokenization** — split text into individual words/tokens
- **Stopword removal** — remove low-meaning words (e.g. "and", "the")
- **Lemmatization** — reduce words to base form (e.g. "running" → "run")
- 🔗 Links are stripped before any text analysis

### ⚙️ 3. Feature Engineering

| Feature | Description |
|---------|-------------|
| 📝 Post Length | Word count per post — captures conciseness of writing style |
| 📖 Readability | Gunning Fog Index — measures complexity via sentence length and word difficulty |
| 🏷️ NER | Named Entity Recognition — captures topical focus differences across types |
| 💬 Sentiment | Polarity scores — analyzes emotional tone of posts |
| 🔢 Vectorization | TF-IDF or word embeddings — converts text to numerical representations |

### ⚖️ 4. Resampling

The dataset is resampled to match the **real-world MBTI population distribution**:
- 📉 **Undersampling** applied to overrepresented types
- 📈 **Oversampling** applied to underrepresented types

### 🏋️ 5. Training

Three classifiers were evaluated per dimension using **grid search cross-validation**:
- Logistic Regression
- K-Nearest Neighbors
- Random Forest

The best-performing model per dimension was selected.

### 🚀 6. Inference and Deployment
- 🔮 Predictions made on new raw text input
- ⚡ REST API built with **FastAPI**
- 🤗 Model hosted on **Hugging Face Spaces**

---

## 📚 References

[Predicting Myers-Briggs Type Indicator with Text Classification](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6839354.pdf)
