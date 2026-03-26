🧬 Antimicrobial Peptide (AMP) Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Bioinformatics](https://img.shields.io/badge/Field-Bioinformatics-green.svg)](https://en.wikipedia.org/wiki/Bioinformatics)

## 📌 Project Overview
The **Antimicrobial Peptide (AMP) Predictor** is a machine learning tool designed to identify peptides with potential activity against Gram-negative bacteria. By integrating **Natural Language Processing (NLP)** techniques with **Biochemical Descriptor** analysis, this project provides a robust computational approach to antibiotic discovery.

The model utilizes a **Mixed-Data Pipeline** to process both the raw amino acid sequences and the physical properties of the peptides simultaneously.

## 🚀 Key Features
* **Mixed-Data Pipeline:** Seamlessly combines text-based sequence features and numerical chemical properties.
* **Stacking Ensemble Classifier:** Uses a multi-layered approach where base models (like Random Forest and SVM) feed into a meta-learner for superior accuracy.
* **K-mer Feature Extraction:** Converts peptide sequences into numerical vectors using NLP-based 3-mer and 4-mer tokenization.
* **Physicochemical Analysis:** Incorporates molecular weight, hydrophobicity, isoelectric point, and charge as critical decision features.

## 🛠️ Technical Stack
* **Data Processing:** `Pandas`, `NumPy`, `Biopython`
* **Machine Learning:** `Scikit-Learn` (StackingClassifier, Pipeline, ColumnTransformer)
* **NLP:** `TfidfVectorizer` for sequence k-mers
* **Visualization:** `Matplotlib`, `Seaborn`

## 📊 How It Works
1.  **Sequence Processing:** Raw FASTA or string sequences are broken into overlapping k-mers.
2.  **Property Calculation:** Biochemical properties are calculated for each peptide sequence.
3.  **The Pipeline:** * Text data goes through a **TF-IDF Vectorizer**.
    * Numerical data goes through a **StandardScaler**.
4.  **Classification:** The **Stacking Ensemble** processes these features to predict whether a peptide is a potent AMP or non-AMP.

## 📂 Repository Structure
```text
├── data/               # Dataset (Peptide sequences and labels)
├── notebooks/          # Jupyter notebooks for model development
├── src/                # Source code for feature engineering
├── models/             # Saved .pkl models
├── requirements.txt    # Installation dependencies
└── README.md           # Project documentation
