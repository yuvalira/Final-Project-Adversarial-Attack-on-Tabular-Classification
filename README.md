# Final Project - Adversarial Attack on Tabular Classification
# RoBERTa vs. GBT

## Overview

This project investigates and compares the robustness of two machine learning models—a Large Language Model (RoBERTa) and a classic Gradient Boosted Tree (GBT)—against adversarial attacks on tabular data classification. The dataset used is the popular **Adult Income dataset**, which contains demographic and employment attributes.

The goal is to evaluate how adversarial attacks can manipulate predictions and to compare the resilience of both models under attack.

---

## Dataset

**Adult Income Dataset (Census Income Dataset)**

* Predicts whether a person earns more than \$50K/year based on attributes such as education, occupation, age, gender, etc.
* Download source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) or Kaggle.

---

## Project Structure

```
/project-root
│
├── data/            → Dataset files 
├── EDA/             → Jupyter Notebook for data analysis and experiments
├── GBT/             → Jupyter Notebook for Training and evaluation GBT model and the trained model
├── LLM/             → Scripts for fine-tuning and evaluating RoBERTa LLM
├── ATTACK/          → Adversarial attack implementations and experiments
├── results/         → Saved results, plots, and reports
├── README.md        → Project description
└── requirements.txt → Python dependencies
```

---

## Methods

### 1. Dataset Analysis

Initial data exploration and preprocessing, including:

* Loading dataset (Kaggle/UCI)
* Cleaning and handling missing values
* Feature encoding (e.g., one-hot for categorical features)

### 2. Exploratory Data Analysis (EDA)

* Summary statistics
* Visualizations of feature distributions
* Correlation heatmaps to understand relationships between variables

### 3. Model Training

* **GBT Model**: trained using scikit-learn GradientBoostingClassifier, which provides excellent stability and interpretability for tabular data classification. Trained on the clean dataset to classify income level.
* **RoBERTa Model**: Fine-tuned with the dataset converted into a token-based format.

### 4. Adversarial Attack

* Applied feature perturbations and neighbor attacks to test model robustness.
* Focused on structured adversarial attacks suitable for tabular data.

### 5. Evaluation

* Comparison of performance (accuracy, F1 score, confusion matrix, etc.) between GBT and RoBERTa under adversarial conditions.

---

## How to Run

1. Clone this repository:

```bash
git clone <repo-url>
cd project-name
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Follow the notebooks and scripts in the order:

   * EDA/ → Data Exploration & Preprocessing
   * GBT\_model/ → Train GBT model
   * LLM\_model/ → fine-tune RoBERTa
   * Attacks/ → Apply adversarial attacks and evaluate

4. Results and visualizations will be saved in `results/`.

---

## Example Results

* Performance metrics for each model
* Comparative information for accuracy 

---

## References

1. Adult Income Dataset ([https://archive.ics.uci.edu/ml/datasets/adult](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset))
2. RoBERTa: Liu, Y. et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach", arXiv:1907.11692 (2019). [https://arxiv.org/abs/1907.11692]
3. Gradient Boosted Trees: Scikit-learn Documentation. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html]
