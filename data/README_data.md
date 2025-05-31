# Data Directory

This folder contains the dataset and preprocessing files used for our project:
"Adversarial Attack on Tabular Classification".

---

## Contents

| File | Description |
|------|-------------|
| `adult.csv` | Full version of the UCI Adult dataset (imbalanced). Used for initial processing and debugging. |
| `DataPreparation.py` | Python script for preprocessing the dataset and creating train/test/validation splits. |
| `label_encoders.pkl` | Saved encoders for categorical columns (used in the GBT pipeline). |

---

## Access the Final Balanced Dataset

The final dataset used for training and evaluation is hosted on Hugging Face:

Link: https://huggingface.co/datasets/ETdanR/Balanced_Adult_Split_DS

This version includes:
- `train_data.csv` – balanced training set  
- `validation_data.csv` – balanced validation set  
- `experiment_data.csv` – balanced test set (used in adversarial evaluation)

Each split was generated using `DataPreparation.py` and uploaded for reproducibility and external access.

---

## Usage

You can download and load the dataset programmatically using the `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("ETdanR/Balanced_Adult_Split_DS")
train_df = dataset["train"].to_pandas()
val_df = dataset["validation"].to_pandas()
test_df = dataset["test"].to_pandas()
```

---

## Notes

- The Hugging Face dataset ensures class balance across all splits (`<=50K` vs `>50K`).
- All preprocessing steps are fully documented in `DataPreparation.py`.

---

## Contributors

- [yuvalira](https://github.com/yuvalira)  
- [ETdanR](https://huggingface.co/ETdanR)
