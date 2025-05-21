import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

from Experiments.Corr_Attack import extract_batch


# Load fine-tuned model directly from Hugging Face Hub
model_path = "ETdanR/RoBERTa_FT_adult"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path)
model.eval()

# Define mask token info
mask_token = tokenizer.mask_token
mask_token_id = tokenizer.mask_token_id

# Token IDs for income prediction (example IDs from your snippet)
greater_than_id = 9312  # "Greater"
less_than_id = 10862    # "Less"


def categorical_gaussian_perturb(datapoint, correlation_dict):

    datapoint = datapoint.iloc[0]
    perturbed = datapoint.copy()

    for feature, sigma in catfeature_sigma_dict.items():

        current_value = datapoint[feature]

        category_scores = correlation_dict[feature]
        categories = list(category_scores.keys())

        positions = np.array([category_scores[cat] for cat in categories])
        center = category_scores[current_value]

        distances = positions - center
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        weights /= weights.sum()

        sampled_index = np.random.choice(len(categories), p=weights)
        new_value = categories[sampled_index]

        perturbed[feature] = new_value

    return perturbed

def numerical_gaussian_perturb(datapoint, numfeature_sigma_dict, value_dict):
    datapoint = datapoint.iloc[0]
    perturbed = datapoint.copy()

    for feature, sigma in numfeature_sigma_dict.items():
        current_value = datapoint[feature]
        values = np.array(value_dict[feature])

        # Center the Gaussian on the current value
        distances = values - current_value
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        weights /= weights.sum()

        sampled_index = np.random.choice(len(values), p=weights)
        new_value = values[sampled_index]

        perturbed[feature] = new_value

    return perturbed

def predict_LM(df):
    sentences = [
        f"age: {row['age']}, workclass: {row['workclass']}, education: {row['education']}, "
        f"educational-num: {row['educational-num']}, marital-status: {row['marital-status']}, "
        f"occupation: {row['occupation']}, relationship: {row['relationship']}, race: {row['race']}, "
        f"gender: {row['gender']}, capital-gain: {row['capital-gain']}, capital-loss: {row['capital-loss']}, "
        f"hours-per-week: {row['hours-per-week']}, native-country: {row['native-country']}, "
        f"income: {mask_token} than 50k"
        for _, row in df.iterrows()
    ]

    encoded = tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    mask_positions = (encoded['input_ids'] == mask_token_id).nonzero(as_tuple=False)
    predictions = torch.argmax(logits, dim=-1)

    results = []
    for i in range(len(df)):
        mask_index = mask_positions[mask_positions[:, 0] == i][0, 1]
        predicted_token_id = predictions[i, mask_index].item()

        if predicted_token_id == greater_than_id:
            results.append(1)
        elif predicted_token_id == less_than_id:
            results.append(0)
        else:
            results.append(-1)

    # Convert to numpy array with shape (batch_size,)
    results_np = np.array(results, dtype=int)
    print("predict called")
    return results_np


def extract_batch(df, batch_length, last_row_handled):
    start = last_row_handled + 1
    # If there's no data left to extract
    if start >= len(df):
        return None, None

    end = start + batch_length
    batch = df.iloc[start:end]
    new_last_row_handled = start + len(batch) - 1

    return batch, new_last_row_handled

def predict_LM(df):
    sentences = [
        f"age: {row['age']}, workclass: {row['workclass']}, education: {row['education']}, "
        f"educational-num: {row['educational-num']}, marital-status: {row['marital-status']}, "
        f"occupation: {row['occupation']}, relationship: {row['relationship']}, race: {row['race']}, "
        f"gender: {row['gender']}, capital-gain: {row['capital-gain']}, capital-loss: {row['capital-loss']}, "
        f"hours-per-week: {row['hours-per-week']}, native-country: {row['native-country']}, "
        f"income: {mask_token} than 50k"
        for _, row in df.iterrows()
    ]

    encoded = tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    mask_positions = (encoded['input_ids'] == mask_token_id).nonzero(as_tuple=False)
    predictions = torch.argmax(logits, dim=-1)

    results = []
    for i in range(len(df)):
        mask_index = mask_positions[mask_positions[:, 0] == i][0, 1]
        predicted_token_id = predictions[i, mask_index].item()

        if predicted_token_id == greater_than_id:
            results.append(1)
        elif predicted_token_id == less_than_id:
            results.append(0)
        else:
            results.append(-1)

    # Convert to numpy array with shape (batch_size,)
    results_np = np.array(results, dtype=int)
    print("predict called")
    return results_np