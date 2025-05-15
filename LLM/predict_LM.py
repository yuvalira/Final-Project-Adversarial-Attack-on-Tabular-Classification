import pandas as pd
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import time

# Load fine-tuned model
model_path = 'C:/Users/ADMIN/PycharmProjects/Final-Project-Adversarial-Attack-on-Tabular-Classification/Models/fine_tuned_roberta_income'
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path)
model.eval()

# Define mask token info
mask_token = tokenizer.mask_token
mask_token_id = tokenizer.mask_token_id

# Token IDs for income prediction
greater_than_id = 9312  # "Greater"
less_than_id = 10862    # "Less"


start_time = time.time()
def predict_LM(df):

    # Build input sentences
    sentences = [
        f"age: {row['age']}, workclass: {row['workclass']}, education: {row['education']}, "
        f"educational-num: {row['educational-num']}, marital-status: {row['marital-status']}, "
        f"occupation: {row['occupation']}, relationship: {row['relationship']}, race: {row['race']}, "
        f"gender: {row['gender']}, capital-gain: {row['capital-gain']}, capital-loss: {row['capital-loss']}, "
        f"hours-per-week: {row['hours-per-week']}, native-country: {row['native-country']}, "
        f"income: {mask_token} than 50k"
        for _, row in df.iterrows()
    ]

    # Tokenize as a batch
    encoded = tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    # Find the index of the [MASK] token in each sequence
    mask_positions = (encoded['input_ids'] == mask_token_id).nonzero(as_tuple=False)

    # Collect predicted token IDs at each masked index
    predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

    # For each sample in the batch, get the token at the mask position
    results = []
    for i in range(len(df)):
        # Get the masked index for sample i
        mask_index = mask_positions[mask_positions[:, 0] == i][0, 1]
        predicted_token_id = predictions[i, mask_index].item()

        if predicted_token_id == greater_than_id:
            results.append(1)
        elif predicted_token_id == less_than_id:
            results.append(0)
        else:
            results.append(-1)  # unknown or invalid token

    return results
end_time = time.time()

df = pd.read_csv("C:/Users/ADMIN/PycharmProjects/Final-Project-Adversarial-Attack-on-Tabular-Classification/data/experiment_data.csv")
batch_df = df.sample(32, random_state=42) # for randomness
predictions = predict_LM(batch_df)

print(predictions)  # e.g., [1, 0, 1, 0, ..., 1]
print(f"Inference time for batch of 16: {end_time - start_time:.4f} seconds")

