from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

# Load the tokenizer and model
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)
model.eval()

# predicted_word = tokenizer.decode(predicted_token_id)
print(f"Token for'great': {tokenizer.encode('great', add_special_tokens=False)}")
print(f"Token for'great ': {tokenizer.encode('great ', add_special_tokens=False)}")
print(f"Token for' great': {tokenizer.encode(' great', add_special_tokens=False)}")
print(f"Token for' great ': {tokenizer.encode(' great ', add_special_tokens=False)}")
print(" ")
print(f"Token for'greater': {tokenizer.encode('greater', add_special_tokens=False)}")
print(f"Token for'greater ': {tokenizer.encode('greater ', add_special_tokens=False)}")
print(f"Token for' greater': {tokenizer.encode(' greater', add_special_tokens=False)}")
print(f"Token for' greater ': {tokenizer.encode(' greater ', add_special_tokens=False)}")
print(" ")
print(f"Token for'Great': {tokenizer.encode('Great', add_special_tokens=False)}")
print(f"Token for'Great ': {tokenizer.encode('Great ', add_special_tokens=False)}")
print(f"Token for' Great': {tokenizer.encode(' Great', add_special_tokens=False)}")
print(f"Token for' Great ': {tokenizer.encode(' Great ', add_special_tokens=False)}")
print(" ")
print(f"Token for'Greater': {tokenizer.encode('Greater', add_special_tokens=False)}")
print(f"Token for'Greater ': {tokenizer.encode('Greater ', add_special_tokens=False)}")
print(f"Token for' Greater': {tokenizer.encode(' Greater', add_special_tokens=False)}")
print(f"Token for' Greater ': {tokenizer.encode(' Greater ', add_special_tokens=False)}")
print(" ")
print(f"Text for 12338 : {tokenizer.decode(12338, add_special_tokens=False)}")
print(f"Text for 2388 : {tokenizer.decode(2388, add_special_tokens=False)}")
print(f"Text for 1437 : {tokenizer.decode(1437, add_special_tokens=False)}")
print(f"Text for 372 : {tokenizer.decode(372, add_special_tokens=False)}")
print(f"Text for 254 : {tokenizer.decode(254, add_special_tokens=False)}")
print(f"Text for 19065 : {tokenizer.decode(19065, add_special_tokens=False)}")
print(f"Text for 9312 : {tokenizer.decode(9312, add_special_tokens=False)}")
print(" ")
print(f"Token for'Less': {tokenizer.encode('Less', add_special_tokens=False)}")
print(f"Token for'Less ': {tokenizer.encode('Less ', add_special_tokens=False)}")
print(f"Token for' Less': {tokenizer.encode(' Less', add_special_tokens=False)}")
print(f"Token for' Less ': {tokenizer.encode(' Less ', add_special_tokens=False)}")
print(" ")
print(f"Tokens for full datapoint: {tokenizer.encode('age: 38, workclass: Private, education: Bachelors, educational-num: 13, marital-status: Married-civ-spouse, occupation: Exec-managerial, relationship: Husband, race: White, gender: Male, capital-gain: 0, capital-loss: 0, hours-per-week: 40, native-country: United States, income: Greater than 50k', add_special_tokens=False)}")
