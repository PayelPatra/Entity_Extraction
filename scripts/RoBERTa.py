import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from preprocessing_utils import clean_text, remove_negative_phrases
from torch.utils.data import Dataset
from utils import evaluate_performance
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

DATA_DIR = "data/"
OUTPUT_DIR = "output/"
MODEL_DIR = "models/roberta/"
ANNOTATED_DATA_PATH = "data/annotated_corpus.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

label2id = {
    'O': 0,
    'B-AGE': 1, 'I-AGE': 2,
    'B-GENDER': 3, 'I-GENDER': 4,
    'B-CANCER STAGE': 5, 'I-CANCER STAGE': 6,
    'B-DISEASE': 7, 'I-DISEASE': 8,
    'B-SYMPTOMS': 9, 'I-SYMPTOMS': 10,
    'B-MEDICATION': 11, 'I-MEDICATION': 12,
    'B-DOSE': 13, 'I-DOSE': 14
}
id2label = {v: k for k, v in label2id.items()}

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)

# Preprocessing
def preprocess_text_files(directory_path):
    preprocessed_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()
            text_data = remove_negative_phrases(text_data)
            cleaned_text = clean_text(text_data)
            preprocessed_data.append({"Filename": filename, "Text": cleaned_text})
    return pd.DataFrame(preprocessed_data)

preprocessed_df = preprocess_text_files(DATA_DIR)
preprocessed_df.to_csv(os.path.join(OUTPUT_DIR, "preprocessed_data.csv"), index=False)

# Dataset Class
class CustomNERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_offsets_mapping=True)
        label_ids = [label2id.get(tag, 0) for tag in self.labels[idx]]
        label_ids = label_ids + [0] * (self.max_len - len(label_ids))
        encoding.pop("offset_mapping")
        encoding["labels"] = torch.tensor(label_ids)
        return {key: torch.tensor(val) for key, val in encoding.items()}

    def __len__(self):
        return len(self.texts)

# Load annotated dataset
df = pd.read_csv(ANNOTATED_DATA_PATH)
texts = df['text'].tolist()
bio_labels = df['bio_labels'].apply(lambda x: x.split()).tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, bio_labels, test_size=0.2, random_state=42)

# Pretrained extraction before fine-tuning
def extract_entities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    return [(tok, id2label.get(pred, 'O')) for tok, pred in zip(tokens, predictions)]

pretrained_results = []
for text in val_texts:
    entities = extract_entities(text)
    pretrained_results.append({"text": text, "predicted_entities": " ".join([f"{tok}:{tag}" for tok, tag in entities])})
pretrained_csv = os.path.join(OUTPUT_DIR, "pretrained_extracted_entities_roberta.csv")
pd.DataFrame(pretrained_results).to_csv(pretrained_csv, index=False)
print(f"Pretrained model extraction results saved to {pretrained_csv}")

# Fine-tuning
train_dataset = CustomNERDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomNERDataset(val_texts, val_labels, tokenizer)
training_args = TrainingArguments(output_dir=MODEL_DIR, num_train_epochs=5, per_device_train_batch_size=16, save_strategy="epoch", evaluation_strategy="epoch")
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
trainer.train()

# Evaluation
eval_results = trainer.predict(val_dataset)
predicted_labels = torch.argmax(torch.tensor(eval_results.predictions), dim=2).flatten().tolist()
true_labels = torch.tensor(eval_results.label_ids).flatten().tolist()
metrics = evaluate_performance(true_labels, predicted_labels)
print("Evaluation Metrics after fine-tuning:", metrics)

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# Fine-tuned extraction after training
fine_tuned_results = []
for text in val_texts:
    entities = extract_entities(text)
    fine_tuned_results.append({"text": text, "predicted_entities": " ".join([f"{tok}:{tag}" for tok, tag in entities])})

pd.DataFrame(fine_tuned_results).to_csv(os.path.join(OUTPUT_DIR, "fine_tuned_extracted_entities_roberta.csv"), index=False)
print("Fine-tuned extracted entities saved.")
