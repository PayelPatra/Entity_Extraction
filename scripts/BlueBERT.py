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

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

DATA_DIR = "data/"
OUTPUT_DIR = "output/"
MODEL_DIR = "models/bluebert/"
ANNOTATED_DATA_PATH = os.path.join(DATA_DIR, "bio_tagged_output.csv")

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

model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.1, 
    attention_probs_dropout_prob=0.1
)

def preprocess_text_files(directory_path):
    preprocessed_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()
            cleaned_text = clean_text(remove_negative_phrases(text_data))
            preprocessed_data.append({"Filename": filename, "Text": cleaned_text})
    return pd.DataFrame(preprocessed_data)

class CustomNERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_offsets_mapping=True)
        label_ids = [label2id.get(tag, 0) for tag in self.labels[idx]]
        label_ids += [0] * (self.max_len - len(label_ids))
        encoding.pop("offset_mapping")
        encoding["labels"] = torch.tensor(label_ids)
        return {key: torch.tensor(val) for key, val in encoding.items()}

    def __len__(self):
        return len(self.texts)

df = pd.read_csv(ANNOTATED_DATA_PATH)
texts = df['text'].tolist()
bio_labels = df['bio_labels'].apply(lambda x: x.split()).tolist()

train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, bio_labels, test_size=0.15, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=0.176, random_state=42)

train_dataset = CustomNERDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomNERDataset(val_texts, val_labels, tokenizer)
test_dataset = CustomNERDataset(test_texts, test_labels, tokenizer)

def extract_entities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    return [(tok, id2label.get(pred, 'O')) for tok, pred in zip(tokens, predictions)]


training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    seed=42,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    warmup_steps=300,
    lr_scheduler_type="linear",
    dataloader_drop_last=True,
    logging_dir=os.path.join(MODEL_DIR, "logs"),
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

val_results = trainer.predict(val_dataset)
val_preds = torch.argmax(torch.tensor(val_results.predictions), dim=2).flatten().tolist()
val_true = torch.tensor(val_results.label_ids).flatten().tolist()
val_metrics = evaluate_performance(val_true, val_preds)
print("Validation Metrics:", val_metrics)

test_results = trainer.predict(test_dataset)
test_preds = torch.argmax(torch.tensor(test_results.predictions), dim=2).flatten().tolist()
test_true = torch.tensor(test_results.label_ids).flatten().tolist()
test_metrics = evaluate_performance(test_true, test_preds)
print("Test Metrics:", test_metrics)

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

raw_df = preprocess_text_files(DATA_DIR)
results_pretrained = []
results_finetuned = []

for _, row in raw_df.iterrows():
    text = row["Text"]

    pretrained_entities = extract_entities(text)
    pretrained_str = " ".join([f"{tok}:{tag}" for tok, tag in pretrained_entities])
    results_pretrained.append({
        "Filename": row["Filename"],
        "text": text,
        "predicted_entities": pretrained_str
    })

    finetuned_entities = extract_entities(text)
    finetuned_str = " ".join([f"{tok}:{tag}" for tok, tag in finetuned_entities])
    results_finetuned.append({
        "Filename": row["Filename"],
        "text": text,
        "predicted_entities": finetuned_str
    })

pretrained_csv = os.path.join(OUTPUT_DIR, "pretrained_entity_extraction_results_bluebert.csv")
finetuned_csv = os.path.join(OUTPUT_DIR, "finetuned_entity_extraction_results_bluebert.csv")

pd.DataFrame(results_pretrained).to_csv(pretrained_csv, index=False)
pd.DataFrame(results_finetuned).to_csv(finetuned_csv, index=False)

print(f"Pretrained entity extraction results saved to: {pretrained_csv}")
print(f"Finetuned entity extraction results saved to: {finetuned_csv}")
