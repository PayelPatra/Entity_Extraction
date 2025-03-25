import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from preprocessing_utils import clean_text, remove_negative_phrases
from utils import evaluate_performance
import nltk
from nltk.corpus import stopwords

# Setup stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Directory paths
DATA_DIR = "data/"
OUTPUT_DIR = "output/"
MODEL_DIR = "models/bioclinicalbert/"
ANNOTATED_DATA_PATH = os.path.join(DATA_DIR, "bio_tagged_output.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# BIO tagging scheme
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

# Load BioClinicalBERT model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# Function to clean and prepare raw .txt files
def preprocess_text_files(directory):
    cleaned_data = []
    for fname in os.listdir(directory):
        if fname.endswith(".txt"):
            fpath = os.path.join(directory, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            cleaned = clean_text(remove_negative_phrases(text))
            cleaned_data.append({"Filename": fname, "Text": cleaned})
    return pd.DataFrame(cleaned_data)

# Load annotated corpus
df = pd.read_csv(ANNOTATED_DATA_PATH)
texts = df["text"].tolist()
bio_labels = df["bio_labels"].apply(lambda x: x.split()).tolist()

# Train-Validation-Test Spliting
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    texts, bio_labels, test_size=0.15, random_state=42
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=0.176, random_state=42  # ~15% of full
)

# Dataset class
class CustomNERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True
        )
        label_ids = [label2id.get(tag, 0) for tag in self.labels[idx]]
        label_ids += [0] * (self.max_len - len(label_ids))
        encoding.pop("offset_mapping")
        encoding["labels"] = torch.tensor(label_ids)
        return {key: torch.tensor(val) for key, val in encoding.items()}

    def __len__(self):
        return len(self.texts)

# Prepare datasets
train_dataset = CustomNERDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomNERDataset(val_texts, val_labels, tokenizer)
test_dataset = CustomNERDataset(test_texts, test_labels, tokenizer)

# Entity extraction function
def extract_entities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    return [(tok, id2label.get(pred, "O")) for tok, pred in zip(tokens, predictions)]

# Training configuration
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Evaluate on validation set
val_output = trainer.predict(val_dataset)
val_preds = torch.argmax(torch.tensor(val_output.predictions), dim=2).flatten().tolist()
val_labels_flat = torch.tensor(val_output.label_ids).flatten().tolist()
val_metrics = evaluate_performance(val_labels_flat, val_preds)
print("Validation Metrics:", val_metrics)

# Evaluate on test set
test_output = trainer.predict(test_dataset)
test_preds = torch.argmax(torch.tensor(test_output.predictions), dim=2).flatten().tolist()
test_labels_flat = torch.tensor(test_output.label_ids).flatten().tolist()
test_metrics = evaluate_performance(test_labels_flat, test_preds)
print("Test Metrics:", test_metrics)

# Save the fine-tuned model
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# Inference on raw .txt files (clinical notes)
raw_data_df = preprocess_text_files(DATA_DIR)
results_pretrained = []
results_finetuned = []

for _, row in raw_data_df.iterrows():
    text = row["Text"]
    
    # Entity extraction using pretrained model
    pretrained_entities = extract_entities(text)
    pretrained_str = " ".join([f"{tok}:{tag}" for tok, tag in pretrained_entities])
    results_pretrained.append({
        "Filename": row["Filename"],
        "text": text,
        "predicted_entities": pretrained_str
    })

    # Entity extraction using finetuned model
    finetuned_entities = extract_entities(text)
    finetuned_str = " ".join([f"{tok}:{tag}" for tok, tag in finetuned_entities])
    results_finetuned.append({
        "Filename": row["Filename"],
        "text": text,
        "predicted_entities": finetuned_str
    })

# Save prediction CSVs
pretrained_csv_path = os.path.join(OUTPUT_DIR, "pretrained_entity_extraction_results.csv")
finetuned_csv_path = os.path.join(OUTPUT_DIR, "finetuned_entity_extraction_results.csv")

pd.DataFrame(results_pretrained).to_csv(pretrained_csv_path, index=False)
pd.DataFrame(results_finetuned).to_csv(finetuned_csv_path, index=False)

print(f"Pretrained entity extraction results saved to: {pretrained_csv_path}")
print(f"Finetuned entity extraction results saved to: {finetuned_csv_path}")
