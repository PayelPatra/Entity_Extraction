import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, set_seed
from sklearn.model_selection import train_test_split
from preprocessing_utils import clean_text, remove_negative_phrases
from utils import evaluate_performance
import nltk
from nltk.corpus import stopwords

#five random seeds
SEEDS = [42, 56, 101, 202, 303]
def set_all_seeds(s: int):
    set_seed(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# Setup stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Paths
DATA_DIR = "data/"
OUTPUT_DIR = "output/"
MODEL_DIR = "models/biobert/"
ANNOTATED_DATA_PATH = os.path.join(DATA_DIR, "bio_tagged_output.csv")

# Creating directories if not present
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

# Loading BioBERT tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def extract_entities_with(model_inst, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model_inst(**inputs).logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    return [(tok, id2label.get(pred, "O")) for tok, pred in zip(tokens, predictions)]

# Function to clean and prepare .txt files
def preprocess_texts_from_dir(directory):
    all_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            cleaned = clean_text(remove_negative_phrases(raw))
            all_texts.append({"Filename": filename, "Text": cleaned})
    return pd.DataFrame(all_texts)

# Load and split annotated data
df = pd.read_csv(ANNOTATED_DATA_PATH)
texts = df['text'].tolist()
labels = df['bio_labels'].apply(lambda x: x.split()).tolist()

# Split into train, val, test
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    texts, labels, test_size=0.15, random_state=42
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=0.176, random_state=42
)

# Dataset class
class CustomNERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True
        )
        label_ids = [label2id.get(tag, 0) for tag in self.labels[idx]]
        label_ids += [0] * (self.max_len - len(label_ids))
        encoded.pop("offset_mapping")
        encoded["labels"] = torch.tensor(label_ids)
        return {key: torch.tensor(val) for key, val in encoded.items()}

    def __len__(self):
        return len(self.texts)

# Prepare datasets
train_dataset = CustomNERDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomNERDataset(val_texts, val_labels, tokenizer)
test_dataset = CustomNERDataset(test_texts, test_labels, tokenizer)


zeroshot_model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

raw_df = preprocess_texts_from_dir(DATA_DIR)
results_pretrained = []
for _, row in raw_df.iterrows():
    text = row["Text"]
    pretrained_entities = extract_entities_with(zeroshot_model, text)
    pretrained_str = " ".join([f"{tok}:{tag}" for tok, tag in pretrained_entities])
    results_pretrained.append({
        "Filename": row["Filename"],
        "text": text,
        "predicted_entities": pretrained_str
    })

pretrained_csv = os.path.join(OUTPUT_DIR, "pretrained_entity_extraction_results_biobert.csv")
pd.DataFrame(results_pretrained).to_csv(pretrained_csv, index=False)
print(f"Pretrained (zero-shot) entity extraction results saved to: {pretrained_csv}")


per_seed_rows = []

for idx, seed in enumerate(SEEDS):
    set_all_seeds(seed)

    # fresh model for this seed
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, f"seed_{seed}"),
        seed=seed,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        eval_strategy="epoch",     # <-- was eval_strategy
        save_strategy="epoch",
        learning_rate=5e-5,
        warmup_steps=300,
        lr_scheduler_type="linear",
        dataloader_drop_last=True,
        logging_dir=os.path.join(MODEL_DIR, f"logs_seed_{seed}"),
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    # Validation evaluation 
    val_output = trainer.predict(val_dataset)
    val_preds = torch.argmax(torch.tensor(val_output.predictions), dim=2).flatten().tolist()
    val_true = torch.tensor(val_output.label_ids).flatten().tolist()
    val_metrics = evaluate_performance(val_true, val_preds)
    print(f"[Seed {seed}] Validation Metrics:", val_metrics)

    # Test evaluation 
    test_output = trainer.predict(test_dataset)
    test_preds = torch.argmax(torch.tensor(test_output.predictions), dim=2).flatten().tolist()
    test_true = torch.tensor(test_output.label_ids).flatten().tolist()
    test_metrics = evaluate_performance(test_true, test_preds)
    print(f"[Seed {seed}] Test Metrics:", test_metrics)


    row = {"seed": seed, **test_metrics}
    per_seed_rows.append(row)


    if idx == 0:
        results_finetuned = []
        for _, r in raw_df.iterrows():
            text = r["Text"]
            finetuned_entities = extract_entities_with(model, text)
            finetuned_str = " ".join([f"{tok}:{tag}" for tok, tag in finetuned_entities])
            results_finetuned.append({
                "Filename": r["Filename"],
                "text": text,
                "predicted_entities": finetuned_str
            })
        finetuned_csv = os.path.join(OUTPUT_DIR, "finetuned_entity_extraction_results_biobert.csv")
        pd.DataFrame(results_finetuned).to_csv(finetuned_csv, index=False)
        print(f"Finetuned entity extraction results (seed {seed}) saved to: {finetuned_csv}")

    # Save model for this seed
    model.save_pretrained(os.path.join(MODEL_DIR, f"seed_{seed}"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, f"seed_{seed}"))


runs_csv = os.path.join(OUTPUT_DIR, "biobert_runs.csv")
pd.DataFrame(per_seed_rows).to_csv(runs_csv, index=False)
print(f"Per-seed test metrics saved to: {runs_csv}")
