import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    Trainer, TrainingArguments, set_seed
)
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
    # (optional) stricter reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# paths
DATA_DIR = "data/"
OUTPUT_DIR = "output/"
MODEL_DIR = "models/pubmedbert/"
ANNOTATED_DATA_PATH = os.path.join(DATA_DIR, "bio_tagged_output.csv")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#labels 
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


# PubMedBERT
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_text_files(directory_path):
    rows = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()
            cleaned_text = clean_text(remove_negative_phrases(text_data))
            rows.append({"Filename": filename, "Text": cleaned_text})
    return pd.DataFrame(rows)

def extract_entities_with(model_inst, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model_inst(**inputs).logits
    preds = torch.argmax(logits, dim=2).squeeze().tolist()
    toks  = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    return [(tok, id2label.get(pred, 'O')) for tok, pred in zip(toks, preds)]


df = pd.read_csv(ANNOTATED_DATA_PATH)
texts = df['text'].tolist()
bio_labels = df['bio_labels'].apply(lambda x: x.split()).tolist()

train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    texts, bio_labels, test_size=0.15, random_state=42
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=0.176, random_state=42
)

class CustomNERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True
        )
        label_ids = [label2id.get(tag, 0) for tag in self.labels[idx]]
        label_ids += [0] * (self.max_len - len(label_ids))
        enc.pop("offset_mapping")
        enc["labels"] = torch.tensor(label_ids)
        return {k: torch.tensor(v) for k, v in enc.items()}
    def __len__(self):
        return len(self.texts)

train_dataset = CustomNERDataset(train_texts, train_labels, tokenizer)
val_dataset   = CustomNERDataset(val_texts,   val_labels,   tokenizer)
test_dataset  = CustomNERDataset(test_texts,  test_labels,  tokenizer)


zeroshot_model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

raw_df = preprocess_text_files(DATA_DIR)
results_pretrained = []
for _, row in raw_df.iterrows():
    ents = extract_entities_with(zeroshot_model, row["Text"])
    results_pretrained.append({
        "Filename": row["Filename"],
        "text": row["Text"],
        "predicted_entities": " ".join([f"{tok}:{tag}" for tok, tag in ents])
    })
pretrained_csv = os.path.join(OUTPUT_DIR, "pretrained_entity_extraction_results_pubmedbert.csv")
pd.DataFrame(results_pretrained).to_csv(pretrained_csv, index=False)
print(f"[PubMedBERT] zero-shot predictions saved to: {pretrained_csv}")

#fine-tuning across 3 seeds 
per_seed_rows = []

for idx, seed in enumerate(SEEDS):
    set_all_seeds(seed)

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
        eval_strategy="epoch",   
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

    # Validation
    val_output = trainer.predict(val_dataset)
    val_preds  = torch.argmax(torch.tensor(val_output.predictions), dim=2).flatten().tolist()
    val_true   = torch.tensor(val_output.label_ids).flatten().tolist()
    print(f"[Seed {seed}] Validation Metrics:", evaluate_performance(val_true, val_preds))

    # Test
    test_output = trainer.predict(test_dataset)
    test_preds  = torch.argmax(torch.tensor(test_output.predictions), dim=2).flatten().tolist()
    test_true   = torch.tensor(test_output.label_ids).flatten().tolist()
    test_metrics = evaluate_performance(test_true, test_preds)
    test_metrics["seed"] = seed
    per_seed_rows.append(test_metrics)
    print(f"[Seed {seed}] Test Metrics:", test_metrics)

    
    if idx == 0:
        results_finetuned = []
        for _, r in raw_df.iterrows():
            ents = extract_entities_with(model, r["Text"])
            results_finetuned.append({
                "Filename": r["Filename"],
                "text": r["Text"],
                "predicted_entities": " ".join([f"{tok}:{tag}" for tok, tag in ents])
            })
        finetuned_csv = os.path.join(OUTPUT_DIR, "finetuned_entity_extraction_results_pubmedbert.csv")
        pd.DataFrame(results_finetuned).to_csv(finetuned_csv, index=False)
        print(f"[PubMedBERT] finetuned predictions (seed {seed}) saved to: {finetuned_csv}")

    
    model.save_pretrained(os.path.join(MODEL_DIR, f"seed_{seed}"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, f"seed_{seed}"))


runs_csv = os.path.join(OUTPUT_DIR, "pubmedbert_runs.csv")
df_runs = pd.DataFrame(per_seed_rows)
df_runs.to_csv(runs_csv, index=False)
print(f"[PubMedBERT] per-seed test metrics saved to: {runs_csv}")

print("\n==== Aggregated PubMedBERT Results ====")
for col in df_runs.columns:
    if col != "seed":
        print(f"{col}: mean={np.mean(df_runs[col]):.4f}, std={np.std(df_runs[col]):.4f}")
