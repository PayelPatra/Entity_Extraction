import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords

# Ensure necessary libraries are available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define paths
DATA_DIR = "data/"
OUTPUT_DIR = "output/"
MODEL_DIR = "models/"
ANNOTATED_DATA_PATH = "data/annotated_corpus.csv"  # Custom dataset for fine-tuning

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load RoBERTa model
def load_pretrained_model():
    model_name = "roberta-base"  # Replace with fine-tuned model if available
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_pretrained_model()

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s/\.\-]", "", text)
    return text

# Function to preprocess data
def preprocess_text_files(directory_path):
    preprocessed_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()
            cleaned_text = clean_text(text_data)
            preprocessed_data.append({"Filename": filename, "Text": cleaned_text})
    return pd.DataFrame(preprocessed_data)

# Preprocess and save data
preprocessed_df = preprocess_text_files(DATA_DIR)
preprocessed_csv = os.path.join(OUTPUT_DIR, "preprocessed_data.csv")
preprocessed_df.to_csv(preprocessed_csv, index=False)
print(f"Preprocessed data saved to {preprocessed_csv}")

# Function to extract entities using RoBERTa
def extract_entities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    return {"Tokens": tokens, "Predictions": predictions}

# Extract entities from preprocessed data
preprocessed_df["Entities"] = preprocessed_df["Text"].apply(lambda x: extract_entities(x))
extracted_csv = os.path.join(OUTPUT_DIR, "extracted_entities.csv")
preprocessed_df.to_csv(extracted_csv, index=False)
print(f"Extracted entities saved to {extracted_csv}")

# Fine-tuning the model with annotated corpus
def fine_tune_model():
    df = pd.read_csv(ANNOTATED_DATA_PATH)  # Load custom annotated corpus
    df["combined_text"] = df.fillna('').apply(lambda row: ' '.join(row.astype(str)), axis=1)
    train_texts, val_texts = train_test_split(df["combined_text"].tolist(), test_size=0.2, random_state=42)
    
    class CustomDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            return {key: val.squeeze(0) for key, val in encoding.items()}
    
    train_dataset = CustomDataset(train_texts, tokenizer)
    val_dataset = CustomDataset(val_texts, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        save_strategy="epoch",
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Fine-tuned model saved successfully.")

fine_tune_model()

# Load fine-tuned model for re-extraction
def load_fine_tuned_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

tokenizer, model = load_fine_tuned_model()

# Extract entities again using fine-tuned model
preprocessed_df["Fine_Tuned_Entities"] = preprocessed_df["Text"].apply(lambda x: extract_entities(x))
fine_tuned_extracted_csv = os.path.join(OUTPUT_DIR, "fine_tuned_extracted_entities.csv")
preprocessed_df.to_csv(fine_tuned_extracted_csv, index=False)
print(f"Fine-tuned extracted entities saved to {fine_tuned_extracted_csv}")
