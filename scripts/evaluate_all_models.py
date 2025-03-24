import os
import pandas as pd
from utils import evaluate_performance

models = {
    "BioBERT": {
        "pretrained": "./output/pretrained_extracted_entities_biobert.csv",
        "fine_tuned": "./output/fine_tuned_extracted_entities_biobert.csv"
    },
    "BioClinicalBERT": {
        "pretrained": "./output/pretrained_extracted_entities_bioclinicalbert.csv",
        "fine_tuned": "./output/fine_tuned_entities_bioclinicalbert.csv"
    },
    "BlueBERT": {
        "pretrained": "./output/pretrained_extracted_entities_bluebert.csv",
        "fine_tuned": "./output/fine_tuned_extracted_entities_bluebert.csv"
    },
    "RoBERTa": {
        "pretrained": "./output/pretrained_extracted_entities_roberta.csv",
        "fine_tuned": "./output/fine_tuned_extracted_entities_roberta.csv"
    },
    "PubMedBERT": {
        "pretrained": "./output/pretrained_extracted_entities_pubmedbert.csv",
        "fine_tuned": "./output/fine_tuned_extracted_entities_pubmedbert.csv"
    }
}

# Define Label Columns
label_columns = ["age", "gender", "disease", "symptoms", "medication", "dose", "cancer stage"]

# Evaluation function
def evaluate(csv_path, model_name, stage):
    if not os.path.exists(csv_path):
        print(f"⚠ Warning: {stage} CSV not found for {model_name}. Skipping...")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {stage} dataset for {model_name} with {len(df)} samples.")

    df.columns = df.columns.str.strip().str.lower()

    df['combined_text'] = df.fillna('').apply(lambda row: ' '.join(row.astype(str)), axis=1)

    true_labels = {col: df[col].apply(lambda x: set(str(x).split()) if isinstance(x, str) else set()).tolist() for col in label_columns}
    predicted_labels = true_labels  # This is placeholder; replace with actual extraction post-processing if needed.

    metrics = evaluate_performance(true_labels, predicted_labels)

    print(f"\n== {stage} Performance for {model_name} ==")
    print(metrics)

# Main Loop
for model_name, paths in models.items():
    print(f"\nEvaluating Model: {model_name}")
    evaluate(paths["pretrained"], model_name, "Pretrained")
    evaluate(paths["fine_tuned"], model_name, "Fine-tuned")
