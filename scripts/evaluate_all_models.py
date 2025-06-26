import os
import pandas as pd
from utils import evaluate_performance

# Path to your ground truth BIO-tagged dataset
GROUND_TRUTH_PATH = "./data/bio_tagged_output.csv"

# Define all models with paths to their prediction CSVs
models = {
    "BioBERT": {
        "pretrained": "./output/pretrained_entity_extraction_results_biobert.csv",
        "fine_tuned": "./output/finetuned_entity_extraction_results_biobert.csv"
    },
    "BioClinicalBERT": {
        "pretrained": "./output/pretrained_entity_extraction_results_bioclinicalbert.csv",
        "fine_tuned": "./output/finetuned_entity_extraction_results_bioclinicalbert.csv"
    },
    "BlueBERT": {
        "pretrained": "./output/pretrained_entity_extraction_results_bluebert.csv",
        "fine_tuned": "./output/finetuned_entity_extraction_results_bluebert.csv"
    },
    "RoBERTa": {
        "pretrained": "./output/pretrained_entity_extraction_results_roberta.csv",
        "fine_tuned": "./output/finetuned_entity_extraction_results_roberta.csv"
    },
    "PubMedBERT": {
        "pretrained": "./output/pretrained_entity_extraction_results_pubmedbert.csv",
        "fine_tuned": "./output/finetuned_entity_extraction_results_pubmedbert.csv"
    }
}

# Load and prepare ground truth from bio_tagged_output.csv
def load_ground_truth():
    df = pd.read_csv(GROUND_TRUTH_PATH)
    df.columns = df.columns.str.strip()
    df["Token_Index"] = df.groupby("Sentence_ID").cumcount()
    return df

# Convert prediction format: "Token:LABEL Token:LABEL ..." → row-wise DataFrame
def explode_predictions(df):
    exploded = []
    for _, row in df.iterrows():
        sent_id = row.get("Sentence_ID") or row.get("Filename") or 0
        predicted = str(row.get("predicted_entities", "")).split()
        for i, pair in enumerate(predicted):
            if ":" not in pair:
                continue
            token, label = pair.rsplit(":", 1)
            exploded.append({
                "Sentence_ID": sent_id,
                "Token_Index": i,
                "Token": token,
                "Label": label
            })
    return pd.DataFrame(exploded)

# Compare predicted labels with ground truth and evaluate
def evaluate_model(model_name, phase, prediction_file, ground_truth_df):
    if not os.path.exists(prediction_file):
        print(f"[!] {phase} predictions for {model_name} not found. Skipping.")
        return

    pred_df = pd.read_csv(prediction_file)
    pred_df = explode_predictions(pred_df)

    # Merge prediction with gold based on Sentence_ID, Token_Index, and Token
    merged = pd.merge(
        ground_truth_df,
        pred_df,
        on=["Sentence_ID", "Token_Index", "Token"],
        how="inner",
        suffixes=("_true", "_pred")
    )

    if merged.empty:
        print(f"[!] No token match found for {model_name} ({phase}).")
        return

    y_true = merged["Label_true"].tolist()
    y_pred = merged["Label_pred"].tolist()

    print(f"\n{model_name} — {phase} Evaluation:")
    metrics = evaluate_performance(y_true, y_pred)
    print(metrics)

    
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Macro F1 Score:     {round(macro_f1, 4)}")
    print(f"Weighted F1 Score:  {round(weighted_f1, 4)}")


# Main script
if __name__ == "__main__":
    print("Starting BIO-tagged Evaluation of All Models...\n")
    ground_truth_df = load_ground_truth()

    for model, paths in models.items():
        print(f"\n==== {model} =====")
        evaluate_model(model, "Pretrained", paths["pretrained"], ground_truth_df)
        evaluate_model(model, "Fine-tuned", paths["fine_tuned"], ground_truth_df)
