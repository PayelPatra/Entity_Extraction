import os
import pandas as pd
from sklearn.metrics import f1_score
from utils import evaluate_performance

GROUND_TRUTH_SEQ = "./data/bio_tagged_output.csv"
GROUND_TRUTH_TOK = "./data/bio_tagged_output_tokens.csv"

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

def load_gold():
    tok = pd.read_csv(GROUND_TRUTH_TOK)
    seq = pd.read_csv(GROUND_TRUTH_SEQ).reset_index().rename(columns={"index": "Sentence_ID"})
    text_to_sid = seq.set_index("text")["Sentence_ID"].to_dict()
    return tok, text_to_sid

def explode_predictions(pred_df, text_to_sid):
    rows = []
    for _, row in pred_df.iterrows():
        text = row.get("text", None)
        if text is None or text not in text_to_sid:
            continue
        sid = text_to_sid[text]
        items = str(row.get("predicted_entities", "")).split()
        for i, pair in enumerate(items):
            if ":" not in pair:
                continue
            tok, lab = pair.rsplit(":", 1)
            rows.append({
                "Sentence_ID": sid,
                "Token_Index": i,
                "Token": tok,
                "Label": lab
            })
    return pd.DataFrame(rows)

def evaluate_model(model_name, phase, pred_path, gold_tok, text_to_sid):
    if not os.path.exists(pred_path):
        print(f"[!] {phase} predictions for {model_name} not found: {pred_path}")
        return
    pred_df = pd.read_csv(pred_path)
    pred_tok = explode_predictions(pred_df, text_to_sid)
    if pred_tok.empty:
        print(f"[!] No alignable predictions for {model_name} ({phase}).")
        return
    merged = pd.merge(
        gold_tok, pred_tok,
        on=["Sentence_ID", "Token_Index", "Token"],
        how="inner",
        suffixes=("_true", "_pred")
    )
    if merged.empty:
        print(f"[!] No token matches after merge for {model_name} ({phase}).")
        return
    y_true = merged["Label_true"].tolist()
    y_pred = merged["Label_pred"].tolist()
    print(f"\n{model_name} — {phase} Evaluation:")
    metrics = evaluate_performance(y_true, y_pred)
    print(metrics)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")

if __name__ == "__main__":
    print("Starting BIO-tagged Evaluation of All Models...\n")
    gold_tok, text_to_sid = load_gold()
    for model, paths in models.items():
        print(f"\n==== {model} ====")
        evaluate_model(model, "Pretrained", paths["pretrained"], gold_tok, text_to_sid)
        evaluate_model(model, "Fine-tuned", paths["fine_tuned"], gold_tok, text_to_sid)
