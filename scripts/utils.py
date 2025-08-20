from sklearn.metrics import classification_report

def evaluate_performance(true_labels, predicted_labels, exclude_o=True):
    report = classification_report(
        true_labels,
        predicted_labels,
        output_dict=True,
        zero_division=0
    )

    keys = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
    if exclude_o and "O" in keys:
        keys.remove("O")

    if keys:
        precision = sum(report[k]["precision"] for k in keys) / len(keys)
        recall = sum(report[k]["recall"] for k in keys) / len(keys)
        f1 = sum(report[k]["f1-score"] for k in keys) / len(keys)
    else:
        precision = report["macro avg"]["precision"]
        recall = report["macro avg"]["recall"]
        f1 = report["macro avg"]["f1-score"]

    accuracy = report["accuracy"]

    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Accuracy": round(accuracy, 4)
    }
