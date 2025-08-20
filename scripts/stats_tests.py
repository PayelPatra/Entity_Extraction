import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

input_file = "./results/all_runs_metrics.csv"
output_dir = "./results"

models = ["BioBERT", "BioClinicalBERT", "BlueBERT", "RoBERTa", "PubMedBERT"]
phases = ["pretrained", "finetuned"]
metrics = [("recall", "Recall"), ("precision", "Precision"), ("f1_macro", "Macro F1")]

def get_pvalues(data, metric, phase):
    model_list = sorted(data["model"].unique().tolist())
    table = pd.DataFrame(index=model_list, columns=model_list, dtype=float)
    for m1 in model_list:
        a = data[(data.model == m1) & (data.phase == phase)][metric].values
        for m2 in model_list:
            if m1 == m2:
                table.loc[m1, m2] = 1.0
            else:
                b = data[(data.model == m2) & (data.phase == phase)][metric].values
                if len(a) >= 2 and len(b) >= 2:
                    _, p = ttest_ind(a, b, equal_var=False)
                    table.loc[m1, m2] = float(p)
                else:
                    table.loc[m1, m2] = np.nan
    return table

def make_latex(table, title):
    names = list(table.index)
    header = " & " + " & ".join(names) + " \\\\"
    rows = []
    for m in names:
        row = [m] + [f"{table.loc[m, n]:.4f}" if pd.notna(table.loc[m, n]) else "" for n in names]
        rows.append(" & ".join(row) + " \\\\")
    body = "\n\\hline\n".join(rows)
    text = (
        "\\begin{table*}[!ht]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{title}}}\n"
        "\\begin{tabular}{|l|" + "c|"*len(names) + "}\n"
        "\\hline\n"
        f"{header}\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )
    return text

def main():
    if not os.path.exists(input_file):
        print(f"Missing {input_file}")
        return

    data = pd.read_csv(input_file)
    os.makedirs(output_dir, exist_ok=True)

    for phase in phases:
        for metric, label in metrics:
            table = get_pvalues(data, metric, phase)
            table.to_csv(os.path.join(output_dir, f"pvals_{metric}_{phase}.csv"), float_format="%.6f")
            latex = make_latex(table, f"P-values of models for {label} ({phase})")
            with open(os.path.join(output_dir, f"pvals_{metric}_{phase}.tex"), "w", encoding="utf-8") as f:
                f.write(latex)
            print(f"\n=== {phase.upper()} — {label} ===")
            print(table.round(4))

if __name__ == "__main__":
    main()
