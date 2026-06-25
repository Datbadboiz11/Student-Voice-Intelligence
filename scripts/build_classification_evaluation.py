from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated classification evaluation reports.")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/evaluation"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_confusion_matrix(source: Path, destination: Path, title: str) -> None:
    matrix = pd.read_csv(source, index_col=0)
    fig, axis = plt.subplots(figsize=(max(6, len(matrix.columns) * 1.15), max(5, len(matrix.index) * 0.95)))
    image = axis.imshow(matrix.values, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    threshold = matrix.values.max() / 2 if matrix.values.size else 0
    for row_index, row in enumerate(matrix.values):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{int(value):,}", ha="center", va="center", fontsize=8,
                      color="white" if value > threshold else "#172033")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def class_report_markdown(path: Path, heading: str) -> list[str]:
    frame = pd.read_csv(path, index_col=0)
    wanted = [index for index in frame.index if index not in {"accuracy", "macro avg", "weighted avg"}]
    lines = [f"### {heading}", "", "| Class | Precision | Recall | F1-score | Support |", "|---|---:|---:|---:|---:|"]
    for label in wanted:
        row = frame.loc[label]
        lines.append(
            f"| {label} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['f1-score']):.4f} | {int(row['support']):,} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    reports = args.reports_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(reports / "baseline/baseline_results.csv")
    baseline_test = baseline[baseline["split"] == "test"]
    sentiment_baseline = baseline_test[
        (baseline_test["task_name"] == "sentiment_3class") & (baseline_test["model_name"] == "tfidf_linear_svm")
    ].iloc[0]
    topic_baseline = baseline_test[
        (baseline_test["task_name"] == "topic_group") & (baseline_test["model_name"] == "tfidf_linear_svm")
    ].iloc[0]

    sentiment_models = pd.read_csv(reports / "transformer/sentiment_model_comparison.csv")
    sentiment_rows = [
        {
            "task": "sentiment",
            "model": "TF-IDF + Linear SVM",
            "family": "baseline",
            "accuracy": sentiment_baseline["accuracy"],
            "macro_f1": sentiment_baseline["macro_f1"],
            "weighted_f1": sentiment_baseline["weighted_f1"],
        }
    ]
    for _, row in sentiment_models[sentiment_models["model_key"] != "tfidf_linear_svm"].iterrows():
        sentiment_rows.append(
            {
                "task": "sentiment",
                "model": row["model_name"],
                "family": "transformer",
                "accuracy": row["test_accuracy"],
                "macro_f1": row["test_macro_f1"],
                "weighted_f1": row["test_weighted_f1"],
            }
        )

    topic_metrics = load_json(reports / "transformer/topic_phobertv2_noweight/metrics.json")["test"]
    topic_rows = [
        {
            "task": "topic",
            "model": "TF-IDF + Linear SVM",
            "family": "baseline",
            "accuracy": topic_baseline["accuracy"],
            "macro_f1": topic_baseline["macro_f1"],
            "weighted_f1": topic_baseline["weighted_f1"],
        },
        {
            "task": "topic",
            "model": "PhoBERT-base-v2 (no class weight)",
            "family": "transformer",
            "accuracy": topic_metrics["accuracy"],
            "macro_f1": topic_metrics["macro_f1"],
            "weighted_f1": topic_metrics["weighted_f1"],
        },
    ]
    metrics = pd.DataFrame([*sentiment_rows, *topic_rows]).sort_values(["task", "macro_f1"], ascending=[True, False])
    metrics_path = output / "classification_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    sentiment_cm = reports / "transformer/phobertv2/test_confusion_matrix.csv"
    topic_cm = reports / "transformer/topic_phobertv2_noweight/test_confusion_matrix.csv"
    save_confusion_matrix(sentiment_cm, output / "sentiment_confusion_matrix.png", "Sentiment — PhoBERT-base-v2")
    save_confusion_matrix(topic_cm, output / "topic_confusion_matrix.png", "Topic — PhoBERT-base-v2 (no class weight)")

    lines = ["# Classification Evaluation", "", "All metrics below are from the held-out test split.", ""]
    for task, group in metrics.groupby("task", sort=False):
        lines.extend([f"## {task.title()} models", "", "| Model | Family | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---|---:|---:|---:|"])
        for _, row in group.iterrows():
            lines.append(f"| {row['model']} | {row['family']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} |")
        best = group.iloc[0]
        lines.extend(["", f"Best {task} model by Macro-F1: **{best['model']}** ({best['macro_f1']:.4f}).", ""])
    lines.extend(class_report_markdown(reports / "transformer/phobertv2/test_classification_report.csv", "Sentiment per-class report"))
    lines.append("")
    lines.extend(class_report_markdown(reports / "transformer/topic_phobertv2_noweight/test_classification_report.csv", "Topic per-class report"))
    lines.extend(["", "## Confusion matrices", "", "- `sentiment_confusion_matrix.png`", "- `topic_confusion_matrix.png`", ""])
    report_path = output / "classification_evaluation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {metrics_path}, {report_path}, and two confusion-matrix PNG files in {output}.")


if __name__ == "__main__":
    main()
