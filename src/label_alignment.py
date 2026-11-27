"""Quantify alignment between rating-proxy labels (0-4 / 5-7 / 8-10) and human annotations.

Reads reports/label_review_annotated.csv (kept separate from reports/label_review_sample.csv
so re-running src.audit never overwrites human annotations) and writes:
- reports/label_alignment_summary.json
- reports/label_alignment_report.md
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix

from src.modeling import CLASSES


def evaluate_label_alignment(
    sample_path: str | Path = "reports/label_review_sample.csv",
    annotated_path: str | Path = "reports/label_review_annotated.csv",
    output_json: str | Path = "reports/label_alignment_summary.json",
    output_md: str | Path = "reports/label_alignment_report.md",
) -> dict:
    sample_df = pd.read_csv(sample_path)
    annotated_df = pd.read_csv(annotated_path)

    if len(annotated_df) != len(sample_df):
        raise ValueError(f"Row count mismatch: expected {len(sample_df)}, got {len(annotated_df)}")
    if not annotated_df["review_id"].equals(sample_df["review_id"]):
        raise ValueError("review_id order/values do not match reports/label_review_sample.csv")
    if not annotated_df["sentiment"].equals(sample_df["sentiment"]):
        raise ValueError("Rating-proxy sentiment column does not match reports/label_review_sample.csv")

    human = annotated_df["human_sentiment"].astype(str).str.strip()
    invalid = set(human) - set(CLASSES)
    if invalid:
        raise ValueError(f"Invalid or blank human_sentiment values found: {sorted(invalid)}")

    proxy = annotated_df["sentiment"].astype(str).str.strip()
    agreement = float(accuracy_score(human, proxy))
    kappa = float(cohen_kappa_score(human, proxy, labels=CLASSES))
    cm = confusion_matrix(proxy, human, labels=CLASSES)
    report = classification_report(human, proxy, labels=CLASSES, output_dict=True, zero_division=0)

    per_bucket = {}
    for bucket in CLASSES:
        subset = annotated_df[proxy == bucket]
        matches = int((subset["human_sentiment"] == bucket).sum())
        per_bucket[bucket] = {
            "sample_count": int(len(subset)),
            "exact_matches": matches,
            "alignment_rate": float(matches / len(subset)) if len(subset) else 0.0,
            "human_distribution": subset["human_sentiment"].value_counts().to_dict(),
        }

    disagreements = annotated_df[proxy != human][
        ["review_id", "score", "sentiment", "human_sentiment", "review_notes"]
    ].to_dict(orient="records")

    summary = {
        "sample_size": int(len(annotated_df)),
        "exact_agreement": agreement,
        "exact_matches": int((proxy == human).sum()),
        "cohen_kappa": kappa,
        "labels": CLASSES,
        "confusion_matrix_proxy_rows_human_cols": cm.tolist(),
        "per_rating_bucket": per_bucket,
        "classification_report_proxy_vs_human": report,
        "disagreement_count": int(len(disagreements)),
        "disagreements": disagreements,
    }

    Path(output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Rating-Proxy vs. Human Sentiment Alignment (60-Row Training Sample)",
        "",
        f"- **Sample size:** {summary['sample_size']} reviews (20 per rating bucket from training split)",
        f"- **Exact agreement:** {summary['exact_matches']}/{summary['sample_size']} ({agreement:.1%})",
        f"- **Cohen's kappa:** {kappa:.4f}",
        "",
        "## Per-Bucket Alignment",
        "",
        "| Rating Bucket | Sampled | Matches | Alignment | Human Label Breakdown |",
        "|---|---:|---:|---:|---|",
    ]
    for bucket in CLASSES:
        info = per_bucket[bucket]
        breakdown = ", ".join(f"{k}: {v}" for k, v in info["human_distribution"].items())
        lines.append(
            f"| {bucket} | {info['sample_count']} | {info['exact_matches']} | "
            f"{info['alignment_rate']:.1%} | {breakdown} |"
        )

    lines.extend([
        "",
        "## Confusion Matrix (Rows = Rating Proxy, Columns = Human Sentiment)",
        "",
        "| Proxy \\\\ Human | Negative | Neutral | Positive |",
        "|---|---:|---:|---:|",
    ])
    for i, bucket in enumerate(CLASSES):
        lines.append(f"| {bucket} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |")

    lines.extend([
        "",
        "## Key Finding",
        "",
        "- **Positive (8–10)** and **Negative (0–4)** align strongly with human text sentiment "
        f"({per_bucket['Positive']['alignment_rate']:.0%} and {per_bucket['Negative']['alignment_rate']:.0%}).",
        f"- **Neutral (5–7)** aligns only **{per_bucket['Neutral']['alignment_rate']:.0%}** of the time; "
        "most 5–7 reviews express predominantly negative criticism in text despite a mid-range numeric score.",
    ])

    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate_label_alignment(), indent=2, ensure_ascii=False))
