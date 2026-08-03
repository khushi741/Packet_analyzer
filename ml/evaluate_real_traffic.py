"""
Real-World Generalization Test.

Evaluates the SYNTHETIC-TRAINED model against REAL CAPTURED traffic data.
This is the honest test: can a model trained on generated data actually
classify real network traffic?

Usage:
    python ml/evaluate_real_traffic.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

CAPTURED_PATH = "ml/dataset/captured_flows.csv"
MODELS_DIR = "ml/models"
EVAL_DIR = "ml/models/evaluation"

FEATURE_COLS = [
    "flow_duration_ms", "total_packets", "total_bytes",
    "forward_packets", "backward_packets", "forward_bytes", "backward_bytes",
    "mean_packet_size", "std_packet_size", "min_packet_size", "max_packet_size",
    "mean_inter_arrival_time_ms", "std_inter_arrival_time_ms",
    "packets_per_second", "bytes_per_second", "tcp_udp_flag",
]


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def evaluate_real():
    # ── Load model ───────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "traffic_classifier.joblib")
    if not os.path.exists(model_path):
        print("ERROR: No trained model found. Run: python ml/train_model.py")
        return

    model = joblib.load(model_path)

    # ── Load captured data ───────────────────────────────────────────────────
    if not os.path.exists(CAPTURED_PATH):
        print(f"ERROR: No captured traffic data at {CAPTURED_PATH}")
        print("Run:  python ml/capture_real_traffic.py --label <app> --duration 600")
        return

    df = pd.read_csv(CAPTURED_PATH)

    print_header("Real-World Generalization Test")
    print(f"  Model: trained on SYNTHETIC data only")
    print(f"  Test data: {len(df)} REAL captured flows")
    print(f"  Source: {CAPTURED_PATH}")

    # Report class distribution
    print(f"\n  Captured class distribution:")
    for label, count in df["label"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {label:16s}: {count:6d} ({pct:.1f}%)")

    # ── Clean data ───────────────────────────────────────────────────────────
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = df[col].replace([np.inf, -np.inf], 0)

    # Filter to only classes the model knows
    known_classes = list(model.classes_)
    df_known = df[df["label"].isin(known_classes)].copy()
    df_unknown = df[~df["label"].isin(known_classes)]

    if len(df_unknown) > 0:
        print(f"\n  ⚠ {len(df_unknown)} flows have labels unknown to the model:")
        for label, count in df_unknown["label"].value_counts().items():
            print(f"    '{label}' ({count} flows) — not in training classes {known_classes}")

    if len(df_known) == 0:
        print("\n  ERROR: No captured flows match training classes. Cannot evaluate.")
        return

    X_real = df_known[FEATURE_COLS]
    y_real = df_known["label"]

    # ── Predict ──────────────────────────────────────────────────────────────
    predictions = model.predict(X_real)
    probabilities = model.predict_proba(X_real)
    confidences = np.max(probabilities, axis=1)

    # ── Metrics ──────────────────────────────────────────────────────────────
    real_acc = accuracy_score(y_real, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_real, predictions, average="weighted", zero_division=0
    )

    print_header("Results: Synthetic Model → Real Traffic")
    print(f"  Accuracy:           {real_acc:.4f} ({real_acc*100:.1f}%)")
    print(f"  Weighted Precision: {precision:.4f}")
    print(f"  Weighted Recall:    {recall:.4f}")
    print(f"  Weighted F1-Score:  {f1:.4f}")
    print(f"  Mean Confidence:    {confidences.mean():.4f}")

    # Per-class report
    print_header("Per-Class Breakdown (Real Data)")
    report_text = classification_report(y_real, predictions, digits=4, zero_division=0)
    print(report_text)

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    present_classes = sorted(set(y_real.unique()) | set(predictions))
    cm = confusion_matrix(y_real, predictions, labels=present_classes)

    print_header("Confusion Matrix (Real Data)")
    df_cm = pd.DataFrame(cm, index=present_classes, columns=present_classes)
    print(df_cm.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=present_classes, yticklabels=present_classes,
        linewidths=0.5, linecolor="gray", ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Synthetic Model on REAL Traffic", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, "confusion_matrix_real.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {cm_path}")

    # ── Error Analysis ───────────────────────────────────────────────────────
    print_header("Error Analysis")

    # Per-class accuracy
    per_class_acc = {}
    for cls in present_classes:
        mask = y_real == cls
        if mask.sum() > 0:
            cls_acc = accuracy_score(y_real[mask], predictions[mask])
            per_class_acc[cls] = cls_acc
            n_correct = (predictions[mask] == cls).sum()
            n_total = mask.sum()
            status = "✓" if cls_acc >= 0.7 else "⚠ STRUGGLING"
            print(f"  {cls:16s}: {cls_acc:.1%} ({n_correct}/{n_total}) {status}")

    # Find worst classes
    if per_class_acc:
        worst_class = min(per_class_acc, key=per_class_acc.get)
        worst_acc = per_class_acc[worst_class]

        print(f"\n  Worst performing class: '{worst_class}' ({worst_acc:.1%})")

        # Analyze misclassifications
        misclassified = predictions[y_real != predictions]
        true_misclassified = y_real[y_real != predictions]
        if len(misclassified) > 0:
            print(f"\n  Most common misclassification patterns:")
            pairs = list(zip(true_misclassified, misclassified))
            from collections import Counter
            for (true_l, pred_l), count in Counter(pairs).most_common(5):
                print(f"    {true_l:16s} → predicted as {pred_l:16s} ({count} times)")

    # Hypotheses for accuracy difference
    # Load synthetic benchmark for comparison
    synth_summary_path = os.path.join(EVAL_DIR, "evaluation_summary.json")
    if os.path.exists(synth_summary_path):
        with open(synth_summary_path) as f:
            synth_summary = json.load(f)
        synth_acc = synth_summary.get("test_accuracy", 0)
        gap = synth_acc - real_acc

        print(f"\n  ┌────────────────────────────────────────────┐")
        print(f"  │  Synthetic benchmark:  {synth_acc:.4f} ({synth_acc*100:.1f}%)        │")
        print(f"  │  Real-world accuracy:  {real_acc:.4f} ({real_acc*100:.1f}%)        │")
        print(f"  │  Generalization gap:   {gap:.4f} ({gap*100:.1f}pp)         │")
        print(f"  └────────────────────────────────────────────┘")

        if gap > 0.15:
            print(f"\n  ⚠ LARGE generalization gap ({gap*100:.0f}pp).")
            print(f"  Likely causes:")
            print(f"    1. Synthetic data has clean, well-separated class boundaries")
            print(f"       while real traffic has overlapping feature distributions")
            print(f"    2. Real network jitter, variable MTU, and background noise")
            print(f"       introduce variance not modeled in synthetic generation")
            print(f"    3. Feature estimation from connection metadata (psutil) is")
            print(f"       coarser than true packet-level statistics")
            print(f"    4. Real traffic includes CDN/proxy intermediaries that shift")
            print(f"       behavioral patterns (e.g., YouTube via Google CDN)")
        elif gap > 0.05:
            print(f"\n  Moderate gap — model generalizes reasonably but has room to improve.")
        else:
            print(f"\n  ✓ Small gap — model generalizes well to real traffic!")

    # ── Save results ─────────────────────────────────────────────────────────
    # Update evaluation_summary.json with real-world results
    if os.path.exists(synth_summary_path):
        with open(synth_summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    summary["real_world_test"] = {
        "test_data": CAPTURED_PATH,
        "test_size": len(df_known),
        "accuracy": round(real_acc, 4),
        "weighted_precision": round(precision, 4),
        "weighted_recall": round(recall, 4),
        "weighted_f1": round(f1, 4),
        "mean_confidence": round(float(confidences.mean()), 4),
        "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
        "generalization_gap": round(synth_acc - real_acc, 4) if 'synth_acc' in dir() else None,
    }
    # Remove None values
    summary["real_world_test"] = {k: v for k, v in summary["real_world_test"].items() if v is not None}

    with open(synth_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Updated: {synth_summary_path}")

    # Save detailed real-world report
    real_report_path = os.path.join(EVAL_DIR, "real_world_report.txt")
    with open(real_report_path, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  Real-World Generalization Test\n")
        f.write(f"  Model trained on: SYNTHETIC data only\n")
        f.write(f"  Evaluated on: {len(df_known)} REAL captured flows\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Accuracy:           {real_acc:.4f}\n")
        f.write(f"Weighted Precision: {precision:.4f}\n")
        f.write(f"Weighted Recall:    {recall:.4f}\n")
        f.write(f"Weighted F1-Score:  {f1:.4f}\n")
        f.write(f"\n{report_text}\n")
    print(f"  Saved: {real_report_path}")


if __name__ == "__main__":
    evaluate_real()
