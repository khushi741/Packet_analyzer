"""
Model Training & Evaluation Pipeline.

Trains RandomForest and GradientBoosting classifiers on the combined traffic
dataset, performs rigorous evaluation with per-class metrics, confusion matrix
heatmap, overfitting detection (train vs test), and DPI baseline comparison.

All evaluation artifacts are saved to: ml/models/evaluation/

Usage:
    python ml/train_model.py
"""

import os
import sys

# Configure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH = "ml/dataset/traffic_dataset.csv"
MODELS_DIR = "ml/models"
EVAL_DIR = "ml/models/evaluation"

FEATURE_COLS = [
    "flow_duration_ms", "total_packets", "total_bytes",
    "forward_packets", "backward_packets", "forward_bytes", "backward_bytes",
    "mean_packet_size", "std_packet_size", "min_packet_size", "max_packet_size",
    "mean_inter_arrival_time_ms", "std_inter_arrival_time_ms",
    "packets_per_second", "bytes_per_second", "tcp_udp_flag",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def has_real_sni(val):
    """Check if a detected_sni value is a real, non-empty domain string."""
    if pd.isna(val):
        return False
    s = str(val).strip()
    return len(s) > 0 and s.lower() not in ("", "nan", "none", "unknown")


def train_model():
    # ── Load dataset ─────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset {DATASET_PATH} not found.")
        print("Run:  python ml/build_dataset.py")
        return

    df = pd.read_csv(DATASET_PATH)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # ── Dataset summary ──────────────────────────────────────────────────────
    print_header("Dataset Summary")
    print(f"  Total samples: {len(df)}")
    print(f"  Features:      {len(FEATURE_COLS)}")
    print(f"  Classes:       {df['label'].nunique()}")

    # Report data source breakdown with explicit warnings
    has_source_col = "source" in df.columns
    if has_source_col:
        source_counts = df["source"].value_counts().to_dict()
        captured = source_counts.get("captured", 0)
        iscx = source_counts.get("iscx", 0)
        synthetic = source_counts.get("synthetic", 0)
        cpp_export = source_counts.get("cpp_export", 0)
        real_total = captured + iscx + cpp_export

        print(f"\n  Data source breakdown:")
        for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            pct = count / len(df) * 100
            print(f"    {src:12s}: {count:6d} ({pct:.1f}%)")

        print()
        if real_total == 0:
            print("  ⚠ CRITICAL: Dataset is 100% SYNTHETIC.")
            print("    No real traffic data. Reported accuracy is a synthetic benchmark")
            print("    and will likely be LOWER (est. 70-85%) on real network traffic.")
        elif synthetic > real_total:
            pct_synth = synthetic / len(df) * 100
            print(f"  ⚠ WARNING: Synthetic data dominates ({pct_synth:.0f}% of dataset).")
            print(f"    Consider capturing more real traffic to improve generalization.")
    else:
        print("\n  ⚠ No 'source' column. Cannot verify data provenance.")

    print(f"\n  Class distribution:")
    for label, count in df["label"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {label:16s}: {count:6d} ({pct:.1f}%)")

    # ── Prepare features and labels ──────────────────────────────────────────
    X = df[FEATURE_COLS].copy()
    y = df["label"].copy()

    # Save feature column order for inference
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_columns.joblib"))

    # ── Train/test split with session awareness ──────────────────────────────
    # If we have a 'source' column and real captured data, use GroupShuffleSplit
    # to prevent session leakage (flows from same capture session in both sets).
    # For synthetic-only data, use standard stratified split (no session correlation).

    use_grouped_split = False
    if has_source_col and "captured" in df["source"].values:
        # Build session groups: captured data is grouped, synthetic/iscx are individual
        # Each captured flow gets a session ID based on label + source
        groups = []
        session_counter = 0
        for idx, row in df.iterrows():
            if row["source"] == "captured":
                # Group by label (all youtube captures = same session, etc.)
                groups.append(f"captured_{row['label']}")
            else:
                # Each synthetic/iscx row is its own "group" (independent)
                groups.append(f"independent_{session_counter}")
                session_counter += 1
        df["_group"] = groups

        n_captured_groups = len(set(g for g in groups if g.startswith("captured_")))
        if n_captured_groups >= 2:
            use_grouped_split = True
            print(f"\n  Split strategy: GroupShuffleSplit (session-aware)")
            print(f"    Captured session groups: {n_captured_groups}")
            gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            train_idx, test_idx = next(gss.split(X, y, groups=df["_group"]))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        df.drop("_group", axis=1, inplace=True)

    if not use_grouped_split:
        split_reason = "synthetic-only data (no session correlation)" if (has_source_col and captured == 0) else "standard"
        print(f"\n  Split strategy: Stratified random ({split_reason})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

    # Keep track of indices for DPI baseline comparison
    test_indices = X_test.index
    df_test = df.loc[test_indices].copy()

    print_header("Training Models")
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")

    # ── Train Random Forest ──────────────────────────────────────────────────
    print(f"\n  [1/3] Training Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    rf_train_preds = rf.predict(X_train)
    rf_test_preds = rf.predict(X_test)
    rf_train_acc = accuracy_score(y_train, rf_train_preds)
    rf_test_acc = accuracy_score(y_test, rf_test_preds)
    print(f"        Train accuracy: {rf_train_acc:.4f}")
    print(f"        Test accuracy:  {rf_test_acc:.4f}")

    # ── Train Gradient Boosting ──────────────────────────────────────────────
    print(f"\n  [2/3] Training Gradient Boosting (100 trees)...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)

    gb_train_preds = gb.predict(X_train)
    gb_test_preds = gb.predict(X_test)
    gb_train_acc = accuracy_score(y_train, gb_train_preds)
    gb_test_acc = accuracy_score(y_test, gb_test_preds)
    print(f"        Train accuracy: {gb_train_acc:.4f}")
    print(f"        Test accuracy:  {gb_test_acc:.4f}")

    # ── Select best model ────────────────────────────────────────────────────
    if rf_test_acc >= gb_test_acc:
        best_model = rf
        best_name = "RandomForest"
        best_train_preds = rf_train_preds
        best_test_preds = rf_test_preds
        best_train_acc = rf_train_acc
        best_test_acc = rf_test_acc
    else:
        best_model = gb
        best_name = "GradientBoosting"
        best_train_preds = gb_train_preds
        best_test_preds = gb_test_preds
        best_train_acc = gb_train_acc
        best_test_acc = gb_test_acc

    print(f"\n  ✓ Selected: {best_name} (test acc: {best_test_acc:.4f})")

    # ── Train Isolation Forest (anomaly detection) ───────────────────────────
    print(f"\n  [3/3] Training Isolation Forest (anomaly detector)...")
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    iso.fit(X)
    joblib.dump(iso, os.path.join(MODELS_DIR, "anomaly_detector.joblib"))
    print(f"        Saved anomaly detector.")

    # ── Save best model ──────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "traffic_classifier.joblib")
    joblib.dump(best_model, model_path)
    print(f"        Saved classifier to {model_path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    classes = sorted(y.unique())

    # ── 1. Per-class classification report ───────────────────────────────────
    print_header("Per-Class Evaluation (Test Set)")

    report_text = classification_report(y_test, best_test_preds, digits=4)
    print(report_text)

    train_report_text = classification_report(y_train, best_train_preds, digits=4)

    # Determine data provenance label
    if has_source_col:
        real_pct = real_total / len(df) * 100
        if real_total == 0:
            data_label = "100% SYNTHETIC (no real traffic)"
        elif real_pct >= 50:
            data_label = f"{real_pct:.0f}% real traffic, {100-real_pct:.0f}% synthetic"
        else:
            data_label = f"{100-real_pct:.0f}% synthetic, {real_pct:.0f}% real traffic"
    else:
        data_label = "Unknown provenance"

    # Save full report
    report_path = os.path.join(EVAL_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  Classification Report — {best_name}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Data composition: {data_label}\n")
        f.write(f"Train/Test split: {1-TEST_SIZE:.0%} / {TEST_SIZE:.0%}\n")
        f.write(f"Split strategy: {'GroupShuffleSplit (session-aware)' if use_grouped_split else 'Stratified random'}\n")

        if has_source_col:
            f.write(f"\nData source breakdown:\n")
            for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
                pct = count / len(df) * 100
                f.write(f"  {src}: {count} ({pct:.1f}%)\n")
            if real_total == 0:
                f.write(f"\n  ⚠ SYNTHETIC-ONLY BENCHMARK\n")
                f.write(f"  Accuracy on real traffic will likely be lower (est. 70-85%).\n")

        f.write(f"\n{'─'*70}\n")
        f.write(f"  MODEL COMPARISON\n")
        f.write(f"{'─'*70}\n")
        f.write(f"  Random Forest     — Train: {rf_train_acc:.4f}  Test: {rf_test_acc:.4f}")
        f.write(f"  {'← OVERFIT!' if rf_train_acc - rf_test_acc > 0.05 else ''}\n")
        f.write(f"  Gradient Boosting — Train: {gb_train_acc:.4f}  Test: {gb_test_acc:.4f}")
        f.write(f"  {'← OVERFIT!' if gb_train_acc - gb_test_acc > 0.05 else ''}\n")
        f.write(f"\n  Selected: {best_name}\n")

        f.write(f"\n{'─'*70}\n")
        f.write(f"  TEST SET PERFORMANCE ({best_name})\n")
        f.write(f"{'─'*70}\n\n")
        f.write(report_text)

        f.write(f"\n{'─'*70}\n")
        f.write(f"  TRAINING SET PERFORMANCE ({best_name}) — for overfitting check\n")
        f.write(f"{'─'*70}\n\n")
        f.write(train_report_text)

        f.write(f"\n{'─'*70}\n")
        f.write(f"  OVERFITTING ANALYSIS\n")
        f.write(f"{'─'*70}\n")
        gap = best_train_acc - best_test_acc
        f.write(f"  Train accuracy: {best_train_acc:.4f}\n")
        f.write(f"  Test accuracy:  {best_test_acc:.4f}\n")
        f.write(f"  Gap:            {gap:.4f}\n")
        if gap > 0.10:
            f.write(f"  ⚠ SIGNIFICANT OVERFITTING DETECTED (gap > 10%)\n")
        elif gap > 0.05:
            f.write(f"  ⚠ MILD OVERFITTING (gap 5-10%)\n")
        else:
            f.write(f"  ✓ No significant overfitting (gap < 5%)\n")

    print(f"  Saved: {report_path}")

    # ── 2. Overfitting analysis ──────────────────────────────────────────────
    print_header("Overfitting Analysis")
    print(f"  {'Model':<22s} {'Train Acc':>10s} {'Test Acc':>10s} {'Gap':>8s}  Status")
    print(f"  {'-'*60}")

    for name, tr_acc, te_acc in [
        ("Random Forest", rf_train_acc, rf_test_acc),
        ("Gradient Boosting", gb_train_acc, gb_test_acc),
    ]:
        gap = tr_acc - te_acc
        status = "✓ OK" if gap < 0.05 else ("⚠ MILD" if gap < 0.10 else "⚠ OVERFIT")
        print(f"  {name:<22s} {tr_acc:>10.4f} {te_acc:>10.4f} {gap:>8.4f}  {status}")

    # ── 3. Confusion matrix heatmap ──────────────────────────────────────────
    print_header("Confusion Matrix")
    cm = confusion_matrix(y_test, best_test_preds, labels=classes)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    print(df_cm.to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor="gray", ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    title_suffix = " [SYNTHETIC DATA]" if (has_source_col and real_total == 0) else ""
    ax.set_title(f"Confusion Matrix — {best_name} (Test Set){title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {cm_path}")

    # ── 4. Feature importance plot ───────────────────────────────────────────
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(FEATURE_COLS))
    ax.bar(range(len(FEATURE_COLS)), importances[indices], color=[colors[i] for i in range(len(FEATURE_COLS))])
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Feature Importances — {best_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(EVAL_DIR, "feature_importance.png")
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    # Also save to old location for backward compat
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(range(len(FEATURE_COLS)), importances[indices], color=[colors[i] for i in range(len(FEATURE_COLS))])
    ax2.set_xticks(range(len(FEATURE_COLS)))
    ax2.set_xticklabels([FEATURE_COLS[i] for i in indices], rotation=45, ha="right")
    ax2.set_title(f"Feature Importances — {best_name}", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Importance")
    plt.tight_layout()
    fig2.savefig(os.path.join(MODELS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig2)
    print(f"  Saved: {fi_path}")

    # ── 5. DPI baseline comparison (CORRECTED) ───────────────────────────────
    print_header("DPI Baseline Comparison")

    if "detected_sni" in df_test.columns and "dpi_app_type" in df_test.columns:
        # Correctly identify rows where DPI has REAL SNI information
        dpi_has_sni = df_test["detected_sni"].apply(has_real_sni)
        sni_subset = df_test[dpi_has_sni]
        no_sni_subset = df_test[~dpi_has_sni]

        print(f"  Test set total:                    {len(df_test)} flows")
        print(f"  Flows WITH real SNI (DPI works):   {len(sni_subset)} ({len(sni_subset)/len(df_test)*100:.1f}%)")
        print(f"  Flows WITHOUT SNI (DPI is blind):  {len(no_sni_subset)} ({len(no_sni_subset)/len(df_test)*100:.1f}%)")

        # Map DPI labels
        dpi_label_map = {
            "YOUTUBE": "youtube",
            "NETFLIX": "netflix",
            "ZOOM": "zoom",
            "FACEBOOK": "social_media",
            "HTTPS": "web_browsing",
            "HTTP": "web_browsing",
            "UNKNOWN": "unknown",
        }

        if len(sni_subset) > 0:
            print(f"\n  SNI values in test subset:")
            for sni, count in sni_subset["detected_sni"].value_counts().items():
                dpi_type = sni_subset[sni_subset["detected_sni"] == sni]["dpi_app_type"].iloc[0]
                print(f"    '{sni}' → DPI={dpi_type} ({count} flows)")

            # DPI accuracy on SNI-visible subset
            dpi_preds = sni_subset["dpi_app_type"].astype(str).str.upper().map(dpi_label_map).fillna("unknown")
            dpi_true = sni_subset["label"]
            dpi_acc_sni = accuracy_score(dpi_true, dpi_preds)

            # ML accuracy on the SAME SNI-visible subset
            X_sni = sni_subset[FEATURE_COLS]
            ml_preds_sni = best_model.predict(X_sni)
            ml_acc_sni = accuracy_score(dpi_true, ml_preds_sni)

            print(f"\n  ┌─────────────────────────────────────────────────┐")
            print(f"  │  Comparison on SNI-VISIBLE subset ({len(sni_subset):3d} flows)   │")
            print(f"  │  (Same test data for both methods)              │")
            print(f"  ├─────────────────────────────────────────────────┤")
            print(f"  │  DPI-only (rule-based SNI):  {dpi_acc_sni:.4f}             │")
            print(f"  │  ML model (behavioral):      {ml_acc_sni:.4f}             │")
            print(f"  └─────────────────────────────────────────────────┘")

            if ml_acc_sni > dpi_acc_sni:
                print(f"\n  ✓ ML outperforms DPI by {(ml_acc_sni - dpi_acc_sni)*100:.1f}pp on SNI-visible traffic")
            elif ml_acc_sni == dpi_acc_sni:
                print(f"\n  ≈ ML matches DPI on SNI-visible traffic")
            else:
                print(f"\n  Note: DPI outperforms ML by {(dpi_acc_sni - ml_acc_sni)*100:.1f}pp on SNI-visible traffic")
                print(f"  (Expected — DPI is exact when SNI is readable)")
        else:
            dpi_acc_sni = None
            ml_acc_sni = None
            print(f"\n  No test flows had real SNI — DPI baseline not computable.")

        # The key value proposition: ML classifies the SNI-blind subset
        if len(no_sni_subset) > 0:
            X_no_sni = no_sni_subset[FEATURE_COLS]
            ml_preds_no_sni = best_model.predict(X_no_sni)
            ml_acc_no_sni = accuracy_score(no_sni_subset["label"], ml_preds_no_sni)

            print(f"\n  ┌─────────────────────────────────────────────────┐")
            print(f"  │  ML on ENCRYPTED subset ({len(no_sni_subset):3d} flows, no SNI)    │")
            print(f"  │  DPI would classify ALL as 'unknown'            │")
            print(f"  │  ML accuracy: {ml_acc_no_sni:.4f}                            │")
            print(f"  └─────────────────────────────────────────────────┘")
        else:
            ml_acc_no_sni = None

        # Save to report
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'─'*70}\n")
            f.write(f"  DPI BASELINE COMPARISON\n")
            f.write(f"{'─'*70}\n")
            f.write(f"  Test set: {len(df_test)} flows\n")
            f.write(f"  With SNI: {len(sni_subset)} ({len(sni_subset)/len(df_test)*100:.1f}%)\n")
            f.write(f"  Without SNI: {len(no_sni_subset)} ({len(no_sni_subset)/len(df_test)*100:.1f}%)\n")
            if dpi_acc_sni is not None:
                f.write(f"\n  On SNI-visible subset ({len(sni_subset)} flows):\n")
                f.write(f"    DPI accuracy: {dpi_acc_sni:.4f}\n")
                f.write(f"    ML accuracy:  {ml_acc_sni:.4f}\n")
            if ml_acc_no_sni is not None:
                f.write(f"\n  On encrypted subset ({len(no_sni_subset)} flows, DPI=unknown):\n")
                f.write(f"    ML accuracy:  {ml_acc_no_sni:.4f}\n")
    else:
        print("  Columns 'detected_sni' / 'dpi_app_type' not in dataset.")

    # ── 6. Save evaluation summary as JSON ───────────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, best_test_preds, average="weighted"
    )

    summary = {
        "model": best_name,
        "dataset_size": len(df),
        "data_composition": data_label if has_source_col else "unknown",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "split_strategy": "GroupShuffleSplit" if use_grouped_split else "stratified_random",
        "train_accuracy": round(best_train_acc, 4),
        "test_accuracy": round(best_test_acc, 4),
        "weighted_precision": round(precision, 4),
        "weighted_recall": round(recall, 4),
        "weighted_f1": round(f1, 4),
        "overfitting_gap": round(best_train_acc - best_test_acc, 4),
        "classes": classes,
        "rf_test_accuracy": round(rf_test_acc, 4),
        "gb_test_accuracy": round(gb_test_acc, 4),
        "is_synthetic_only": has_source_col and real_total == 0,
        "caveat": "Trained on synthetic data only. Real-traffic accuracy will differ." if (has_source_col and real_total == 0) else None,
    }

    # Remove None values
    summary = {k: v for k, v in summary.items() if v is not None}

    summary_path = os.path.join(EVAL_DIR, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {summary_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    print_header("Final Results")
    print(f"  Model:               {best_name}")
    print(f"  Data:                {data_label}")
    print(f"  Test Accuracy:       {best_test_acc:.4f}")
    print(f"  Weighted Precision:  {precision:.4f}")
    print(f"  Weighted Recall:     {recall:.4f}")
    print(f"  Weighted F1-Score:   {f1:.4f}")
    print(f"  Overfitting Gap:     {best_train_acc - best_test_acc:.4f}")

    if has_source_col and real_total == 0:
        print()
        print(f"  ⚠ SYNTHETIC BENCHMARK ONLY")
        print(f"    To get real-world accuracy, capture traffic with:")
        print(f"    python ml/capture_traffic.py --interface \"Wi-Fi\" --label <app> --duration 600")

    print(f"\n  Artifacts saved to: {EVAL_DIR}/")
    print(f"    ├── classification_report.txt")
    print(f"    ├── confusion_matrix.png")
    print(f"    ├── feature_importance.png")
    print(f"    └── evaluation_summary.json")


if __name__ == "__main__":
    train_model()
