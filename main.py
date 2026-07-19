"""
main.py — Full Pipeline Entry Point
======================================
Cloud Carbon Footprint Evaluation — ML Pipeline
Obia John Simon | Bells University of Technology, Ota

Pipeline steps:
  1. Load & merge the REAL imported dataset (archive__3_.zip CSVs in data/)
  2. Preprocess data (clean, engineer features, encode, split, scale)
  3. Train Linear Regression + Random Forest
  4. Evaluate with regression AND classification metrics
  5. Print carbon reduction recommendations
  6. (Optional) Launch Flask web dashboard

Usage:
    python main.py              # Full pipeline
    python main.py --serve      # Pipeline + start web server
    python main.py --data-dir path/to/csvs   # Custom dataset location
"""

import os
import sys
import argparse

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
SRC_DIR   = os.path.join(BASE_DIR, "src")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, DATA_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Cloud Carbon Footprint ML Pipeline — Obia John Simon"
    )
    parser.add_argument("--serve", action="store_true",
                        help="Start Flask web server after training")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="Directory containing the dataset CSV files")
    args = parser.parse_args()

    print("=" * 62)
    print("  Cloud Carbon Footprint — ML Pipeline")
    print("  Obia John Simon | Bells University of Technology")
    print("  Dataset: Google Cloud real-world emission records")
    print("=" * 62)

    # ── Step 1: Load real dataset ────────────────────────────────
    print("\n📂 STEP 1: Loading real dataset from", args.data_dir)
    from load_dataset import build_master_dataset
    df_raw = build_master_dataset(data_dir=args.data_dir)

    # ── Step 2: Preprocess ───────────────────────────────────────
    print("\n🔧 STEP 2: Preprocessing data ...")
    from preprocess import preprocess
    (
        X_train, X_test, y_train, y_test, y_test_orig,
        scaler, encoders,
        feature_names, df_full
    ) = preprocess(df_raw, output_dir=MODEL_DIR)

    # ── Steps 3 & 4: Train + Evaluate ───────────────────────────
    print("\n🤖 STEP 3–4: Training & Evaluating models ...")
    from train import train_and_evaluate
    lr, rf, reg_df, cls_df = train_and_evaluate(
        X_train, X_test, y_train, y_test, y_test_orig,
        feature_names, df_full,
        model_dir=MODEL_DIR,
        output_dir=OUT_DIR
    )

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  REGRESSION RESULTS SUMMARY")
    print("=" * 62)
    print(reg_df[["model", "MAE", "RMSE", "R2",
                   "Accuracy_10pct", "CV_R2_mean"]].to_string(index=False))

    print("\n" + "=" * 62)
    print("  CLASSIFICATION RESULTS SUMMARY (Emission Bands)")
    print("=" * 62)
    print(cls_df[["model", "Accuracy", "Precision",
                   "Recall", "F1_Score"]].to_string(index=False))

    # ── Step 5: Carbon reduction recommendations ─────────────────
    print("\n" + "=" * 62)
    print("  CARBON REDUCTION RECOMMENDATIONS")
    print("=" * 62)
    recommendations = [
        "1. Migrate workloads to low-carbon regions "
          "(europe-north1: 180 gCO₂/kWh, europe-west1: 280).",
        "2. Replace Compute Engine VMs with Cloud Run / Cloud Functions "
          "for burst workloads — up to 80% CO₂ reduction.",
        "3. Optimise Vertex AI training: use TPUs instead of GPUs for "
          "large models; schedule training in europe-north1.",
        "4. Right-size GKE clusters — low-utilisation nodes waste energy; "
          "enable cluster autoscaler.",
        "5. Use BigQuery slot reservations to cap query energy use; "
          "avoid on-demand scans on multi-TB tables.",
        "6. Enable Cloud Storage lifecycle policies to archive/delete "
          "idle data — reduces storage energy by up to 40%.",
        "7. Time-shift batch BigQuery & Vertex AI jobs to off-peak hours "
          "when regional grid carbon is lowest.",
        "8. Track project-level emissions via the Carbon Footprint export "
          "to BigQuery and set emission budgets per team.",
    ]
    for tip in recommendations:
        print(f"  {tip}")

    print(f"\n✅ Pipeline complete.  Outputs → {OUT_DIR}")
    print(f"                        Models  → {MODEL_DIR}")

    # ── Step 6 (optional): serve ─────────────────────────────────
    if args.serve:
        print("\n🌐 Starting Flask web server on http://127.0.0.1:5000 ...")
        os.chdir(SRC_DIR)
        from app import app
        app.run(debug=False, port=5000)


if __name__ == "__main__":
    main()
