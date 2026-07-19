"""
train.py — Model Training & Evaluation
========================================
Implements Chapter 3 Objectives:
  - Train Linear Regression and Random Forest models
  - Evaluate with Regression metrics: MAE, MSE, RMSE, R²   (Section 3.4.2)
  - Evaluate with Classification metrics: Accuracy, Precision, Recall, F1
    (Section 3.4.1) — achieved by bucketing CO₂ into emission bands
  - Cross-validation (5-fold)
  - Feature importance plot (Random Forest)
  - Comparison charts between models

Emission bands used for classification metrics (kg CO₂):
  Low    : < 0.1
  Medium : 0.1 – 2.0
  High   : > 2.0
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

# ── Emission classification bands ─────────────────────────────
EMISSION_BINS   = [0, 0.1, 2.0, np.inf]
EMISSION_LABELS = ["Low", "Medium", "High"]


def _to_emission_class(y_kg: np.ndarray) -> np.ndarray:
    """Convert continuous CO₂ values (kg) to Low/Medium/High classes."""
    result = pd.cut(y_kg, bins=EMISSION_BINS, labels=EMISSION_LABELS, right=True)
    # Fill any NaN (values exactly at boundary edge) with 'Medium'
    return np.array(result.fillna("Medium").astype(str))


# ─────────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────────

def regression_metrics(y_true_log, y_pred_log, label="Model") -> dict:
    """Compute regression metrics; convert back to kg scale first."""
    y_true = np.expm1(np.array(y_true_log))
    y_pred = np.expm1(np.clip(np.array(y_pred_log), -10, 30))

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    acc_10 = (np.abs((y_pred - y_true) / (y_true + 1e-9)) <= 0.10).mean() * 100

    print(f"\n{'─'*52}")
    print(f"  {label} — Regression Metrics")
    print(f"{'─'*52}")
    print(f"  MAE            : {mae:.6f} kg CO₂")
    print(f"  MSE            : {mse:.6f}")
    print(f"  RMSE           : {rmse:.6f} kg CO₂")
    print(f"  R²             : {r2:.4f}")
    print(f"  MAPE           : {mape:.2f}%")
    print(f"  Accuracy ±10%  : {acc_10:.2f}%")

    return {
        "model": label,
        "MAE": mae, "MSE": mse, "RMSE": rmse,
        "R2": r2, "MAPE": mape, "Accuracy_10pct": acc_10,
    }


def classification_metrics(y_true_log, y_pred_log, label="Model") -> dict:
    """
    Convert CO₂ predictions to emission classes and compute
    Accuracy, Precision, Recall, F1-Score — as specified in Objectives.
    """
    y_true_kg = np.expm1(np.array(y_true_log))
    y_pred_kg = np.expm1(np.clip(np.array(y_pred_log), -10, 30))

    y_true_cls = _to_emission_class(y_true_kg)
    y_pred_cls = _to_emission_class(y_pred_kg)

    acc  = accuracy_score(y_true_cls, y_pred_cls)
    prec = precision_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
    rec  = recall_score(y_true_cls, y_pred_cls,    average="weighted", zero_division=0)
    f1   = f1_score(y_true_cls, y_pred_cls,        average="weighted", zero_division=0)

    print(f"\n  {label} — Classification Metrics (emission bands)")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(classification_report(y_true_cls, y_pred_cls,
                                 labels=EMISSION_LABELS, zero_division=0))

    return {
        "model":     label,
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1_Score":  f1,
    }


# ─────────────────────────────────────────────────────────────────
#  Model trainers
# ─────────────────────────────────────────────────────────────────

def train_linear_regression(X_train, y_train) -> LinearRegression:
    print("\n⏳ Training Linear Regression ...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("✅ Linear Regression trained.")
    return lr


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    print("\n⏳ Training Random Forest ...")
    rf = RandomForestRegressor(
        n_estimators     = 200,
        max_depth        = 20,
        min_samples_split= 5,
        min_samples_leaf = 2,
        max_features     = "sqrt",
        random_state     = 42,
        n_jobs           = -1,
    )
    rf.fit(X_train, y_train)
    print("✅ Random Forest trained.")
    return rf


# ─────────────────────────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_true_log, y_pred_log, label, out_dir):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, -10, 30))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.45, s=18, color="#2c7be5")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual CO₂ Emissions (kg)")
    ax.set_ylabel("Predicted CO₂ Emissions (kg)")
    ax.set_title(f"{label} — Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"{label.replace(' ', '_')}_actual_vs_pred.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Plot saved: {path}")


def plot_residuals(y_true_log, y_pred_log, label, out_dir):
    y_true    = np.expm1(y_true_log)
    y_pred    = np.expm1(np.clip(y_pred_log, -10, 30))
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="#e85d04")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{label} — Residuals vs Fitted")
    axes[1].hist(residuals, bins=40, color="#457b9d", edgecolor="white")
    axes[1].set_xlabel("Residual (kg CO₂)")
    axes[1].set_title(f"{label} — Residual Distribution")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{label.replace(' ', '_')}_residuals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Plot saved: {path}")


def plot_confusion_matrix(y_true_log, y_pred_log, label, out_dir):
    y_true_kg  = np.expm1(np.array(y_true_log))
    y_pred_kg  = np.expm1(np.clip(np.array(y_pred_log), -10, 30))
    y_true_cls = _to_emission_class(y_true_kg)
    y_pred_cls = _to_emission_class(y_pred_kg)

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=EMISSION_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMISSION_LABELS, yticklabels=EMISSION_LABELS, ax=ax)
    ax.set_xlabel("Predicted Emission Band")
    ax.set_ylabel("Actual Emission Band")
    ax.set_title(f"{label} — Confusion Matrix (Emission Bands)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{label.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Plot saved: {path}")


def plot_feature_importance(rf_model, feature_names, out_dir):
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_names)))
    ax.bar(range(len(feature_names)),
           importances[indices],
           color=[colors[i] for i in range(len(feature_names))])
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices],
                        rotation=45, ha="right", fontsize=8)
    ax.set_title("Random Forest — Feature Importance")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(out_dir, "RF_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Plot saved: {path}")


def plot_model_comparison(reg_results, cls_results, out_dir):
    reg_df = pd.DataFrame(reg_results)
    cls_df = pd.DataFrame(cls_results)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Row 1: Regression metrics
    reg_metrics = [("MAE", "MAE (kg CO₂)"), ("RMSE", "RMSE (kg CO₂)"),
                   ("R2", "R² Score"),      ("Accuracy_10pct", "Accuracy ±10% (%)")]
    colors = ["#2c7be5", "#e85d04"]
    for ax, (col, lbl) in zip(axes[0], reg_metrics):
        vals = reg_df[col].values
        bars = ax.bar(reg_df["model"], vals, color=colors)
        ax.set_title(lbl, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02,
                    f"{v:.4f}", ha="center", fontsize=8)

    # Row 2: Classification metrics
    cls_metrics = [("Accuracy", "Accuracy"), ("Precision", "Precision"),
                   ("Recall",   "Recall"),   ("F1_Score",  "F1-Score")]
    for ax, (col, lbl) in zip(axes[1], cls_metrics):
        vals = cls_df[col].values
        bars = ax.bar(cls_df["model"], vals, color=colors)
        ax.set_title(lbl, fontsize=9)
        ax.set_ylim(0, 1.15)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{v:.4f}", ha="center", fontsize=8)

    fig.suptitle("Model Comparison — Carbon Footprint Prediction\n"
                 "Top: Regression Metrics | Bottom: Classification Metrics",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot saved: {path}")


def plot_co2_by_service_region(df_full, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    region_co2 = (df_full.groupby("region")["co2_emissions_kg"]
                          .mean().sort_values(ascending=False))
    axes[0].bar(region_co2.index, region_co2.values, color="#2a9d8f")
    axes[0].set_title("Avg CO₂ Emissions by Region (kg)")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Avg CO₂ (kg)")
    axes[0].tick_params(axis="x", rotation=35)

    svc_co2 = (df_full.groupby("service")["co2_emissions_kg"]
                       .mean().sort_values(ascending=False))
    axes[1].bar(svc_co2.index, svc_co2.values, color="#e76f51")
    axes[1].set_title("Avg CO₂ Emissions by Service (kg)")
    axes[1].set_xlabel("Service")
    axes[1].set_ylabel("Avg CO₂ (kg)")
    axes[1].tick_params(axis="x", rotation=40)

    plt.tight_layout()
    path = os.path.join(out_dir, "co2_by_service_region.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Plot saved: {path}")


# ─────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test, y_test_orig,
    feature_names, df_full,
    model_dir="./models", output_dir="./outputs"
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    reg_results = []
    cls_results = []

    # ── 1. Linear Regression ──────────────────────────────────
    lr       = train_linear_regression(X_train, y_train)
    lr_pred  = lr.predict(X_test)

    lr_reg = regression_metrics(y_test, lr_pred, "Linear Regression")
    lr_cls = classification_metrics(y_test, lr_pred, "Linear Regression")
    reg_results.append(lr_reg)
    cls_results.append(lr_cls)

    cv_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring="r2")
    print(f"  LR Cross-Val R² : {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
    lr_reg["CV_R2_mean"] = cv_lr.mean()
    lr_reg["CV_R2_std"]  = cv_lr.std()

    plot_actual_vs_predicted(y_test, lr_pred, "Linear Regression", output_dir)
    plot_residuals(y_test, lr_pred, "Linear Regression", output_dir)
    plot_confusion_matrix(y_test, lr_pred, "Linear Regression", output_dir)
    joblib.dump(lr, os.path.join(model_dir, "linear_regression.pkl"))

    # ── 2. Random Forest ──────────────────────────────────────
    rf      = train_random_forest(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_reg = regression_metrics(y_test, rf_pred, "Random Forest")
    rf_cls = classification_metrics(y_test, rf_pred, "Random Forest")
    reg_results.append(rf_reg)
    cls_results.append(rf_cls)

    cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
    print(f"  RF Cross-Val R² : {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
    rf_reg["CV_R2_mean"] = cv_rf.mean()
    rf_reg["CV_R2_std"]  = cv_rf.std()

    plot_actual_vs_predicted(y_test, rf_pred, "Random Forest", output_dir)
    plot_residuals(y_test, rf_pred, "Random Forest", output_dir)
    plot_confusion_matrix(y_test, rf_pred, "Random Forest", output_dir)
    plot_feature_importance(rf, feature_names, output_dir)
    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))

    # ── Comparison & EDA plots ────────────────────────────────
    plot_model_comparison(reg_results, cls_results, output_dir)
    plot_co2_by_service_region(df_full, output_dir)

    # ── Save results tables ───────────────────────────────────
    reg_df = pd.DataFrame(reg_results)
    cls_df = pd.DataFrame(cls_results)
    reg_df.to_csv(os.path.join(output_dir, "regression_results.csv"), index=False)
    cls_df.to_csv(os.path.join(output_dir, "classification_results.csv"), index=False)
    print(f"\n✅ Results saved to {output_dir}")

    return lr, rf, reg_df, cls_df
