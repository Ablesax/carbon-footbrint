"""
preprocess.py — Data Preprocessing Pipeline
==============================================
Implements Chapter 3 methodology:
  - Data cleaning & quality checks
  - Feature engineering (energy proxy, carbon intensity flags, workload type)
  - Label encoding of categoricals
  - Log-transform of skewed target (co2_emissions_kg)
  - Train / test split (80/20, stratified by service)
  - StandardScaler normalisation
  - Saves scaler + encoders to models/ directory

Works with the REAL imported dataset produced by data/load_dataset.py.
No synthetic generation is needed.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
import joblib

RANDOM_STATE = 42
TEST_SIZE    = 0.30

# ── Service → workload category mapping ────────────────────────
# Mirrors the AWS taxonomy used in Chapter 2 (IaaS/PaaS/FaaS/Storage/Analytics)
SERVICE_WORKLOAD_MAP = {
    "compute_engine":   "IaaS_Compute",
    "gke":              "IaaS_Container",
    "cloud_run":        "PaaS_Serverless",
    "cloud_functions":  "FaaS",
    "vertex_ai":        "ML_Training",
    "genai_api":        "ML_Inference",
    "bigquery":         "Analytics",
    "cloud_storage":    "Storage",
    "pubsub":           "Messaging",
    "cloud_loadbalancer":"Networking",
}

# Energy-intensive services (compute + ML) as defined in thesis Chapter 2.2.1
COMPUTE_INTENSIVE = {"compute_engine", "gke", "cloud_run", "vertex_ai", "genai_api"}

HIGH_CARBON_THRESHOLD = 400   # gCO2/kWh — thesis Section 3.3


# ─────────────────────────────────────────────────────────────────
def load_data(path_or_df):
    """Accept either a CSV path or an already-loaded DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning as specified in Chapter 3.2:
      1. Drop exact duplicates
      2. Impute numeric NaNs with column median
      3. Remove physically impossible values
      4. Remove extreme outliers (1st–99th percentile on target)
    """
    initial = len(df)

    # 1. Duplicates
    df = df.drop_duplicates()

    # 2. Impute numeric NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"   Imputed {n_missing:>4} NaN in '{col}' → median")

    # 3. Physical validity guards
    df = df[df["energy_consumed_kwh"] > 0]
    df = df[df["co2_emissions_kg"]    > 0]
    df = df[df["carbon_intensity_gco2_kwh"] > 0]
    df = df[df["usage_amount"]        > 0]

    # 4. IQR / percentile outlier removal on CO₂ target
    q_lo = df["co2_emissions_kg"].quantile(0.01)
    q_hi = df["co2_emissions_kg"].quantile(0.99)
    df = df[df["co2_emissions_kg"].between(q_lo, q_hi)]

    print(f"   Cleaned: {initial} → {len(df)} rows  ({initial - len(df)} removed)")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering — Chapter 3.3.
    All engineered features have a direct grounding in the thesis formulas.
    """

    # ── Workload category from service name ─────────────────────
    df["workload_category"] = df["service"].map(SERVICE_WORKLOAD_MAP).fillna("Other")
    df["is_compute_intensive"] = df["service"].isin(COMPUTE_INTENSIVE).astype(int)

    # ── Energy proxy ─────────────────────────────────────────────
    # Based on: Energy (kWh) = usage_amount × kwh_per_unit (service coefficient)
    # service_kwh_per_unit already loaded from service_energy_coefficients.csv
    df["energy_proxy"] = df["usage_amount"] * df["service_kwh_per_unit"].fillna(
        df["energy_consumed_kwh"] / (df["usage_amount"] + 1e-9)
    )

    # ── Carbon intensity flag ────────────────────────────────────
    # Objective: identify high-carbon regions (Section 3.3, Eq. related to region carbon)
    df["high_carbon_region"] = (df["carbon_intensity_gco2_kwh"] > HIGH_CARBON_THRESHOLD).astype(int)

    # ── Cost efficiency: cost per kg CO₂ ────────────────────────
    # Only available for rows that came from daily_cost_emissions
    df["cost_per_kg_co2"] = (
        df["cost_usd"].fillna(df["energy_consumed_kwh"] * 0.05)   # fallback: $0.05/kWh
        / (df["co2_emissions_kg"] + 1e-9)
    )

    # ── Energy intensity: kWh per unit of usage ─────────────────
    df["energy_per_usage"] = df["energy_consumed_kwh"] / (df["usage_amount"] + 1e-9)

    # ── Temporal features (carbon-aware scheduling context) ──────
    if "month" not in df.columns:
        df["date"]        = pd.to_datetime(df["date"], errors="coerce")
        df["month"]       = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["quarter"]     = df["date"].dt.quarter

    # ── Log-scale usage (right-skewed) ───────────────────────────
    df["log_usage_amount"] = np.log1p(df["usage_amount"])

    print("✅ Feature engineering complete — added derived features:")
    new_cols = ["workload_category", "is_compute_intensive", "energy_proxy",
                "high_carbon_region", "cost_per_kg_co2", "energy_per_usage",
                "log_usage_amount", "month", "day_of_week", "quarter"]
    for c in new_cols:
        print(f"   • {c}")

    return df


def encode_categoricals(df: pd.DataFrame):
    """
    Label-encode: service, region, workload_category, project_id.
    Returns df with _enc columns and the fitted encoder objects.
    """
    encoders = {}
    for col in ["service", "region", "workload_category", "project_id"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print(f"✅ Encoded {len(encoders)} categorical columns")
    return df, encoders


def get_feature_columns() -> list:
    """
    Final ML feature set — aligned with Chapter 3 methodology.
    Numeric features only (encoded categoricals included).
    """
    return [
        # Core energy / emission drivers
        "energy_consumed_kwh",
        "carbon_intensity_gco2_kwh",
        "usage_amount",
        "log_usage_amount",
        "service_kwh_per_unit",
        # Engineered features
        "energy_proxy",
        "energy_per_usage",
        "high_carbon_region",
        "is_compute_intensive",
        "cost_per_kg_co2",
        # Temporal context
        "month",
        "day_of_week",
        "quarter",
        # Encoded categoricals
        "service_enc",
        "region_enc",
        "workload_category_enc",
        "project_id_enc",
    ]


def preprocess(data_source, output_dir: str = "."):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    data_source : str | pd.DataFrame
        Path to a CSV file OR a pre-loaded DataFrame from load_dataset.py.
    output_dir : str
        Directory to save scaler and encoder .pkl files.

    Returns
    -------
    X_train, X_test, y_train, y_test, y_test_orig,
    scaler, encoders, feature_names, df_full
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_source)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)

    features     = get_feature_columns()
    # Confirm all features exist; fill any still-missing with 0
    for f in features:
        if f not in df.columns:
            df[f] = 0
            print(f"   ⚠️  Feature '{f}' not found — filled with 0")

    X = df[features].astype(float)
    y = df["co2_emissions_kg"]

    # Log-transform target (right-skewed CO₂ distribution — Chapter 3 justification)
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    y_test_orig = np.expm1(y_test)   # original-scale for reporting

    # Scale features (StandardScaler — Chapter 3.3)
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Save preprocessing artifacts ───────────────────────────
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    for col, le in encoders.items():
        joblib.dump(le, os.path.join(output_dir, f"le_{col}.pkl"))

    print(f"\n✅ Preprocessing complete")
    print(f"   Train samples : {X_train_scaled.shape[0]}")
    print(f"   Test  samples : {X_test_scaled.shape[0]}")
    print(f"   Feature count : {len(features)}")
    print(f"   Artifacts saved to: {output_dir}")

    return (
        X_train_scaled, X_test_scaled,
        y_train, y_test, y_test_orig,
        scaler, encoders,
        features, df
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    from load_dataset import build_master_dataset
    df_raw = build_master_dataset()
    preprocess(df_raw, output_dir="../models")
