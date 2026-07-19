"""
load_dataset.py — Real Dataset Loader
=======================================
Loads and merges the locally imported dataset files from archive__3_.zip:
  - daily_cost_emissions.csv   (500 rows: date, project, service, region, usage, cost, energy, g_co2_per_kwh, emissions)
  - daily_emissions.csv        (500 rows: same minus cost column)
  - daily_usage.csv            (500 rows: usage only — used for feature enrichment)
  - emission_factors.csv       (9 rows:  region → g_co2_per_kwh lookup)
  - service_energy_coefficients.csv (10 rows: service → kwh_per_unit)
  - service_cost_coefficients.csv   (10 rows: service → cost info)
  - services.csv               (10 rows: service metadata)
  - projects.csv               (10 rows: project → owner)

Output: a single merged pandas DataFrame ready for preprocessing.

This replaces the old synthetic generate_dataset.py and fulfills
Objective 1: "Collect relevant datasets related to cloud computing,
energy consumption and carbon emission".
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# High-carbon threshold used in feature engineering (gCO2/kWh)
HIGH_CARBON_THRESHOLD = 400


def load_raw_tables(data_dir: str = DATA_DIR) -> dict:
    """Load every CSV in the dataset into a dict of DataFrames."""
    files = {
        "daily_cost_emissions":        "daily_cost_emissions.csv",
        "daily_emissions":             "daily_emissions.csv",
        "daily_usage":                 "daily_usage.csv",
        "emission_factors":            "emission_factors.csv",
        "service_energy_coefficients": "service_energy_coefficients.csv",
        "service_cost_coefficients":   "service_cost_coefficients.csv",
        "services":                    "services.csv",
        "projects":                    "projects.csv",
    }

    tables = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required dataset file not found: {path}\n"
                "Please place the CSV files from archive__3_.zip inside the data/ folder."
            )
        tables[key] = pd.read_csv(path)
        print(f"   ✅ Loaded {fname:45s}  ({len(tables[key])} rows)")

    return tables


def build_master_dataset(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Merge all tables into one master DataFrame with the following columns:

    Identifiers   : date, project_id, owner, service, region
    Usage         : usage_amount, unit, description
    Energy        : energy_kwh, kwh_per_unit
    Emissions     : g_co2_per_kwh, emissions_kgco2
    Cost          : cost_usd  (NaN for rows sourced from daily_emissions only)

    These columns are then used by preprocess.py to engineer ML features.
    """
    print("\n📂 Loading dataset files ...")
    tables = load_raw_tables(data_dir)

    # ── 1. Combine the two daily observation tables ──────────────
    # daily_cost_emissions has 'cost_usd'; daily_emissions does not.
    dce = tables["daily_cost_emissions"].copy()
    de  = tables["daily_emissions"].copy()
    de["cost_usd"] = np.nan          # add missing column so concat aligns

    df = pd.concat([dce, de], ignore_index=True)

    # Deduplicate — same (date, project_id, service, region) can appear in both files
    df = df.drop_duplicates(subset=["date", "project_id", "service", "region"])
    print(f"\n   Combined observation rows  : {len(df):>6}")

    # ── 2. Enrich with service metadata ─────────────────────────
    svc = tables["services"][["service", "unit", "description"]]
    df  = df.merge(svc, on="service", how="left")

    # ── 3. Enrich with energy coefficients ──────────────────────
    sec = tables["service_energy_coefficients"][["service", "kwh_per_unit", "notes"]]
    df  = df.merge(sec, on="service", how="left", suffixes=("", "_sec"))

    # ── 4. Enrich with cost coefficients ────────────────────────
    scc = tables["service_cost_coefficients"]
    if "service" in scc.columns:
        df = df.merge(scc, on="service", how="left", suffixes=("", "_scc"))

    # ── 5. Enrich with project owners ────────────────────────────
    proj = tables["projects"]
    df   = df.merge(proj, on="project_id", how="left")

    # ── 6. Verify emission factors align ────────────────────────
    ef = tables["emission_factors"].rename(columns={"g_co2_per_kwh": "ef_g_co2_per_kwh"})
    df = df.merge(ef[["region", "ef_g_co2_per_kwh"]], on="region", how="left")
    # If the observation's carbon intensity differs from the reference factor, flag it
    df["carbon_factor_delta"] = (df["g_co2_per_kwh"] - df["ef_g_co2_per_kwh"]).abs()

    # ── 7. Parse dates ────────────────────────────────────────────
    df["date"]       = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    df["month"]      = df["date"].dt.month
    df["day_of_week"]= df["date"].dt.dayofweek   # 0=Mon … 6=Sun
    df["quarter"]    = df["date"].dt.quarter

    # ── 8. Rename for consistency with thesis terminology ────────
    df = df.rename(columns={
        "g_co2_per_kwh":   "carbon_intensity_gco2_kwh",
        "emissions_kgco2": "co2_emissions_kg",
        "energy_kwh":      "energy_consumed_kwh",
        "kwh_per_unit":    "service_kwh_per_unit",
    })

    print(f"   Master dataset rows        : {len(df):>6}")
    print(f"   Master dataset columns     : {df.shape[1]:>6}")
    print(f"   Services                   : {sorted(df['service'].unique())}")
    print(f"   Regions                    : {sorted(df['region'].unique())}")
    print(f"   Date range                 : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   CO₂ range (kg)             : {df['co2_emissions_kg'].min():.4f} → {df['co2_emissions_kg'].max():.4f}")

    return df


if __name__ == "__main__":
    df = build_master_dataset()
    print("\nSample rows:")
    print(df[["date", "project_id", "service", "region",
              "energy_consumed_kwh", "carbon_intensity_gco2_kwh",
              "co2_emissions_kg"]].head(8).to_string(index=False))