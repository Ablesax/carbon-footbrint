# Cloud Carbon Footprint ML — Obia John Simon
## Bells University of Technology, Ota | B.Tech Computer Science

---

## Project Overview

A machine learning system that **predicts and evaluates the carbon footprint of AWS cloud workloads**
using **Linear Regression** and **Random Forest** models.

Aligns with UN SDGs 12 (Responsible Consumption) and 13 (Climate Action).

---

## Project Structure

```
carbon_footprint_ml/
├── main.py                     ← Full pipeline entry point
├── requirements.txt
├── data/
│   ├── generate_dataset.py     ← Synthetic dataset generator (emission-factor based)
│   └── cloud_carbon_dataset.csv← Generated dataset (2000 records)
├── src/
│   ├── preprocess.py           ← Data cleaning, feature engineering, scaling
│   ├── train.py                ← Model training + evaluation + visualisations
│   └── app.py                  ← Flask web application (real-time prediction)
├── templates/
│   └── index.html              ← Web dashboard UI
├── models/                     ← Saved .pkl model artifacts
└── outputs/                    ← Charts, results CSV
```

---

## Dataset Features

| Feature | Description |
|---|---|
| `cpu_utilization` | CPU usage (%) |
| `memory_utilization` | Memory usage (%) |
| `storage_gb` | Storage consumed (GB) |
| `data_transfer_gb` | Network data transferred (GB) |
| `duration_hours` | Workload runtime (hours) |
| `pue` | Power Usage Effectiveness |
| `carbon_intensity_gco2_kwh` | Grid carbon intensity (gCO₂/kWh) |
| `instance_tdp_watts` | Instance Thermal Design Power (W) |
| `workload_type` | EC2 / Lambda / S3 / RDS |
| `aws_region` | AWS deployment region |
| **`co2_emissions_kg`** | **Target: CO₂ emitted (kg)** |

---

## Emission Formula (Green Algorithms / CodeCarbon Based)

```
Energy_compute (kWh) = (cpu_util / 100) × TDP_W × PUE × duration_h / 1000
Energy_storage (kWh) = storage_GB × 0.000017 × duration_h
Energy_network (kWh) = transfer_GB × 0.06

Total_Energy (kWh) = Energy_compute + Energy_storage + Energy_network
CO2 (kg) = (Total_Energy × carbon_intensity_gCO2/kWh) / 1000
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (generate data → preprocess → train → evaluate)
python main.py

# 3. Run pipeline AND launch web server
python main.py --serve

# 4. If you already have the CSV, skip generation
python main.py --no-generate --serve
```

---

## Model Evaluation Metrics

- **MAE** — Mean Absolute Error (kg CO₂)
- **RMSE** — Root Mean Squared Error
- **R² Score** — Coefficient of determination
- **MAPE** — Mean Absolute Percentage Error
- **Accuracy ±10%** — Predictions within 10% of actual value
- **5-Fold Cross-Validation R²**

---

## Web Application

After running `python main.py --serve`, open: **http://127.0.0.1:5000**

Input your workload parameters and receive:
- CO₂ predictions from both models
- Real-world CO₂ equivalences (driving km, smartphone charges, trees)
- Personalised carbon reduction recommendations

---

## Outputs Generated

| File | Description |
|---|---|
| `outputs/Linear_Regression_actual_vs_pred.png` | Actual vs Predicted plot |
| `outputs/Random_Forest_actual_vs_pred.png` | Actual vs Predicted plot |
| `outputs/RF_feature_importance.png` | Feature importance chart |
| `outputs/model_comparison.png` | Side-by-side model comparison |
| `outputs/co2_by_region_workload.png` | CO₂ by AWS region and workload type |
| `outputs/model_results.csv` | All evaluation metrics |
| `models/linear_regression.pkl` | Trained LR model |
| `models/random_forest.pkl` | Trained RF model |
| `models/scaler.pkl` | StandardScaler |

---

## References

- Green Algorithms Project: https://www.green-algorithms.org/
- CodeCarbon: https://github.com/mlco2/codecarbon
- IEA Data Centres Report (2023): https://www.iea.org/reports/data-centres-and-data-transmission-networks
- Electricity Maps: https://www.electricitymaps.com/
- SPEC Power: https://www.spec.org/power_ssj2008/
- Panwar et al. (2022): doi:10.1186/s13677-022-00368-5
