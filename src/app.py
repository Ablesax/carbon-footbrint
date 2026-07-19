"""
app.py — Flask Web Dashboard
==============================
Routes:
  GET  /              Dashboard with dataset summary & charts
  POST /predict       Real-time CO₂ prediction
  GET  /api/stats     JSON model performance stats
  GET  /api/services  JSON list of services and their coefficients
  GET  /api/regions   JSON region carbon intensities
  GET  /api/recommendations   JSON reduction tips
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import pandas as pd

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates")
)

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
DATA_DIR  = os.path.join(BASE_DIR, "data")


def _load(name):
    p = os.path.join(MODEL_DIR, name)
    return joblib.load(p) if os.path.exists(p) else None


lr_model         = _load("linear_regression.pkl")
rf_model         = _load("random_forest.pkl")
scaler           = _load("scaler.pkl")
le_service       = _load("le_service.pkl")
le_region        = _load("le_region.pkl")
le_workload_cat  = _load("le_workload_category.pkl")
le_project       = _load("le_project_id.pkl")

# ── Reference look-ups ───────────────────────────────────────────
REGION_CARBON = {
    "europe-north1":        180,
    "europe-west1":         280,
    "us-central1":          386,
    "us-east1":             410,
    "asia-northeast1":      470,
    "me-central1":          550,
    "southamerica-east1":   520,
    "australia-southeast1": 640,
    "asia-south1":          708,
}

SERVICE_KWH_PER_UNIT = {
    "compute_engine":    0.50,
    "gke":               0.60,
    "cloud_run":         0.40,
    "cloud_functions":   0.0001,
    "vertex_ai":         2.50,
    "genai_api":         0.00002,
    "bigquery":          2.00,
    "cloud_storage":     0.001,
    "pubsub":            0.00001,
    "cloud_loadbalancer":0.000001,
}

SERVICE_WORKLOAD_MAP = {
    "compute_engine":    "IaaS_Compute",
    "gke":               "IaaS_Container",
    "cloud_run":         "PaaS_Serverless",
    "cloud_functions":   "FaaS",
    "vertex_ai":         "ML_Training",
    "genai_api":         "ML_Inference",
    "bigquery":          "Analytics",
    "cloud_storage":     "Storage",
    "pubsub":            "Messaging",
    "cloud_loadbalancer":"Networking",
}

COMPUTE_INTENSIVE = {"compute_engine", "gke", "cloud_run", "vertex_ai", "genai_api"}
HIGH_CARBON_THRESHOLD = 400

FEATURE_ORDER = [
    "energy_consumed_kwh", "carbon_intensity_gco2_kwh",
    "usage_amount", "log_usage_amount", "service_kwh_per_unit",
    "energy_proxy", "energy_per_usage", "high_carbon_region",
    "is_compute_intensive", "cost_per_kg_co2",
    "month", "day_of_week", "quarter",
    "service_enc", "region_enc", "workload_category_enc", "project_id_enc",
]


def _encode(le_obj, value: str) -> int:
    if le_obj is None:
        return 0
    try:
        return int(le_obj.transform([value])[0])
    except Exception:
        # Unseen label — use midpoint
        return len(le_obj.classes_) // 2


def build_feature_vector(form):
    service      = form.get("service", "compute_engine")
    region       = form.get("region",  "us-east1")
    usage_amount = float(form.get("usage_amount", 100))
    cost_usd     = float(form.get("cost_usd", 0) or 0)
    month        = int(form.get("month", 6))
    day_of_week  = int(form.get("day_of_week", 1))
    quarter      = int(form.get("quarter", 2))

    carbon_int       = REGION_CARBON.get(region, 400)
    kwh_per_unit     = SERVICE_KWH_PER_UNIT.get(service, 0.5)
    energy_kwh       = usage_amount * kwh_per_unit
    energy_proxy     = energy_kwh
    energy_per_usage = energy_kwh / (usage_amount + 1e-9)
    log_usage        = np.log1p(usage_amount)
    high_carbon      = int(carbon_int > HIGH_CARBON_THRESHOLD)
    is_compute       = int(service in COMPUTE_INTENSIVE)
    workload_cat     = SERVICE_WORKLOAD_MAP.get(service, "Other")
    co2_est          = (energy_kwh * carbon_int) / 1000
    cost_per_co2     = (cost_usd if cost_usd > 0
                        else energy_kwh * 0.05) / (co2_est + 1e-9)

    svc_enc = _encode(le_service,      service)
    reg_enc = _encode(le_region,       region)
    wkl_enc = _encode(le_workload_cat, workload_cat)
    prj_enc = 0   # project unknown at inference time

    vec = np.array([[
        energy_kwh, carbon_int, usage_amount, log_usage, kwh_per_unit,
        energy_proxy, energy_per_usage, high_carbon, is_compute, cost_per_co2,
        month, day_of_week, quarter,
        svc_enc, reg_enc, wkl_enc, prj_enc,
    ]])
    return vec


def get_recommendations(co2_kg, region, service):
    tips = []
    carbon_int = REGION_CARBON.get(region, 400)

    if carbon_int > HIGH_CARBON_THRESHOLD:
        low_region = min(REGION_CARBON, key=REGION_CARBON.get)
        tips.append({
            "icon": "🌍",
            "title": "Switch to a Low-Carbon Region",
            "detail": (f"'{region}' emits {carbon_int} gCO₂/kWh. "
                       f"Migrating to {low_region} "
                       f"({REGION_CARBON[low_region]} gCO₂/kWh) could cut "
                       f"emissions by up to {round((1 - REGION_CARBON[low_region]/carbon_int)*100)}%.")
        })

    if service in ("compute_engine", "gke"):
        tips.append({
            "icon": "⚡",
            "title": "Migrate to Serverless",
            "detail": ("Cloud Run or Cloud Functions only use resources during "
                       "execution, reducing idle-power waste by up to 80%.")
        })

    if service == "vertex_ai":
        tips.append({
            "icon": "🧠",
            "title": "Optimise ML Training",
            "detail": ("Schedule Vertex AI training in europe-north1 (180 gCO₂/kWh). "
                       "Use spot/preemptible instances to cut compute costs and emissions.")
        })

    if service == "bigquery":
        tips.append({
            "icon": "📊",
            "title": "Reduce BigQuery Scan Volume",
            "detail": ("Partition tables by date and cluster by key columns "
                       "to avoid full-table scans. Each TB not scanned saves ~2 kWh.")
        })

    tips.append({
        "icon": "🕐",
        "title": "Time-Shift Batch Workloads",
        "detail": ("Use Cloud Scheduler to run non-urgent jobs at night when "
                   "renewable energy share on the grid is typically higher.")
    })

    if co2_kg > 1.0:
        tips.append({
            "icon": "♻️",
            "title": "Purchase Carbon Offsets",
            "detail": (f"Estimated emission: {co2_kg:.4e} kg CO₂. "
                       "Consider Google Cloud's Carbon Offset programme or Gold Standard credits.")
        })

    return tips


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if scaler is None or lr_model is None:
            return jsonify({"error": "Models not trained yet. Run main.py first."}), 503

        X_raw    = build_feature_vector(request.form)
        X_scaled = scaler.transform(X_raw)

        lr_pred  = float(np.expm1(lr_model.predict(X_scaled)[0]))
        rf_pred  = float(np.expm1(rf_model.predict(X_scaled)[0]))
        ensemble = (lr_pred + rf_pred) / 2

        service = request.form.get("service", "compute_engine")
        region  = request.form.get("region",  "us-east1")
        recs    = get_recommendations(ensemble, region, service)

        return jsonify({
            "linear_regression_kg": round(lr_pred, 6),
            "random_forest_kg":     round(rf_pred, 6),
            "ensemble_kg":          round(ensemble, 6),
            "recommendations":      recs,
            "region_carbon_intensity": REGION_CARBON.get(region, "N/A"),
            "co2_equivalent": {
                "driving_km":              round(ensemble / 0.000192, 2),
                "smartphone_charges":      round(ensemble / 0.0000084, 0),
                "trees_offset_per_year":   round(ensemble / 21.77, 4),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/stats")
def api_stats():
    reg_path = os.path.join(OUT_DIR, "regression_results.csv")
    cls_path = os.path.join(OUT_DIR, "classification_results.csv")
    if not os.path.exists(reg_path):
        return jsonify({"error": "Run main.py to train models first."}), 404
    reg = pd.read_csv(reg_path).to_dict(orient="records")
    cls = pd.read_csv(cls_path).to_dict(orient="records") if os.path.exists(cls_path) else []
    return jsonify({"regression": reg, "classification": cls})


@app.route("/api/regions")
def api_regions():
    return jsonify(REGION_CARBON)


@app.route("/api/services")
def api_services():
    return jsonify({
        svc: {
            "kwh_per_unit":    kwh,
            "workload_category": SERVICE_WORKLOAD_MAP.get(svc, "Other"),
            "compute_intensive": svc in COMPUTE_INTENSIVE,
        }
        for svc, kwh in SERVICE_KWH_PER_UNIT.items()
    })


@app.route("/api/recommendations")
def api_recommendations():
    return jsonify([
        "Migrate to europe-north1 (180 gCO₂/kWh) for lowest-carbon compute.",
        "Replace Compute Engine with Cloud Run/Functions for intermittent workloads.",
        "Partition BigQuery tables to eliminate full-table scans.",
        "Schedule Vertex AI training in low-carbon regions and off-peak windows.",
        "Enable Cloud Storage lifecycle rules to auto-archive cold data.",
        "Use GKE cluster autoscaler to eliminate idle node energy.",
        "Export Carbon Footprint data to BigQuery and set per-project budgets.",
        "Purchase carbon offsets via Google Cloud or Gold Standard for residual emissions.",
    ])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
