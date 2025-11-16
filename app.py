import os
import json
import sqlite3
from datetime import datetime, timedelta
from collections import OrderedDict

import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify, render_template, g, send_file
from reportlab.pdfgen import canvas

# -------------------------
# Config
# -------------------------
APP = Flask(__name__, template_folder="templates")
DB_PATH = "history.db"
API_KEY = "devkey"

# -------------------------
# Load training columns (feature order)
# -------------------------
if not os.path.exists("data/creditcard.csv"):
    raise RuntimeError("data/creditcard.csv missing - run convert_dataset.py first")

df = pd.read_csv("data/creditcard.csv")
FEATURES = list(df.drop("Class", axis=1).columns)
print("FEATURES loaded:", FEATURES)

# -------------------------
# Load models & scaler
# -------------------------
MODELS = {}

def safe_load_model(name, path):
    if os.path.exists(path):
        try:
            MODELS[name] = joblib.load(path)
            print(f"Loaded model: {name} from {path}")
            return True
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return False

# primary model (required)
if not safe_load_model("rf", "model.pkl"):
    safe_load_model("rf", "credit_fraud_model.pkl")

# optional alternatives
safe_load_model("lr", "model_lr.pkl")
safe_load_model("xgb", "model_xgb.pkl")

if not MODELS:
    raise RuntimeError("No ML models loaded. Run train_model.py first")

if not os.path.exists("scaler.pkl"):
    raise RuntimeError("scaler.pkl missing - run train_model.py first")
SCALER = joblib.load("scaler.pkl")
print("Scaler loaded")

# extract feature importances
FEATURE_IMPORTANCES = []
try:
    if "rf" in MODELS:
        rf = MODELS["rf"]
        imps = getattr(rf, "feature_importances_", None)
        if imps is not None:
            FEATURE_IMPORTANCES = sorted(zip(FEATURES, map(float, imps)), key=lambda x: x[1], reverse=True)
except Exception:
    FEATURE_IMPORTANCES = []

# -------------------------
# SQLite helpers
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            model TEXT,
            prediction INTEGER,
            probability REAL,
            amount REAL,
            inputs TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("SQLite DB initialized.")

def get_db():
    if "_db" not in g:
        g._db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    return g._db

@APP.teardown_appcontext
def close_conn(exc):
    db = g.pop("_db", None)
    if db:
        db.close()

# -------------------------
# Routes
# -------------------------
@APP.route("/")
def home():
    return render_template("index.html", keys=FEATURES)

@APP.route("/features")
def features():
    return jsonify({"features": FEATURES})

@APP.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(force=True)
        model_choice = data.pop("model_choice", "rf")
        if model_choice not in MODELS:
            model_choice = "rf"
        model = MODELS[model_choice]

        # build feature vector in exact order
        values = []
        for k in FEATURES:
            try:
                val = data.get(k, 0)
                # Handle categorical features (convert to numeric if string)
                if isinstance(val, str):
                    # For categorical columns, we need to encode them
                    # Since we're predicting, just convert to hash or use simple encoding
                    values.append(float(hash(val) % 1000))
                else:
                    values.append(float(val))
            except:
                values.append(0.0)

        arr = np.array(values).reshape(1, -1)
        
        # Create DataFrame to match training format
        df_pred = pd.DataFrame(arr, columns=FEATURES)
        
        # Scale
        scaled = SCALER.transform(df_pred)

        # predict probability
        proba = None
        try:
            proba = float(model.predict_proba(scaled)[0][1])
        except Exception:
            try:
                p = model.predict(scaled)[0]
                proba = float(p)
            except Exception:
                proba = 0.0

        pred = 1 if proba >= 0.5 else 0

        # persist to DB
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO history (ts, model, prediction, probability, amount, inputs) VALUES (?,?,?,?,?,?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_choice,
                pred,
                proba,
                float(data.get("Amount", 0)),
                json.dumps(data),
            )
        )
        conn.commit()

        return jsonify({
            "Prediction": "FRAUDULENT" if pred == 1 else "LEGITIMATE",
            "raw_prediction": int(pred),
            "probability": proba,
            "model_used": model_choice
        })

    except Exception as e:
        print("predict_api error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@APP.route("/history", methods=["GET"])
def history():
    if request.args.get("api_key") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 403
    conn = get_db()
    cur = conn.cursor()
    rows = cur.execute("SELECT ts, model, prediction, probability, amount FROM history ORDER BY id DESC LIMIT 100").fetchall()
    out = []
    for ts, model, pred, prob, amt in rows:
        out.append({
            "timestamp": ts,
            "model": model.upper(),
            "prediction": "FRAUD" if pred == 1 else "LEGIT",
            "probability": prob,
            "amount": amt
        })
    return jsonify(out)

@APP.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM history")
        conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@APP.route("/download_report", methods=["POST"])
def download_report():
    try:
        data = request.get_json(force=True)
        filename = f"fraud_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        path = os.path.join(os.getcwd(), filename)
        
        # Create PDF
        c = canvas.Canvas(path)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(80, 800, "Credit Card Fraud Detection Report")
        
        c.setFont("Helvetica", 11)
        c.drawString(80, 780, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(80, 765, "-" * 80)
        
        y = 740
        c.setFont("Helvetica-Bold", 12)
        c.drawString(80, y, "Transaction Details:")
        y -= 25
        
        c.setFont("Helvetica", 11)
        for k, v in data.items():
            if k == "model_choice":
                continue
            c.drawString(100, y, f"{k}: {v}")
            y -= 18
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = 780
        
        c.save()
        
        # Send file to browser for download
        return send_file(
            path, 
            as_attachment=True, 
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Download error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@APP.route("/admin_stats", methods=["GET"])
def admin_stats():
    if request.args.get("api_key") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db()
    cur = conn.cursor()
    rows = cur.execute("SELECT ts, prediction, probability, amount FROM history").fetchall()

    total = len(rows)
    frauds = sum(1 for r in rows if r[1] == 1)
    avg_amount = float(sum((r[3] or 0) for r in rows) / total) if total else 0.0
    fraud_rate = (frauds / total) if total else 0.0

    # last 7 days
    now = datetime.now()
    last7 = OrderedDict()
    for i in range(6, -1, -1):
        d = (now - timedelta(days=i)).date().isoformat()
        last7[d] = {"date": d, "total": 0, "frauds": 0}

    # last 24 hours
    last24 = OrderedDict()
    for i in range(23, -1, -1):
        dt = now - timedelta(hours=i)
        hour_key = dt.strftime("%Y-%m-%d %H:00")
        last24[hour_key] = {"hour": hour_key, "total": 0, "frauds": 0}

    for ts, pred, prob, amt in rows:
        try:
            row_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            try:
                row_dt = datetime.fromisoformat(ts)
            except Exception:
                continue
        date_key = row_dt.date().isoformat()
        hour_key = row_dt.strftime("%Y-%m-%d %H:00")

        if date_key in last7:
            last7[date_key]["total"] += 1
            if pred == 1:
                last7[date_key]["frauds"] += 1
        if hour_key in last24:
            last24[hour_key]["total"] += 1
            if pred == 1:
                last24[hour_key]["frauds"] += 1

    last7_list = list(last7.values())
    last24_list = []
    for k, v in last24.items():
        total_h = v["total"]
        fr = v["frauds"]
        last24_list.append({"hour": v["hour"], "total": total_h, "frauds": fr, "fraud_rate": (fr/total_h) if total_h else 0.0})

    top_features = [{"feature": f, "importance": imp} for f, imp in FEATURE_IMPORTANCES[:12]]

    return jsonify({
        "total_scanned": total,
        "total_frauds": frauds,
        "fraud_rate": fraud_rate,
        "avg_amount": avg_amount,
        "last7days": last7_list,
        "last24hours": last24_list,
        "top_features": top_features
    })

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    init_db()
    print("Server ready at http://127.0.0.1:5000")
    APP.run(debug=True)