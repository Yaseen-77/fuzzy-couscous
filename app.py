from flask import Flask, request, jsonify , render_template
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

app =Flask(__name__)

model = tf.keras.models.load_model('best_model.keras')
scaler= joblib.load('scaler.joblib')
encoders = joblib.load("encoders.joblib")
numeric_features = joblib.load("numeric_features.joblib")
categorical_features = joblib.load("categorical_features.joblib")

def preprocess(order):
    created_at = pd.to_datetime(order["created_at"])

    row = {
        "market_id": order.get("market_id", -1),
        "store_id": order.get("store_id", -1),
        "store_primary_category": order.get("store_primary_category", "unknown"),
        "order_protocol": order.get("order_protocol", -1),
        "subtotal": order["subtotal"],
        "total_items": order["total_items"],
        "num_distinct_items": order["num_distinct_items"],
        "min_item_price": order["min_item_price"],
        "max_item_price": order["max_item_price"],
        "total_onshift_partners": order["total_onshift_partners"],
        "total_busy_partners": order["total_busy_partners"],
        "total_outstanding_orders": order["total_outstanding_orders"],
    }

    row["price_range"] = row["max_item_price"] - row["min_item_price"]
    row["avg_item_price"] = row["subtotal"] / max(row["total_items"], 1)
    row["items_per_distinct_item"] = row["total_items"] / max(row["num_distinct_items"], 1)

    row["busy_partner_ratio"] = row["total_busy_partners"] / max(row["total_onshift_partners"], 1)
    row["available_partners"] = row["total_onshift_partners"] - row["total_busy_partners"]
    row["orders_per_available_partner"] = row["total_outstanding_orders"] / (row["available_partners"] + 1)

    hour = created_at.hour
    dow = created_at.dayofweek
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    row["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    row["is_weekend"] = int(dow in [5, 6])

    cat_inputs = []
    raw_cats = {
        "market_id": row["market_id"],
        "store_id_bucketed": row["store_id"],
        "store_primary_category": row["store_primary_category"],
        "order_protocol": row["order_protocol"],
    }
    for col in categorical_features:
        le = encoders[col]
        val = str(raw_cats[col])
        val = val if val in set(le.classes_) else le.classes_[0]
        encoded = le.transform([val])[0]
        cat_inputs.append(np.array([[encoded]]))

    numeric_row = pd.DataFrame([row])[numeric_features]
    numeric_scaled = scaler.transform(numeric_row)

    return cat_inputs + [numeric_scaled]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        order = request.get_json()
        model_inputs = preprocess(order)
        pred_log = model.predict(model_inputs, verbose=0)[0][0]
        pred_minutes = float(np.expm1(pred_log))
        return jsonify({
            "predicted_delivery_time_minutes": round(pred_minutes, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/about_me")
def about_me():
    return render_template("about_me.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)