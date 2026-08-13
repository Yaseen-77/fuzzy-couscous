
# Porter Delivery Time Predictor

A neural network-based regression model that predicts intra-city delivery time (in minutes) for Porter orders, deployed as a Flask web application with a styled HTML frontend.

## 🚀 Live Demo

**[porter-delivery-predictor-git-main-yaseen-sk.vercel.app](https://porter-delivery-predictor-git-main-yaseen-sk.vercel.app/)**

> ⏳ **Note:** This is deployed on a free-tier serverless host, so the first request after a period of inactivity may take **~5–10 seconds** to respond while the server "cold starts." Subsequent requests will be fast. If the page looks unresponsive on first load, just wait a few seconds before retrying.

## Problem Statement

Porter is an intra-city logistics platform connecting customers, restaurants/businesses, and delivery partners. Accurately predicting delivery time helps:

- Improve customer experience
- Optimize delivery-partner allocation
- Identify operational bottlenecks
- Reduce delayed deliveries
- Improve resource utilization

Given order details (items, pricing, store info) and live marketplace conditions (partner availability, outstanding orders) at the moment an order is placed, the model predicts how many minutes the delivery will take.

## Dataset

Each row represents one delivery, with fields including:

- `market_id`, `store_id`, `store_primary_category`, `order_protocol`
- `created_at`, `actual_delivery_time` (used to compute the target)
- `subtotal`, `total_items`, `num_distinct_items`, `min_item_price`, `max_item_price`
- `total_onshift_partners`, `total_busy_partners`, `total_outstanding_orders`

## Project Structure

```
LAB2/
├── app.py                        # Flask application (API + server)
├── delivery_time_model.keras     # Trained neural network
├── scaler.joblib                 # Fitted RobustScaler for numeric features
├── encoders.joblib                # Fitted LabelEncoders for categorical features
├── numeric_features.joblib        # List of numeric feature names, in order
├── categorical_features.joblib    # List of categorical feature names, in order
├── requirements.txt
├── templates/
│   └── index.html                # Frontend form UI
└── static/
    └── style.css                 # Frontend styling
```

## Methodology

### 1. Data Loading & Exploration

Loaded the dataset, checked missing values, distributions, and categorical cardinality to inform preprocessing decisions.

### 2. Target Computation

```
delivery_time_minutes = (actual_delivery_time - created_at) in minutes
```

Rows with missing timestamps or non-positive durations were dropped; extreme outliers (>300 minutes) were capped. A `log1p` transform was applied to the target to reduce right-skew before training.

### 3. Feature Engineering

- **Temporal**: hour-of-day and day-of-week encoded cyclically (`sin`/`cos`), plus `is_weekend`.
- **Categorical**: `market_id`, `store_id` (rare IDs bucketed), `store_primary_category`, `order_protocol` — encoded as integer indices for embedding layers.
- **Order-content**: `price_range`, `avg_item_price`, `items_per_distinct_item`.
- **Operational load**: `busy_partner_ratio`, `available_partners`, `orders_per_available_partner` — ratio features that generalize better across markets of different sizes than raw counts.

### 4. Data Cleaning

Missing numeric values imputed with the median (with missingness indicators); missing categoricals filled with `"unknown"`. Outliers in price/item-count columns capped via IQR. Logically inconsistent rows (e.g., `min_item_price > max_item_price`) removed.

### 5. Preprocessing

Chronological train/validation/test split (to avoid leaking future demand patterns). Numeric features scaled with `RobustScaler`; categorical features label-encoded for embedding lookup — both fit on the training set only.

### 6. Modeling

- **Baselines**: mean predictor, Ridge regression, LightGBM — used to set a performance floor and validate feature usefulness.
- **Neural network**: a wide-and-deep-style MLP — each categorical feature passed through its own `Embedding` layer, concatenated with scaled numeric features, followed by `Dense → BatchNorm → Dropout` blocks and a linear output. Trained with Huber loss, Adam optimizer, early stopping, and learning-rate scheduling.

### 7. Evaluation

Reported MAE, MSE, RMSE, MAPE, and R² (converted back to real minutes via `expm1`), plus residual diagnostics and segment-wise error breakdowns (by market, store category, hour) to catch localized model weaknesses.

### 8. Interpretability

SHAP values (`TreeExplainer` for LightGBM, `KernelExplainer` for the neural network) used to explain global feature importance and individual predictions.

### 9. Deployment

The trained model and all preprocessing artifacts (scaler, encoders, feature lists) are saved and loaded once at Flask startup. A `/predict` endpoint accepts raw order details as JSON, applies identical preprocessing, and returns the predicted delivery time. A styled HTML form (`templates/index.html`) provides a browser-based interface.

## Setup & Running

```bash
# create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the app
python app.py
```

Then open **http://localhost:5000/** in a browser.

## API

**POST** `/predict`

Request body (JSON):

```json
{
  "created_at": "2026-08-11T19:30:00",
  "market_id": 2,
  "store_id": 1583,
  "store_primary_category": "burger",
  "order_protocol": 3,
  "subtotal": 3200,
  "total_items": 4,
  "num_distinct_items": 3,
  "min_item_price": 400,
  "max_item_price": 1200,
  "total_onshift_partners": 12,
  "total_busy_partners": 9,
  "total_outstanding_orders": 15
}
```

Response:

```json
{
  "predicted_delivery_time_minutes": 32.6
}
```

**GET** `/health` — returns `{"status": "ok"}` for a basic liveness check.

## Tech Stack

- **Modeling**: TensorFlow/Keras, scikit-learn, LightGBM, SHAP
- **Backend**: Flask
- **Frontend**: HTML, CSS, vanilla JavaScript (fetch API)
- **Artifact persistence**: Keras native format (`.keras`) for the model, `joblib` for the scaler/encoders/feature lists

## Notes / Limitations

- `store_id` bucketing (grouping rare stores into an "other" category) is applied during training but not fully replicated at inference time — unseen store IDs fall back to the model's "unknown" category rather than a frequency-based bucket.
- The model is trained on a static historical snapshot; in production, operational features (`total_busy_partners`, etc.) should be sourced live rather than as static inputs.
- No authentication/rate-limiting is implemented — this is a demo/academic deployment, not production-hardened.
