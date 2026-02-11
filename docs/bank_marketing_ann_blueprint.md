# ANN Project Blueprint — Bank Marketing Term Deposit Prediction (MLP)

## 1) Project Summary
Build an **ANN (Multilayer Perceptron / Feed‑Forward Neural Network)** to predict whether a banking customer will **subscribe to a term deposit** after a marketing campaign contact.  
This is a **real-world tabular classification** problem (business + analytics friendly), ideal for demonstrating deep-learning fundamentals without synthetic data.

---

## 2) Dataset

### Dataset name (common sources)
- **Bank Marketing Dataset** (Portuguese bank direct marketing campaigns)
- Widely available via **Kaggle** and the **UCI Machine Learning Repository**

### Typical features (what you’ll see)
- **Customer profile**: age, job, marital status, education, default, housing loan, personal loan
- **Contact context**: contact type, month, day of week, duration, number of contacts (campaign)
- **History / prior outcomes**: previous contacts, previous outcome
- **Economic context (in some versions)**: employment variation rate, consumer price index, EURIBOR, etc.

### Target column
- **y** (or similar): *did the customer subscribe?* → `yes/no` (binary)

### Why this dataset is good for ANN
- Real business problem with clear ROI narrative
- Mix of numeric + categorical features (good to show proper preprocessing)
- Can demonstrate **class imbalance**, **calibration**, and **threshold tuning**

---

## 3) Model (DL technique)

### Technique used
✅ **ANN = Feed‑Forward Neural Network (FFNN) / Multilayer Perceptron (MLP)**

### Why MLP for this use case
- Handles **non-linear interactions** (e.g., combinations of customer profile + timing + campaign behavior)
- Strong baseline for tabular classification when engineered properly
- Easy to deploy: fast inference, small model size

### Suggested architecture (starter)
- Input: preprocessed feature vector
- Dense(128) → ReLU → Dropout(0.3)
- Dense(64)  → ReLU → Dropout(0.2)
- Dense(32)  → ReLU
- Output: Dense(1) → Sigmoid

### Loss / Optimizer
- Loss: **Binary Cross‑Entropy**
- Optimizer: **Adam**
- Metrics: **AUC**, **F1**, **Precision/Recall**, **PR‑AUC**

### Training notes (important)
- Standardize numeric features (e.g., StandardScaler)
- Handle categorical features:
  - Option 1 (simple): One‑Hot Encoding
  - Option 2 (advanced): Learned **Embeddings** for high‑cardinality categoricals
- Use **early stopping** (avoid overfitting)
- Tune probability threshold (banking decisions are cost-sensitive)

---

## 4) Evaluation (what to report)
Minimum recommended:
- Confusion Matrix
- Precision, Recall, F1
- ROC‑AUC and PR‑AUC
- Calibration curve (optional, but impressive)
- Business translation:
  - “If we call top X% ranked customers, expected conversion improves by Y%”
  - Lift / gains chart (optional)

---

## 5) System Architecture (end-to-end)

### High-level architecture
```text
                ┌─────────────────────────────┐
                │         Frontend            │
                │  (Streamlit or React UI)    │
                └──────────────┬──────────────┘
                               │ HTTPS (REST)
                               ▼
                ┌─────────────────────────────┐
                │           Backend           │
                │   FastAPI (Prediction API)  │
                │  /predict  /batch_predict   │
                └──────────────┬──────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│  Model Artifacts   │  │     Database      │  │    Monitoring     │
│  (Saved ANN +      │  │  PostgreSQL       │  │  Logs + metrics   │
│  preprocessors)    │  │ (users, requests, │  │  (Prometheus/     │
│  local/S3)         │  │  predictions)     │  │   Grafana optional)│
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

---

## 6) Backend

### Recommended backend
- **FastAPI (Python)** for:
  - `/predict` single customer prediction
  - `/batch_predict` CSV upload for batch scoring
  - `/health` service health check

### Backend responsibilities
- Load preprocessing pipeline (scaler + encoders)
- Load trained ANN model (`.keras` / `.h5` / SavedModel)
- Validate request schema (Pydantic)
- Return:
  - probability (0–1)
  - predicted class (`yes/no`)
  - optional top contributing features (if using explainability)

---

## 7) Frontend

### Option A (fastest): Streamlit
- Simple UI:
  - Manual input form (customer fields)
  - “Predict” button
  - Show probability + decision
  - Upload CSV for batch predictions
- Best if you want speed + clarity.

### Option B (portfolio): React + Tailwind
- Modern UI with:
  - form + CSV upload + results table
  - charts (conversion probability distribution)
- Calls FastAPI endpoints.

---

## 8) Database

### Recommended database
- **PostgreSQL**
  - Store prediction requests + results
  - Track model version + timestamp
  - Store user sessions (if you add login)

### Suggested tables
- `prediction_requests`:
  - `id, timestamp, input_payload_json, probability, predicted_label, model_version`
- `batch_jobs` (optional):
  - `job_id, uploaded_filename, status, created_at, completed_at`
- `model_registry` (optional):
  - `model_version, trained_on_date, metrics_json, artifact_path`

If you want minimal setup, you can start with **SQLite**, then move to PostgreSQL.

---

## 9) Project Structure (GitHub-ready)
```text
bank-marketing-ann/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
│  └─ 01_eda_and_baselines.ipynb
├─ src/
│  ├─ train.py
│  ├─ preprocess.py
│  ├─ evaluate.py
│  └─ config.py
├─ models/
│  ├─ ann_model.keras
│  └─ preprocess_pipeline.pkl
├─ api/
│  ├─ main.py          # FastAPI app
│  └─ schemas.py       # Pydantic schemas
├─ app/
│  └─ streamlit_app.py # or react-ui/
├─ requirements.txt
├─ README.md
└─ bank_marketing_ann_blueprint.md
```

---

## 10) Implementation Plan (quick)
1. **Data ingestion + EDA**
2. Build preprocessing pipeline (OHE + scaling)
3. Train **MLP ANN** + early stopping
4. Evaluate with AUC / F1 + threshold tuning
5. Save model + preprocessors
6. Create FastAPI `/predict` endpoint
7. Build UI (Streamlit or React)
8. Log predictions to DB
9. (Optional) Dockerize for clean deployment

---

## 11) Stretch Enhancements (if you want it to stand out)
- Use **categorical embeddings** instead of one-hot
- Add **calibration** (Platt scaling / isotonic) for better probability reliability
- Add **cost-sensitive threshold** (maximize expected profit)
- Add model monitoring (drift detection on feature distributions)
- Add explainability (SHAP for tabular baseline)

---

## 12) Deliverables (what you’ll show)
- Trained ANN model + metrics report
- API service + UI demo
- Short business summary: “targeting top X% improves conversions by Y%”
- GitHub repo with reproducible steps
