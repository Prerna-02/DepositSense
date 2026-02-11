"""
Streamlit UI for the Bank Marketing Term Deposit Prediction.
Connects to the FastAPI backend for predictions.
Features:
  - Single customer prediction form
  - Batch CSV upload & download
  - Clean, modern dark-themed design
"""

import streamlit as st
import requests
import pandas as pd
import json

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing — Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-yes {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
    }
    .result-no {
        background: linear-gradient(135deg, #dc2626 0%, #f87171 100%);
        color: white;
    }
    .result-card .prob {
        font-size: 3rem;
        font-weight: 700;
    }
    .result-card .label {
        font-size: 1.3rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .metric-box {
        background: rgba(100, 116, 240, 0.08);
        border: 1px solid rgba(100, 116, 240, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Term Deposit Predictor</h1>
    <p>Predict customer subscription using an Artificial Neural Network</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Check API health ─────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

api_up = check_api()
if not api_up:
    st.warning("⚠️ API is offline. Start the backend with: `uvicorn api.main:app --reload`")


# ── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧑 Single Prediction", "📂 Batch Prediction"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Personal Info")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox("Job", [
            "admin.", "blue-collar", "entrepreneur", "housemaid",
            "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed", "unknown"
        ])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("Credit Default", ["no", "yes"])
        balance = st.number_input("Balance (€)", min_value=-10000, max_value=200000, value=1500)

    with col2:
        st.markdown("##### 🏠 Loans")
        housing = st.selectbox("Housing Loan", ["no", "yes"])
        loan = st.selectbox("Personal Loan", ["no", "yes"])

        st.markdown("##### 📞 Contact")
        contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
        month = st.selectbox("Month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        duration = st.number_input("Duration (sec)", min_value=0, max_value=5000, value=250)

    with col3:
        st.markdown("##### 📊 Campaign")
        campaign = st.number_input("# Contacts This Campaign", min_value=1, max_value=100, value=2)
        pdays = st.number_input("Days Since Prev Contact", min_value=-1, max_value=999, value=-1,
                                help="-1 means not previously contacted")
        previous = st.number_input("# Previous Contacts", min_value=0, max_value=100, value=0)
        poutcome = st.selectbox("Previous Outcome", ["unknown", "success", "failure", "other"])

    st.divider()

    if st.button("🔮 Predict", type="primary", use_container_width=True, disabled=(not api_up)):
        payload = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
        }

        with st.spinner("Running prediction..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                result = resp.json()

                prob = result["probability"]
                label = result["prediction"]

                css_class = "result-yes" if label == "yes" else "result-no"
                emoji = "✅" if label == "yes" else "❌"

                st.markdown(f"""
                <div class="result-card {css_class}">
                    <div class="prob">{prob:.1%}</div>
                    <div class="label">{emoji} Will {'Subscribe' if label == 'yes' else 'Not Subscribe'}</div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Probability", f"{prob:.4f}")
                c2.metric("Decision", label.upper())
                c3.metric("Model Version", result["model_version"])

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload CSV for Batch Predictions")
    st.caption("CSV must contain the same columns as the training data "
               "(age, job, marital, education, default, balance, housing, loan, "
               "contact, day, month, duration, campaign, pdays, previous, poutcome).")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_csv")

    if uploaded and api_up:
        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner("Processing batch..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                    resp = requests.post(f"{API_URL}/batch_predict", files=files, timeout=60)

                    if resp.status_code != 200:
                        st.error(f"API error: {resp.json().get('detail', resp.text)}")
                    else:
                        data = resp.json()
                        total = data["total"]
                        preds = data["predictions"]

                        st.success(f"✔ Processed **{total}** rows")

                        df_results = pd.DataFrame(preds)
                        yes_count = (df_results["prediction"] == "yes").sum()
                        no_count = total - yes_count

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total", total)
                        m2.metric("Will Subscribe", yes_count)
                        m3.metric("Won't Subscribe", no_count)

                        st.dataframe(df_results, use_container_width=True, height=400)

                        csv_out = df_results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download Results CSV",
                            data=csv_out,
                            file_name="predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
