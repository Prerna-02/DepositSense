# 🏦 Bank Marketing — Term Deposit Predictor (ANN)

An end-to-end **deep learning project** that predicts whether a bank customer will subscribe to a term deposit, built with a Multilayer Perceptron (MLP) neural network.

## 📊 Results

| Metric        | Value  |
|---------------|--------|
| Accuracy      | 84%    |
| ROC-AUC       | 0.92   |
| PR-AUC        | 0.88   |
| Precision (Yes) | 0.80 |
| Recall (Yes)  | 0.90   |

## 🏗️ Architecture

```
data/raw/bank.csv → src/preprocess.py → src/train.py → models/ann_model.keras
                                                            ↓
                              app/streamlit_app.py ← api/main.py (FastAPI)
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess data
```bash
python -m src.preprocess
```

### 3. Train the model
```bash
python -m src.train
```

### 4. Evaluate
```bash
python -m src.evaluate
```

### 5. Start API server
```bash
uvicorn api.main:app --reload
```

### 6. Launch Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
Bank_Marketting_Term_Deposit/
├── data/
│   ├── raw/bank.csv              # Raw dataset
│   └── processed/                # Preprocessed numpy arrays
├── src/
│   ├── config.py                 # Central configuration
│   ├── preprocess.py             # Data preprocessing pipeline
│   ├── train.py                  # MLP model training
│   └── evaluate.py               # Model evaluation & plots
├── models/
│   ├── ann_model.keras           # Saved model
│   ├── preprocess_pipeline.pkl   # Fitted sklearn pipeline
│   ├── confusion_matrix.png      # Evaluation plots
│   ├── roc_curve.png
│   └── pr_curve.png
├── api/
│   ├── main.py                   # FastAPI prediction server
│   └── schemas.py                # Pydantic schemas
├── app/
│   └── streamlit_app.py          # Streamlit frontend
├── requirements.txt
└── README.md
```

## 🧠 Model Details

- **Type**: Feed-Forward Neural Network (MLP)
- **Layers**: Dense(128) → Dense(64) → Dense(32) → Dense(1, sigmoid)
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam with learning rate reduction
- **Regularization**: Dropout (0.3, 0.2) + Early Stopping

## 📌 Dataset

**Bank Marketing Dataset** (UCI / Kaggle) — Portuguese bank direct marketing campaigns.  
11,162 records × 16 features + 1 binary target (`deposit`: yes/no).
