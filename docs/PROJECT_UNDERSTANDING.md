# DepositSense — Project Understanding

## 1. Problem Statement

Banks run direct marketing campaigns (typically phone calls) to convince customers to subscribe to **term deposits**. These campaigns are expensive — calling every customer wastes time and money on low-probability leads. The goal is to **predict which customers are most likely to subscribe**, so the bank can focus its resources on high-conversion prospects.

> **Business question**: *"Given a customer's profile and campaign context, will they subscribe to a term deposit?"*

This is a **binary classification** problem: the answer is either **Yes** (subscribe) or **No** (don't subscribe).

---

## 2. Dataset Overview

We use the **Bank Marketing Dataset** originally from a Portuguese banking institution, widely available on Kaggle and the UCI ML Repository.

| Property | Detail |
|----------|--------|
| **Rows** | 11,162 |
| **Features** | 16 input features + 1 target |
| **Target** | `deposit` (yes / no) |
| **Imbalance** | Slight — 52.6% No, 47.4% Yes |

### Feature Categories

| Category | Features | Why They Matter |
|----------|----------|-----------------|
| **Customer Profile** | age, job, marital, education, default, balance, housing, loan | Captures the customer's financial stability and demographics |
| **Campaign Context** | contact, day, month, duration, campaign | Captures *how* and *when* the bank reached out — duration is the strongest predictor |
| **Historical** | pdays, previous, poutcome | Captures *past interactions* — previous success is a strong positive signal |

---

## 3. Why Deep Learning (ANN)?

### What is an ANN?

An **Artificial Neural Network (ANN)**, specifically a **Multilayer Perceptron (MLP)** or **Feed-Forward Neural Network (FFNN)**, is a deep learning architecture composed of layers of interconnected neurons. Each neuron applies a weighted sum followed by a non-linear activation function.

```
Input → [Dense Layer] → [Activation] → [Dense Layer] → ... → Output
```

### Why ANN / MLP for This Problem?

| Reason | Explanation |
|--------|-------------|
| **Non-linear interactions** | Traditional models (logistic regression) assume linear feature relationships. An ANN can learn complex interactions — e.g., "a retired person contacted in October with a long call duration has very high conversion probability" — without manual feature engineering. |
| **Handles mixed data** | After encoding, the MLP processes both numeric and categorical features seamlessly in a unified vector space. |
| **Scalable** | As data grows, the same architecture can be retrained without redesign. |
| **Deployable** | The trained model is compact (~500 KB), loads in seconds, and gives predictions in milliseconds — ideal for real-time API serving. |

### Why Not Simpler Models?

| Model | Limitation |
|-------|-----------|
| Logistic Regression | Assumes linear boundaries — misses complex feature interactions |
| Decision Tree | Prone to overfitting; fragile to small data variations |
| Random Forest / XGBoost | Strong alternatives for tabular data (often competitive), but we specifically use ANN to demonstrate **deep learning fundamentals** |

> **Note**: For tabular data, tree-based models (XGBoost, LightGBM) often perform comparably. We choose ANN here to demonstrate DL concepts in a real-world setting.

---

## 4. DL Techniques Used

### 4.1 Network Architecture

```
Input (51 features)
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(64, ReLU)  → Dropout(0.2)
    ↓
Dense(32, ReLU)
    ↓
Dense(1, Sigmoid) → Output (probability 0–1)
```

| Component | What It Does | Why We Use It |
|-----------|-------------|---------------|
| **Dense layers** | Fully connected layers that learn feature representations | Core building block of an MLP — each layer captures progressively abstract patterns |
| **ReLU activation** | `max(0, x)` — introduces non-linearity | Prevents vanishing gradient problem; computationally efficient; industry standard |
| **Sigmoid output** | Squashes output to [0, 1] | Gives a **probability** interpretation for binary classification |
| **Dropout** | Randomly zeros out neurons during training | **Regularization** technique that prevents overfitting by forcing the network to learn redundant representations |

### 4.2 Loss Function — Binary Cross-Entropy

```
Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

- Measures how far the predicted probability (ŷ) is from the actual label (y)
- Perfect for binary classification — penalizes confident wrong predictions heavily
- Works naturally with sigmoid output

### 4.3 Optimizer — Adam

- **Adam** (Adaptive Moment Estimation) combines momentum + adaptive learning rates
- Converges faster than vanilla SGD
- Less sensitive to learning rate selection
- We use initial LR = 0.001 with **ReduceLROnPlateau** to halve the LR when validation loss plateaus

### 4.4 Regularization Techniques

| Technique | Purpose |
|-----------|---------|
| **Dropout (0.3, 0.2)** | Prevents co-adaptation of neurons — forces robust feature learning |
| **Early Stopping** | Monitors validation loss; stops training when it stops improving (patience=10). Restores best weights. Prevents the model from memorizing training data |
| **Class Weighting** | Adjusts loss to give more weight to the minority class, countering any class imbalance |
| **LR Reduction** | Reduces learning rate by 50% when val loss plateaus for 5 epochs — helps escape local minima |

### 4.5 Data Preprocessing

| Step | Technique | Why |
|------|-----------|-----|
| Numeric features | **StandardScaler** (zero mean, unit variance) | Neural networks converge faster and perform better when inputs are on similar scales |
| Categorical features | **One-Hot Encoding** | Converts categories to binary vectors so the network doesn't impose false ordinal relationships |
| Train/Val/Test split | 70% / 15% / 15% (stratified) | Stratification ensures each split preserves the class ratio |

---

## 5. Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 84% | 84 out of 100 predictions are correct |
| **ROC-AUC** | 0.92 | Excellent discrimination — the model ranks subscribers above non-subscribers 92% of the time |
| **PR-AUC** | 0.88 | Strong performance even when focusing on the positive class |
| **Precision (Yes)** | 0.80 | Of customers predicted as "Yes", 80% actually subscribe |
| **Recall (Yes)** | 0.90 | Of actual subscribers, the model catches 90% of them |

### Business Impact

> *"If the bank calls only the top 50% of customers ranked by model probability, it would capture ~90% of actual subscribers while saving 50% of campaign costs."*

---

## 6. End-to-End Architecture

```
┌─────────────────────────────────┐
│        Streamlit Frontend       │
│  (Single + Batch Prediction)    │
└──────────────┬──────────────────┘
               │ HTTP (REST)
               ▼
┌─────────────────────────────────┐
│        FastAPI Backend          │
│  /predict  /batch_predict       │
│  /health                        │
└──────┬──────────┬───────────────┘
       │          │
       ▼          ▼
┌────────────┐  ┌──────────────┐
│ Model +    │  │ SQLite DB    │
│ Pipeline   │  │ (prediction  │
│ (.keras +  │  │  logging)    │
│  .pkl)     │  │              │
└────────────┘  └──────────────┘
```

---

## 7. Key Takeaways

1. **ANNs can effectively solve real business classification problems** — even with tabular data
2. **Proper preprocessing is crucial** — scaling + encoding directly impacts convergence
3. **Regularization matters** — dropout + early stopping prevent overfitting on small-medium datasets
4. **Probability output enables flexible decisions** — the bank can adjust the threshold based on cost-benefit analysis
5. **End-to-end deployment** — from raw data to a live API + UI in a single project

---

## 8. Technologies Used

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Preprocessing | scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Database | SQLite |
| Language | Python 3.10+ |
