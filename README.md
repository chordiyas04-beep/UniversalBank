# 🏦 Universal Bank — Loan Marketing Intelligence Dashboard

A professional Streamlit dashboard for Universal Bank's Marketing team to predict personal loan acceptance and design hyper-personalised campaigns.

## 🚀 Live Demo
Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud)

## 📊 Features

| Section | Description |
|---|---|
| 📊 Overview & Descriptive Analytics | KPIs, distributions, customer profiling |
| 🔍 Diagnostic Analytics | Correlation heatmaps, segment deep-dives |
| 🤖 Predictive Modelling | Decision Tree, Random Forest, Gradient Boosted Tree with ROC, Confusion Matrix |
| 🎯 Prescriptive Analytics | Target segments, budget allocation, campaign recommendations |
| 📤 Predict New Customers | Upload customer file → download predictions with risk tiers |

## 🗂 Project Structure

```
universal_bank_app/
├── app.py                  # Main Streamlit application
├── UniversalBank.xlsx      # Training dataset
├── sample_test_data.xlsx   # Sample file for predictions
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚙️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/universal-bank-loan-dashboard.git
cd universal-bank-loan-dashboard

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (ensure `UniversalBank.xlsx` is included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — live in ~2 minutes

## 🤖 Models Used

- **Decision Tree** — `max_depth=6`, interpretable baseline
- **Random Forest** — `n_estimators=200`, robust ensemble
- **Gradient Boosted Tree** — `n_estimators=200, lr=0.05`, best performance

All models trained with **SMOTE** oversampling to handle 9.6% class imbalance.

## 📋 Dataset Columns

| Column | Description |
|---|---|
| Age | Customer age (years) |
| Experience | Years of professional experience |
| Income | Annual income ($000) |
| Family | Family size (1–4) |
| CCAvg | Avg monthly credit card spend ($000) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced |
| Mortgage | Mortgage value ($000) |
| Securities Account | Has securities account (0/1) |
| CD Account | Has certificate of deposit account (0/1) |
| Online | Uses internet banking (0/1) |
| CreditCard | Has UB credit card (0/1) |
| **Personal Loan** | **Target: Accepted loan (0/1)** |

## 💡 Key Insights

- Income is the #1 predictor (correlation 0.50)
- CD Account holders convert at 46% vs 7% average
- $150–200k income band shows 50.5% acceptance rate
- Targeting top 2 segments with half budget outperforms broad campaigns

---
*Built for Universal Bank Marketing Intelligence | Powered by Streamlit + scikit-learn*
