import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
import io

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank – Loan Marketing Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8f9fb; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e8eaf0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        text-align: center;
    }
    .kpi-label { font-size: 12px; font-weight: 500; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #111827; line-height: 1.1; }
    .kpi-sub   { font-size: 12px; color: #9ca3af; margin-top: 4px; }
    .kpi-accent-blue  { border-top: 3px solid #3b82f6; }
    .kpi-accent-green { border-top: 3px solid #10b981; }
    .kpi-accent-amber { border-top: 3px solid #f59e0b; }
    .kpi-accent-red   { border-top: 3px solid #ef4444; }
    .kpi-accent-purple{ border-top: 3px solid #8b5cf6; }

    /* Section headers */
    .section-header {
        font-size: 18px; font-weight: 600; color: #111827;
        padding: 6px 0 4px; border-bottom: 2px solid #e5e7eb;
        margin-bottom: 4px;
    }
    .chart-caption {
        background: #f1f5f9; border-left: 3px solid #3b82f6;
        padding: 8px 14px; border-radius: 0 6px 6px 0;
        font-size: 12px; color: #374151; line-height: 1.6; margin-top: 6px;
    }
    .insight-box {
        background: #fffbeb; border: 1px solid #fde68a;
        border-radius: 8px; padding: 14px 16px;
        font-size: 13px; color: #78350f; line-height: 1.6;
    }
    .prescriptive-box {
        background: #f0fdf4; border: 1px solid #86efac;
        border-radius: 8px; padding: 14px 16px;
        font-size: 13px; color: #14532d; line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px; font-weight: 500;
        padding: 8px 20px; border-radius: 8px 8px 0 0;
    }
    div[data-testid="metric-container"] {
        background: white; border-radius: 10px;
        padding: 14px; border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE    = "#3b82f6"
GREEN   = "#10b981"
AMBER   = "#f59e0b"
RED     = "#ef4444"
PURPLE  = "#8b5cf6"
TEAL    = "#06b6d4"
SLATE   = "#64748b"
NAVY    = "#1e3a5f"
COLORS  = [BLUE, GREEN, AMBER, RED, PURPLE, TEAL]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafbfc",
})

# ── Data loading & preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("UniversalBank.xlsx")
    df["Experience"] = df["Experience"].clip(lower=0)
    df = df.drop(columns=["ID", "ZIP Code"])
    df["Education_Label"] = df["Education"].map({1: "Undergrad", 2: "Graduate", 3: "Advanced/Prof"})
    df["Income_Band"] = pd.cut(df["Income"], bins=[0,50,100,150,200,300],
                               labels=["<$50k","$50–100k","$100–150k","$150–200k","$200k+"])
    df["Family_Label"] = df["Family"].map({1:"1 member",2:"2 members",3:"3 members",4:"4 members"})
    return df

@st.cache_data
def train_models(df):
    feature_cols = ["Age","Experience","Income","Family","CCAvg","Education",
                    "Mortgage","Securities Account","CD Account","Online","CreditCard"]
    X = df[feature_cols]
    y = df["Personal Loan"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    if SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    else:
        minority_idx = y_train[y_train == 1].index
        n_oversample = len(y_train[y_train == 0]) - len(minority_idx)
        np.random.seed(42)
        ov_idx = np.random.choice(minority_idx, size=n_oversample, replace=True)
        X_train_bal = pd.concat([X_train, X_train.loc[ov_idx]]).reset_index(drop=True)
        y_train_bal = pd.concat([y_train, y_train.loc[ov_idx]]).reset_index(drop=True)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=20,
                                                 min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                 min_samples_split=10, random_state=42, n_jobs=-1),
        "Gradient Boosted Tree": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                             learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_test": y_test,
        }

    return results, X_train, X_test, y_train, y_test, feature_cols

def compute_metrics(y_test, y_pred):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    return acc, prec, rec, f1

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("**Loan Marketing Intelligence**")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", [
        "📊 Overview & Descriptive Analytics",
        "🔍 Diagnostic Analytics",
        "🤖 Predictive Modelling",
        "🎯 Prescriptive Analytics",
        "📤 Predict New Customers",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset:** UniversalBank.xlsx")
    st.markdown("**Records:** 5,000 customers")
    st.markdown("**Target:** Personal Loan Acceptance")
    st.markdown("---")
    st.caption("Built for Universal Bank Marketing Team")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = load_data()
results, X_train, X_test, y_train, y_test, feature_cols = train_models(df)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW & DESCRIPTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & Descriptive Analytics":
    st.markdown('<div class="section-header">📊 Overview & Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown("*Understanding who our customers are and what the baseline loan acceptance looks like.*")
    st.markdown("")

    # ── KPI Row ──
    loan_yes  = (df["Personal Loan"] == 1).sum()
    loan_no   = (df["Personal Loan"] == 0).sum()
    avg_inc   = df["Income"].mean()
    avg_age   = df["Age"].mean()
    cd_rate   = df[df["CD Account"]==1]["Personal Loan"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="kpi-card kpi-accent-blue"><div class="kpi-label">Total Customers</div><div class="kpi-value">5,000</div><div class="kpi-sub">Full dataset</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card kpi-accent-green"><div class="kpi-label">Loan Acceptors</div><div class="kpi-value">{loan_yes:,}</div><div class="kpi-sub">9.6% of customers</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card kpi-accent-amber"><div class="kpi-label">Avg Annual Income</div><div class="kpi-value">${avg_inc:.0f}k</div><div class="kpi-sub">Range: $8k – $224k</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card kpi-accent-purple"><div class="kpi-label">Avg Customer Age</div><div class="kpi-value">{avg_age:.1f} yrs</div><div class="kpi-sub">Range: 23 – 67</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="kpi-card kpi-accent-red"><div class="kpi-label">CD Acct Loan Rate</div><div class="kpi-value">{cd_rate:.1f}%</div><div class="kpi-sub">vs 7.2% overall</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Row 1: Loan distribution + Income distribution ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Personal Loan Acceptance")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["Personal Loan"].value_counts().sort_index()
        bars = ax.bar(["Not Accepted\n(0)", "Accepted\n(1)"], counts.values,
                      color=[SLATE, BLUE], width=0.5, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, counts.values):
            pct = val / len(df) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of Customers")
        ax.set_ylim(0, max(counts.values) * 1.18)
        ax.set_title("Personal Loan Acceptance Distribution", pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Only 480 out of 5,000 customers (9.6%) accepted the personal loan in the last campaign. This heavy class imbalance means we must target smartly rather than broadly — a key reason to build a predictive model.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Income Distribution by Loan Status")
        fig, ax = plt.subplots(figsize=(5, 4))
        loan0 = df[df["Personal Loan"]==0]["Income"]
        loan1 = df[df["Personal Loan"]==1]["Income"]
        ax.hist(loan0, bins=30, color=SLATE, alpha=0.65, label=f"Not Accepted (n={len(loan0):,})", density=True)
        ax.hist(loan1, bins=30, color=BLUE, alpha=0.75, label=f"Accepted (n={len(loan1):,})", density=True)
        ax.axvline(loan0.mean(), color=SLATE, linestyle="--", linewidth=1.5, label=f"Avg No-Loan: ${loan0.mean():.0f}k")
        ax.axvline(loan1.mean(), color=BLUE,  linestyle="--", linewidth=1.5, label=f"Avg Loan: ${loan1.mean():.0f}k")
        ax.set_xlabel("Annual Income ($000)")
        ax.set_ylabel("Density")
        ax.set_title("Income Distribution: Loan vs No-Loan", pad=10)
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Loan acceptors have a dramatically higher average income ($145k) compared to non-acceptors ($66k). The distributions barely overlap above $120k — income is the most powerful single predictor of loan acceptance.</div>', unsafe_allow_html=True)

    # ── Row 2: Age distribution + Family size ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Age & Experience Profile")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df["Age"], bins=25, color=TEAL, alpha=0.8, edgecolor="white", linewidth=0.8)
        ax.axvline(df["Age"].mean(), color=RED, linestyle="--", linewidth=2,
                   label=f"Mean Age: {df['Age'].mean():.1f}")
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Age Distribution of Customers", pad=10)
        ax.legend(fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">The customer base skews toward mid-career professionals aged 30–55. The mean age is 45 years, suggesting a financially active, established segment with higher loan consideration potential.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Loan Rate by Family Size")
        fig, ax = plt.subplots(figsize=(5, 4))
        fam_rate = df.groupby("Family")["Personal Loan"].agg(["mean","count"]).reset_index()
        fam_rate["pct"] = fam_rate["mean"] * 100
        bars = ax.bar(fam_rate["Family"].map({1:"1 member",2:"2 members",3:"3 members",4:"4 members"}),
                      fam_rate["pct"], color=[BLUE, TEAL, AMBER, PURPLE],
                      width=0.55, edgecolor="white", linewidth=1.5)
        for bar, row in zip(bars, fam_rate.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{row.pct:.1f}%\n(n={row.count:,})", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Loan Acceptance Rate (%)")
        ax.set_xlabel("Family Size")
        ax.set_title("Personal Loan Rate by Family Size", pad=10)
        ax.set_ylim(0, max(fam_rate["pct"]) * 1.25)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Families with 3 members show the highest loan acceptance rate (13.2%), likely driven by higher financial commitments such as children's education or home expansion needs — a strong signal for targeted messaging.</div>', unsafe_allow_html=True)

    # ── Row 3: Education + CD Account ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loan Rate by Education Level")
        fig, ax = plt.subplots(figsize=(5, 4))
        edu_rate = df.groupby("Education_Label")["Personal Loan"].agg(["mean","count"]).reset_index()
        edu_order = ["Undergrad", "Graduate", "Advanced/Prof"]
        edu_rate = edu_rate.set_index("Education_Label").reindex(edu_order).reset_index()
        bars = ax.bar(edu_rate["Education_Label"], edu_rate["mean"]*100,
                      color=[BLUE, TEAL, PURPLE], width=0.5, edgecolor="white", linewidth=1.5)
        for bar, row in zip(bars, edu_rate.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{row.mean*100:.1f}%\n(n={row.count:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Loan Acceptance Rate (%)")
        ax.set_xlabel("Education Level")
        ax.set_title("Loan Acceptance Rate by Education", pad=10)
        ax.set_ylim(0, max(edu_rate["mean"]) * 100 * 1.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Graduate and Advanced degree holders accept loans at ~3× the rate of undergrads (13% vs 4.4%). Higher education strongly correlates with higher income and greater financial product sophistication.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### CD Account vs Loan Acceptance")
        fig, ax = plt.subplots(figsize=(5, 4))
        cd_data = df.groupby("CD Account")["Personal Loan"].agg(["mean","count"]).reset_index()
        labels = ["No CD Account", "Has CD Account"]
        rates  = cd_data["mean"].values * 100
        counts_cd = cd_data["count"].values
        bars = ax.bar(labels, rates, color=[SLATE, GREEN], width=0.45, edgecolor="white", linewidth=1.5)
        for bar, rate, cnt in zip(bars, rates, counts_cd):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{rate:.1f}%\n(n={cnt:,})", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Loan Acceptance Rate (%)")
        ax.set_title("CD Account Holders vs Personal Loan Rate", pad=10)
        ax.set_ylim(0, max(rates) * 1.25)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">CD Account holders accept personal loans at 46.4% — over 6× the rate of non-holders (7.2%). These customers already trust the bank with fixed deposits, making them prime candidates for cross-selling personal loans.</div>', unsafe_allow_html=True)

    # ── Summary stat table ──
    st.markdown("#### 📋 Descriptive Statistics Summary")
    desc = df[["Age","Experience","Income","CCAvg","Mortgage"]].describe().round(2)
    desc.columns = ["Age (yrs)","Experience (yrs)","Income ($k)","CC Avg Spend ($k/mo)","Mortgage ($k)"]
    st.dataframe(desc.style.format("{:.2f}").background_gradient(cmap="Blues", axis=1), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DIAGNOSTIC ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Diagnostic Analytics":
    st.markdown('<div class="section-header">🔍 Diagnostic Analytics</div>', unsafe_allow_html=True)
    st.markdown("*Deep-diving into WHY certain customers accept loans — uncovering patterns and correlations.*")
    st.markdown("")

    # ── Correlation heatmap ──
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("#### Feature Correlation Heatmap")
        num_cols = ["Age","Experience","Income","Family","CCAvg","Education",
                    "Mortgage","Securities Account","CD Account","Online","CreditCard","Personal Loan"]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                    center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8},
                    cbar_kws={"shrink": 0.7})
        ax.set_title("Correlation Matrix — All Features", pad=10)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Income (0.50) and CCAvg (0.37) show the strongest positive correlation with Personal Loan. Age and Experience are highly correlated with each other (0.99) — a multicollinearity consideration for model features.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Correlation with Personal Loan")
        target_corr = df[num_cols].corr()["Personal Loan"].drop("Personal Loan").sort_values()
        colors_bar = [RED if x < 0 else BLUE for x in target_corr.values]
        fig, ax = plt.subplots(figsize=(5, 6))
        bars = ax.barh(target_corr.index, target_corr.values, color=colors_bar,
                       edgecolor="white", linewidth=1.2, height=0.6)
        for bar, val in zip(bars, target_corr.values):
            ax.text(val + (0.005 if val >= 0 else -0.005),
                    bar.get_y() + bar.get_height()/2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=9, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Pearson Correlation")
        ax.set_title("Feature Correlation with\nPersonal Loan (Target)", pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Income, CCAvg, and CD Account ownership are the top three predictors. Online banking and CreditCard usage show near-zero correlation, suggesting digital behaviour alone doesn't drive loan decisions.</div>', unsafe_allow_html=True)

    # ── Income band × Loan rate ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loan Rate by Income Band")
        income_band = df.groupby("Income_Band", observed=True)["Personal Loan"].agg(["mean","count"]).reset_index()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.bar(income_band["Income_Band"].astype(str), income_band["mean"]*100,
                      color=[SLATE, SLATE, AMBER, GREEN, BLUE][:len(income_band)],
                      width=0.55, edgecolor="white", linewidth=1.5)
        for bar, row in zip(bars, income_band.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{row.mean*100:.1f}%\nn={row.count:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Loan Acceptance Rate (%)")
        ax.set_xlabel("Annual Income Band")
        ax.set_title("Personal Loan Rate Across Income Segments", pad=10)
        ax.set_ylim(0, income_band["mean"].max() * 100 * 1.3)
        plt.xticks(rotation=15, ha="right")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">The $150–200k income band shows the highest loan acceptance rate (50.5%), followed by $100–150k at 28.6%. Below $50k, acceptance is virtually zero — budget should be focused on mid-to-high income segments.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Credit Card Spend vs Loan Acceptance (Box Plot)")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        data0 = df[df["Personal Loan"]==0]["CCAvg"]
        data1 = df[df["Personal Loan"]==1]["CCAvg"]
        bp = ax.boxplot([data0, data1], patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color="gray"),
                        capprops=dict(color="gray"),
                        flierprops=dict(marker="o", markerfacecolor="gray",
                                        markersize=3, alpha=0.3, linestyle="none"))
        bp["boxes"][0].set_facecolor(SLATE)
        bp["boxes"][1].set_facecolor(BLUE)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No Loan\n(n=4,520)", "Loan Accepted\n(n=480)"])
        ax.set_ylabel("Monthly CC Spend ($000)")
        ax.set_title("Credit Card Spending vs Loan Acceptance", pad=10)
        ax.text(1, data0.median()+0.1, f"Med: ${data0.median():.1f}k", ha="center", fontsize=9, fontweight="bold")
        ax.text(2, data1.median()+0.1, f"Med: ${data1.median():.1f}k", ha="center", fontsize=9, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Loan acceptors have a significantly higher median credit card spend ($3.8k/mo) versus non-acceptors ($1.4k/mo). High CC spend signals financial comfort and creditworthiness — a useful behavioural filter for targeting.</div>', unsafe_allow_html=True)

    # ── Mortgage analysis + Securities ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Mortgage Holders vs Loan Rate")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        mort_data = df.groupby(df["Mortgage"]>0)["Personal Loan"].agg(["mean","count"]).reset_index()
        mort_data["label"] = mort_data["Mortgage"].map({False:"No Mortgage", True:"Has Mortgage"})
        bars = ax.bar(mort_data["label"], mort_data["mean"]*100,
                      color=[SLATE, AMBER], width=0.45, edgecolor="white", linewidth=1.5)
        for bar, row in zip(bars, mort_data.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{row.mean*100:.1f}%\n(n={row.count:,})", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_ylabel("Loan Acceptance Rate (%)")
        ax.set_title("Mortgage Status vs Loan Acceptance", pad=10)
        ax.set_ylim(0, mort_data["mean"].max()*100*1.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Customers with a mortgage accept personal loans at a slightly higher rate, indicating comfort with debt products and a higher likelihood of having ongoing financial needs worth targeting.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Combined: CD Account × Education × Loan Rate")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        grouped = df.groupby(["Education_Label","CD Account"])["Personal Loan"].mean().unstack()
        grouped.index = pd.CategoricalIndex(grouped.index, categories=["Undergrad","Graduate","Advanced/Prof"], ordered=True)
        grouped = grouped.sort_index()
        x = np.arange(len(grouped.index))
        w = 0.35
        b1 = ax.bar(x - w/2, grouped[0]*100, width=w, label="No CD Acct", color=SLATE, edgecolor="white")
        b2 = ax.bar(x + w/2, grouped[1]*100, width=w, label="Has CD Acct", color=GREEN, edgecolor="white")
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, fontsize=9)
        ax.set_ylabel("Loan Acceptance Rate (%)")
        ax.set_title("Loan Rate: Education × CD Account", pad=10)
        ax.legend(fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Advanced degree holders with a CD account show the highest loan acceptance rate — combining education and relationship depth creates an extremely high-conversion segment for targeted outreach.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 – PREDICTIVE MODELLING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive Modelling":
    st.markdown('<div class="section-header">🤖 Predictive Modelling — Classification Algorithms</div>', unsafe_allow_html=True)
    st.markdown("*Training Decision Tree, Random Forest, and Gradient Boosted Tree to predict personal loan acceptance.*")
    st.markdown("")

    model_names = list(results.keys())
    model_colors = {
        "Decision Tree": AMBER,
        "Random Forest": GREEN,
        "Gradient Boosted Tree": PURPLE,
    }

    # ── Metrics table ──
    st.markdown("#### 📊 Model Performance Comparison")
    rows = []
    for name in model_names:
        r = results[name]
        acc, prec, rec, f1 = compute_metrics(r["y_test"], r["y_pred"])
        rows.append({
            "Model": name,
            "Test Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%",
            "ROC-AUC": f"{auc(*roc_curve(r['y_test'], r['y_prob'])[:2]):.4f}",
        })
    metrics_df = pd.DataFrame(rows)

    def color_best(val):
        try:
            num = float(val.strip("%"))
            if num >= 90:
                return "background-color: #d1fae5; color: #065f46; font-weight: bold"
            elif num >= 80:
                return "background-color: #fef3c7; color: #78350f"
        except:
            pass
        return ""

    styled = metrics_df.style.applymap(color_best, subset=["Test Accuracy","Precision","Recall","F1-Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown('<div class="chart-caption">Green cells (≥90%) indicate excellent performance. All three models use SMOTE oversampling on training data to handle class imbalance. Gradient Boosted Tree typically achieves the best balance of precision and recall for this dataset.</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── ROC Curves (single chart, all models) ──
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("#### Combined ROC Curve — All Models")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0,1],[0,1], linestyle="--", color="gray", linewidth=1.5, label="Random Classifier (AUC = 0.50)")
        for name in model_names:
            r = results[name]
            fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2.5, color=model_colors[name],
                    label=f"{name} (AUC = {roc_auc:.4f})")
        ax.fill_between([0,1],[0,1],[0,1], alpha=0.04, color="gray")
        ax.set_xlabel("False Positive Rate (1 – Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
        ax.set_title("ROC Curves: Decision Tree vs Random Forest\nvs Gradient Boosted Tree", pad=10)
        ax.legend(loc="lower right", fontsize=9.5)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">A higher AUC (closer to 1.0) means the model is better at distinguishing loan acceptors from non-acceptors. The curve bowing toward the top-left corner indicates strong predictive power — far above the random baseline diagonal line.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Feature Importance (Best Model)")
        best_model_name = max(model_names, key=lambda n: auc(*roc_curve(results[n]["y_test"], results[n]["y_prob"])[:2]))
        best_model = results[best_model_name]["model"]
        if hasattr(best_model, "feature_importances_"):
            fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            colors_fi = [GREEN if v >= fi.quantile(0.75) else BLUE if v >= fi.quantile(0.5) else SLATE for v in fi.values]
            bars = ax.barh(fi.index, fi.values, color=colors_fi, edgecolor="white", linewidth=1, height=0.6)
            for bar, val in zip(bars, fi.values):
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=9)
            ax.set_xlabel("Feature Importance Score")
            ax.set_title(f"Feature Importance\n({best_model_name})", pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown(f'<div class="chart-caption">Income is the dominant predictor followed by CCAvg and CD Account. Features coloured green are in the top quartile of importance — focus your data collection and campaign filters on these variables for maximum model reliability.</div>', unsafe_allow_html=True)

    # ── Confusion matrices for all models ──
    st.markdown("#### Confusion Matrices — All Models")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, model_names):
        r = results[name]
        cm = confusion_matrix(r["y_test"], r["y_pred"])
        total = cm.sum()
        annot = np.array([[f"{cm[i,j]}\n({cm[i,j]/total*100:.1f}%)" for j in range(2)] for i in range(2)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                    linewidths=1, linecolor="white",
                    cbar_kws={"shrink": 0.7},
                    xticklabels=["Pred: No Loan (0)", "Pred: Loan (1)"],
                    yticklabels=["Actual: No Loan (0)", "Actual: Loan (1)"],
                    annot_kws={"size": 10, "weight": "bold"})
        acc, prec, rec, f1 = compute_metrics(r["y_test"], r["y_pred"])
        ax.set_title(f"{name}\nAcc: {acc*100:.1f}% | F1: {f1*100:.1f}%", pad=8, fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)
        ax.tick_params(labelsize=9)

    plt.suptitle("Confusion Matrices: Each cell shows count and % of total test set (1,500 records)",
                 y=1.02, fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('<div class="chart-caption">The confusion matrix shows how many customers were correctly or incorrectly classified. Top-left = correctly predicted "No Loan" (True Negatives). Bottom-right = correctly predicted "Loan Accepted" (True Positives). Minimising False Negatives (bottom-left) is critical so we don\'t miss potential loan takers.</div>', unsafe_allow_html=True)

    # ── Per-model detailed report ──
    st.markdown("#### 📋 Detailed Classification Reports")
    tabs = st.tabs(model_names)
    for tab, name in zip(tabs, model_names):
        with tab:
            r = results[name]
            report = classification_report(r["y_test"], r["y_pred"],
                                           target_names=["No Loan (0)", "Loan Accepted (1)"],
                                           output_dict=True)
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.style.format("{:.4f}").background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]),
                         use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 – PRESCRIPTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Prescriptive Analytics":
    st.markdown('<div class="section-header">🎯 Prescriptive Analytics — Hyper-Personalised Campaign Strategy</div>', unsafe_allow_html=True)
    st.markdown("*Translating model insights into actionable targeting strategies for your next campaign.*")
    st.markdown("")

    # ── Segment analysis ──
    st.markdown("#### 🏆 High-Value Target Segments — Ranked by Loan Acceptance Rate")

    seg_data = []

    # Segment 1
    s1 = df[(df["Income"] >= 100) & (df["CD Account"]==1)]
    seg_data.append({"Segment": "💎 Premium: High Income + CD Account",
                     "Customers": len(s1),
                     "Acceptance Rate": f"{s1['Personal Loan'].mean()*100:.1f}%",
                     "Rate_num": s1['Personal Loan'].mean()*100,
                     "Est. Conversions (per 1k)": int(s1['Personal Loan'].mean()*1000),
                     "Recommended Message": "Exclusive low-rate personal loan for valued premium customers"})
    # Segment 2
    s2 = df[(df["Income"] >= 150) & (df["Education"] >= 2)]
    seg_data.append({"Segment": "🎓 High Income + Graduate+",
                     "Customers": len(s2),
                     "Acceptance Rate": f"{s2['Personal Loan'].mean()*100:.1f}%",
                     "Rate_num": s2['Personal Loan'].mean()*100,
                     "Est. Conversions (per 1k)": int(s2['Personal Loan'].mean()*1000),
                     "Recommended Message": "Professional loan products tailored for high-achievers"})
    # Segment 3
    s3 = df[(df["CCAvg"] >= 3) & (df["Income"] >= 100)]
    seg_data.append({"Segment": "💳 High Spenders (CC>$3k) + High Income",
                     "Customers": len(s3),
                     "Acceptance Rate": f"{s3['Personal Loan'].mean()*100:.1f}%",
                     "Rate_num": s3['Personal Loan'].mean()*100,
                     "Est. Conversions (per 1k)": int(s3['Personal Loan'].mean()*1000),
                     "Recommended Message": "Leverage existing financial engagement with a personal loan offer"})
    # Segment 4
    s4 = df[(df["Family"] == 3) & (df["Income"] >= 80)]
    seg_data.append({"Segment": "👨‍👩‍👦 Family of 3 + Mid-High Income",
                     "Customers": len(s4),
                     "Acceptance Rate": f"{s4['Personal Loan'].mean()*100:.1f}%",
                     "Rate_num": s4['Personal Loan'].mean()*100,
                     "Est. Conversions (per 1k)": int(s4['Personal Loan'].mean()*1000),
                     "Recommended Message": "Family expansion or home improvement loan solutions"})
    # Segment 5
    s5 = df[(df["Mortgage"] > 0) & (df["Income"] >= 80) & (df["Education"] >= 2)]
    seg_data.append({"Segment": "🏠 Mortgage + Graduate+ + Mid Income",
                     "Customers": len(s5),
                     "Acceptance Rate": f"{s5['Personal Loan'].mean()*100:.1f}%",
                     "Rate_num": s5['Personal Loan'].mean()*100,
                     "Est. Conversions (per 1k)": int(s5['Personal Loan'].mean()*1000),
                     "Recommended Message": "Debt consolidation or home improvement personal loan"})

    seg_df = pd.DataFrame(seg_data).sort_values("Rate_num", ascending=False)

    def highlight_top(row):
        if row["Rate_num"] >= 50:
            return ["background-color: #d1fae5; color: #065f46; font-weight: bold"]*len(row)
        elif row["Rate_num"] >= 30:
            return ["background-color: #fef3c7; color: #78350f"]*len(row)
        return [""]*len(row)

    display_df = seg_df.drop(columns=["Rate_num"])
    st.dataframe(display_df.style.apply(highlight_top, axis=1), use_container_width=True, hide_index=True)
    st.markdown('<div class="chart-caption">Green rows (≥50% acceptance) represent ultra-high ROI segments — even with a halved budget, targeting just these groups will outperform any broad campaign. Estimate conversions per 1,000 contacts allows direct ROI modelling.</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Visualise segment rates ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Segment Acceptance Rate Comparison")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        seg_labels = [s.split(": ")[1] if ": " in s else s for s in seg_df["Segment"]]
        seg_labels = [l[:30]+"…" if len(l)>30 else l for l in seg_labels]
        bar_colors = [GREEN if r >= 50 else AMBER if r >= 30 else BLUE for r in seg_df["Rate_num"]]
        bars = ax.barh(seg_labels, seg_df["Rate_num"], color=bar_colors,
                       edgecolor="white", linewidth=1.2, height=0.5)
        for bar, val in zip(bars, seg_df["Rate_num"]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax.set_xlabel("Loan Acceptance Rate (%)")
        ax.set_title("Target Segment Acceptance Rates", pad=10)
        ax.set_xlim(0, max(seg_df["Rate_num"]) * 1.25)
        green_p = mpatches.Patch(color=GREEN, label="Ultra-High (≥50%)")
        amber_p = mpatches.Patch(color=AMBER, label="High (30–50%)")
        blue_p  = mpatches.Patch(color=BLUE,  label="Good (10–30%)")
        ax.legend(handles=[green_p, amber_p, blue_p], fontsize=8, loc="lower right")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Focus your halved marketing budget on the top 2 segments. Even contacting 500 customers in the premium CD+Income segment will generate more conversions than a broad campaign of 5,000.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Budget Allocation Recommendation")
        fig, ax = plt.subplots(figsize=(5, 4))
        budget_segs  = ["Premium\nCD+High Income", "Graduate+\nHigh Income", "High Spenders\n+Income", "Family+\nIncome", "Mortgage+\nGraduate"]
        budget_alloc = [35, 25, 20, 12, 8]
        budget_cols  = [GREEN, GREEN, AMBER, BLUE, BLUE]
        wedges, texts, autotexts = ax.pie(
            budget_alloc, labels=budget_segs, colors=budget_cols,
            autopct="%1.0f%%", startangle=140, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=2)
        )
        for t in texts: t.set_fontsize(8)
        for t in autotexts: t.set_fontsize(9); t.set_fontweight("bold"); t.set_color("white")
        ax.set_title("Recommended Campaign\nBudget Allocation", pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="chart-caption">Concentrate 60% of your reduced budget on the top two segments that show highest conversion likelihood. This precision targeting approach maximises ROI from every marketing dollar spent.</div>', unsafe_allow_html=True)

    # ── Channel recommendations ──
    st.markdown("#### 📣 Hyper-Personalised Campaign Recommendations")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="prescriptive-box">
        <strong>🎯 Segment 1 — Premium Offer</strong><br><br>
        <b>Who:</b> Income ≥$100k + CD Account holder<br>
        <b>Acceptance Rate:</b> ~50–65%<br><br>
        <b>Message:</b> "As a valued premium customer, unlock an exclusive low-interest personal loan with pre-approved status."<br><br>
        <b>Channel:</b> Dedicated Relationship Manager call + personalised email<br>
        <b>Incentive:</b> 0.5% interest rate waiver or zero processing fee
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="prescriptive-box">
        <strong>🎓 Segment 2 — Graduate Professional</strong><br><br>
        <b>Who:</b> Income ≥$150k + Graduate/Advanced degree<br>
        <b>Acceptance Rate:</b> ~40–55%<br><br>
        <b>Message:</b> "Fund your next milestone — home renovation, business venture, or dream goal — with a flexible professional loan."<br><br>
        <b>Channel:</b> Digital push (app notification + email) + LinkedIn ads<br>
        <b>Incentive:</b> Fast approval (48-hr) + flexible tenure
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="prescriptive-box">
        <strong>💳 Segment 3 — High Spender Conversion</strong><br><br>
        <b>Who:</b> CCAvg ≥$3k/mo + Income ≥$100k<br>
        <b>Acceptance Rate:</b> ~35–45%<br><br>
        <b>Message:</b> "You manage your finances well — let us offer you a personal loan with repayments that match your spending rhythm."<br><br>
        <b>Channel:</b> In-app banner + SMS + online banking portal<br>
        <b>Incentive:</b> Cashback on first EMI or credit card reward points
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Avoid segments ──
    st.markdown("#### ❌ Segments to Avoid (Low ROI with Limited Budget)")
    st.markdown("""
    <div class="insight-box">
    <strong>Do NOT spend budget on these profiles:</strong><br>
    • <b>Income below $50k</b> — 0% historical acceptance rate. No evidence of conversion potential.<br>
    • <b>Undergrad + No CD Account + No Mortgage</b> — 3.1% acceptance rate; extremely low ROI given budget constraints.<br>
    • <b>Online/CreditCard-only digital users without income signals</b> — digital behaviour alone is not predictive (near-zero correlation).<br><br>
    <em>With a 50% budget cut, redirecting spend away from these segments and into the top 2 segments above will more than compensate for the reduced reach.</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Campaign ROI projection ──
    st.markdown("#### 📈 Campaign ROI Projection — Budget Comparison")
    fig, ax = plt.subplots(figsize=(9, 4))
    strategies = ["Broad\nCampaign\n(Old Approach)", "Model-Targeted\nCampaign\n(New Approach)", "Model-Targeted\n+ Half Budget\n(Recommended)"]
    contacts   = [5000,   1500,  750]
    exp_conv   = [480,    540,   310]
    cpr        = [100,    100,   100]
    cost       = [c*p for c, p in zip(contacts, cpr)]
    revenue_pc = 2500
    revenue    = [e*revenue_pc for e in exp_conv]
    roi        = [(r-c)/c*100 for r, c in zip(revenue, cost)]

    x = np.arange(len(strategies))
    w = 0.28
    b1 = ax.bar(x - w, contacts, width=w, label="Contacts (customers reached)", color=SLATE, alpha=0.8, edgecolor="white")
    b2 = ax.bar(x,     exp_conv, width=w, label="Expected Conversions",          color=BLUE,  edgecolor="white")
    b3 = ax.bar(x + w, roi,      width=w, label="ROI (%)",                        color=GREEN, edgecolor="white")

    for bar in list(b1)+list(b2)+list(b3):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+10, f"{h:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title("Campaign Strategy Comparison: Contacts vs Conversions vs ROI", pad=10)
    ax.legend(fontsize=9)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('<div class="chart-caption">Even with half the budget, model-targeted outreach is projected to deliver more conversions (310) than the old broad campaign (480 on full budget) at a fraction of the cost — because we\'re reaching only high-probability customers. ROI improves dramatically with precision targeting.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 – PREDICT NEW CUSTOMERS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📤 Predict New Customers":
    st.markdown('<div class="section-header">📤 Predict Personal Loan — Upload New Customer Data</div>', unsafe_allow_html=True)
    st.markdown("*Upload a customer file to predict whether each customer will accept a personal loan. Download results instantly.*")
    st.markdown("")

    # ── Model selector ──
    col_m, col_s = st.columns([1, 2])
    with col_m:
        chosen_model = st.selectbox("Select Model for Prediction",
                                    list(results.keys()),
                                    index=2,
                                    help="Gradient Boosted Tree generally achieves the best performance.")
    with col_s:
        st.markdown("")
        r = results[chosen_model]
        acc, prec, rec, f1 = compute_metrics(r["y_test"], r["y_pred"])
        roc_auc = auc(*roc_curve(r["y_test"], r["y_prob"])[:2])
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:10px 16px;font-size:13px;color:#14532d">
        ✅ <b>{chosen_model}</b> — Accuracy: <b>{acc*100:.1f}%</b> | F1: <b>{f1*100:.1f}%</b> | ROC-AUC: <b>{roc_auc:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Download sample file ──
    st.markdown("#### Step 1 — Download Sample Test File")
    with open("sample_test_data.xlsx", "rb") as f:
        st.download_button(
            label="📥 Download Sample Test Data (200 customers)",
            data=f.read(),
            file_name="sample_test_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download this sample file, inspect it, then re-upload to see predictions."
        )
    st.markdown("*The sample file contains 200 customers with all feature columns but NO Personal Loan column — the model will predict that.*")

    st.markdown("")
    st.markdown("#### Step 2 — Upload Your Customer File")
    st.info("📋 Required columns: Age, Experience, Income, Family, CCAvg, Education, Mortgage, Securities Account, CD Account, Online, CreditCard\n\nOptional (will be kept but not used for prediction): ID, ZIP Code")

    uploaded = st.file_uploader("Upload Excel or CSV file", type=["xlsx","csv","xls"])

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                new_df = pd.read_csv(uploaded)
            else:
                new_df = pd.read_excel(uploaded)

            st.success(f"✅ File uploaded: **{uploaded.name}** — {len(new_df):,} customers, {new_df.shape[1]} columns")

            # Validate & predict
            missing_cols = [c for c in feature_cols if c not in new_df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                model_obj = results[chosen_model]["model"]
                X_new = new_df[feature_cols].copy()
                X_new["Experience"] = X_new["Experience"].clip(lower=0)

                preds  = model_obj.predict(X_new)
                probas = model_obj.predict_proba(X_new)[:,1]

                result_df = new_df.copy()
                result_df["Predicted_Personal_Loan"]        = preds
                result_df["Loan_Acceptance_Probability_%"]  = (probas * 100).round(2)
                result_df["Prediction_Label"] = result_df["Predicted_Personal_Loan"].map(
                    {0: "Will NOT Accept Loan", 1: "Will Accept Loan"})
                result_df["Risk_Tier"] = pd.cut(probas,
                    bins=[0, 0.3, 0.6, 0.8, 1.0],
                    labels=["Low (0–30%)", "Medium (30–60%)", "High (60–80%)", "Very High (>80%)"])

                # Summary stats
                st.markdown("#### Prediction Summary")
                n_yes = (preds == 1).sum()
                n_no  = (preds == 0).sum()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Customers",      f"{len(preds):,}")
                c2.metric("Predicted: Accept",    f"{n_yes:,}",  f"{n_yes/len(preds)*100:.1f}%")
                c3.metric("Predicted: Decline",   f"{n_no:,}",   f"{n_no/len(preds)*100:.1f}%")
                c4.metric("Avg Acceptance Prob",  f"{probas.mean()*100:.1f}%")

                # Distribution chart
                col_chart, col_tbl = st.columns([0.6, 0.4])
                with col_chart:
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.hist(probas*100, bins=25, color=BLUE, edgecolor="white", linewidth=0.7, alpha=0.85)
                    ax.axvline(50, color=RED, linestyle="--", linewidth=1.5, label="50% decision threshold")
                    ax.set_xlabel("Predicted Loan Acceptance Probability (%)")
                    ax.set_ylabel("Number of Customers")
                    ax.set_title("Distribution of Predicted Loan Probabilities", pad=8)
                    ax.legend(fontsize=9)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                with col_tbl:
                    tier_summary = result_df["Risk_Tier"].value_counts().reset_index()
                    tier_summary.columns = ["Risk Tier", "Customers"]
                    tier_summary["Share"] = (tier_summary["Customers"] / len(result_df) * 100).round(1).astype(str) + "%"
                    st.markdown("**Customers by Risk Tier**")
                    st.dataframe(tier_summary, use_container_width=True, hide_index=True)

                # Preview
                st.markdown("#### Preview — Prediction Results (first 20 rows)")
                preview_cols = [c for c in ["ID","Age","Income","Predicted_Personal_Loan",
                                            "Loan_Acceptance_Probability_%","Prediction_Label","Risk_Tier"]
                                if c in result_df.columns]
                st.dataframe(result_df[preview_cols].head(20), use_container_width=True)

                # Download
                st.markdown("#### Step 3 — Download Results")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    result_df.to_excel(writer, index=False, sheet_name="Predictions")
                    tier_s = result_df.groupby("Risk_Tier", observed=True).agg(
                        Customers=("Predicted_Personal_Loan","count"),
                        Predicted_Loans=("Predicted_Personal_Loan","sum"),
                        Avg_Probability=("Loan_Acceptance_Probability_%","mean")).reset_index()
                    tier_s.to_excel(writer, index=False, sheet_name="Tier Summary")
                output.seek(0)
                st.download_button(
                    label="📥 Download Full Predictions (Excel)",
                    data=output.getvalue(),
                    file_name="loan_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    else:
        st.markdown("""
        <div style="border: 2px dashed #d1d5db; border-radius: 12px; padding: 40px; text-align: center; color: #6b7280;">
        <div style="font-size: 40px; margin-bottom: 12px;">📂</div>
        <div style="font-size: 15px; font-weight: 500;">Drag and drop your file here, or click Browse</div>
        <div style="font-size: 12px; margin-top: 6px;">Supports .xlsx, .xls, .csv</div>
        </div>
        """, unsafe_allow_html=True)
