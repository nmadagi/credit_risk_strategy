# credit_risk_german.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import streamlit as st

# -------------------------------
# 1. LOAD GERMAN CREDIT DATA
# -------------------------------
# download and save german.data as data/german_credit.csv
# space-delimited file; no header
colnames = [
    "checking","duration","credit_history","purpose","credit_amount",
    "savings","employment","installment_rate","personal_status","other_debtors",
    "residence_since","property","age","other_installment_plans","housing",
    "num_existing_credits","job","dependents","telephone","foreign_worker","credit_rating"
]

df0 = pd.read_csv("data/german_credit.csv", sep=" ", header=None, names=colnames)

# -------------------------------
# 2. CLEAN & MAP TO “loan-like” schema
# -------------------------------
# Create “loan_status” from credit_rating: assume 1=good → Fully Paid, 2=bad → Charged Off
df0["loan_status"] = np.where(df0["credit_rating"] == 2, "Charged Off", "Fully Paid")

# Map some fields to approximate your schema
df0["loan_amnt"] = df0["credit_amount"]
df0["term"] = df0["duration"]
# approximate interest rate fixed
df0["int_rate"] = 12.0  
df0["installment"] = df0["credit_amount"] * (df0["int_rate"]/100) / df0["duration"]
# approximate annual income, dti, fico
df0["annual_inc"] = df0["credit_amount"] * 10
df0["dti"] = (df0["installment"]*12 / df0["annual_inc"]) * 100
df0["fico_range_high"] = np.clip(500 + df0["age"]*3, 580, 850)
df0["revol_util"] = 30.0
df0["purpose"] = df0["purpose"].astype(str)
df0["home_ownership"] = df0["housing"].astype(str)

# keep only needed columns
cols = [
    "loan_amnt","term","int_rate","installment","annual_inc","dti",
    "fico_range_high","revol_util","purpose","home_ownership","loan_status"
]
df = df0[cols].copy()
df["target"] = np.where(df["loan_status"] == "Charged Off", 1, 0)

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------
df["log_annual_inc"] = np.log(df["annual_inc"])
df["income_to_installment"] = df["annual_inc"] / (df["installment"] * 12)
df = pd.get_dummies(df, columns=["purpose","home_ownership"], drop_first=True)

# -------------------------------
# 4. SPLIT & SCALE
# -------------------------------
X = df.drop(["loan_status","target"], axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_cols = X_train.columns.tolist()

# -------------------------------
# 5. TRAIN MODELS
# -------------------------------
logit_model = LogisticRegression(max_iter=1000)
logit_model.fit(X_train_scaled, y_train)
logit_pred = logit_model.predict_proba(X_test_scaled)[:,1]
logit_auc = roc_auc_score(y_test, logit_pred)

ratio = (len(y_train) - y_train.sum()) / y_train.sum()
xgb_clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    scale_pos_weight=ratio, eval_metric="auc", random_state=42
)
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict_proba(X_test)[:,1]
xgb_auc = roc_auc_score(y_test, xgb_pred)

champion = "XGBoost" if xgb_auc >= logit_auc else "Logistic"

def predict_pd(df_input, model_choice="auto"):
    if model_choice == "auto":
        model_choice = champion
    if model_choice == "Logistic":
        return logit_model.predict_proba(scaler.transform(df_input[feature_cols]))[:,1]
    else:
        return xgb_clf.predict_proba(df_input[feature_cols])[:,1]

# -------------------------------
# 6. RISK STRATEGY & SUMMARY
# -------------------------------
X_test_df = X_test.copy()
X_test_df["PD"] = predict_pd(X_test_df, model_choice="auto")
X_test_df["loan_amnt"] = df.loc[X_test_df.index, "loan_amnt"]
X_test_df["LGD"] = 0.4
X_test_df["EAD"] = X_test_df["loan_amnt"]
X_test_df["EL"] = X_test_df["PD"] * X_test_df["LGD"] * X_test_df["EAD"]
bins = [0, 0.02, 0.05, 0.10, 1]
labels = ["AA","A","B","C"]
X_test_df["risk_band"] = pd.cut(X_test_df["PD"], bins=bins, labels=labels)
X_test_df["decision"] = np.where(
    X_test_df["risk_band"].isin(["AA","A"]), "Approve",
    np.where(X_test_df["risk_band"]=="B", "Review", "Decline")
)

summary = X_test_df.groupby("risk_band").agg(
    applicants=("PD","count"),
    avg_PD=("PD","mean"),
    avg_EL=("EL","mean"),
    total_EL=("EL","sum")
).reset_index()

# -------------------------------
# 7. STREAMLIT DASHBOARD
# -------------------------------
st.set_page_config(page_title="Credit Risk – German", layout="wide")
st.title("Credit Risk Strategy — German Credit Dataset")
st.markdown(f"Logistic AUC = {logit_auc:.3f} | XGBoost AUC = {xgb_auc:.3f} | Champion = {champion}")

st.sidebar.header("Applicant Input")
loan_amnt = st.sidebar.number_input("Loan Amount", min_value=100, value=1000)
int_rate  = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
annual_inc = st.sidebar.number_input("Annual Income", value=30000)
dti        = st.sidebar.slider("DTI (%)", 0.0, 100.0, 20.0)
fico       = st.sidebar.slider("FICO proxy", 580, 850, 700)
model_choice = st.sidebar.selectbox("Model for Scoring", ["auto","XGBoost","Logistic"])

sample = pd.DataFrame({
    "loan_amnt": [loan_amnt],
    "int_rate": [int_rate],
    "installment": [loan_amnt * (int_rate/100) / 12],
    "annual_inc": [annual_inc],
    "dti": [dti],
    "fico_range_high": [fico],
    "revol_util": [30.0],
    "term": [12],  # short term for demo
    "log_annual_inc": [np.log(annual_inc)],
    "income_to_installment": [annual_inc / ((loan_amnt*(int_rate/100)/12)*12)]
})
for c in feature_cols:
    if c not in sample.columns:
        sample[c] = 0
sample = sample[feature_cols]

pd_pred = predict_pd(sample, model_choice=model_choice)[0]
if pd_pred < 0.02:
    decision = "Approve (AA)"
elif pd_pred < 0.05:
    decision = "Approve (A)"
elif pd_pred < 0.10:
    decision = "Review (B)"
else:
    decision = "Decline (C)"

col1, col2, col3 = st.columns(3)
col1.metric("Predicted PD", f"{pd_pred*100:.2f}%")
col2.metric("Champion Model", champion)
col3.metric("Decision", decision)

st.markdown("---")
st.subheader("Portfolio Summary")
st.dataframe(summary)
st.bar_chart(summary.set_index("risk_band")["avg_PD"])
st.caption("Risk band policy: Approve AA/A, Review B, Decline C")
