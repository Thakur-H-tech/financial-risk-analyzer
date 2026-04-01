import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.ensemble import IsolationForest

# Load models
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
nlp_model = joblib.load("nlp_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Financial Risk Analyzer", layout="wide")

st.title("💳 Personal Financial Risk Analyzer")

# Upload
uploaded_file = st.file_uploader("Upload Bank Statement", type=["csv", "xlsx"])

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# NLP categorization
def predict_category(desc):
    desc = clean_text(desc)
    vec = vectorizer.transform([desc])
    return nlp_model.predict(vec)[0]

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # Categorize
    df["Category"] = df["Description"].apply(predict_category)

    st.subheader("📊 Categorized Data")
    st.dataframe(df)

    # Financial features
    income = df[df["Amount"] > 0]["Amount"].sum()
    expenses = abs(df[df["Amount"] < 0]["Amount"].sum())
    savings = income - expenses
    debt = 10000

    savings_ratio = savings / income if income != 0 else 0
    expense_ratio = expenses / income if income != 0 else 0
    debt_income_ratio = debt / income if income != 0 else 0

    features = np.array([[income, expenses, savings, debt,
                          savings_ratio, expense_ratio, debt_income_ratio]])

    features = scaler.transform(features)

    # Prediction
    prediction = model.predict_proba(features)[0][1]
    risk_score = int(prediction * 100)

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Income", f"₹{int(income)}")
    col2.metric("💸 Expenses", f"₹{int(expenses)}")
    col3.metric("💾 Savings", f"₹{int(savings)}")

    # Risk
    st.subheader("📊 Risk Score")
    st.progress(risk_score)

    if risk_score > 70:
        st.error("🔴 High Risk")
    elif risk_score > 40:
        st.warning("🟠 Moderate Risk")
    else:
        st.success("🟢 Stable")

    # Anomaly Detection
    iso = IsolationForest(contamination=0.05)
    df["Anomaly"] = iso.fit_predict(df[["Amount"]])

    st.subheader("🚨 Unusual Transactions")
    st.dataframe(df[df["Anomaly"] == -1])

    # Trends
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month

    st.subheader("📈 Monthly Trend")
    st.line_chart(df.groupby("Month")["Amount"].sum())

    # Category spending
    st.subheader("📊 Spending by Category")
    st.bar_chart(df[df["Amount"] < 0].groupby("Category")["Amount"].sum().abs())

    # Suggestions
    st.subheader("💡 Recommendations")

    if risk_score > 70:
        st.info("Reduce non-essential spending and focus on debt reduction.")
    elif risk_score > 40:
        st.info("Track spending and increase savings.")
    else:
        st.info("You are financially stable. Consider investing.")

    # Insights
    st.subheader("📌 Insights")
    st.write(f"Savings Ratio: {savings_ratio:.2f}")
    st.write(f"Expense Ratio: {expense_ratio:.2f}")
    st.write(f"Debt Ratio: {debt_income_ratio:.2f}")

else:
    st.info("👆 Upload a file to begin analysis")