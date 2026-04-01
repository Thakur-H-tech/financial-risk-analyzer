import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.ensemble import IsolationForest

# PDF + OCR
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# Load models
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
nlp_model = joblib.load("nlp_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Financial Risk Analyzer", layout="wide")
st.title("💳 Personal Financial Risk Analyzer")

# Upload
uploaded_file = st.file_uploader("Upload CSV / Excel / PDF", type=["csv", "xlsx", "pdf"])

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- NLP ----------------
def predict_category(desc):
    desc = clean_text(desc)
    vec = vectorizer.transform([desc])
    return nlp_model.predict(vec)[0]

# ---------------- PDF HANDLING ----------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def extract_text_from_image_pdf(file):
    images = convert_from_bytes(file.read())
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text

def extract_text(file):
    try:
        text = extract_text_from_pdf(file)
        if text.strip():
            return text
    except:
        pass
    return extract_text_from_image_pdf(file)

# ---------------- TEXT → DF ----------------
def text_to_df(text):
    lines = text.split("\n")
    data = []

    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            date = parts[0]
            amount = parts[-1]
            desc = " ".join(parts[1:-1])

            try:
                amount = float(amount.replace(",", ""))
                data.append([date, desc, amount])
            except:
                continue

    return pd.DataFrame(data, columns=["Date", "Description", "Amount"])

# ---------------- MAIN ----------------
if uploaded_file:

    # -------- FILE HANDLING --------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    elif uploaded_file.name.endswith(".pdf"):
        text = extract_text(uploaded_file)
        df = text_to_df(text)

    # -------- STANDARDIZE COLUMNS --------
    df.columns = df.columns.str.strip().str.lower()

    # Description mapping
    if "transaction details" in df.columns:
        df["Description"] = df["transaction details"]
    elif "description" in df.columns:
        df["Description"] = df["description"]
    else:
        st.error("❌ No description column found")
        st.stop()

    # Amount mapping
    if "amount" in df.columns:
        df["Amount"] = df["amount"]
    else:
        st.error("❌ No amount column found")
        st.stop()

    # Date mapping
    if "date" in df.columns:
        df["Date"] = df["date"]

    # -------- CLEAN AMOUNT --------
    df["Amount"] = df["Amount"].astype(str)
    df["Amount"] = df["Amount"].str.replace(",", "").str.replace("₹", "")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # -------- FIX CREDIT / DEBIT --------
    df["Amount"] = df.apply(
        lambda x: -abs(x["Amount"]) if any(word in str(x["Description"]).lower()
        for word in ["paid", "sent", "debit", "dr"]) else abs(x["Amount"]),
        axis=1
    )

    # -------- USE TAGS (PAYTM BONUS) --------
    if "tags" in df.columns:
        df["Description"] = df["Description"] + " " + df["tags"].astype(str)

    st.subheader("📄 Processed Data")
    st.dataframe(df)

    # -------- NLP CATEGORY --------
    df["Category"] = df["Description"].apply(predict_category)

    st.subheader("📊 Categorized Data")
    st.dataframe(df)

    # -------- FEATURE ENGINEERING --------
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

    # -------- PREDICTION --------
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

    # -------- ANOMALY DETECTION --------
    iso = IsolationForest(contamination=0.05)
    df["Anomaly"] = iso.fit_predict(df[["Amount"]])

    st.subheader("🚨 Unusual Transactions")
    st.dataframe(df[df["Anomaly"] == -1])

    # -------- TRENDS --------
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Month"] = df["Date"].dt.month

        st.subheader("📈 Monthly Trend")
        st.line_chart(df.groupby("Month")["Amount"].sum())
    except:
        st.warning("Date format issue")

    # -------- CATEGORY GRAPH --------
    st.subheader("📊 Spending by Category")
    st.bar_chart(df[df["Amount"] < 0].groupby("Category")["Amount"].sum().abs())

    # -------- RECOMMENDATIONS --------
    st.subheader("💡 Recommendations")

    if risk_score > 70:
        st.info("Reduce unnecessary spending and focus on debt repayment.")
    elif risk_score > 40:
        st.info("Track expenses and increase savings.")
    else:
        st.info("You are financially stable. Consider investing.")

    # -------- INSIGHTS --------
    st.subheader("📌 Insights")
    st.write(f"Savings Ratio: {savings_ratio:.2f}")
    st.write(f"Expense Ratio: {expense_ratio:.2f}")
    st.write(f"Debt Ratio: {debt_income_ratio:.2f}")

else:
    st.info("👆 Upload CSV, Excel, or PDF to begin")
