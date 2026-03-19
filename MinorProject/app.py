
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("MinorProject/loan_approval_data.csv")

# ---------------------------
# Basic Cleaning
# ---------------------------
df = df.drop("Applicant_ID", axis=1)
df["Education_Level"] = df["Education_Level"].astype(str).str.strip()

# ---------------------------
# Fix Target
# ---------------------------
df["Loan_Approved"] = df["Loan_Approved"].replace({
    "Yes": 1, "No": 0, "Y": 1, "N": 0
})
df["Loan_Approved"] = pd.to_numeric(df["Loan_Approved"], errors="coerce")
df = df.dropna(subset=["Loan_Approved"])
df["Loan_Approved"] = df["Loan_Approved"].astype(int)

# ---------------------------
# Missing Values
# ---------------------------
num_cols = df.select_dtypes(include=["float", "int"]).columns
df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])

cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# ---------------------------
# Encoding
# ---------------------------
education_map = {
    "Undergraduate": 0,
    "Graduate": 1,
    "Postgraduate": 2
}
df["Education_Level"] = df["Education_Level"].map(education_map).fillna(0)

cols = ["Employment_Status","Marital_Status","Loan_Purpose",
        "Property_Area","Gender","Employer_Category"]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)

df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

# ---------------------------
# Feature Engineering
# ---------------------------
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

df = df.drop(columns=["Credit_Score", "DTI_Ratio", "Applicant_Income"])
df = df.fillna(0)

# ---------------------------
# Train Model
# ---------------------------
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# ---------------------------
# UI
# ---------------------------
st.title("💼 CreditWise Loan Approval Dashboard")

# ---------------------------
# Dataset Graphs
# ---------------------------
if st.checkbox("📊 Show Dataset Insights"):

    st.subheader("Loan Approval Distribution")
    fig1, ax1 = plt.subplots()
    df["Loan_Approved"].value_counts().plot.pie(
        autopct="%1.1f%%",
        labels=["Rejected","Approved"],
        ax=ax1
    )
    ax1.set_ylabel("")
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ---------------------------
# USER INPUT
# ---------------------------
st.subheader("Enter Applicant Details")

Applicant_Income = st.number_input("Applicant Income", 0)
Coapplicant_Income = st.number_input("Coapplicant Income", 0)
Credit_Score = st.number_input("Credit Score", 300, 900)
DTI_Ratio = st.number_input("DTI Ratio", 0.0)
Savings = st.number_input("Savings", 0.0)
Loan_Amount = st.number_input("Loan Amount", 0.0)

Education_Level = st.selectbox("Education Level",
    ["Undergraduate","Graduate","Postgraduate"])

Employment_Status = st.selectbox("Employment Status",
    ["Salaried","Self-Employed","Business"])

Marital_Status = st.selectbox("Marital Status",
    ["Married","Single"])

Loan_Purpose = st.selectbox("Loan Purpose",
    ["Home","Education","Personal","Business"])

Property_Area = st.selectbox("Property Area",
    ["Urban","Semi-Urban","Rural"])

Gender = st.selectbox("Gender", ["Male","Female"])
Employer_Category = st.selectbox("Employer Category", ["Govt","Private","Self"])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Credit_Score": Credit_Score,
        "DTI_Ratio": DTI_Ratio,
        "Savings": Savings,
        "Loan_Amount": Loan_Amount,
        "Education_Level": Education_Level,
        "Employment_Status": Employment_Status,
        "Marital_Status": Marital_Status,
        "Loan_Purpose": Loan_Purpose,
        "Property_Area": Property_Area,
        "Gender": Gender,
        "Employer_Category": Employer_Category
    }])

    # Encoding
    input_data["Education_Level"] = input_data["Education_Level"].map(education_map).fillna(0)

    encoded_input = ohe.transform(input_data[cols])
    encoded_input_df = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out(cols))

    input_data = pd.concat([input_data.drop(columns=cols), encoded_input_df], axis=1)

    # Feature Engineering
    input_data["DTI_Ratio_sq"] = input_data["DTI_Ratio"] ** 2
    input_data["Credit_Score_sq"] = input_data["Credit_Score"] ** 2
    input_data["Applicant_Income_log"] = np.log1p(input_data["Applicant_Income"])

    input_data = input_data.drop(columns=["Credit_Score","DTI_Ratio","Applicant_Income"])

    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    # ---------------------------
    # OUTPUT
    # ---------------------------
    if prediction == 1:
        st.success(f"✅ Loan Approved ({prob[1]:.2f})")
    else:
        st.error(f"❌ Loan Rejected ({prob[1]:.2f})")

    # ---------------------------
    # User vs Dataset Graph
    # ---------------------------
    st.subheader("📊 Your Income vs Dataset")

    fig3, ax3 = plt.subplots()
    sns.histplot(df["Applicant_Income_log"], bins=20, ax=ax3)

    ax3.axvline(np.log1p(Applicant_Income), linestyle="--")
    st.pyplot(fig3)

    # ---------------------------
    # Probability Graph
    # ---------------------------
    st.subheader("📈 Prediction Probability")

    fig4, ax4 = plt.subplots()
    ax4.bar(["Rejected","Approved"], prob)
    st.pyplot(fig4)
