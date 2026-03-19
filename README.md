# 💼 CreditWise Loan Approval System

🚀 *End-to-End Machine Learning Project for Predicting Loan Approval*

---

## 📌 Overview

This project builds an **Intelligent Loan Approval System** using Machine Learning to predict whether a loan application should be **Approved (1)** or **Rejected (0)**.

The system is designed to replace manual decision-making with a **data-driven, fast, and unbiased approach**.

---

## 🎯 Problem Statement

Traditional loan approval systems rely on manual verification, which leads to:

* ❌ Slow processing time
* ❌ Human bias and inconsistency
* ❌ Incorrect approvals/rejections

This project solves these issues by using historical applicant data to automate decision-making.

---

## ⚙️ Project Workflow

### 🔹 1. Data Loading

* Dataset loaded using **Pandas**
* Initial inspection using `.head()`, `.info()`, `.describe()`

---

### 🔹 2. Data Preprocessing

* Handling missing values using **SimpleImputer**

  * Numerical → Mean imputation
* Dropped irrelevant column:

  * `Applicant_ID`

---

### 🔹 3. Exploratory Data Analysis (EDA)

* Class distribution (Loan Approved vs Rejected)
* Gender distribution analysis
* Income distribution (Applicant & Co-applicant)
* Outlier detection using **Boxplots**
* Feature relationships using:

  * Histograms
  * Correlation Heatmap

---

### 🔹 4. Feature Engineering

* Created new features:

  * `DTI_Ratio_sq`
  * `Credit_Score_sq`
  * `Applicant_Income_log`
* Removed redundant features to improve model performance

---

### 🔹 5. Encoding

* **Label Encoding**

  * `Education_Level`, `Loan_Approved`
* **One-Hot Encoding**

  * Employment Status
  * Marital Status
  * Loan Purpose
  * Property Area
  * Gender
  * Employer Category

---

### 🔹 6. Feature Scaling

* Applied **StandardScaler** to normalize data

---

### 🔹 7. Model Training

Trained multiple ML models:

#### ✅ Logistic Regression

#### ✅ K-Nearest Neighbors (KNN)

#### ✅ Naive Bayes

---

### 🔹 8. Model Evaluation

Performance measured using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📊 Key Insights

* Credit Score and DTI Ratio strongly influence loan approval
* Income distribution impacts approval probability
* Feature engineering improved model performance

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Seaborn, Matplotlib**
* **Scikit-learn**

---

## 📈 Results

* Built a complete ML pipeline from preprocessing to evaluation
* Compared multiple models for best performance
* Achieved reliable prediction performance for loan approval

---

## 👨‍💻 Author

**Durgesh Nandan**


