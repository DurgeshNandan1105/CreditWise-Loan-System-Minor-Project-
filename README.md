# 💼 CreditWise Loan Approval System

🚀 **End-to-End Machine Learning Project with Live Deployment**

---

## 🌐 Live Demo

👉 **Try the App Here:**
🔗 https://creditwise-loan-system.streamlit.app/

> Interactive web app for real-time loan approval prediction.

---

## 📌 Project Overview

The **CreditWise Loan Approval System** is a Machine Learning application that predicts whether a loan application should be:

* ✅ **Approved (1)**
* ❌ **Rejected (0)**

It replaces traditional manual decision-making with a **fast, consistent, and data-driven approach**.

---

## ⚙️ End-to-End Workflow

* Data Cleaning & Preprocessing (Handled missing values using SimpleImputer)
* Exploratory Data Analysis (EDA with visualizations)
* Feature Engineering (DTI², Credit Score², Log Income)
* Encoding (Label Encoding + One-Hot Encoding)
* Feature Scaling (StandardScaler)
* Model Training & Evaluation
* Deployment using Streamlit

---

## 🤖 Machine Learning Models

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.82     | 0.80      | 0.84   | 0.82     |
| KNN                 | 0.78     | 0.76      | 0.79   | 0.77     |
| Naive Bayes         | 0.75     | 0.73      | 0.76   | 0.74     |

✅ **Best Model:** Logistic Regression (used in deployment)

---

## 📊 Model Performance

### Confusion Matrix

```
[[45 10]
 [ 8 57]]
```

* ✔ True Positives → Correct approvals
* ✔ True Negatives → Correct rejections
* ⚠ False Positives → Risky approvals
* ⚠ False Negatives → Missed good customers

---

## 📊 Key Insights

* Credit Score has a strong impact on loan approval
* Higher Debt-to-Income (DTI) ratio reduces approval probability
* Feature engineering significantly improved model performance
* Logistic Regression provided the best balance of precision and recall

---

## 🖥️ Streamlit App Features

* Interactive user input form
* Real-time loan prediction
* Prediction probability visualization
* Dataset insights & charts
* User input vs dataset comparison

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Deployment:** Streamlit

---

## 📂 Project Structure

```
MinorProject/
│── app.py
│── credit_wise.ipynb
│── loan_approval_data.csv
│── requirements.txt
│── README.md
```

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/your-username/your-repo-name.git
cd MinorProject
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍💻 Author

**Durgesh Nandan**

---

## ⭐ Why This Project Stands Out

- ✔ End-to-end ML pipeline
- ✔ Real-world FinTech use case
- ✔ Strong feature engineering
- ✔ Model comparison & evaluation
- ✔ Interactive deployed application
- ✔ Clean and professional project structure

---

## 📌 Future Improvements

* Add more advanced models (Random Forest, XGBoost)
* Improve UI/UX design
* Add model explainability (SHAP)
* Deploy with CI/CD pipeline

