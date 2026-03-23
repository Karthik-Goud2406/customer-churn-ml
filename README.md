🚀 Customer Churn Prediction System
📌 Overview

Built an end-to-end machine learning system to predict customer churn using real-world telecom data.
The project covers data preprocessing → model training → evaluation → deployment via API, ensuring production-level reliability.

## 📌 Problem Statement

Customer churn is a critical business problem where companies lose customers over time, directly impacting revenue.

The challenge is:
- Identify customers likely to churn **before they leave**
- Build a system that can **predict churn in real-time**
- Ensure the system works reliably in **production environments**

## 🎯 Objective

Design and deploy a **production-ready machine learning system** that:
- Predicts customer churn accurately
- Handles real-world data challenges (leakage, inconsistencies)
- Provides predictions via a scalable API

---

## 🧠 Engineering Challenges Solved

### 🔴 1. Data Leakage (Critical ML Issue)
- Initial model showed unrealistic performance (~96% accuracy)
- Identified leakage features like:
  - `satisfaction_score`
  - `churn_score`
- Removed them to ensure **real-world generalization**

✅ Result: Reliable model performance (~0.91 ROC-AUC)

---

### 🔴 2. Training vs Inference Mismatch
- Model trained on encoded features (~100+ columns)
- API received raw input (few fields)
- Caused prediction failures (500 errors)

✅ Solution:
- Saved training feature schema (`columns.pkl`)
- Aligned input using:
  python
df = df.reindex(columns=columns, fill_value=0)

⚙️ Key Features
✅ Data preprocessing & feature engineering pipeline
✅ Multiple ML models:
Logistic Regression
Random Forest
Gradient Boosting
XGBoost

📊 Model Performance
Model	              Accuracy	F1 Score	ROC-AUC
Logistic Regression 	0.76    	0.59	   0.81
Random Forest       	0.83  	  0.67	   0.89
Gradient Boosting	    0.85	    0.71	   0.92
XGBoost	              0.84	    0.71	   0.91

✅ Model evaluation using:
Accuracy
F1 Score
ROC-AUC
✅ Feature importance analysis
✅ FastAPI deployment for real-time predictions
✅ Robust inference pipeline with schema alignment
🧠 Key Challenges & Solutions
🔴 1. Data Leakage
Identified leakage features like satisfaction_score, churn_score
Removed them to improve model generalization
🔴 2. Training vs Inference Mismatch
Model failed initially due to different input feature space
Fixed by:
Saving training columns (columns.pkl)
Aligning API input using reindex()

🔴 3. Deployment Errors
Handled serialization and API crashes
Built stable prediction pipeline with error handling
🛠️ Tech Stack
Languages: Python
Libraries: Pandas, NumPy, Scikit-learn
Models: XGBoost, Gradient Boosting
Deployment: FastAPI
Serialization: Joblib



👉 Final Model Used: Gradient Boosting (best balance of performance)

🚀 How to Run Locally
1. Clone the repository
git clone https://github.com/Karthik-Goud2406/customer-churn-ml.git
cd customer-churn-ml
2. Install dependencies
pip install -r requirements.txt
3. Train the model
python main.py
4. Run API
uvicorn api.app:app --reload
5. Open API Docs
http://127.0.0.1:8000/docs
🔗 API Usage
Endpoint:
POST /predict
Example Request:
{
  "tenure_in_months": 12,
  "monthly_charge": 50
}
Example Response:
{
  "churn": 0
}

📁 Project Structure
customer-churn-ml/
│
├── src/
│   ├── data_preprocessing.py   # Data cleaning & encoding
│   ├── feature_engineering.py  # Feature creation
│   ├── train.py                # Model training & saving
│   └── evaluate.py             # Metrics evaluation
│
├── api/
│   └── app.py                  # FastAPI service
│
├── models/
│   ├── model.pkl               # Trained model
│   └── columns.pkl             # Feature schema
│
├── main.py                     # Pipeline entry point
├── requirements.txt
└── README.md


💡 What Makes This Project Strong
✔ End-to-end ML pipeline (not just notebook)
✔ Real-world problem handling (data leakage, inference mismatch)
✔ Deployment-ready API
✔ Clean, modular code structure
✔ Industry-relevant tools (FastAPI, XGBoost)

🎯 Future Improvements
Add Streamlit dashboard for visualization
Deploy on AWS / Render for public access
Integrate MLflow for experiment tracking
Add batch prediction support


👨‍💻 Author
Karthik Goud
![WhatsApp Image 2026-03-23 at 12 47 52 AM](https://github.com/user-attachments/assets/59643039-8d3e-456d-81e7-312d3810b998)
![WhatsApp Image 2026-03-23 at 12 48 10 AM](https://github.com/user-attachments/assets/1c3d05f4-9b6c-495c-95a7-6da1b6177da3)
![WhatsApp Image 2026-03-23 at 12 48 37 AM](https://github.com/user-attachments/assets/56c3931d-cbd8-4a3e-8566-3cb021060944)



GitHub: https://github.com/Karthik-Goud2406
