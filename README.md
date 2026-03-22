🚀 Customer Churn Prediction System
📌 Overview

Built an end-to-end machine learning system to predict customer churn using real-world telecom data.
The project covers data preprocessing → model training → evaluation → deployment via API, ensuring production-level reliability.

⚙️ Key Features
✅ Data preprocessing & feature engineering pipeline
✅ Multiple ML models:
Logistic Regression
Random Forest
Gradient Boosting
XGBoost
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

📊 Model Performance
Model	Accuracy	F1 Score	ROC-AUC
Logistic Regression	~0.76	~0.59	~0.81
Random Forest	~0.83	~0.67	~0.89
Gradient Boosting	~0.85	~0.71	~0.92
XGBoost	~0.84	~0.71	~0.91

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
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── api/
│   └── app.py
│
├── models/
│   ├── model.pkl
│   └── columns.pkl
│
├── main.py
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

GitHub: https://github.com/Karthik-Goud2406