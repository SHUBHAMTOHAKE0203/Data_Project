# 📊 Sales Forecasting System (Next 8 Weeks Prediction)

---

## 🚀 Overview

This project predicts the **next 8 weeks of sales for each state** using historical time-series data.
It is a complete end-to-end pipeline covering:

* Data preprocessing
* Feature engineering
* Model training & evaluation
* API deployment using FastAPI

---

## 🎯 Problem Statement

Forecast future sales for each state using past sales data while ensuring:

* No data leakage
* Proper time-series modeling
* Scalable prediction system

---

## 🧾 Dataset Description

The dataset contains the following columns:

| Column  | Description          |
| ------- | -------------------- |
| `date`  | Date of sales record |
| `state` | State name           |
| `sales` | Sales value          |

---

## ⚙️ Data Preprocessing

Steps performed:

* Converted `date` column to datetime format
* Sorted dataset by `state` and `date`
* Handled missing values
* Removed inconsistencies

---

## 🔥 Feature Engineering (Core Part)

### 📌 Time-Based Features

* Day of week
* Month

---

### 📌 Holiday Feature

* Binary flag for holidays using calendar data

---

### 📌 Lag Features

* `lag_1` → Previous day sales
* `lag_7` → Previous week sales
* `lag_30` → Previous month sales

---

### 📌 Rolling Statistics

* Rolling Mean (7 days)
* Rolling Standard Deviation (7 days)

👉 **Important:**
Shift (`t-1`) is applied before rolling to prevent **data leakage**.

---

## 🧠 Train-Test Split

* Used chronological split (time-series logic)
* No random splitting
* Future data is never used in training

---

## 🤖 Models Used

* **Prophet** → Handles seasonality & trends
* **XGBoost** → Works with engineered features
* **ARIMA** (optional)
* **LSTM** (optional)

---

## 📊 Model Evaluation

### Metric:

* **Mean Absolute Error (MAE)**

👉 Measures average prediction error

---

## 🏆 Model Selection

* Models trained per state
* Best model selected using lowest MAE
* Saved using `joblib`

---

## 🔮 Prediction System

### Input:

* State name

### Output:

* Best model used
* Next 8 weeks forecast

---

## 🚀 API Deployment

Built using **FastAPI**

### 🔗 Endpoint:

```
GET /predict/{state}
```

### 📌 Example:

```
/predict/Alabama
```

### 📌 Response:

```json
{
  "state": "Alabama",
  "model_used": "prophet",
  "forecast": [120, 130, 140, ...]
}
```

---

## 📁 Project Structure

```
project/
│── data/
│── models/
│── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│── api/
│   ├── app.py
│   ├── predict.py
│── requirements.txt
│── README.md
```

---

## ⚙️ How to Run the Project

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 2️⃣ Train models

```
python -m src.train
```

---

### 3️⃣ Run API

```
uvicorn api.app:app --reload
```

---

### 4️⃣ Test API

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## ⚠️ Challenges & Solutions

| Challenge            | Solution                  |
| -------------------- | ------------------------- |
| Data leakage         | Used shift before rolling |
| Time ordering        | Sorted dataset            |
| Multi-state handling | Used groupby              |
| Model selection      | Compared MAE              |

---

## 🔮 Future Improvements

* Real-time data integration
* Cloud deployment (AWS / Azure)
* Dashboard visualization
* Ensemble models for better accuracy

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* Prophet
* XGBoost
* FastAPI

---

## 📌 Conclusion

This project demonstrates a **complete real-world time-series forecasting pipeline** with proper ML practices and deployment.

---

## ⭐ Author

**Shubham Tohake**

---

## 💡 Final Note

This solution is scalable and can be extended to real-world applications like:

* Retail demand forecasting
* Supply chain optimization
* Sales analytics systems
