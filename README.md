# Glass Type Identification Using Machine Learning

When broken glass becomes a clue at a crime scene, identifying its type can turn that tiny shard into a powerful piece of evidence. This project uses machine learning to predict the type of glass based on its chemical composition—helping forensic analysts connect the dots faster and smarter.

---

## What This Project Does

This project builds an intelligent system to **classify glass fragments** into different types (e.g., window, tableware, containers) based on their chemical makeup. The ultimate goal: **assist forensic investigations** by providing fast, accurate predictions about the origin of glass samples.

---

## 📦 Dataset Details

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/glass+identification)

---

## 🎯 Project Goals

- Explore the data (EDA): uncover patterns, check class distributions, and identify outliers
- Build and compare several machine learning models
- Reduce complexity with **PCA** for easy visualization
- Rank features using **RFE** to understand what drives predictions
- Fine-tune models for higher accuracy
- Create a user-friendly web app with **Streamlit** for live predictions

---

## 🛠 Models Used

| Model               | Why It Was Chosen                         |
|--------------------|--------------------------------------------|
| K-Nearest Neighbors| Simple, interpretable, effective for small data |
| Decision Tree      | Fast and easy to visualize decision logic |
| Random Forest      | Great for handling feature importance      |
| Gradient Boosting  | Boosts performance by reducing error       |
| XGBoost            | State-of-the-art performance in tabular data |

Each model was evaluated using:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Cross-validation (for reliability)

---


## 🌐 App Deployment (Streamlit)

We turned our best-performing model into a **real-time prediction app** using Streamlit.

### How to Run It:

```bash
streamlit run app.py


glass_type_classification/
│
├── app.py                # Streamlit web app
├── model.pkl             # Saved ML model
├── scaler.pkl            # StandardScaler for input preprocessing
├── encoder.pkl           # Label encoder (if used)
├── pca.pkl               # PCA model for dimensionality reduction
├── notebook.ipynb        # Full development notebook
├── README.md             # This file

## ⚙️ Tools & Libraries

Python

Scikit-learn

XGBoost

Matplotlib & Seaborn (visuals)

Streamlit (deployment)
