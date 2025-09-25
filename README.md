# Sonar-Rock-Mine-Classifier 📡

## Project Overview

This project implements a machine learning solution to classify underwater **sonar returns** as either a **Rock (R)** or an **Explosive Mine (M)**. The core of the system is a highly optimized **Support Vector Machine (SVM)** model, deployed via a high-fidelity, command-center-style web application built using **Streamlit**.

The application prioritizes **safety** by maximizing the detection of the critical "Mine" class.

---

## 🚀 The Sonar Classification Application

The Streamlit application simulates a submarine's sonar analysis console. Users can select pre-loaded sonar data and instantly see the model's classification, confidence level, threat assessment, and corresponding tactical recommendations.

### Application Screenshots

| Selecting a Mine Target |
| :---: |
|<img width="1817" height="967" alt="s1" src="https://github.com/user-attachments/assets/2768cdcb-d008-4a56-8689-a197f82f2c67" />
| Selecting a Rock Target |
 |<img width="1782" height="817" alt="s2" src="https://github.com/user-attachments/assets/ab6cc8e2-08c5-477d-8163-8a58b769a4e5" />
 | Handling an Unknown Target |
|<img width="1773" height="969" alt="s3" src="https://github.com/user-attachments/assets/f7bae01a-01b1-43e9-b64c-17ac0a416928" />|

---

## 🧠 Machine Learning Model Details

The model was trained on the **Sonar, Mines vs. Rocks** dataset, which consists of 60 normalized energy features from sonar signals.

### ✅ Final Model Selection and Conclusion

The classification task is highly sensitive, where missing a Mine (False Negative) is much more costly than falsely identifying a Rock as a Mine (False Positive). The **Support Vector Machine (SVM)** was chosen for its superior performance, particularly its perfect recall for the critical Mine class.

#### 📊 Cross-Validation Comparison

The Cross-Validation (CV) scores provide the most reliable estimate of real-world performance, indicating the **Tuned SVM** as the clear winner:

| **Model** | **Best CV Accuracy** |
| :--- | :--- |
| **Tuned SVM** | **87.13%** |
| Tuned Random Forest | 82.98% |
| Tuned Logistic Regression | 77.08% |

#### ⚖️ Tuned SVM Evaluation Summary

| **Metric** | **Score** | **Insight** |
| :--- | :--- | :--- |
| **Best Reliable Accuracy (CV)** | **87.13%** | The most trustworthy estimate of the model's performance on unseen data. |
| **Test Accuracy** | **85.71%** | Confirms strong generalization on the small held-out test set. |
| **Mine (M) Recall** | **1.00 (100%)** | **No false negatives** — all actual mines were correctly identified. **Critical for safety-sensitive tasks.** |
| **Mine (M) Precision** | **0.79** | Of all predicted "Mine" instances, 79% were correct. (Acceptable trade-off for 100% Recall). |

> **Conclusion:** This model is both **accurate and safe**, making it the best choice for reliable classification in sonar-based mine detection.

---

## 🛠️ Project Structure and Technology Stack

### Repository Contents

| File/Folder | Description |
| :--- | :--- |
| `sonar_app.py` | The main Streamlit application providing the command-center UI and prediction logic. |
| `sonar_svm_model.pkl` | The trained **Tuned SVM** classification model. |
| `sonar_scaler.pkl` | The **StandardScaler** object used to preprocess the data (required for prediction). |
| `sonar_dataset_notebook.ipynb` | Jupyter Notebook detailing EDA, data preprocessing, model tuning, and final evaluation. |
| `requirements.txt` | Lists the necessary Python packages  |
| `.gitignore` | Standard file to ignore virtual environment|

### Technologies

* **Deployment:** Streamlit
* **Model:** Scikit-learn (Support Vector Machine)
* **Language:** Python
* **Styling:** Custom CSS (Injected via Streamlit Markdown)

---

## ⚙️ Setup and Run Locally

To run the Sonar Classification App on your local machine, follow these steps:

1.  **Clone the Repository:**
   

2.  **Create and Activate Environment:**
   

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run sonar_app.py
    ```
The application will open automatically in your browser.
