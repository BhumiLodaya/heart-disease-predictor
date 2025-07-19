# ❤️ Heart Disease Prediction using Machine Learning

This project builds a **Random Forest Classifier** to predict the presence of heart disease based on various medical attributes. It uses the popular `heart.csv` dataset and evaluates the model's performance using standard metrics like accuracy, confusion matrix, and classification report.

## 📂 Project Files

├── disease.py # Main Python script for training and evaluating the model
├── heart.csv # Dataset containing patient health records
├── README.md # Project documentation (this file)

---

## 🧠 Machine Learning Pipeline

### 1. Dataset Loading
The dataset is loaded from `heart.csv`, containing health features like:
- Age, sex, cholesterol, chest pain type, fasting blood sugar, resting ECG, max heart rate, etc.
- The `target` column indicates heart disease presence:  
  - `1` = disease present  
  - `0` = no disease

### 2. Preprocessing
- Features (`X`) and target (`y`) are split.
- The data is split into **training (80%)** and **testing (20%)** sets.

### 3. Model
- **Algorithm**: `RandomForestClassifier` from `scikit-learn`
- The model is trained on the training set.
- Predictions are made on the test set.

### 4. Evaluation
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

---

## 📊 Sample Output
✅ Dataset loaded and split successfully!
Training set shape: (242, 13)
Testing set shape: (61, 13)
✅ Model training completed!

✅ Accuracy of the model: 0.87

📊 Confusion Matrix:
[[24 5]
[ 2 30]]

📋 Classification Report:
precision recall f1-score support
0 0.92 0.83 0.87 29
1 0.86 0.94 0.90 32
accuracy 0.88 61

yaml
Copy
Edit

---

## 📦 Dependencies

Install required libraries using:

bash
pip install pandas scikit-learn
🚀 How to Run
bash
Copy
Edit
python disease.py
Make sure the heart.csv file is in the same directory as disease.py.

👩‍💻 Author
Bhumi Lodaya
📌 Skilled in Python, ML, and Health Analytics

📌 License
This project is for educational and research use only.





