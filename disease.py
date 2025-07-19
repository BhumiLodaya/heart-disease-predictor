import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("C:/Users/bhumi/OneDrive/Desktop/professional/internship/ml_heart/heart.csv")

# 2. Feature-Target split
X = df.drop('target', axis=1)
y = df['target']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)

print("✅ Dataset loaded and split successfully!")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("✅ Model training completed!")

print(f"\n✅ Accuracy of the model: {accuracy_score(y_test, y_pred):.2f}")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
