import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📊 Load dataset
df = pd.read_csv("data/student_performance_final.csv")

# 🔍 Basic check
print("Dataset Shape:", df.shape)

# 🎯 Features & Target
X = df.drop(["pass", "grade", "final_score"], axis=1)
y = df["pass"]

# 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🤖 Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# 🎯 Train
model.fit(X_train, y_train)

# 📈 Predictions
y_pred = model.predict(X_test)

# 📊 Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 💾 Save Model
joblib.dump(model, "models/student_model.pkl")

print("\n✅ Model saved successfully!")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save image
plt.savefig("outputs/confusion_matrix.png")

# Show (optional)
plt.show()

print("✅ Confusion matrix saved in outputs/")