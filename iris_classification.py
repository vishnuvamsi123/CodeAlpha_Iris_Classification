# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

print("--- Step 1: Loading the Dataset ---")
# load the classic Iris dataset from scikit-learn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species_id'] = iris.target
df['species'] = df['species_id'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset successfully loaded.")
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Summary Statistics:")
print(df.describe())

print("\n--- Step 2: Exploratory Data Analysis (EDA) ---")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8))
pairplot = sns.pairplot(df.drop(columns=['species_id']), hue='species', palette='muted')
pairplot_path = os.path.join(output_dir, "iris_pairplot.png")
pairplot.savefig(pairplot_path)
plt.close()
print(f"EDA Pairplot saved to {pairplot_path}")
print("\n--- Step 3: Data Preprocessing ---")

X = df.drop(columns=['species_id', 'species'])
# Target (y): Species ID (0, 1, or 2)
y = df['species_id']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Data split completed: Training size = {X_train.shape[0]}, Testing size = {X_test.shape[0]}")

print("\n--- Step 4: Model Training ---")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Random Forest Classifier model trained successfully!")

print("\n--- Step 5: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix - Iris Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
plt.close()
print(f"Confusion Matrix heatmap saved to {confusion_matrix_path}")

print("\n--- Step 6: Saving the Trained Model and Scaler ---")
# Serialize and save the trained model and scale object to files using joblib
# This allows us to load the model later for production/deployment without retraining it
model_path = os.path.join(output_dir, "iris_model.pkl")
scaler_path = os.path.join(output_dir, "iris_scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Trained model saved to {model_path}")
print(f"Standard scaler saved to {scaler_path}")
print("Classification pipeline completed successfully!")
