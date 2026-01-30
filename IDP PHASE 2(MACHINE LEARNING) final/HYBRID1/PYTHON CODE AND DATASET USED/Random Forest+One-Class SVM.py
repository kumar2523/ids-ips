import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("anomaly_traffic_clean.csv")  

# Convert categorical features to numeric using One-Hot Encoding
df = pd.get_dummies(df)

# Assuming the last column is the target
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target (0 = Normal, 1 = Anomaly)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1️⃣ Train Random Forest Classifier (Supervised Learning)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions using Random Forest
y_pred_rf = rf_model.predict(X_test)

# 2️⃣ Train One-Class SVM (Unsupervised Anomaly Detection)
ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="auto")
ocsvm.fit(X_train)

# Predict anomalies with One-Class SVM (-1 = anomaly, 1 = normal)
y_pred_ocsvm = ocsvm.predict(X_test)
y_pred_ocsvm = np.where(y_pred_ocsvm == -1, 1, 0)  # Convert to 0 (normal) and 1 (anomaly)

# Combine predictions using logical OR (either model flags it as anomaly)
final_predictions = np.logical_or(y_pred_rf, y_pred_ocsvm).astype(int)

# 3️⃣ PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

# Plot results (Random Forest Classification)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_rf, cmap="coolwarm", edgecolors="k")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("Random Forest Classification")
plt.colorbar(label="Predicted Label")
plt.show()

# Plot results (One-Class SVM Anomaly Detection)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_ocsvm, cmap="coolwarm", edgecolors="k")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("One-Class SVM Anomaly Detection")
plt.colorbar(label="Predicted Label")
plt.show()

# 4️⃣ Confusion Matrix & Classification Report
print("Confusion Matrix:\n", confusion_matrix(y_test, final_predictions))
print("\nClassification Report:\n", classification_report(y_test, final_predictions))
