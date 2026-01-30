import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("anomaly_traffic_clean.csv")

# Check column names
print("Columns in dataset:", df.columns)

# Identify label column
label_column = "attack_type"

# Encode categorical target variable
le = LabelEncoder()
df[label_column] = le.fit_transform(df[label_column])

# Identify categorical feature columns
categorical_cols = ["protocol_type", "service", "flag"]

# Apply One-Hot Encoding to categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features (X) and target (y)
X = df.drop(columns=[label_column])
y = df[label_column]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Decision Tree Model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train_scaled, y_train)
y_pred_tree = decision_tree_model.predict(X_test_scaled)

# Hybrid Model: Combining Predictions
y_pred_hybrid = (y_pred_logistic + y_pred_tree) / 2
y_pred_hybrid = np.round(y_pred_hybrid).astype(int)

# Performance Metrics
print("\nLogistic Regression Metrics:")
print(classification_report(y_test, y_pred_logistic))

print("\nDecision Tree Metrics:")
print(classification_report(y_test, y_pred_tree))

print("\nHybrid Model Metrics:")
print(classification_report(y_test, y_pred_hybrid))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_hybrid)
print("\nConfusion Matrix (Hybrid Model):\n", conf_matrix)
print("Hybrid Model Accuracy:", accuracy_score(y_test, y_pred_hybrid))

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Hybrid Model")
plt.show()

# PCA for Visualization
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_pred_hybrid, palette={0: 'blue', 1: 'red'}, alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Logistic Regression + Decision Tree Anomaly Detection")
plt.legend(title="Prediction")
plt.show()

# Feature Importance from Decision Tree
feature_importance = decision_tree_model.feature_importances_
feature_names = X.columns

# Sort feature importance
sorted_idx = np.argsort(feature_importance)[::-1]

# Select top N features for better readability
N = 20  # Display top 20 most important features
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance[sorted_idx][:N], y=np.array(feature_names)[sorted_idx][:N], palette="viridis")
plt.title("Top 20 Feature Importance from Decision Tree")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.xticks(rotation=45)
plt.show()
