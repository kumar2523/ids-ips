import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("anomaly_traffic_clean.csv")
print("Dataset Loaded. Columns:", data.columns)

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and labels
X = data.drop(columns=['attack_type'])
y = data['attack_type']

# Convert attack_type to binary (1 for anomaly, 0 for normal)
y = np.where(y != 'normal', 1, 0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict using XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Apply Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(X_train)
y_pred_lof = lof.predict(X_test)
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)  # Convert to binary (1 = anomaly, 0 = normal)

# Combine XGBoost and LOF predictions
final_predictions = np.logical_or(y_pred_xgb, y_pred_lof).astype(int)

# Evaluate Model
conf_matrix = confusion_matrix(y_test, final_predictions)
report = classification_report(y_test, final_predictions)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)

# Visualizing Feature Importance from XGBoost
plt.figure(figsize=(10,6))
sns.barplot(x=xgb_model.feature_importances_, y=X.columns)
plt.title("Feature Importance from XGBoost")
plt.show()

# PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=final_predictions, palette='coolwarm')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("XGBoost + LOF Anomaly Detection")
plt.show()
