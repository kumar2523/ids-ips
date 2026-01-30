import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load the dataset
data = pd.read_csv("anomaly_traffic_clean.csv")
print("Columns in dataset:", data.columns)

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

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define Autoencoder
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(8, activation='relu')(encoded)

decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=1)

# Extract encoded features
encoder = Model(input_layer, encoded)
X_encoded = encoder.predict(X_scaled)

# Apply DBSCAN
clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_encoded)
labels = clustering.labels_

# Convert clustering labels to binary anomaly labels (1: Anomaly, 0: Normal)
predictions = np.where(labels == -1, 1, 0)
y_true = np.where(y != 'normal', 1, 0)

# Evaluate the model
accuracy = accuracy_score(y_true, predictions)
conf_matrix = confusion_matrix(y_true, predictions)
report = classification_report(y_true, predictions)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Autoencoder + DBSCAN")
plt.show()

# Visualize clustering results using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_encoded)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='viridis', legend=True)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clustering on Encoded Data")
plt.show()