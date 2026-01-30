import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('/home/kali/Desktop/IDP/HYBRID1/anomaly_traffic_clean.csv')

# Drop non-numeric columns (if any)
data_numeric = data.select_dtypes(include=[np.number])

# Standardize the dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Apply PCA for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
data_numeric['PCA1'] = data_pca[:, 0]
data_numeric['PCA2'] = data_pca[:, 1]

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
data_numeric['IsolationForest_Anomaly'] = iso_forest.fit_predict(data_scaled)
data_numeric['IsolationForest_Anomaly'] = data_numeric['IsolationForest_Anomaly'].map({1: 0, -1: 1})  # Convert to 0 (normal) and 1 (anomaly)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
data_numeric['KMeans_Cluster'] = kmeans.fit_predict(data_scaled)

# Visualization of K-Means Clustering
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data_numeric['PCA1'], y=data_numeric['PCA2'], hue=data_numeric['KMeans_Cluster'], palette='coolwarm')
plt.title('K-Means Clustering')
plt.show()

# Visualization of Isolation Forest Anomalies
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data_numeric['PCA1'], y=data_numeric['PCA2'], hue=data_numeric['IsolationForest_Anomaly'], palette={0: 'blue', 1: 'red'})
plt.title('Isolation Forest Anomalies (Red = Anomalies)')
plt.show()

# Confusion Matrix and Report
true_labels = data['attack_type'].apply(lambda x: 1 if x != 'normal' else 0)  # Considering attack types as anomalies
cm = confusion_matrix(true_labels, data_numeric['IsolationForest_Anomaly'])
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(true_labels, data_numeric['IsolationForest_Anomaly']))
