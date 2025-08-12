## Perform k-means clustering for different values of k. Evaluate the above scores for each k value. Make a plot of the values against the k value to determine the optimal cluster count.  

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load dataset
df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")

# Prepare features
X_clustering = df.drop(columns=['patientid', 'target'])

silhouette_scores = []
ch_scores = []
db_scores = []
k_range = range(2, 11)  # Try k from 2 to 10

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_clustering)
    labels = km.labels_
    silhouette_scores.append(silhouette_score(X_clustering, labels))
    ch_scores.append(calinski_harabasz_score(X_clustering, labels))
    db_scores.append(davies_bouldin_score(X_clustering, labels))

import numpy as np

# Normalize scores to [0, 1] for plotting
sil_norm = (np.array(silhouette_scores) - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores))
ch_norm = (np.array(ch_scores) - min(ch_scores)) / (max(ch_scores) - min(ch_scores))
db_norm = (np.array(db_scores) - min(db_scores)) / (max(db_scores) - min(db_scores))

plt.figure(figsize=(8,6))
plt.plot(k_range, sil_norm, marker='o', label="Silhouette Score (normalized)")
plt.plot(k_range, ch_norm, marker='o', label="Calinski-Harabasz Score (normalized)")
plt.plot(k_range, db_norm, marker='o', label="Davies-Bouldin Index (normalized)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Normalized Score Value")
plt.title("Clustering Metrics vs k (Normalized) - Cardiovascular Dataset")
plt.legend()
plt.grid(True)
plt.show()
