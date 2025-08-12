## A7. Using elbow plot, determine the optimal k value for k-means clustering. Use below code.


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")

# Prepare features
X_clustering = df.drop(columns=['patientid', 'target'])

# Elbow method
distortions = []
K = range(2, 20)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_clustering)
    distortions.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(8,6))
plt.plot(K, distortions, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.title('Elbow Method - Cardiovascular Dataset')
plt.grid(True)
plt.show()
