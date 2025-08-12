## A5. For the clustering done in A4, calculate the: (i) Silhouette Score, (ii) CH Score and (iii) DB Index.
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load dataset
df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")

# Prepare features
X_clustering = df.drop(columns=['patientid', 'target'])

# Train KMeans
kmeans_model = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_clustering)

# Evaluate metrics
sil_score = silhouette_score(X_clustering, kmeans_model.labels_)
ch_score = calinski_harabasz_score(X_clustering, kmeans_model.labels_)
db_index = davies_bouldin_score(X_clustering, kmeans_model.labels_)

# Print results
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")
