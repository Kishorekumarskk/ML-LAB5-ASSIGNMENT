##Perform k-means clustering on your data. Please remove / ignore the target variable for performing clustering. Sample code suggested below

import pandas as pd
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")

def perform_kmeans_clustering(df, n_clusters=2):
    # Drop non-feature columns: ID and target
    X = df.drop(columns=['patientid', 'target'])
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    return kmeans

# Run clustering
kmeans_model = perform_kmeans_clustering(df, n_clusters=2)

# Print results
print("Cluster labels:", kmeans_model.labels_)
print("Cluster centers:\n", kmeans_model.cluster_centers_)
