## A7. Using the elbow plot, determine the optimal k value for k-means clustering. Use below code.

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function for Elbow plot
def elbow_plot(X_train, k_max):
    distortions = []
    for k in range(2, k_max):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_train)
        distortions.append(kmeans.inertia_)
    return distortions

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
    drop_cols = ['cardio']
    drop_cols = [col for col in drop_cols if col in df.columns]
    X_cluster = df.drop(columns=drop_cols)

    distortions = elbow_plot(X_cluster, 20)

    plt.plot(range(2, 20), distortions, marker='o')
    plt.xlabel("k")
    plt.ylabel("Distortion (Inertia)")
    plt.title("Elbow Plot for K-Means")
    plt.show()
