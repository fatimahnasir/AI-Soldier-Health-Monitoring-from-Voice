import os
import numpy as np
from sklearn.cluster import KMeans
from extract_features import extract_features
import pickle
# -Load audio files 
data_path = "data"
X = []
files = []

for file in os.listdir(data_path):
    if file.endswith(".wav"):
        file_path = os.path.join(data_path, file)
        features = extract_features(file_path)
        X.append(features)
        files.append(file)

X = np.array(X)
print(f"Extracted features from {len(X)} audio files")

# K-Means clustering 
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Save the model
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Save file-to-cluster mapping
file_cluster_mapping = dict(zip(files, labels))
with open("file_cluster_mapping.pkl", "wb") as f:
    pickle.dump(file_cluster_mapping, f)

print("K-Means clustering done! File-to-cluster mapping saved.")
for file, cluster in file_cluster_mapping.items():
    print(f"{file} -> Cluster {cluster}")
