from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap

import numpy as np

def cluster_tabular_data(X):
    ## Assume the input data is in the form of a pandas dataframe and without the target column

    def find_gl_max_in_lc_max(sil):
        lc_max = []
        for i in range(1, len(sil)-1):
            if sil[i] > sil[i-1] and sil[i] > sil[i+1]:
                lc_max.append(sil[i])
        if lc_max == []:
            gl_max = np.max(sil)
        else:
            gl_max = np.max(lc_max)
        return gl_max

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(X)

    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(normalized_data)

    # Determine k using The Silhouette Method
    sil = []
    k_values = range(2, 10)

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in k_values:
        kmeans = KMeans(n_clusters = k).fit(reduced_embeddings)
        labels = kmeans.labels_
        sil.append(silhouette_score(reduced_embeddings, labels, metric = 'euclidean'))

    # cluster_k_score = np.max(sil)
    cluster_k_score = find_gl_max_in_lc_max(sil)
    cluster_k = sil.index(cluster_k_score) + 2

    kmeans = KMeans(n_clusters=cluster_k, random_state=42)
    kmeans.fit(reduced_embeddings)
    labels = kmeans.labels_

    if len(np.unique(labels)) != cluster_k:
        difference = cluster_k - len(np.unique(labels))
        for _ in range(difference):
            for i in range(len(np.unique(labels))):
                if i not in np.unique(labels):
                    for j in [x for x in np.unique(labels) if x > i]:
                        labels[labels == (j)] = j - 1
                    break
        assert np.max(labels) == cluster_k, "Maximum label is not equal to the number of clusters"
        cluster_k = len(np.unique(labels))
        print(f"Adjusted Optimal k: {cluster_k}")
    else:
        print(f"Optimal k: {cluster_k}")

    assert len(labels) == X.shape[0], "Number of labels does not match the number of samples"
    assert len(np.unique(labels)) == cluster_k, "Number of unique labels is not equal to the number of clusters"

    return labels, cluster_k

def lim_cluster_tabular_data(X, max_k):
    ## Assume the input data is in the form of a pandas dataframe and without the target column

    def find_gl_max_in_lc_max(sil):
        lc_max = []
        for i in range(1, len(sil)-1):
            if sil[i] > sil[i-1] and sil[i] > sil[i+1]:
                lc_max.append(sil[i])
        if lc_max == []:
            gl_max = np.max(sil)
        else:
            gl_max = np.max(lc_max)
        return gl_max

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(X)

    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(normalized_data)

    # Determine k using The Silhouette Method
    sil = []
    k_values = range(2, max_k + 1)

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in k_values:
        kmeans = KMeans(n_clusters = k, n_init=25).fit(reduced_embeddings)
        labels = kmeans.labels_
        sil.append(silhouette_score(reduced_embeddings, labels, metric = 'euclidean'))

    # cluster_k_score = np.max(sil)
    cluster_k_score = find_gl_max_in_lc_max(sil)
    cluster_k = sil.index(cluster_k_score) + 2

    kmeans = KMeans(n_clusters=cluster_k, random_state=42)
    kmeans.fit(reduced_embeddings)
    labels = kmeans.labels_

    if len(np.unique(labels)) != cluster_k:
        difference = cluster_k - len(np.unique(labels))
        for _ in range(difference):
            for i in range(len(np.unique(labels))):
                if i not in np.unique(labels):
                    for j in [x for x in np.unique(labels) if x > i]:
                        labels[labels == (j)] = j - 1
                    break
        assert np.max(labels) == cluster_k, "Maximum label is not equal to the number of clusters"
        cluster_k = len(np.unique(labels))
        print(f"Adjusted Optimal k: {cluster_k}")
    else:
        print(f"Optimal k: {cluster_k}")

    assert len(labels) == X.shape[0], "Number of labels does not match the number of samples"
    assert len(np.unique(labels)) == cluster_k, "Number of unique labels is not equal to the number of clusters"

    return labels, cluster_k