import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from clustering.confusion import Confusion


def get_kmeans(all_features, all_labels, num_classes, random_state=None):

    all_features = all_features.numpy()
    all_features = preprocessing.normalize(all_features)
    
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes, random_state=random_state)
    clustering_model.fit(all_features)
    cluster_assignment = clustering_model.labels_

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)
    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    
    return confusion