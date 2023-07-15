import torch
from clustering.confusion import Confusion
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import preprocessing


def get_kmeans(all_features, all_labels, num_classes, random_state=None, return_confusion=False):

    all_features = all_features.numpy()
    all_features = preprocessing.normalize(all_features)
    print('Clustering with kmeans...')
    
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes, random_state=random_state)
    clustering_model.fit(all_features)
    cluster_assignment = clustering_model.labels_

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)
    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    
    return confusion