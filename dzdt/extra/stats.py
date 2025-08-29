#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2025 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2025	Abdelkrime Aries <kariminfo0@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import sys
import os
from typing import Tuple, List
import pandas as pd
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

from dzdt.model.chdzdt_tok import CharTokenizer
from dzdt.tools.const import char_tokenizer_config, word_tokenizer_config
from dzdt.model.chdzdt_mdl import MLMLMBertModel

from dzdt.extra.plms import arabert_preprocess, load_bertlike_model, load_canine_model, load_chdzdt_model, get_oneword_embeddings
from dzdt.extra.data import get_word_cluster_data, get_word_noisy_data


# =============================================
#          Statistical functions 
# =============================================

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculates the average cosine similarity between two aligned sets of embeddings.

    Args:
        emb1 (np.ndarray): First set of embeddings, shape (n_samples, n_features).
        emb2 (np.ndarray): Second set of embeddings, shape (n_samples, n_features).

    Returns:
        float: The mean cosine similarity between corresponding rows of emb1 and emb2.

    Raises:
        ValueError: If emb1 and emb2 do not have the same shape.

    Example:
        >>> import numpy as np
        >>> emb1 = np.array([[1, 0], [0, 1]])
        >>> emb2 = np.array([[0, 1], [1, 0]])
        >>> cosine_similarity(emb1, emb2)
        0.0
    """
    
    return (np.sum(emb1 * emb2, axis=1) / (np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1))).mean()

def euclidean(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculates the mean Euclidean distance between corresponding rows of two embedding arrays.

    Args:
        emb1 (np.ndarray): First array of embeddings with shape (n_samples, n_features).
        emb2 (np.ndarray): Second array of embeddings with shape (n_samples, n_features).

    Returns:
        float: The mean Euclidean distance between each pair of corresponding rows in emb1 and emb2.
    """
    return np.linalg.norm(emb1 - emb2, axis=1).mean()


def kmeans_ari(embeddings: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Performs KMeans clustering on the given embeddings and computes the Adjusted Rand Index (ARI) 
    between the predicted cluster labels and the provided true labels.

    Args:
        embeddings (np.ndarray): Array of shape (n_samples, n_features) representing the data points to cluster.
        true_labels (np.ndarray): Array of shape (n_samples,) containing the ground truth labels for each data point.

    Returns:
        float: The Adjusted Rand Index (ARI) score measuring the similarity between the true labels and the predicted cluster labels.
    """

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=np.unique(true_labels).size, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)

    return adjusted_rand_score(true_labels, predicted_labels)


def silhouette(embeddings: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Calculates the silhouette score for the given embeddings and true labels.

    The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.
    It ranges from -1 to 1, where a higher value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

    Args:
        embeddings (np.ndarray): Array of shape (n_samples, n_features) representing the data points.
        true_labels (np.ndarray): Array of shape (n_samples,) containing the cluster labels for each data point.

    Returns:
        float: The silhouette score for the given clustering.
    """
    return silhouette_score(embeddings, true_labels)


def cluster_cos_euc(embeddings: np.ndarray, true_labels: np.ndarray) -> Tuple[float, float]:
    """
    Computes the average cosine similarity and Euclidean distance between embeddings within the same cluster.

    For each cluster (as indicated by `true_labels`), the function selects the first embedding as the representative.
    It then calculates the cosine similarity and Euclidean distance between the representative and each subsequent embedding in the same cluster.
    The averages are computed over all intra-cluster pairs.

    Args:
        embeddings (np.ndarray): Array of embedding vectors.
        true_labels (np.ndarray): Array of cluster labels corresponding to each embedding.

    Returns:
        Tuple[float, float]: A tuple containing:
            - Average cosine similarity within clusters.
            - Average Euclidean distance within clusters.
    """
    sim = 0.0
    euc = 0.0

    current_label = None
    rep_embedding = None

    for embedding, label in zip(embeddings, true_labels):
        if current_label != label:                 
            current_label = label
            rep_embedding = embedding
            continue

        sim += np.dot(embedding, rep_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(rep_embedding))
        euc += np.linalg.norm(embedding - rep_embedding)

    #  TODO you can add Box plot of similarity and distance 

    nbr = len(true_labels) - len(np.unique(true_labels))

    return sim/nbr, euc/nbr