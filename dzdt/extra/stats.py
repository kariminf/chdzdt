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
    """Calculate the cosine similarity between two aligned sets of embeddings (row-wise)."""
    return (np.sum(emb1 * emb2, axis=1) / (np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1))).mean()

def euclidean(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return np.linalg.norm(emb1 - emb2, axis=1).mean()

# returns adjusted rand score over the clusters
def kmeans_ars(embeddings: np.ndarray, true_labels: np.ndarray) -> float:

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=np.unique(true_labels).size, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)

    return adjusted_rand_score(true_labels, predicted_labels)

# returns sillouette
def silhouette(embeddings: np.ndarray, true_labels: np.ndarray) -> float:
    return silhouette_score(embeddings, true_labels)

# returns average cosine similarity and euclidean distance  #
# of elements of each cluster with their first element (representative)
def cluster_cos_euc(embeddings: np.ndarray, true_labels: np.ndarray) -> Tuple[float, float]:
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