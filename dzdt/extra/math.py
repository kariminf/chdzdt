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



def log_scaling(X: np.array) -> np.array:
    """
    Applies logarithmic scaling to the input array.

    This function shifts the input array so that its minimum value becomes slightly above zero,
    preventing issues with taking the logarithm of zero or negative values. It then applies the
    natural logarithm to each element.

    Parameters
    ----------
    X : np.array
        Input array to be scaled.

    Returns
    -------
    np.array
        Logarithmically scaled array.
    """
    X_pos = X - X.min() + 1e-6
    return np.log(X_pos)