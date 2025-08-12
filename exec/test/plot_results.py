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

import numpy as np
import os, json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Example data (replace with your Frobenius norms)
# Format: {language: {model_name: {"mapconcat": (p, r, s), "mapadd": (p, r, s)}}}
# ---------------------------

URL = "~/Data/DZDT/results/morpholex/_composition/"

path = os.path.expanduser(URL)

with open(path + "frobenius.json", "r", encoding="utf-8") as f:
    data = json.load(f)


# ---------------------------
# Settings
# ---------------------------
colors = ["red", "green", "blue", "purple"]
markers = ["o", "s", "^", "D"]  # one per model

normalize = False  # set to False if you want raw values

# ---------------------------
# Plot 3D
# ---------------------------
fig = plt.figure(figsize=(12, 5))

for i, lang in enumerate(["EN", "FR"], 1):
    ax = fig.add_subplot(1, 2, i, projection="3d")
    for m_idx, (model, comps) in enumerate(data[lang].items()):
        vec = np.array(comps["mapconcat"], dtype=float)
        if normalize and np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)
        ax.scatter(
            vec[0], vec[1], vec[2],
            color=colors[m_idx],
            marker=markers[m_idx],
            s=70,
            label=f"{model}"
        )

    ax.set_title(lang)
    ax.set_xlabel("Prefix Frobenius")
    ax.set_ylabel("Root Frobenius")
    ax.set_zlabel("Suffix Frobenius")
    ax.view_init(elev=20, azim=35)

# Create a shared legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
plt.savefig(path + "frobenius_3d.pdf", bbox_inches="tight")
plt.close()


# ---------------------------
# Plot 1D
# ---------------------------

models_EN = list(data["EN"].keys())
models_FR = list(data["FR"].keys())

prefix_EN = [data["EN"][m]["mapadd"][0] for m in models_EN]
prefix_FR = [data["FR"][m]["mapadd"][0] for m in models_FR]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# English
axes[0].bar(models_EN, prefix_EN, color=colors)
axes[0].set_title("English - mapadd prefix Frobenius")
axes[0].set_ylabel("Frobenius norm")
axes[0].tick_params(axis='x', rotation=45)

# French
axes[1].bar(models_FR, prefix_FR, color=colors)
axes[1].set_title("French - mapadd prefix Frobenius")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(path + "frobenius_1d.pdf", bbox_inches="tight")
plt.close()