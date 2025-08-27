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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dzdt.model.chdzdt_tok import CharTokenizer
from dzdt.tools.const import char_tokenizer_config, word_tokenizer_config
from dzdt.model.chdzdt_mdl import MLMLMBertModel

from dzdt.extra.plms import arabert_preprocess, load_bertlike_model, load_canine_model, load_chdzdt_model, get_oneword_embeddings
from dzdt.extra.data import get_word_cluster_data, get_word_noisy_data
from dzdt.extra.stats import cosine_similarity, kmeans_ari, silhouette, cluster_cos_euc




def graphical_plot(xy: np.ndarray, labels: np.ndarray, output_url: str):
    plt.figure(figsize=(10, 10))
    plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap='hsv', alpha=0.8)
    plt.title("Visualization of Word Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label='Cluster Label')
    plt.legend()
    plt.savefig(output_url)
    plt.close()  

def graphical_clustering(embeddings: np.ndarray, true_labels: np.ndarray, output_url: str):
    """
    Visualize the clustering of embeddings using t-SNE and save the plot to output_url.
    """

    output_url = os.path.expanduser(output_url)

    print("calculating TSNE ...")
    tsne = TSNE(n_components=2, random_state=42, max_iter=1000, verbose=0, n_jobs=-1)
    tsne_emb = tsne.fit_transform(embeddings)
    graphical_plot(tsne_emb, true_labels, output_url + "_tsne.png")

    print("calculating UMAP ...")
    reducer = umap.UMAP(n_neighbors=29, min_dist=0.1, metric='cosine', random_state=42)
    umap_emb = reducer.fit_transform(embeddings)
    graphical_plot(umap_emb, true_labels, output_url + "_umap.png")


# =============================================
#          Testing functions
# =============================================

def test_word_clustering(args):

    if "chdzdt" in args.m:
        print("loading CHDZDT model ...")
        tokenizer, model = load_chdzdt_model(args.m)
    elif "canine" in args.m:
        print("loading Canine model ...")
        tokenizer, model = load_canine_model(args.m)
    else:
        print("loading BERT-like model ...")
        tokenizer, model = load_bertlike_model(args.m)

    print("loading data ...")
    Data = get_word_cluster_data(args.input)

    words       = Data["word"].tolist()
    true_labels = Data["cluster"].to_numpy()

    if "arabert" in args.m: 
        print("normalizing using AraBERT ...")
        words = arabert_preprocess(words, args.m)

    print("encoding words ...")
    words_cls_emb, words_tok_emb = get_oneword_embeddings(words, tokenizer, model)
    words_cls_emb = words_cls_emb.detach().numpy()
    words_tok_emb = words_tok_emb.detach().numpy()

 
    print("calculating ASR over KMeans ...")
    cls_ari = kmeans_ari(words_cls_emb, true_labels)
    tok_ari = kmeans_ari(words_tok_emb, true_labels)

    print("calculating sil ...")
    cls_sil = silhouette(words_cls_emb, true_labels)
    tok_sil = silhouette(words_tok_emb, true_labels)

    print("calculating average cosine similarity and euclidean distance ...")
    cls_cos, cls_euc = cluster_cos_euc(words_cls_emb, true_labels)
    tok_cos, tok_euc = cluster_cos_euc(words_tok_emb, true_labels)

    print("writing results ...")
    with open(os.path.expanduser(args.output) + ".txt", "w", encoding="utf8") as out_f:

        out_f.write("\n\nResults:\n")
        out_f.write("==================================\n")
        out_f.write("\nembedding\tKMeans+ARI\tSil\tAvg. Cos.\Avg. Euc.\n")
        out_f.write(f"CLS\t{cls_ari}\t{cls_sil}\t{cls_cos}\t{cls_euc}\n")

        # if "chdzdt" not in args.m:
        out_f.write(f"Centroid\t{tok_ari}\t{tok_sil}\t{tok_cos}\t{tok_euc}\n")

    print("graphical clustering ...")
    # graphical_clustering(words_cls_emb, true_labels, args.output)


def test_word_noise(args):

    if "chdzdt" in args.m:
        print("loading CHDZDT model ...")
        tokenizer, model = load_chdzdt_model(args.m)
    elif "canine" in args.m:
        print("loading Canine model ...")
        tokenizer, model = load_canine_model(args.m)
    else:
        print("loading BERT-like model ...")
        tokenizer, model = load_bertlike_model(args.m)

    print("loading data ...")
    words, ofus1fix, ofus1var = get_word_noisy_data(args.input)

    if "arabert" in args.m: 
        print("normalizing using AraBERT ...")
        words = arabert_preprocess(words, args.m)
        ofus1fix = arabert_preprocess(ofus1fix, args.m)
        ofus1var = arabert_preprocess(ofus1var, args.m)

    print("encoding words ...")
    words_cls_emb, words_tok_emb = get_oneword_embeddings(words, tokenizer, model)
    words_cls_emb = words_cls_emb.detach().numpy()
    words_tok_emb = words_tok_emb.detach().numpy()

    print("encoding ofus1fix ...")
    ofus1fix_cls_emb, ofus1fix_tok_emb = get_oneword_embeddings(ofus1fix, tokenizer, model)
    ofus1fix_cls_emb = ofus1fix_cls_emb.detach().numpy()
    ofus1fix_tok_emb = ofus1fix_tok_emb.detach().numpy()

    print("encoding ofus2fix ...")
    ofus2fix = [w.replace("*", "#") for w in ofus1fix]  # 
    ofus2fix_cls_emb, ofus2fix_tok_emb = get_oneword_embeddings(ofus2fix, tokenizer, model)
    ofus2fix_cls_emb = ofus2fix_cls_emb.detach().numpy()
    ofus2fix_tok_emb = ofus2fix_tok_emb.detach().numpy()

    print("encoding ofus1var ...")
    ofus1var_cls_emb, ofus1var_tok_emb = get_oneword_embeddings(ofus1var, tokenizer, model)
    ofus1var_cls_emb = ofus1var_cls_emb.detach().numpy()
    ofus1var_tok_emb = ofus1var_tok_emb.detach().numpy()

    del tokenizer, model, words, ofus1fix, ofus1var, ofus2fix

    print("cosine similarities ...")
    cls_cos_ofus1fix = cosine_similarity(words_cls_emb, ofus1fix_cls_emb)
    cls_cos_ofus2fix = cosine_similarity(words_cls_emb, ofus2fix_cls_emb)
    cls_cos_ofus1var = cosine_similarity(words_cls_emb, ofus1var_cls_emb)
    tok_cos_ofus1fix = cosine_similarity(words_tok_emb, ofus1fix_tok_emb)
    tok_cos_ofus2fix = cosine_similarity(words_tok_emb, ofus2fix_tok_emb)   
    tok_cos_ofus1var = cosine_similarity(words_tok_emb, ofus1var_tok_emb)

    print("writing results ...")
    with open(os.path.expanduser(args.output) + ".txt", "w", encoding="utf8") as out_f:

        out_f.write("\n\nResults:\n")
        out_f.write("==================================\n")
        out_f.write("\nembedding\tfix*\tfix#\tvar\n")
        out_f.write(f"CLS\t{cls_cos_ofus1fix}\t{cls_cos_ofus2fix}\t{cls_cos_ofus1var}\n")
        out_f.write(f"Tok\t{tok_cos_ofus1fix}\t{tok_cos_ofus2fix}\t{tok_cos_ofus1var}\n")

# =============================================
#          Command line parser
# =============================================     


def test_main(args):
    if args.t == "cluster":
        test_word_clustering(args)
    elif args.t == "noise":
        test_word_noise(args)
    else:
        print(f"Unknown task {args.t}.")

parser = argparse.ArgumentParser(description="test morphological clustering using a pre-trained model")
parser.add_argument("-t", help="task name", default="cluster", choices=["cluster", "noise"])
parser.add_argument("-m", help="model name/path")
parser.add_argument("input", help="input clusters' file")
parser.add_argument("output", help="output txt file containing the results")
parser.set_defaults(func=test_main)


if __name__ == "__main__":

    # argv = sys.argv[1:]
    # args = parser.parse_args(argv)
    # # print(args)
    # # parser.print_help()
    # args.func(args)

    # d, ss = "ar_infl_wr", ["avg75", "min100"]
    # d, ss = "ar_deriv_nr", ["avg15", "min30"]
    # src = "~/Data/DZDT/test/morphology/arabic/"

    # d, ss = "en_infl_wr", ["avg6", "min7"]
    # d, ss = "en_deriv_wr", ["avg30", "min78"]
    # src = "~/Data/DZDT/test/morphology/english/"

    # d, ss = "fr_infl_wr", ["avg40", "min42"]
    # d, ss = "fr_deriv_wr", ["avg15", "min30"]
    # src = "~/Data/DZDT/test/morphology/french/"


    # dst = f"~/Data/DZDT/results/morph-consist/{d}/"


    # ========================================
    ss = ["cls"] # ["noise"]
    # d = "ar_taboo"
    # src = "~/Data/DZDT/test/lexicon/arabic/"
    # dst = "~/Data/DZDT/results/lexicon/arabic/noise/"

    d = "dz_taboo"
    src = "~/Data/DZDT/test/lexicon/arabizi/"
    dst = "~/Data/DZDT/results/lexicon/arabizi/"

    # d = "en_taboo"
    # src = "~/Data/DZDT/test/lexicon/english/"
    # dst = "~/Data/DZDT/results/lexicon/english/"

    # d = "fr_taboo"
    # src = "~/Data/DZDT/test/lexicon/french/"
    # dst = "~/Data/DZDT/results/lexicon/french/"

    
    mdls = [
        # ("chdzdt_5x4x128_20it", "~/Data/DZDT/models/chdzdt_5x4x128_20it"),
        # ("chdzdt_4x4x64_20it", "~/Data/DZDT/models/chdzdt_4x4x64_20it"),
        # ("chdzdt_4x4x32_20it", "~/Data/DZDT/models/chdzdt_4x4x32_20it"),
        ("chdzdt_3x2x16_20it", "~/Data/DZDT/models/chdzdt_3x2x16_20it"),
        ("chdzdt_2x1x16_20it", "~/Data/DZDT/models/chdzdt_2x1x16_20it"),
        ("chdzdt_2x4x16_20it", "~/Data/DZDT/models/chdzdt_2x4x16_20it"),
        ("chdzdt_2x2x32_20it", "~/Data/DZDT/models/chdzdt_2x2x32_20it"),
        ("chdzdt_2x2x16_20it", "~/Data/DZDT/models/chdzdt_2x2x16_20it"),
        ("chdzdt_2x2x8_20it", "~/Data/DZDT/models/chdzdt_2x2x8_20it"),
        ("chdzdt_1x2x16_20it", "~/Data/DZDT/models/chdzdt_1x2x16_20it"),
        # ("arabert", "aubmindlab/bert-base-arabertv02-twitter"),
        # ("bert", "google-bert/bert-base-uncased"),
        # ("flaubert", "flaubert/flaubert_base_uncased"),
        # ("dziribert", "alger-ia/dziribert"),
        # ("caninec", "google/canine-c"),
    ]

    for s in ss: #"full", 
        for mdl in mdls:
            print(f"Testing model {mdl[0]} on data with {d}_{s} clusters ...")
            argv = [
                "-t", "cluster",
                "-m", mdl[1],
                f"{src}{d}_{s}.csv",
                f"{dst}/{s}/{d}_{s}_{mdl[0]}"
                ]
            args = parser.parse_args(argv)
            args.func(args)