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
import pandas as pd
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import numpy as np
from typing import List, Tuple

from sklearn.linear_model import LinearRegression, RidgeCV


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.extra.plms import arabert_preprocess, load_bertlike_model, load_canine_model, load_chdzdt_model, get_oneword_embeddings
from dzdt.extra.data import get_csv_string_data
from dzdt.extra.stats import cosine_similarity, euclidean


def test_additive_composition(w: np.array, p: np.array, r: np.array, s: np.array) -> Tuple[float, float]:
    w_comp = p + r + s

    return cosine_similarity(w, w_comp), euclidean(w, w_comp)

def test_weighted_additive_composition(w: np.array, 
                                       p: np.array, 
                                       r: np.array, 
                                       s: np.array, 
                                       abg: Tuple[float, float, float]) -> Tuple[float, float]:
    w_comp = (abg[0] * p) + (abg[1] * r) + (abg[2] * s)

    return cosine_similarity(w, w_comp), euclidean(w, w_comp)


def train_weighted_additive_composition(w: np.array, p: np.array, r: np.array, s: np.array) -> Tuple[float, float, float]:
    N, d = p.shape
    X_stack = np.zeros((N * d, 3))
    X_stack[:, 0] = p.reshape(-1)   # prefix dims
    X_stack[:, 1] = r.reshape(-1)   # root dims
    X_stack[:, 2] = s.reshape(-1)   # suffix dims

    w_flat = w.reshape(-1)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_stack, w_flat)

    return reg.coef_

def train_lin_map_concat_composition(w: np.array, p: np.array, r: np.array, s: np.array) -> RidgeCV:
    X = np.hstack([p, r, s])

    alphas = np.logspace(-6, 3, 10)
    reg = RidgeCV(alphas=alphas, cv=5, fit_intercept=False)
    reg.fit(X, w)

    return reg

def test_lin_map_concat_composition(w: np.array, 
                                    p: np.array, 
                                    r: np.array, 
                                    s: np.array, 
                                    mdl: RidgeCV) -> Tuple[float, float]:
    w_comp = mdl.predict(np.hstack([p, r, s]))

    return cosine_similarity(w, w_comp), euclidean(w, w_comp)

def test_multiplicative_composition(w: np.array, p: np.array, r: np.array, s: np.array) -> Tuple[float, float]:
    w_comp = p * r * s

    return cosine_similarity(w, w_comp), euclidean(w, w_comp)

def load_encode_data(url, tokenizer, model, arabert=False):
    print("loading data ...")
    Data = get_csv_string_data(url)
    words    = Data["word"].tolist()
    prefixes = Data["prefix"].tolist()
    roots    = Data["root"].tolist()
    suffixes = Data["suffix"].tolist()

    # if arabert: 
    #     print("normalizing using AraBERT ...")
    #     words = arabert_preprocess(words, args.m)
    #     prefixes = arabert_preprocess(prefixes, args.m)
    #     roots = arabert_preprocess(roots, args.m)
    #     suffixes = arabert_preprocess(suffixes, args.m)

    print("encoding words ...")
    words_cls_emb, words_tok_emb = get_oneword_embeddings(words, tokenizer, model, out="numpy")
    prefixes_cls_emb, prefixes_tok_emb = get_oneword_embeddings(prefixes, tokenizer, model, out="numpy")
    roots_cls_emb, roots_tok_emb = get_oneword_embeddings(roots, tokenizer, model, out="numpy")
    suffixes_cls_emb, suffixes_tok_emb = get_oneword_embeddings(suffixes, tokenizer, model, out="numpy")

    return {
        "cls": {
            "word": words_cls_emb, "prefix": prefixes_cls_emb, "root": roots_cls_emb, "suffix": suffixes_cls_emb
        },
        "tok": {
            "word": words_tok_emb, "prefix": prefixes_tok_emb, "root": roots_tok_emb, "suffix": suffixes_tok_emb
        }
    }

def train_trainable(args, tokenizer, model):
    data = load_encode_data(args.input + "_train.csv", tokenizer, model, "arabert" in args.p)
    weights = {}
    for enc in data:
        w = data[enc]["word"]
        p = data[enc]["prefix"]
        r = data[enc]["root"]
        s = data[enc]["suffix"]
        weights[enc] = {}
        weights[enc]["wadd"] = train_weighted_additive_composition(w, p, r, s)
        weights[enc]["mapconcat"] = train_lin_map_concat_composition(w, p, r, s)
    return weights

def plot_weights(mdl: RidgeCV, url: str):

    W = mdl.coef_

    d = W.shape[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(W, aspect='auto', cmap='coolwarm', interpolation='nearest')

    ax.axvline(d - 0.5, color='black', linewidth=2)
    ax.axvline(2*d - 0.5, color='black', linewidth=2)

    ax.set_title(r"Weight matrix $W = [W_p \; W_r \; W_s]$")
    ax.set_xlabel("Input dimensions (prefix | root | suffix)")
    ax.set_ylabel("Output dimensions")
    fig.colorbar(im, ax=ax, label="Weight value")

    plt.savefig(url)
    plt.close()  


def frobenius_norms(mdl: RidgeCV) -> Tuple[float, float, float]:
    W = mdl.coef_   # shape (d, 3d)
    d = W.shape[0]

    # Split W into its three blocks
    W_p = W[:, :d]
    W_r = W[:, d:2*d]
    W_s = W[:, 2*d:]

    # Compute Frobenius norms
    fro_p = np.linalg.norm(W_p, 'fro')
    fro_r = np.linalg.norm(W_r, 'fro')
    fro_s = np.linalg.norm(W_s, 'fro')

    return fro_p, fro_r, fro_s


# =============================================
#          Testing functions
# =============================================

def test_word_composition(args):

    if "chdzdt" in args.p:
        print("loading CHDZDT model ...")
        tokenizer, model = load_chdzdt_model(args.p)
    elif "canine" in args.p:
        print("loading Canine model ...")
        tokenizer, model = load_canine_model(args.p)
    else:
        print("loading BERT-like model ...")
        tokenizer, model = load_bertlike_model(args.p)

    weights = train_trainable(args, tokenizer, model)

    results = {}
    # testing
    data = load_encode_data(args.input + "_test.csv", tokenizer, model, "arabert" in args.p)
    for emb_type in data:
        emb = data[emb_type]
        results[emb_type] = {}
        cos, euc = test_additive_composition(emb["word"], emb["prefix"], emb["root"], emb["suffix"])
        results[emb_type]["add"] = [cos, euc]
        cos, euc = test_multiplicative_composition(emb["word"], emb["prefix"], emb["root"], emb["suffix"])
        results[emb_type]["mul"] = [cos, euc]
        cos, euc = test_weighted_additive_composition(emb["word"], emb["prefix"], emb["root"], emb["suffix"], weights[emb_type]["wadd"])
        results[emb_type]["wadd"] = [cos, euc]
        cos, euc = test_lin_map_concat_composition(emb["word"], emb["prefix"], emb["root"], emb["suffix"], weights[emb_type]["mapconcat"])
        results[emb_type]["mapconcat"] = [cos, euc]


    output: str = os.path.expanduser(args.output)

    plot_weights(weights["cls"]["mapconcat"], output.replace(".txt", "_cls_mapconcat.png"))
    plot_weights(weights["tok"]["mapconcat"], output.replace(".txt", "_tok_mapconcat.png"))

    fro_p, fro_r, fro_s = frobenius_norms(weights["cls"]["mapconcat"])

    with open(output, "w", encoding="utf8") as f:
        f.write(f"Testing composition for model {args.m}\n\n\n")
        f.write("comp\tcls\t\ttok\t\n")
        f.write("\tcos\teuc\tcos\teuc\n")
        for comp in ["add", "mul", "wadd", "mapconcat"]: #  
            f.write(f"{comp}")
            for enc in ["cls", "tok"]:
                f.write(f"\t{results[enc][comp][0]}\t{results[enc][comp][1]}")
            f.write("\n")

        f.write("\n\n")
        w_cls, w_tok = weights["cls"]["wadd"], weights["tok"]["wadd"]
        f.write("Weighted sum\n")
        f.write(f"weights cls = {str(w_cls)}\n")
        f.write(f"weights tok = {str(w_tok)}\n")

        f.write("\n\n")
        f.write("Map concat\n")
        f.write(f"{fro_p}, {fro_r}, {fro_s}\n")







# =============================================
#          Command line parser
# =============================================     


def test_main(args):
    test_word_composition(args)



parser = argparse.ArgumentParser(description="test morphological clustering using a pre-trained model")
parser.add_argument("-m", help="model label")
parser.add_argument("-p", help="model name/path")
parser.add_argument("input", help="input clusters' file")
parser.add_argument("output", help="output txt file containing the results")
parser.set_defaults(func=test_main)


if __name__ == "__main__":

    # argv = sys.argv[1:]
    # args = parser.parse_args(argv)
    # # print(args)
    # # parser.print_help()
    # args.func(args)

    d = "morpholex111_fr"
    src = "~/Data/DZDT/test/morpholex/"
    dst = "~/Data/DZDT/results/morpholex/_composition/"

    mdls = [
        ("chdzdt_5x4x128_20it", "~/Data/DZDT/models/chdzdt_5x4x128_20it"),
        ("chdzdt_4x4x64_20it", "~/Data/DZDT/models/chdzdt_4x4x64_20it"),
        ("chdzdt_4x4x32_20it", "~/Data/DZDT/models/chdzdt_4x4x32_20it"),
        # ("chdzdt_3x2x16_20it", "~/Data/DZDT/models/chdzdt_3x2x16_20it"),
        # ("chdzdt_2x1x16_20it", "~/Data/DZDT/models/chdzdt_2x1x16_20it"),
        # ("chdzdt_2x4x16_20it", "~/Data/DZDT/models/chdzdt_2x4x16_20it"),
        # ("chdzdt_2x2x32_20it", "~/Data/DZDT/models/chdzdt_2x2x32_20it"),
        # ("chdzdt_2x2x16_20it", "~/Data/DZDT/models/chdzdt_2x2x16_20it"),
        # ("chdzdt_2x2x8_20it", "~/Data/DZDT/models/chdzdt_2x2x8_20it"),
        # ("chdzdt_1x2x16_20it", "~/Data/DZDT/models/chdzdt_1x2x16_20it"),
        # ("arabert", "aubmindlab/bert-base-arabertv02-twitter"),
        # ("bert", "google-bert/bert-base-uncased"),
        ("flaubert", "flaubert/flaubert_base_uncased"),
        ("dziribert", "alger-ia/dziribert"),
        ("caninec", "google/canine-c"),
    ]

    for mdl in mdls:
        print(f"Testing model {mdl[0]} on data with {d} clusters ...")
        argv = [
            "-m", mdl[0],
            "-p", mdl[1],
            f"{src}{d}",
            f"{dst}{d}_comp_{mdl[0]}.txt"
            ]
        args = parser.parse_args(argv)
        args.func(args)