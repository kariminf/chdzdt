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
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import CanineTokenizer, CanineModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import joblib 

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dzdt.model.chdzdt_tok import CharTokenizer
from dzdt.tools.const import char_tokenizer_config, word_tokenizer_config
from dzdt.model.chdzdt_mdl import MLMLMBertModel

# =============================================
#          Models loading 
# =============================================

def load_chdzdt_model(plm_loc: str) -> Tuple[CharTokenizer, MLMLMBertModel]:
    """ Load a CHDZDT model from the specified path.
    Args:
        plm_loc (str): The path to the pre-trained model or the model identifier from Hugging Face's model hub.
    Returns:
        Tuple[CharTokenizer, MLMLMBertModel]: A tuple containing the character tokenizer and the model.
    """
    plm_loc = os.path.expanduser(plm_loc)
    # print("loading characters tokenizer")
    char_tokenizer: CharTokenizer = CharTokenizer.load(os.path.join(plm_loc, "char_tokenizer.pickle"))

    # print("loading characters encoder")
    char_tokenizer_config()
    char_encoder = MLMLMBertModel.from_pretrained(plm_loc)
    word_tokenizer_config()

    return char_tokenizer, char_encoder

def load_bertlike_model(plm_loc: str) -> Tuple[BertTokenizer, BertModel]:
    """ Load a BERT-like model from the specified path.
    This function supports models like BERT, RoBERTa, and others that are compatible with the Hugging Face Transformers library.
    Args:
        plm_loc (str): The path to the pre-trained model or the model identifier from Hugging Face's model hub.
    Returns:
        Tuple[BertTokenizer, BertModel]: A tuple containing the tokenizer and the model.
    """
    plm_loc = os.path.expanduser(plm_loc)
    tokenizer = AutoTokenizer.from_pretrained(plm_loc)
    model = AutoModelForMaskedLM.from_pretrained(plm_loc)
    if isinstance(model, BertModel):
        return tokenizer, model
    
    if hasattr(model, 'base_model'):    
        return tokenizer, model.base_model
    
    return tokenizer, model.bert

def load_canine_model(plm_loc: str) -> Tuple[CanineTokenizer, CanineModel]:
    """ Load a Canine model from the specified path.
    Args:
        plm_loc (str): The path to the pre-trained model or the model identifier from Hugging Face's model hub.
    Returns:
        Tuple[CanineTokenizer, CanineModel]: A tuple containing the Canine tokenizer and the model.
    """
    plm_loc = os.path.expanduser(plm_loc)
    tokenizer = CanineTokenizer.from_pretrained(plm_loc)
    model = CanineModel.from_pretrained(plm_loc)
    if isinstance(model, BertModel):
        return tokenizer, model
    
    if hasattr(model, 'base_model'):    
        return tokenizer, model.base_model
    
    return tokenizer, model.bert 


def arabert_preprocess(texts: List[str], model_name: str) -> List[str]:
    from arabert.preprocess import ArabertPreprocessor
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    result = []
    for text in texts: 
        result.append(arabert_prep.preprocess(text))
    return result



# =============================================
#          Data loading and processing
# =============================================
   

def get_data(url: str) ->  Tuple[List[str], np.ndarray, List[str]]:
    url = os.path.expanduser(url)
    data = pd.read_csv(url, sep='\t')
    data["word"] = data["word"].astype(str)

    return data["word"].tolist(), data.iloc[:, data.columns != "word"].values, data.columns[1:]



def get_embeddings(words: List[str], tokenizer: BertTokenizer, bert: BertModel) -> Tuple[torch.Tensor, torch.Tensor]:
    
    tokens = tokenizer(words, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
    with torch.no_grad():
        outputs = bert(**tokens)

    return outputs.last_hidden_state[:, 0, :], outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)

# =============================================
#          Model  functions
# =============================================

def train_probing_model(in_url: str, out_url: str, tokenizer: BertTokenizer, bert: BertModel) -> Tuple[OneVsRestClassifier, OneVsRestClassifier]:

    X, y, names = get_data(in_url + "_train.csv")

    words_cls_emb, words_tok_emb = get_embeddings(X, tokenizer, bert)
    words_cls_emb = words_cls_emb.detach().numpy()
    words_tok_emb = words_tok_emb.detach().numpy()

    # Multi-label logistic regression
    clf_cls = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf_tok = OneVsRestClassifier(LogisticRegression(max_iter=1000))

    clf_cls.fit(words_cls_emb, y)
    clf_tok.fit(words_tok_emb, y)

    # Save the models
    joblib.dump(clf_cls, out_url + "_clf_cls.pkl")
    joblib.dump(clf_tok, out_url + "_clf_tok.pkl")

    return clf_cls, clf_tok

def test_probing_model(in_url: str, out_url: str, tokenizer: BertTokenizer, bert: BertModel, clf_cls: OneVsRestClassifier, clf_tok: OneVsRestClassifier) -> None:
    """ Test the probing models on the test data.
    Args:
        in_url (str): The input URL for the test data.
        out_url (str): The output URL for saving results.
        tokenizer (BertTokenizer): The tokenizer used for encoding words.
        bert (BertModel): The BERT model used for generating embeddings.
        clf_cls (OneVsRestClassifier): The classifier for class embeddings.
        clf_tok (OneVsRestClassifier): The classifier for token embeddings.
    """
    X, y, names = get_data(in_url + "_test.csv")

    words_cls_emb, words_tok_emb = get_embeddings(X, tokenizer, bert)
    words_cls_emb = words_cls_emb.detach().numpy()
    words_tok_emb = words_tok_emb.detach().numpy()

    y_pred_cls = clf_cls.predict(words_cls_emb)
    y_pred_tok = clf_tok.predict(words_tok_emb)

    # Save predictions
    pd.DataFrame(y_pred_cls, columns=names).to_csv(out_url + "_pred_cls.csv", index=False)
    pd.DataFrame(y_pred_tok, columns=names).to_csv(out_url + "_pred_tok.csv", index=False)

    # Save the classification reports
    
    report_cls = classification_report(y, y_pred_cls, target_names=names, zero_division=0)
    report_tok = classification_report(y, y_pred_tok, target_names=names, zero_division=0) 
    with open(out_url + "_report.txt", "w") as f:
        f.write("Classification Report for Class Embeddings:\n")
        f.write(report_cls)
        f.write("\n\n")
        f.write("Classification Report for Token Embeddings:\n")
        f.write(report_tok)


# =============================================
#          Testing functions
# =============================================

def test_morph_probing(args):

    if "chdzdt" in args.p:
        print("loading CHDZDT model ...")
        tokenizer, model = load_chdzdt_model(args.p)
    elif "canine" in args.p:
        print("loading Canine model ...")
        tokenizer, model = load_canine_model(args.p)
    else:
        print("loading BERT-like model ...")
        tokenizer, model = load_bertlike_model(args.p)

    out_url = os.path.expanduser(args.output)
    out_url = os.path.join(out_url, args.m)
    if not os.path.exists(out_url):
        os.makedirs(out_url)
    
    out_url = os.path.join(out_url, args.l)

    print("training the prob models ...")
    clf_cls, clf_tok = train_probing_model(args.input, out_url, tokenizer, model)

    print("testing the prob models ...")
    test_probing_model(args.input, out_url, tokenizer, model, clf_cls, clf_tok)
    


# =============================================
#          Command line parser
# =============================================     


def test_main(args):
    test_morph_probing(args)

parser = argparse.ArgumentParser(description="test morphological probing")
parser.add_argument("-m", help="model out name")
parser.add_argument("-l", help="language code")
parser.add_argument("-p", help="model name/path")
parser.add_argument("input", help="input clusters' file")
parser.add_argument("output", help="output folder for results")
parser.set_defaults(func=test_main)


if __name__ == "__main__":

    # argv = sys.argv[1:]
    # args = parser.parse_args(argv)
    # # print(args)
    # # parser.print_help()
    # args.func(args)


    src = "~/Data/DZDT/test/morpholex/morpholex_en2"
    dst = "~/Data/DZDT/results/morpholex/"

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
        ("bert", "google-bert/bert-base-uncased"),
        # ("flaubert", "flaubert/flaubert_base_uncased"),
        ("dziribert", "alger-ia/dziribert"),
        ("caninec", "google/canine-c"),
    ]

    for lang in ["en2"]: 
        for mdl in mdls:
            print(f"Testing model {mdl[0]} on {lang} ...")
            argv = [
                "-m", mdl[0],
                "-p", mdl[1],
                "-l", lang,
                src,
                dst
                ]
            args = parser.parse_args(argv)
            args.func(args)