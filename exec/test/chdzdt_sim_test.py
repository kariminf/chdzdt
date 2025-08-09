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

import sys
import os
import timeit
import pandas as pd

import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.model.chdzdt_tok import CharTokenizer
from dzdt.tools.const import char_tokenizer_config, word_tokenizer_config
from dzdt.model.chdzdt_mdl import MLMLMBertModel

def load_chdzdt_model(model_path: str) -> MLMLMBertModel:
    """
    Load the CHDZDT model from the specified path.
    """
    # print("loading characters tokenizer")
    char_tokenizer: CharTokenizer = CharTokenizer.load(os.path.join(model_path, "char_tokenizer.pickle"))

    # print("loading characters encoder")
    char_tokenizer_config()
    char_encoder = MLMLMBertModel.from_pretrained(model_path)
    word_tokenizer_config()

    return char_tokenizer, char_encoder

# Model loading
LOC = os.path.expanduser("~/Data/DZDT/models/chdzdt_2x2x16_20it")

print("loading the model ...")
char_tokenizer, char_encoder  = load_chdzdt_model(LOC)


# Dataset loading 
data_url = os.path.expanduser("~/Data/DZDT/test/Multilingual_Wordpairs/Gold_Standards/en-ws353.dataset")
Data = pd.read_csv(data_url, sep=";")

print(Data.head())

print("encoding ...")
word1_tokens = char_tokenizer.encode_words(Data["word1"], return_tensors="pt")
word1_emb   = char_encoder(**word1_tokens, return_dict=True)["last_hidden_state"][:, 0, :]


word2_tokens = char_tokenizer.encode_words(Data["word2"], return_tensors="pt")
word2_emb   = char_encoder(**word2_tokens, return_dict=True)["last_hidden_state"][:, 0, :]

print("codes=", word1_emb.shape, word2_emb.shape)

# Cosine similarity bedween word embeddings
cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
similarity_scores = cosine_sim(word1_emb, word2_emb)

print("similarity scores=", similarity_scores)

# corelation with the gold standard
corr_pearson = Data["score"].corr(pd.Series(similarity_scores.detach().numpy()), method='pearson')
corr_kendall = Data["score"].corr(pd.Series(similarity_scores.detach().numpy()), method='kendall')
corr_spearman = Data["score"].corr(pd.Series(similarity_scores.detach().numpy()), method='spearman')
print("correlation (pearson, kendall, spearman)", corr_pearson, corr_kendall, corr_spearman)
