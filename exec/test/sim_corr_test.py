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

from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dzdt.model.chdzdt_tok import CharTokenizer
from dzdt.tools.const import char_tokenizer_config, word_tokenizer_config
from dzdt.model.chdzdt_mdl import MLMLMBertModel



def arabert_preprocess(texts: List[str], model_name: str) -> List[str]:
    from arabert.preprocess import ArabertPreprocessor
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    result = []
    for text in texts: 
        result.append(arabert_prep.preprocess(text))
    return result

def load_chdzdt_model(plm_loc: str) -> Tuple[CharTokenizer, MLMLMBertModel]:
    """
    Load the CHDZDT model from the specified path.
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
    plm_loc = os.path.expanduser(plm_loc)
    tokenizer = AutoTokenizer.from_pretrained(plm_loc)
    model = AutoModelForMaskedLM.from_pretrained(plm_loc)
    if isinstance(model, BertModel):
        return tokenizer, model
    
    if hasattr(model, 'base_model'):    
        return tokenizer, model.base_model
    
    return tokenizer, model.bert

def get_data(url: str) -> pd.DataFrame:
    url = os.path.expanduser(url)
    return pd.read_csv(url, sep=";")


def get_embeddings(words: List[str], tokenizer: BertTokenizer, bert: BertModel) -> Tuple[torch.Tensor, torch.Tensor]:
    
    tokens = tokenizer(words, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
    with torch.no_grad():
        outputs = bert(**tokens)

    return outputs.last_hidden_state[:, 0, :], outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)

def get_correlation(scores: pd.Series, similarity_scores: torch.Tensor) -> Tuple[float, float, float]:
    scores_series = pd.Series(similarity_scores.detach().numpy())
    corr_pearson = scores.corr(scores_series, method='pearson')
    corr_kendall = scores.corr(scores_series, method='kendall')
    corr_spearman = scores.corr(scores_series, method='spearman')
    return corr_pearson, corr_kendall, corr_spearman


# =============================================
#          Command line functions
# =============================================

def test_word_similarity(args):

    print("loading model ...")
    if "chdzdt" in args.m:
        tokenizer, model = load_chdzdt_model(args.m)
    else:
        tokenizer, model = load_bertlike_model(args.m)

    print("loading data ...")
    Data = get_data(args.input)

    out_url = os.path.expanduser(args.output)
    out_f = open(out_url, "w", encoding="utf8")

    out_f.write(f"WordSimilarity for {args.m}")
    out_f.write("\n==================================\n\n\n")
    out_f.write(Data.head().to_csv(sep="\t", index=False))

    word1 = Data["word1"].tolist()
    word2 = Data["word2"].tolist()

    if "arabert" in args.m: 
        print("normalizing using AraBERT ...")
        word1 = arabert_preprocess(word1, args.m)
        word2 = arabert_preprocess(word2, args.m)

    print("encoding words ...")
    word1_cls_emb, word1_tok_emb = get_embeddings(word1, tokenizer, model)
    word2_cls_emb, word2_tok_emb = get_embeddings(word2, tokenizer, model)

    # out_f.write("\n\nWords1 CLS Embeddings:\n")
    # out_f.write("==================================\n")
    # out_f.write(word1_cls_emb.detach().numpy().tolist().__str__())
    # out_f.write("\n\nWords2 CLS Embeddings:\n")
    # out_f.write("==================================\n")
    # out_f.write(word2_cls_emb.detach().numpy().tolist().__str__())
    # out_f.write("\n\nWords1 Centroid Embeddings:\n")
    # out_f.write("==================================\n")
    # out_f.write(word1_tok_emb.detach().numpy().tolist().__str__())
    # out_f.write("\n\nWords2 Centroid Embeddings:\n")
    # out_f.write("==================================\n")
    # out_f.write(word2_tok_emb.detach().numpy().tolist().__str__())

    print("calculating cosine similarities ...")
    cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity_cls = cosine_sim(word1_cls_emb, word2_cls_emb)
    cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6) # better be carefull
    similarity_tok = cosine_sim(word1_tok_emb, word2_tok_emb)

    out_f.write("\n\nCLS similarities:\n")
    out_f.write("==================================\n")
    out_f.write(similarity_cls.detach().numpy().tolist().__str__())
    out_f.write("\n\nCentroid similarities:\n")
    out_f.write("==================================\n")
    out_f.write(similarity_tok.detach().numpy().tolist().__str__())

    print("calculating correlation ...")
    
    out_f.write("\n\ncorrelation:\n")
    out_f.write("==================================\n")
    out_f.write("\nembedding\tpearson\tkendall\tspearman\n")

    corr_pearson, corr_kendall, corr_spearman = get_correlation(Data["score"], similarity_cls)
    out_f.write(f"CLS\t{corr_pearson}\t{corr_kendall}\t{corr_spearman}\n")

    if "chdzdt" not in args.m:
        corr_pearson, corr_kendall, corr_spearman = get_correlation(Data["score"], similarity_tok)
        out_f.write(f"Cedntroid\t{corr_pearson}\t{corr_kendall}\t{corr_spearman}\n")

    out_f.close()



parser = argparse.ArgumentParser(description="test word similarity correlation using a pre-trained model")

parser.add_argument("-m", help="model name/path")
parser.add_argument("input", help="input wordsim file")
parser.add_argument("output", help="output txt file containing the results")
parser.set_defaults(func=test_word_similarity)


if __name__ == "__main__":

    argv = sys.argv[1:]

    argv = [
        "-m", "~/Data/DZDT/models/chdzdt_5x4x128_20it",
        "~/Data/DZDT/test/Multilingual_Wordpairs/Gold_Standards/fr-ws353.dataset",
        "~/Data/DZDT/models/sim_test_chdzdt_5x4x128_20it_fr.txt"
    ]

    # argv = [
    #     "-m", "flaubert/flaubert_base_uncased",
    #     "~/Data/DZDT/test/Multilingual_Wordpairs/Gold_Standards/fr-ws353.dataset",
    #     "~/Data/DZDT/models/sim_test_flaubert_fr.txt"
    # ]

    # argv = [
    #     "-m", "google-bert/bert-base-uncased",
    #     "~/Data/DZDT/test/Multilingual_Wordpairs/Gold_Standards/en-ws353.dataset",
    #     "~/Data/DZDT/models/sim_test_bert_en.txt"
    # ]

    # argv = [
    #     "-m", "aubmindlab/bert-base-arabertv02-twitter",
    #     "~/Data/DZDT/test/Multilingual_Wordpairs/Gold_Standards/ar-ws353.dataset",
    #     "~/Data/DZDT/models/sim_test_arabertv02_ar.txt"
    # ]

    args = parser.parse_args(argv)
    # print(args)
    # parser.print_help()
    args.func(args)