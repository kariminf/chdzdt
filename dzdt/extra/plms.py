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
import torch
from typing import Tuple, List
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import CanineTokenizer, CanineModel

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
    char_tokenizer: CharTokenizer = CharTokenizer.load(os.path.join(plm_loc, "char_tokenizer.pkl"))

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
    tokenizer = AutoTokenizer.from_pretrained(plm_loc, use_fast=True)
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


def get_oneword_embeddings(words: List[str], 
                           tokenizer: BertTokenizer, 
                           bert: BertModel,
                           out="torch") -> Tuple[torch.Tensor, torch.Tensor]:
    
    tokens = tokenizer(words, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
    with torch.no_grad():
        outputs = bert(**tokens)

    emb_cls, emb_tok = outputs.last_hidden_state[:, 0, :], outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)

    if out == "numpy":
        emb_cls, emb_tok = emb_cls.detach().numpy(), emb_tok.detach().numpy()

    return emb_cls, emb_tok

def get_oneword_embeddings_cuda(words: List[str], 
                           tokenizer: BertTokenizer, 
                           bert: BertModel,
                           device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    
    tokens = tokenizer(words, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    if bert.device != device:
        bert.to(device)
    with torch.no_grad():
        outputs = bert(**tokens)

    emb_cls, emb_tok = outputs.last_hidden_state[:, 0, :], outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)

    return emb_cls, emb_tok


def get_sent_embeddings(sentences: List[str], tokenizer: BertTokenizer, bert: BertModel):

    tokens = tokenizer(sentences, 
                       return_tensors="pt", 
                       padding=True, 
                       truncation=True, 
                       add_special_tokens=True
                       )
    with torch.no_grad():
        outputs = bert(**tokens)

    return outputs.last_hidden_state[:, 0, :]

def get_sent_embeddings_cuda(sentences: List[str], tokenizer: BertTokenizer, bert: BertModel, device=None):

    # this gives the user the choice to use a device implicitly or explicitly
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert.to(device)
    bert.eval()

    tokens = tokenizer(sentences, 
                       return_tensors="pt", 
                       padding=True, 
                       truncation=True, 
                       add_special_tokens=True
                       ).to(device)
    with torch.no_grad():
        outputs = bert(**tokens)

    return outputs.last_hidden_state[:, 0, :]

def get_sent_seq_embeddings0(sentences: List[str], tokenizer: BertTokenizer, encoder: BertModel, max_words=None):

    if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
        emb_d = encoder.config.hidden_size
    elif hasattr(encoder, "hidden_size"):
        emb_d = encoder.hidden_size
    elif hasattr(encoder, "d_model"):
        emb_d = encoder.d_model
    else:
        raise AttributeError("Cannot determine embedding dimension from encoder/model.")

    result = []

    for sent in sentences:
        sent_words = sent.split()
        if (max_words is not None) and (len(sent_words) > max_words):
            sent_words = sent_words[:max_words]
    
        tokens = tokenizer(sent_words, 
                           return_tensors="pt", 
                           padding=True, 
                           truncation=True, 
                           add_special_tokens=True
                           )

        with torch.no_grad():
            outputs = encoder(**tokens)
        sent_enc = outputs.last_hidden_state[:, 0, :]
        if (max_words is not None) and (len(sent_words) < max_words):
            padding = torch.zeros((max_words - len(sent_words), emb_d))
            sent_enc = torch.concat([sent_enc, padding])
        
        result.append(sent_enc)

    return torch.stack(result)

def get_sent_seq_embeddings(sentences: List[str], tokenizer: BertTokenizer, encoder: BertModel, max_words=30):

    # if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
    #     emb_d = encoder.config.hidden_size
    # elif hasattr(encoder, "hidden_size"):
    #     emb_d = encoder.hidden_size
    # elif hasattr(encoder, "d_model"):
    #     emb_d = encoder.d_model
    # else:
    #     raise AttributeError("Cannot determine embedding dimension from encoder/model.")

    result = []

    for sent in sentences:
        sent_words = sent.split()
        l = len(sent_words)
        if max_words is not None and l != max_words:
            if l > max_words:
                sent_words = sent_words[:max_words]
            else:
                sent_words = sent_words + [""] * (max_words - l)
    
        tokens = tokenizer(sent_words, 
                           return_tensors="pt", 
                           padding=True, 
                           truncation=True, 
                           add_special_tokens=True
                           )

        with torch.no_grad():
            outputs = encoder(**tokens)
        sent_enc = outputs.last_hidden_state[:, 0, :]
        result.append(sent_enc)

    return torch.stack(result)


class AraBertTokenizer:
    def __init__(self, tokenizer, araberturl):
        self.tokenizer = tokenizer 
        self.araberturl = araberturl
        self.special_tokens_map = tokenizer.special_tokens_map

    def __call__(self, text, **kwds):
        text = arabert_preprocess(text, self.araberturl)
        return self.tokenizer(text, **kwds)
    
    def convert_ids_to_tokens(self, *args):
        return self.tokenizer.convert_ids_to_tokens(*args)
    

def load_model(url, pretokenized=False):
    if "chdzdt" in url:
        print("loading CHDZDT model ...")
        tokenizer, model = load_chdzdt_model(url)
    elif "canine" in url:
        print("loading Canine model ...")
        tokenizer, model = load_canine_model(url)
    else:
        print("loading BERT-like model ...")
        tokenizer, model = load_bertlike_model(url)

        if ("arabert" in url) and (not pretokenized): 
            print("adding AraBERT normalization ...")
            tokenizer = AraBertTokenizer(tokenizer, url)

    return tokenizer, model


# class DualTokenizer:
#     def __init__(self, char_tokenizer, word_tokenizer):
#         self.char_tokenizer = char_tokenizer
#         self.word_tokenizer = word_tokenizer

#     def __call__(self, text, **kwds):
#         return {
#             "char_tokens": self.char_tokenizer(text, **kwds), 
#             "word_tokens": self.word_tokenizer(text, **kwds)
#             }
    
# class DualEncoder:
#     def __init__(self, seq_encoder, seq_sent_encoder, bert_encoder):
#         self.seq_encoder = seq_encoder
#         self.bert_encoder = bert_encoder
#         self.seq_sent_encoder = seq_sent_encoder

#     def __call__(self, char_tokens, word_tokens):
#         seq_encoding = self.seq_encoder()

#         return self.tokenizer1(text, **kwds), self.tokenizer2(text, **kwds)


def get_embedding_size(encoder):
    if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
        emb_d = encoder.config.hidden_size
    elif hasattr(encoder, "hidden_size"):
        emb_d = encoder.hidden_size
    elif hasattr(encoder, "d_model"):
        emb_d = encoder.d_model
    else:
        raise AttributeError("Cannot determine embedding dimension from encoder/model.")
    return emb_d