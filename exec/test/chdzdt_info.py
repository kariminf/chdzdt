#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2024 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2024	Abdelkrime Aries <kariminfo0@gmail.com>
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

LOC = os.path.expanduser("~/Data/DZDT/models/chdzdt_2x4x16_20it")

words = ["qaꞢwa", "gatꞢau", "plꞢy", "خدꞢت", "يꞢمل", "kittꞢn", "أكتوبر"]

print("loading characters tokenizer")
# char_tokenizer: CharTokenizer = CharTokenizer.load(os.path.join(LOC, "char_tokenizer.pkl"))

TOKEN_PATH = os.path.expanduser("~/Data/DZDT/hug_token_read.txt")
with open(TOKEN_PATH, "r") as f:
    for l in f:
        hugingface_token = l.rstrip("\n")
        break

print(hugingface_token)

char_tokenizer: CharTokenizer = CharTokenizer.from_pretrained("huggingface:kariminf/chdzdt", token=hugingface_token)

print("loading characters encoder")
# char_tokenizer_config()
# char_encoder = MLMLMBertModel.from_pretrained(LOC)
# word_tokenizer_config()

char_encoder = MLMLMBertModel.from_pretrained("huggingface:kariminf/chdzdt:chdzdt_1x2x16_20it", token=hugingface_token)

print("encoding ...")
char_tokens = char_tokenizer.encode_words(words, return_tensors="pt")
char_code = char_encoder(**char_tokens, return_dict=True)

print("codes=", char_code.keys(), char_code["last_hidden_state"].shape)

pred = char_code["prediction_logits"].argmax(dim=2)

pred2 = char_code["seq_labels_logits"]#.argmax(dim=1)

# row, col = np.unravel_index(np.argsort(char_code["prediction_logits"].detach().numpy().ravel()), char_code["prediction_logits"].detach().numpy().shape)

# pred = char_code["prediction_logits"][row]
# print(row)

sizes = [len(word) for word in words]

print("original=", words)
print("decoded=", char_tokenizer.decode_words(pred, sizes=sizes))

# print(char_code["seq_labels_logits"].sum(dim=1))
print(pred2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("char_encoder", count_parameters(char_encoder))
print("char_encoder.bert", count_parameters(char_encoder.bert))
