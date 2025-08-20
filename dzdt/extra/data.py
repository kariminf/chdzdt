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

import os
from typing import Tuple, List
import pandas as pd

# =============================================
#          Data loading and processing
# =============================================

   

def get_word_cluster_data(url: str) -> pd.DataFrame:
    url = os.path.expanduser(url)
    data = pd.read_csv(url, sep="\t", encoding="utf8", keep_default_na=False, na_values=[])
    data["word"] = data["word"].astype(str)
    data["cluster"] = data["cluster"].astype(int)
    return data

def get_csv_string_data(url: str, sep="\t") -> pd.DataFrame:
    url = os.path.expanduser(url)
    data = pd.read_csv(url, sep=sep, encoding="utf8", keep_default_na=False, na_values=[])
    data = data.astype(str)
    return data


def get_word_noisy_data(url: str) -> Tuple[List[str], List[str], List[str]]:
    url = os.path.expanduser(url)
    data = pd.read_csv(url, sep="\t", encoding="utf8", keep_default_na=False, na_values=[])
    data = data.astype(str)
    return (data["word"].tolist(), 
            data["obfus1fix"].tolist(), 
            data["obfus1var"].tolist())


def get_tagging_data(url: str, max_words=None) -> Tuple[List[List[str]], List[List[str]]]:
    url = os.path.expanduser(url)
    X_words, Y_tags = [], []
    with open(url, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            words, tags = line.split("\t")
            words, tags = words.split(), tags.split()
            
            if max_words is not None:
                l = len(words)
                if l > max_words:
                    words, tags = words[:max_words], tags[:max_words]
                else: # >=
                    l = max_words - l
                    words, tags = words + (["<PAD>"] * l), tags + (["<PAD>"] * l)
            X_words.append(words)
            Y_tags.append(tags)
    return X_words, Y_tags
