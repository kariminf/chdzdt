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
    data = pd.read_csv(url, sep="\t", encoding="utf8")
    data["word"] = data["word"].astype(str)
    data["cluster"] = data["cluster"].astype(int)
    return data

def get_csv_string_data(url: str, sep="\t") -> pd.DataFrame:
    url = os.path.expanduser(url)
    data = pd.read_csv(url, sep=sep, encoding="utf8")
    data = data.astype(str)
    return data


def get_word_noisy_data(url: str) -> Tuple[List[str], List[str], List[str]]:
    url = os.path.expanduser(url)
    data = pd.read_csv(url, sep="\t", encoding="utf8")
    data = data.astype(str)
    return (data["word"].tolist(), 
            data["obfus1fix"].tolist(), 
            data["obfus1var"].tolist())