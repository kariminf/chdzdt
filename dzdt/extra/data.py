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
    """
    Reads a tab-separated values (TSV) file containing word cluster data and returns it as a pandas DataFrame.

    Parameters:
        url (str): The file path or URL to the TSV file. The path can include '~' for the user's home directory.

    Returns:
        pd.DataFrame: A DataFrame with columns 'word' (as string) and 'cluster' (as integer).

    Notes:
        - Assumes the TSV file contains at least 'word' and 'cluster' columns.
        - Missing values are not treated as NaN.
    """
    url = os.path.expanduser(url)
    data = pd.read_csv(url, sep="\t", encoding="utf8", keep_default_na=False, na_values=[])
    data["word"] = data["word"].astype(str)
    data["cluster"] = data["cluster"].astype(int)
    return data

def get_csv_string_data(url: str, sep="\t") -> pd.DataFrame:
    """
    Reads a CSV file from the specified URL or file path and returns its contents as a pandas DataFrame,
    with all values converted to strings.

    Args:
        url (str): The path or URL to the CSV file. The path can include '~' for the home directory.
        sep (str, optional): The delimiter to use for separating columns. Defaults to tab ('\t').

    Returns:
        pd.DataFrame: A DataFrame containing the CSV data, with all values as strings.
    """
    url = os.path.expanduser(url)
    data = pd.read_csv(url, sep=sep, encoding="utf8", keep_default_na=False, na_values=[])
    data = data.astype(str)
    return data


def get_word_noisy_data(url: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Reads a tab-separated values (TSV) file containing word data and returns lists of words and their obfuscated forms.

    Args:
        url (str): Path to the TSV file. Can include '~' for user home directory.

    Returns:
        Tuple[List[str], List[str], List[str]]: 
            - List of words from the "word" column.
            - List of fixed obfuscated forms from the "obfus1fix" column.
            - List of variable obfuscated forms from the "obfus1var" column.

    Notes:
        - Assumes the file has columns: "word", "obfus1fix", and "obfus1var".
        - All values are read as strings.
        - Missing values are treated as empty strings.
    """
    url = os.path.expanduser(url)
    data = pd.read_csv(url, sep="\t", encoding="utf8", keep_default_na=False, na_values=[])
    data = data.astype(str)
    return (data["word"].tolist(), 
            data["obfus1fix"].tolist(), 
            data["obfus1var"].tolist())


def get_tagging_data(url: str, max_words=None) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Loads and processes tagging data from a file.
    Each line in the file should contain words and their corresponding tags, separated by a tab character.
    Words and tags are space-separated within their respective fields.
    Args:
        url (str): Path to the data file. Can include '~' for user home directory.
        max_words (int, optional): Maximum number of words per sentence. If specified, sentences longer than
            max_words are truncated, and shorter sentences are padded with empty strings for words and '<PAD>' for tags.
    Returns:
        Tuple[List[List[str]], List[List[str]]]: A tuple containing two lists:
            - List of sentences, where each sentence is a list of words.
            - List of tag sequences, where each sequence is a list of tags corresponding to the words.
    Notes:
        - Lines with mismatched number of words and tags are skipped.
        - Empty lines are ignored.
    """
    url = os.path.expanduser(url)
    X_words, Y_tags = [], []
    i = 1
    with open(url, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            words, tags = line.split("\t")
            words, tags = words.split(), tags.split()

            if len(words) != len(tags):
                continue
            
            if max_words is not None:
                l = len(words)
                if l > max_words:
                    words, tags = words[:max_words], tags[:max_words]
                else: # >=
                    l = max_words - l
                    words, tags = words + ([""] * l), tags + (["<PAD>"] * l)
            X_words.append(words)
            Y_tags.append(tags)
    return X_words, Y_tags
