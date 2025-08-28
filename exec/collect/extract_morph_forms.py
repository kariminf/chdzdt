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
import re
from typing        import List, Union
import getopt
import os
import sys
import sqlite3
import pandas as pd
from typing import Dict, Set

# =============================================

def get_sqlite3_data(url: str, table: str) -> pd.DataFrame:
    url = os.path.expanduser(url)
    conn = sqlite3.connect(url)
    Data = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return Data

# =============================================

def arramooz_extract_table(clusters: Dict[str, Set[str]], in_url: str, table: str):
    Data = get_sqlite3_data(in_url, table)
    for index, row in Data.iterrows():
        root, word = row["root"], row["unvocalized"]
        if root not in clusters:
            clusters[root] = set()
        if word != root:
            clusters[root].add(word)

def arramooz_extract_table_nounwordtype(clusters: Dict[str, Set[str]], in_url: str):
    """extract nouns by form (type)"""
    Data = get_sqlite3_data(in_url, "nouns")
    for index, row in Data.iterrows():
        wtype, word = row["wordtype"], row["unvocalized"]
        if wtype not in clusters:
            clusters[wtype] = set()
        clusters[wtype].add(word)

def morphynet_extract(in_url: str) -> Dict[str, Set[str]]:
    clusters = {}
    in_url = os.path.expanduser(in_url)
    Data = pd.read_csv(in_url, sep="\t", encoding="utf8", header=None, names=["sword", "tword", "spos", "tpos", "morph", "type"])
    for index, row in Data.iterrows():
        root, word = row["sword"], row["tword"]
        if root not in clusters:
            clusters[root] = set()
        if word != root:
            clusters[root].add(word)
    return clusters

def qutrub_extract(in_url: str) -> Dict[str, Set[str]]:
    clusters = {}
    in_url = os.path.expanduser(in_url)
    Data = pd.read_csv(in_url, sep="\t", encoding="utf8", header=None, names=["vbcnj", "vbcnjshakl", "pron", "tmp", "trans", "vb", "shakl"])
    # print(Data.columns)
    # exit(0)
    for index, row in Data.iterrows():
        root, word = row["vb"], row["vbcnj"]
        if root not in clusters:
            clusters[root] = set()
        if word != root:
            clusters[root].add(word)
    return clusters

def arramooz_extract(in_url: str, form: str = "all") -> Dict[str, Set[str]]:
    
    clusters = {}

    if form == "fnouns":
        print("extracting arramooz nouns by type ...")
        arramooz_extract_table_nounwordtype(clusters, in_url)
    else:
        if form in ["all", "verbs"]:
            print("extracting arramooz verbs ...")
            arramooz_extract_table(clusters, in_url, "verbs")
        
        if form in ["all", "nouns"]:
            print("extracting arramooz nouns ...")
            arramooz_extract_table(clusters, in_url, "nouns")

    return clusters

def extract(clusters: Dict[str, Set[str]], out_url:str, min_length: int = 2, form: str = "all"):

    save_key = form != "fnouns"

    print("saving ...")
    out_url = os.path.expanduser(out_url)
    i = -1
    with open(out_url, "w", encoding="utf8") as out_f:
        
        out_f.write("word\tcluster\n")
        for root, words in clusters.items():
            if (not root ) or (len(words) < min_length): continue
            i += 1
            if save_key:
                out_f.write(f"{root}\t{i}\n")
            for word in words:
                if not word: continue
                out_f.write(f"{word}\t{i}\n")

    print("extraction finished")



# =============================================
#          Command line functions
# =============================================

def process_extract(args):
    clusters = {}
    if args.t == "arramooz":
        clusters = arramooz_extract(args.input, form=args.f)
    elif args.t == "morphynet":
        clusters = morphynet_extract(args.input)
    elif args.t == "qutrub":
        clusters = qutrub_extract(args.input)
    else:
        print(f"unknown source type {args.t}")
        return
    
    extract(clusters, args.output, min_length=args.n, form=args.f)
    


parser = argparse.ArgumentParser(description="extract sentences from tatoeba by language labels")

parser.add_argument("-t", help="source type", default="arramooz", choices=["arramooz", "morphynet", "qutrub"])
parser.add_argument("-f", help="form", default="all", choices=["all", "nouns", "fnouns", "verbs"],)
parser.add_argument("-n", help="min number of cluster elements", default=1, type=int)
parser.add_argument("input", help="input the source file")
parser.add_argument("output", help="output csv file containing words and their clusters")
parser.set_defaults(func=process_extract)


if __name__ == "__main__":

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args.func(args)