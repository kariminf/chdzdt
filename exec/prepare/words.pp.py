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

import argparse
from pandas import DataFrame
from typing import List, Dict
import os
import pandas as pd 
import re
import sys
from itertools import combinations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.tools.const import MAX_WORD_SIZE
from dzdt.tools.io import list_files, list_subfolders

SCRIPT = os.path.basename(__file__)
WORDS_FILE_EXT = ".csv"
LINES_NBR = 300000
CHUNKSIZE = 100000


# =============================================
#        Words' extraction functions
# =============================================

def extract_words(data: DataFrame, words_dict: Dict[str, List[int]], labels_table: List[str]):

    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str)

    for index, row in data.iterrows():
        label  = row["label"]
        text = row["text"]
        if label is None:
            print("error in class name")
            continue
        if label not in labels_table:
            labels_table.append(label)
        for word in re.split(r" +", text):
            if len(word) > MAX_WORD_SIZE: continue
            if word not in words_dict:
                words_dict[word] = [label]
            elif label not in words_dict[word]:
                words_dict[word].append(label)

def extract_words_freq(data: DataFrame, words_dict: Dict[str, Dict[str, int]], labels_table: List[str]):

    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str)

    for index, row in data.iterrows():
        label  = row["label"]
        text = row["text"]
        if (label is None) or (not len(label)):
            print("error in class name")
            continue
        if label not in labels_table:
            labels_table.append(label)
        
        for word in re.split(r" +", text):
            if len(word) > MAX_WORD_SIZE: continue
            if word not in words_dict:
                words_dict[word] = {label: 0}
            elif label not in words_dict[word]:
                words_dict[word][label] = 0
            words_dict[word][label] += 1

def csv_extract_words(in_file: str, words_dict: Dict[str, List[int]], labels_table: List[str]):
    data = pd.read_csv(in_file, sep="\t")
    extract_words(data, words_dict, labels_table)


def csv_extract_words_chunks(in_file: str, words_dict: Dict[str, List[int]], labels_table: List[str], freq: bool = False):
    i=0
    for data in pd.read_csv(in_file, sep="\t", chunksize=CHUNKSIZE):
        i+= 1
        print(f"Processing chunk {i} ...")
        if freq:
            extract_words_freq(data, words_dict, labels_table)
        else:
            extract_words(data, words_dict, labels_table)


def words2dataframe(words_dict: Dict[str, List[int]], labels_table: List[str], freq: bool=False) -> DataFrame:
    data = {"word": []}
    labels_table = sorted(labels_table)
    for label in labels_table:
        data[label] = []

    for word, labels in words_dict.items():
        data["word"].append(word)
        for label in labels_table:
            if freq:
                if label in labels:
                    data[label].append(int(labels[label]))
                else:
                    data[label].append(0)
            else:
                data[label].append(int(label in labels))
    
    return DataFrame(data)


def folder_extract_words(in_folder: str, out_file: str, ext=None, folders=False, freq: bool=False):
    words_dict = {}
    labels_table = []
    if folders:
        for sub_url in list_subfolders(in_folder):
            sub_url = os.path.join(in_folder, sub_url)
            for in_file in list_files(sub_url, suffix=ext):
                print("extracting words from ", in_file)
                csv_extract_words_chunks(os.path.join(sub_url,in_file), words_dict, labels_table, freq=freq)
    else:
        for in_file in list_files(in_folder, suffix=ext):
            print("extracting words from ", in_file)
            csv_extract_words_chunks(os.path.join(in_folder,in_file), words_dict, labels_table, freq=freq)
    data = words2dataframe(words_dict, labels_table, freq=freq)
    data.to_csv(out_file, sep="\t", index=False)

# =============================================
#        Words' batch split functions
# =============================================

def split2batches(in_url: str, out_url: str, lines_nbr: int, shuffle=True):
    fname = os.path.basename(in_url)[:-4]
    data = pd.read_csv(in_url, sep='\t')
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    max_pos  = data.shape[0]
    count = 0 
    begin = 0

    while begin <  max_pos:
        print("preparing chunk ", count)
        end = begin + lines_nbr
        data.iloc[begin:end, :].to_csv(os.path.join(out_url, fname + str(count) + ".csv"), index=False, sep='\t')
        count += 1
        begin = end


def bin_onehot_to_dec(vec):
    base = 1
    res = 0
    for e in vec:
        res += base * e
        base = base * 2
    return res

def bin_idx_to_dec(idx):
    res = 0
    for i in idx:
        res += 2**i
    return res

def list_idx_to_str(l, idx):
    res = ""
    for i in idx:
        res += ("" if i == 0 else "-") + l[i]
    return res


#  TODO complete
def words_info(in_url: str, out_url: str):
    data = pd.read_csv(in_url, sep='\t')
    langs = data.columns[1:].tolist()
    idx_list = range(len(langs))
    hash_map = {}
    for i in idx_list:
        for cmb in combinations(idx_list, i):
            hash_map[list_idx_to_str(cmb)] = bin_idx_to_dec(cmb)

    data['hash'] = list(map(lambda l: bin_onehot_to_dec(l), data.iloc[:, 1:].values.tolist()))

    hashes = data['hash'].unique()
    counts = data['hash'].value_counts()

    with open(out_url, "w", encoding="utf8") as f:
        f.write("combination\tcount\n")
        for hash in hashes:
            

        




# =============================================
#          Command line functions
# =============================================

def process_extract(args):
    input = os.path.expanduser(args.input)
    output = os.path.expanduser(args.output)
    folder_extract_words(input, output, ext=args.i, folders=args.f, freq=args.q)

def process_split(args):
    input = os.path.expanduser(args.input)
    output = os.path.expanduser(args.output)
    split2batches(input, output, args.l, shuffle=args.s)

def process_info(args):
    input = os.path.expanduser(args.input)
    output = os.path.expanduser(args.output)
    words_info(input, output)


parser = argparse.ArgumentParser(description="prepare words dataset")
subparsers = parser.add_subparsers(help="choose preparing process", required=True)

parser_extract = subparsers.add_parser("extract", help="Extract words from csv files in a folder and save them into another")
parser_extract.add_argument("-f", help="the files are in subfolders", default=False, action=argparse.BooleanOptionalAction)
parser_extract.add_argument("-q", help="if True, the output will have frequencies", default=False, action=argparse.BooleanOptionalAction)
parser_extract.add_argument("-i", help="input file suffix (extension)", default=".csv")
parser_extract.add_argument("input", help="input folder containing csv sentences files")
parser_extract.add_argument("output", help="output csv file containing words")
parser_extract.set_defaults(func=process_extract)


parser_split = subparsers.add_parser("split", help="split a word file into multiple ones by specifying th number of lines")
parser_split.add_argument("-s", help="shuffle before splitting", default=False, action=argparse.BooleanOptionalAction)
parser_split.add_argument("-l", help="number of lines per file", type=int, default=LINES_NBR)
parser_split.add_argument("input", help="input csv file containing words")
parser_split.add_argument("output", help="output folder containing splitted files")
parser_split.set_defaults(func=process_split)

parser_info = subparsers.add_parser("info", help="Information about word dataset")
parser_info.add_argument("input", help="input csv file containing words")
parser_info.add_argument("output", help="output file containing the info")
parser_info.set_defaults(func=process_info)


if __name__ == "__main__":

    argv = sys.argv[1:]

    argv = [
        "extract",
        "-f",
        "-q",
        "-i", "_norm.csv",
        "~/Data/DZDT/data/1_all",
        "~/Data/DZDT/data/2_words/words_label_freq.csv"
    ]

    # argv = [
    #     "split",
    #     "-s",
    #     "-l", "300000",
    #     "~/Data/DZDT/data/2_words/words.csv",
    #     "~/Data/DZDT/data/2_words/chunks"
    # ]

    args = parser.parse_args(argv)
    # print(args)
    args.func(args)


 
