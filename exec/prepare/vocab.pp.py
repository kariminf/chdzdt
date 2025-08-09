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

from typing import Dict, Iterable, List, Tuple
import argparse
import os
import pandas as pd 
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dzdt.preprocess import LOWER_NORMALIZER
from dzdt.preprocess.filterer import RegexFilterer
from dzdt.preprocess.normalizer import ENTITY_WORD_NORMALIZER
from dzdt.preprocess.stemmer import DzDtStemmer
from dzdt.preprocess.tokenizer import RegexSplitter
from dzdt.tools.chars import ARABIC_TASHKIIL_CHARSET, ARABIC_TATWEEL_ORD, charset2string
from dzdt.tools.const import TAG_LIST, MAX_WORD_SIZE
from dzdt.tools.io import list_files
from dzdt.tools.process import SeqPipeline

CHUNKSIZE = 100000


sent_regex_filterer = RegexFilterer([
    r"[" + charset2string(ARABIC_TASHKIIL_CHARSET) + chr(ARABIC_TATWEEL_ORD) + "]"
])

sent_regex_splitter = RegexSplitter()


TAG_NORMSPLIT = SeqPipeline([
    LOWER_NORMALIZER,
    sent_regex_filterer,
    sent_regex_splitter,
    ENTITY_WORD_NORMALIZER,
        # SPACE_JOINER,
])


# =============================================
#        Affixes' functions
# =============================================

def file2list(url: str) -> List[str]:
    result = set()
    with open(url, "r") as f:
        for l in f:
            l = l.rstrip("\n")
            if l and not l.startswith("#"): result.add(l)
    return list(result)

def prepare_affixes(in_url: str, out_url: str):
    prefixes = set()
    suffixes = set()
    for file in list_files(in_url, suffix="_prefix.txt"):
        print("processing prefix", file)
        prefixes.update(file2list(os.path.join(in_url, file)))

    for file in list_files(in_url, suffix="_suffix.txt"):
        print("processing suffix", file)
        suffixes.update(file2list(os.path.join(in_url, file)))

    with open(os.path.join(out_url, "prefix.txt"), "w") as f:
        f.write("\n".join(sorted(prefixes)))
    with open(os.path.join(out_url, "suffix.txt"), "w") as f:
        f.write("\n".join(sorted(suffixes)))


def load_affixes(url: str) -> Tuple[List[str], List[str]]:

    pref_url = os.path.join(url, "prefix.txt")
    suff_url = os.path.join(url, "suffix.txt")
    if not os.path.isfile(pref_url):
        raise ValueError(
            f"Can't find a prefix file at path '{pref_url}'. To load the prefix from a pretrained"
            " model use `tokenizer = DzDtTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
    if not os.path.isfile(suff_url):
        raise ValueError(
            f"Can't find a suffix file at path '{suff_url}'. To load the suffix from a pretrained"
            " model use `tokenizer = DzDtTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
        )
        
    return file2list(pref_url), file2list(suff_url)
    

# =============================================
#        Frequency functions
# =============================================

class VocabTokenizer:
    def __init__(self, stemmer: DzDtStemmer, norm=False) -> None:
        self.stemmer = stemmer
        self.tokenize = self.__tokenize_simple
        if norm:
            self.tokenize = self.__tokenize_norm
    
    def __tokenize_simple(self, text: str) -> Iterable[Tuple[str, str, str]]:
        for token in re.split("[ \t]+", text):
            if len(token) > MAX_WORD_SIZE: continue
            yield self.stemmer.stem(token) 

    def __tokenize_norm(self, text: str) -> Iterable[Tuple[str, str, str]]:
        for token in TAG_NORMSPLIT.process([text]):
            if len(token) > MAX_WORD_SIZE: continue
            if token in TAG_LIST:
                yield "", token, ""
            prefix, root, suffix = self.stemmer.stem(token)
            yield prefix, ENTITY_WORD_NORMALIZER.next(root)[0], suffix


def extract_freq(data: pd.DataFrame, 
                 freq: Dict[str, Dict[str, int]],  
                 tokenizer: VocabTokenizer):
    
    data["text"] = data["text"].astype(str)
    for index, row in data.iterrows():
        text = row["text"]
        for pref, root, suff in tokenizer.tokenize(text):
            if pref:
                freq["pref"][pref] = freq["pref"].get(pref, 0) + 1
            if root:
                freq["root"][root] = freq["root"].get(root, 0) + 1
            if suff:
                freq["suff"][suff] = freq["suff"].get(suff, 0) + 1


def file_extract_freq(in_url: str, freq: Dict[str, Dict[str, int]], tokenizer: VocabTokenizer, chunksize=None):

    if chunksize is None or chunksize < 1:
        extract_freq(pd.read_csv(in_url, sep="\t"), freq, tokenizer)
    else:
        i=0
        for data in pd.read_csv(in_url, sep="\t", chunksize=CHUNKSIZE):
            i+= 1
            print(f"Processing chunk {i} ...")
            extract_freq(data, freq, tokenizer)
    

def freq2dataframe(token_freq: Dict[str, int]) -> pd.DataFrame:
    data = {"token": [], "freq": []}
    for token, freq in sorted(token_freq.items()):
        data["token"].append(token)
        data["freq"].append(freq)
    
    return pd.DataFrame(data)


def folder_extract_freq(in_url: str, out_url:str, affix_url: str = None, in_ext=".csv", out_pref = "", norm=False, chunksize=None):
    if affix_url is None: affix_url = in_url
    tokenizer = VocabTokenizer(DzDtStemmer(*load_affixes(affix_url)), norm=norm)
    freq = {
        "root": {},
        "pref": {},
        "suff": {},
    }

    for file in list_files(in_url, suffix=in_ext):
        print("processing file", file)
        # prefix = file[:-len(in_ext)]
        file_extract_freq(os.path.join(in_url, file), freq, tokenizer, chunksize=chunksize)

    if not out_pref: out_pref = os.path.basename(os.path.dirname(os.path.join(in_url, "")))
    freq2dataframe(freq["pref"]).to_csv(os.path.join(out_url, out_pref + "_pref.tf.csv"), sep="\t", index=False)
    freq2dataframe(freq["root"]).to_csv(os.path.join(out_url, out_pref + "_root.tf.csv"), sep="\t", index=False)
    freq2dataframe(freq["suff"]).to_csv(os.path.join(out_url, out_pref + "_suff.tf.csv"), sep="\t", index=False)


# =============================================
#        extract functions
# =============================================

def file_reduce_vocab(in_url: str, out_url: str, min_freq: int = 1):
    data = pd.read_csv(in_url, sep="\t")
    data = data[data["freq"]>=min_freq]
    data.to_csv(out_url, sep="\t", index=False)

# =============================================
#          Joining functions
# =============================================

def join_affix(urls: List[str], out_url: str):
    token_freq = {}
    for url in urls:
        data = pd.read_csv(url, sep="\t")
        for index, row in data.iterrows():
            token_freq[row["token"]] = token_freq.get(row["token"], 0) + row["freq"]
    
    freq2dataframe(token_freq).to_csv(out_url, sep="\t", index=False)
            

def folder_combine_stats(in_url: str, out_url: str, suffix=".csv"):
    token_freq = {}
    for file in list_files(in_url, suffix=suffix):
        print("combining file", file)
        data = pd.read_csv(os.path.join(in_url, file), sep="\t")
        data["token"] = data["token"].astype(str)
        data["freq"] = data["freq"].astype(int)
        for index, row in data.iterrows():
            token_freq[row["token"]] = token_freq.get(row["token"], 0) + row["freq"]
    freq2dataframe(token_freq).to_csv(out_url, sep="\t", index=False)



    # join_affix([os.path.join(url, f) for f in list_files(url, suffix="_pref.tf.csv")], 
    #            os.path.join(url, "pref.tf.csv")
    #            )
    # join_affix([os.path.join(url, f) for f in list_files(url, suffix="_suff.tf.csv")], 
    #            os.path.join(url, "suff.tf.csv")
    #            )

def file_extract_vocab(in_url: str, out_url: str):
    data = pd.read_csv(in_url, sep="\t")
    data["token"] = data["token"].astype(str)
    with open(out_url, "w") as f:
        f.write("\n".join(data["token"].tolist()))

# =============================================
#          Command line functions
# =============================================

def process_affix(args):
    prepare_affixes(args.input, args.output)

def process_freq(args):
    folder_extract_freq(args.input, args.output, affix_url=args.a, in_ext=args.i, norm=args.n, chunksize=args.c)

def process_reduce(args):
    file_reduce_vocab(args.input, args.output, min_freq=args.m)

def process_combine(args):
    folder_combine_stats(args.input, args.output, suffix=args.i)

def process_vocab(args):
    file_extract_vocab(args.input, args.output)

parser = argparse.ArgumentParser(description="prepare vocabulary")
subparsers = parser.add_subparsers(help="choose preparing process", required=True)

parser_affix = subparsers.add_parser("affix", help="Combine affix files into one without duplicates")
parser_affix.add_argument("input", help="input folder containing *._suffix.txt and *._prefix.txt files")
parser_affix.add_argument("output", help="output folder to store suffix.txt and prefix.txt files")
parser_affix.set_defaults(func=process_affix)

parser_freq = subparsers.add_parser("freq", help="get prefix, root, suffix frequencies")
parser_freq.add_argument("-a", help="folder containing prefix and suffix files; if None input will be used", default=None)
parser_freq.add_argument("-i", help="input files extension", default=".csv")
# parser_freq.add_argument("-o", help="output files name prefixe", default=".csv")
parser_freq.add_argument("-n", help="normalized tokens", default=False, action=argparse.BooleanOptionalAction)
parser_freq.add_argument("-c", help="chunksize", type=int, default=None)
parser_freq.add_argument("input", help="input folder containing sentence files")
parser_freq.add_argument("output", help="output folder containing frequency files")
parser_freq.set_defaults(func=process_freq)

parser_reduce = subparsers.add_parser("reduce", help="extract most frequent roots reducing vocabulary")
parser_reduce.add_argument("-m", help="minimum frequency", type=int, default=1)
parser_reduce.add_argument("input", help="input file")
parser_reduce.add_argument("output", help="output file")
parser_reduce.set_defaults(func=process_reduce)

parser_combine= subparsers.add_parser("combine", help="combine stats")
parser_combine.add_argument("-i", help="input files extension", default=".csv")
parser_combine.add_argument("input", help="input folder")
parser_combine.add_argument("output", help="output file")
parser_combine.set_defaults(func=process_combine)

parser_vocab= subparsers.add_parser("vocab", help="extract vocabulary from stats")
parser_vocab.add_argument("input", help="input file")
parser_vocab.add_argument("output", help="output file")
parser_vocab.set_defaults(func=process_vocab)

if __name__ == "__main__":
    argv = sys.argv[1:]

    dir = os.path.expanduser("~/Data/DZDT")
    # argv = [
    #     "affix",
    #     os.path.join(dir, "collect/affix"),
    #     os.path.join(dir, "data/3_vocab")
    # ]

    # L = "ber"

    # argv = [
    #     "freq",
    #     "-a", os.path.join(dir, "data/3_vocab"),
    #     "-n",
    #     "-i", "_norm.csv",
    #     "-c", str(CHUNKSIZE),
    #     os.path.join(dir, "data/1_all/dz"),
    #     os.path.join(dir, "data/3_vocab")
    # ]

    # argv = [
    #     "reduce",
    #     "-m", "50",
    #     # os.path.join(dir, f"data/3_vocab/{L}_root.tf.csv"),
    #     # os.path.join(dir, f"data/3_vocab/{L}_vocab_reduced.csv")
    #     os.path.join(dir, f"data/3_vocab/suff.csv"),
    #     os.path.join(dir, f"data/3_vocab/suff_reduced.csv")
    # ]

    # argv = [
    #     "combine",
    #     "-i", "_suff.tf.csv",
    #     os.path.join(dir, "data/3_vocab"),
    #     os.path.join(dir, "data/3_vocab/suff.csv")
    # ]

    argv = [
        "vocab",
        os.path.join(dir, "data/3_vocab/reduced/vocab_reduced.csv"),
        os.path.join(dir, "data/3_vocab/final/vocab.txt")
    ]

    args = parser.parse_args(argv)
    args.func(args)
