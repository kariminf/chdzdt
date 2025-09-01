#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023-2024	Abdelkrime Aries <kariminfo0@gmail.com>
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
from typing import List
import os
import pandas as pd 
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.preprocess import NOTAG_NORMALIZER, TAG_NORMALIZER
from dzdt.tools.io import list_files 


# =============================================
#        Duplication filtering functions
# =============================================

def folder_vid_text_nodup(in_folder: str, out_folder: str, subset:List[str]=["text"], in_ext = ".csv", out_ext = ".csv"):
    """Filter duplicate rows from csv files in a folder.
    
    Args:
        in_folder (str): original CSV files' folder.
        out_folder (str): filtered CSV files' folder.
        subset (List[str], optional): list of similarity columns. Defaults to ["text"].
    """
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    for in_file in list_files(in_folder, suffix=in_ext):
        print("Deleting duplicates of ", in_file)
        data = pd.read_csv(os.path.join(in_folder, in_file), sep="\t")
        data.drop_duplicates(subset=subset, keep="last", inplace=True)
        data.to_csv(os.path.join(out_folder, in_file[:-len(in_ext)] + out_ext), sep="\t", index=False)


# =============================================
#         Keyword filtering functions
# =============================================

def folder_filter_keywords(in_folder: str, out_folder: str, kw_file: str, in_ext = ".csv", out_ext = ".csv"):
    """Filter CSV files in a given folder using some keywords stored in a file.

    Args:
        in_folder (str): original CSV files' folder.
        out_folder (str): filtered CSV files' folder.
        kw_file (str): keywords' file; a file containing each keyword in a line. A keyword can be a RE.
    """
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    with open(kw_file, "r") as f:
        kw = "|".join([l.rstrip("\n") for l in f.readlines() if len(l) > 2])
    kw = r"(" + kw + ")"

    rs = re.compile(kw)

    for in_file in list_files(in_folder, in_ext):
        print("Filtering ", in_file)
        data = pd.read_csv(os.path.join(in_folder, in_file), sep="\t")
        data["text"] = data["text"].astype(str)
        nokeep = data.apply(lambda row : bool(rs.search(row["text"])), axis = 1)
        data[~nokeep].to_csv(os.path.join(out_folder, in_file[:-len(in_ext)] + out_ext), sep="\t", index=False)

# =============================================
#            Files normalization
# =============================================

def folder_vid_text_norm(in_folder: str, out_folder: str, tag_norm=False, in_ext = ".csv", out_ext = ".csv"):
    """Filter duplicate rows from csv files in a folder.

    Args:
        in_folder (str): original CSV files' folder.
        out_folder (str): filtered CSV files' folder.
        tag_norm (bool, optional): normalize tags. Defaults to False.
    """
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    processor = TAG_NORMALIZER if tag_norm else NOTAG_NORMALIZER
    for in_file in list_files(in_folder, in_ext):
        print("Normalizing ", in_file)
        data = pd.read_csv(os.path.join(in_folder, in_file), sep="\t")
        data["text"] = data["text"].astype(str)
        data["text"] = data.apply(lambda row : processor.process([row["text"]])[0], axis = 1)
        data.to_csv(os.path.join(out_folder, in_file[:-len(in_ext)] + out_ext), sep="\t", index=False)

# =============================================
#         Index files joining functions
# =============================================

def idx_join(in_folder: str, out_folder: str, in_ext = ".csv", out_ext = ".csv"):

    res_data = pd.DataFrame({"vid": [], "lang_type": [], "polarity": [], "text": []})
    for in_file in list_files(in_folder, in_ext):
        print("Joining ", in_file)
        data = pd.read_csv(os.path.join(in_folder, in_file), sep="\t")
        data.drop(["channel", "url"], axis=1, inplace=True)
        data.rename(columns={"title": "text"}, inplace=True)
        data["vid"] = 0
        data["lang_type"] = ""
        data["polarity"] = ""
        data.reindex(["vid", "lang_type", "polarity", "text"])
        res_data = pd.concat([res_data, data]).reset_index(drop=True)

    res_data.to_csv(os.path.join(out_folder, "__IDX" + out_ext), sep="\t", index=False)

# =============================================
#         Combining functions
# =============================================

def youtube_combine(in_folder: str, out_file: str, ytb_cls="DZ", in_ext=".csv"):
    res_data = pd.DataFrame({})
    for in_file in list_files(in_folder, suffix=in_ext):
        print("Combining ", in_file)
        data = pd.read_csv(os.path.join(in_folder, in_file), sep="\t")
        data["text"] = data["text"].astype(str)
        res_data = pd.concat([res_data, data["text"]]).reset_index(drop=True)

    # res_data.rename(index={0: "text"})

    res_data["cls"] = ytb_cls

    res_data.columns = ["text", "cls"]
    
    res_data.to_csv(out_file, sep="\t", index=False)

# =============================================
#          Command line functions
# =============================================

def process_nodup(args):
    subset = ["text"]
    if args.v: subset=["vid", "text"]
    folder_vid_text_nodup(args.input, args.output, subset=subset, in_ext=args.i, out_ext=args.o)

def process_idxjoin(args):
    idx_join(args.input, args.output, in_ext=args.i, out_ext=args.o)

def process_kwfilter(args):
    folder_filter_keywords(args.input, args.output, args.keywords, in_ext=args.i, out_ext=args.o)

def process_normalize(args):
    if args.o is None: 
        args.o = "_normt.csv" if args.t else "_norm.csv"
    folder_vid_text_norm(args.input, args.output, tag_norm=args.t, in_ext=args.i, out_ext=args.o)

def process_combine(args):
    youtube_combine(args.input, args.output, ytb_cls=args.l, in_ext=args.i)

parser = argparse.ArgumentParser(description="prepare youtube dataset")
subparsers = parser.add_subparsers(help="choose preparing process", required=True)

parser_nodup = subparsers.add_parser("nodup", help="delete duplicate comments from each file")
parser_nodup.add_argument("-i", help="input files extension", default="_cmt.csv")
parser_nodup.add_argument("-o", help="output files extension", default="_nodup.csv")
parser_nodup.add_argument("-v", help="duplicates by video ID", default=False, action=argparse.BooleanOptionalAction)
parser_nodup.add_argument("input", help="input folder containing csv files (youtube comments)")
parser_nodup.add_argument("output", help="output folder text files")
parser_nodup.set_defaults(func=process_nodup)

parser_idxjoin = subparsers.add_parser("idxjoin", help="merge all youtube index files into one")
parser_idxjoin.add_argument("-i", help="input files extension", default="_idx.csv")
parser_idxjoin.add_argument("-o", help="output files extension", default="_nodup.csv")
parser_idxjoin.add_argument("input", help="input folder containing csv files (youtube videos captions)")
parser_idxjoin.add_argument("output", help="output folder text files")
parser_idxjoin.set_defaults(func=process_idxjoin)

parser_kwfilter = subparsers.add_parser("kwfilter", help="filter text using a set of keywords (regular expression)")
parser_kwfilter.add_argument("-i", help="input files extension", default="_nodup.csv")
parser_kwfilter.add_argument("-o", help="output files extension", default="_kw.csv")
parser_kwfilter.add_argument("input", help="input folder containing commnts files")
parser_kwfilter.add_argument("output", help="output folder for keyword-filtered text files")
parser_kwfilter.add_argument("keywords", help="file containing filtring keywords")
parser_kwfilter.set_defaults(func=process_kwfilter)

parser_normalize = subparsers.add_parser("normalize", help="normalize all files in the folder")
parser_normalize.add_argument("-i", help="input files extension", default="_kw.csv")
parser_normalize.add_argument("-o", help="output files extension", default=None)
parser_normalize.add_argument("-t", help="normalize tags", default=False, action=argparse.BooleanOptionalAction)
parser_normalize.add_argument("input", help="input folder containing commnts files")
parser_normalize.add_argument("output", help="output folder for normalizd text files")
parser_normalize.set_defaults(func=process_normalize)

parser_combine = subparsers.add_parser("combine", help="merge all files into one and add one column for a class")
parser_combine.add_argument("-i", help="input files extension", default="_norm.csv")
parser_combine.add_argument("-l", help="text label", default="DZ")
parser_combine.add_argument("input", help="input folder containing commnts files")
parser_combine.add_argument("output", help="output file")
parser_combine.set_defaults(func=process_combine)


if __name__ == "__main__":

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args.func(args)
