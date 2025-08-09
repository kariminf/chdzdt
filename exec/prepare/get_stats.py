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

import getopt
import os
import sys
import pandas as pd 
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
SCRIPT = os.path.basename(__file__)

from dzdt.tools.io import list_files 

lengths = {}

# =============================================
#          Youtube stats
# =============================================

def extract_stats(in_folder: str, out_file: str, label: str):
    with open(out_file, "w") as out_f:
        out_f.write(f"file\t{label}\n")
        for in_file in list_files(in_folder):
            print("extracting stats", in_file)
            data = pd.read_csv(os.path.join(in_folder, in_file), sep="\t")
            out_f.write(f"{in_file}\t{data.shape[0]}\n")


def combine_stats(in_folder: str, out_file: str):
    result = None
    for in_file in list_files(in_folder):
        print("Combining statistics ", in_file)
        data = pd.read_csv(os.path.join(in_folder, in_file), sep="\t")
        data["file"] = data["file"].astype("string")
        if result is None:
            result = data
        else:
            result = result.merge(data, how="outer", on=["file"])
    result.to_csv(out_file, sep="\t", index=False)

# =============================================
#          Command line functions
# =============================================

def help():
    print("=========== HELP! ===========")
    print("Preprocess wikipedia files")
    print("COMMAND:")
    print(f"\t {SCRIPT} [OPTIONS] [In-folder URL] [Out-folder URL]")

    print("OPTIONS:")

    print("\t -h \t \tShow this help")

    print("\t -p prepare \tPrepare wikipedia files: tokenuzation and normalization")
    print("\t \t \t-p prepare -l [language] [In-folder URL] [Out-folder URL]")
    print("\t \t \t[language] = arabic, english, french, etc.")
    print()
    print("\t -p combine \tCombine many files into one")
    print("\t \t \t-p combine -c [CLASS] [In-folder URL] [Out-file URL]")
    print("\t \t \t-p combine [In-folder URL] [Out-file URL]")


def main(argv: List[str]):

    process  = "extract"
    label     = "stats"

    opts, args = getopt.getopt(argv,"hp:l:")
    # print(opts, args)
    for opt, arg in opts:
        if   opt == "-h":
            help()
            sys.exit()
        elif opt == "-p":
            process = arg
        elif opt == "-l":
            label = arg

    if len(args) < 2:
        print("You have to afford an in-folder and an out-folder")
        help()
        exit(1)
    
    if process == "extract":
        extract_stats(args[0], args[1], label)
    elif process == "combine":
        combine_stats(args[0], args[1])
    else:
        print("Wrong command")
        help()

if __name__ == "__main__":
    # main(sys.argv[1:])

    # main([
    #     "-p", "extract",
    #     "-l", "idx",
    #     "/home/karim/Data/DZDT/collect/youtube/0_IDX_raw",
    #     "/home/karim/Data/DZDT/collect/youtube/6_stats/stats_idx.csv"
    # ])

    stats = {}

    d = False
    with open("/home/karim/Data/DZDT/data/2_words/words.csv") as f:
        for l in f:
            if d:
                ln = len(l.split("\t")[0])
                stats[ln] = stats.get(ln, 0) + 1
            d = True

    with open("/home/karim/Data/DZDT/data/2_words/words_stats.csv", "w") as f:
        for ln in stats:
            f.write(str(ln) + "\t" + str(stats[ln]) + "\n")


