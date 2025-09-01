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
import re
from typing        import List, Union
import getopt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dzdt.preprocess import NOTAG_NORMALIZER, TAG_NORMALIZER
from dzdt.tools.io import read_filelines


def csv_line_quote_norm(line: str) -> str:
    if not re.search(r'["\n]', line): return line
    line = '"' + line.replace('"', '""') + '"'
    return line

def get_processor(label: Union[str, None], tagged: bool):
    if label is None:
        return lambda text: text
    processor = TAG_NORMALIZER if tagged else NOTAG_NORMALIZER
    return lambda text: csv_line_quote_norm(processor.process([text.rstrip("\n")])[0]) + f"\t{label}\n"

def tatoeba_extract(in_url: str, out_url:str, in_labels: str, label: str, tagged: bool):
    in_labels = in_labels.split(",")
    process = get_processor(label, tagged)
    with open(out_url, "w", encoding="utf8") as out_f:
        if label is not None: out_f.write("text\tlabel\n")
        for line in read_filelines(in_url):
            if line.count("\t") != 2: continue
            id, lang, text = line.split("\t")
            if lang not in in_labels: continue
            out_f.write(process(text))
    print("extraction finished")



# =============================================
#          Command line functions
# =============================================

def process_extract(args):
    tatoeba_extract(args.input, args.output, args.i, args.o, args.t)


parser = argparse.ArgumentParser(description="extract sentences from tatoeba by language labels")

parser.add_argument("-i", help="input labels separated by a comma", default="ber,kab")
parser.add_argument("-o", help="output label", default="BER")
parser.add_argument("-t", help="normalize tags", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("input", help="input csv sentences file")
parser.add_argument("output", help="output csv file containing just the selected labels")
parser.set_defaults(func=process_extract)


if __name__ == "__main__":

    argv = sys.argv[1:]

    argv = [
        "-i", "ber,kab",
        "-o", "BER",
        "-t",
        "/home/karim/Data/DZDT/collect/tatoeba/sentences.csv",
        "/home/karim/Data/DZDT/collect/tatoeba/ber.tatoeba_norm.csv"
    ]

    args = parser.parse_args(argv)
    # print(args)
    # parser.print_help()
    args.func(args)