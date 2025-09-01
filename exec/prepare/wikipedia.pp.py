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
from multiprocessing import cpu_count
from nltk.tokenize import sent_tokenize
from typing import List
from wikiextractor import WikiExtractor
from wikiextractor.WikiExtractor import process_dump
import getopt
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.preprocess import NOTAG_NORMALIZER, TAG_NORMALIZER
from dzdt.preprocess.filterer import filter_web_tags_in_line
from dzdt.tools.io import list_files, list_subfolders, read_filelines
from dzdt.tools.process import SeqPipeline


MIN_CHARS = 20
SKIP_RE = re.compile("^(</?doc|\[\[[^:]*:).*")
SCRIPT = os.path.basename(__file__)

WikiExtractor.expand_templates = True
WikiExtractor.HtmlFormatting = True
WikiExtractor.Extractor.keepLinks = True
WikiExtractor.Extractor.to_json = False
WikiExtractor.expand_templates = True

# =============================================
#          Preparing functions
# =============================================

def keep_line(line: str, min_chars:int= MIN_CHARS) -> bool:
    if len(line) < min_chars: return False
    if len(line.split(" ")) < 3: return False
    if line.startswith(" "): return False
    return True

def filter_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = re.sub(r"[_]+", " ", line)
    return filter_web_tags_in_line(line)[0]

def tokenize_line(line: str, language="english") -> List[str]:

    other = None
    if language == "arabic":
        language = "english"
        other = ("؟", "؟")

    result = sent_tokenize(line, language=language)

    if other:
        res = []
        exp = "([^{0}]+(?:{1}|$)) ?".format(other[0], other[1])
        for s in result:
            res.extend(re.findall(exp, s))
        result = res

    return result

def wiki_prepare(line: str, language="english") -> str:
    result = ""
    lines = tokenize_line(re.sub(r"\n{2,}", r"\n", filter_line(line)),  language=language)
    for line in lines:
        if keep_line(line): result += line + "\n"
    return result

def wiki_file_prepare(in_file: str, out_file: str, language="english"):
    with open(out_file, "w", encoding="utf8") as out_f:
        for line in read_filelines(in_file):
            out_f.write(wiki_prepare(line, language=language))

def wiki_folder_prepare(in_folder: str, out_folder: str, out_ext:str = "_pp.txt", language="english"):
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    for sub_folder in list_subfolders(in_folder):
        sub_path = os.path.join(in_folder, sub_folder)
        out_path = os.path.join(out_folder, sub_folder + "_")
        for in_file in list_files(sub_path):
            print("Preparing ", in_file)
            wiki_file_prepare(os.path.join(sub_path, in_file), out_path + in_file + out_ext, language=language)

# =============================================
#          Normalization functions
# =============================================

def wiki_normalize(line: str, processor: SeqPipeline) -> str:
    line = processor.process([line.rstrip("\n") + " "])[0]
    # line = re.sub(r"(\.+)$", r" \1", line)
    # line = re.sub(r"^[-\.]+", r"", line)
    return line + "\n"

def wiki_file_normalize(in_file: str, out_file: str, processor: SeqPipeline):
    with open(out_file, "w", encoding="utf8") as out_f:
        for line in read_filelines(in_file):
            out_f.write(wiki_normalize(line, processor))

def wiki_folder_normalize(in_folder: str, out_folder: str, in_ext:str = "_pp.txt", out_ext:str = "_norm.txt", tag_norm=False):
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    # processor = NORMALIZER2 if tag_norm else NORMALIZER1
    processor = TAG_NORMALIZER if tag_norm else NOTAG_NORMALIZER
    for in_file in list_files(in_folder, suffix=in_ext):
        print("Normalizing ", in_file)
        out_file = os.path.join(out_folder, in_file[:-len(in_ext)] + out_ext) 
        in_file = os.path.join(in_folder, in_file)
        wiki_file_normalize(in_file, out_file, processor)

# =============================================
#          Combining functions
# =============================================

def csv_line_quote_norm(line: str) -> str:
    if not re.search(r'["\n]', line): return line
    line = '"' + line.replace('"', '""') + '"'
    return line


def wiki_folder_combine(in_folder: str, out_file: str, text_cls: str=None, in_ext:str = ".txt"):
    out_plus = f"\n" if text_cls is None else f"\t{text_cls}\n"
    quote_norm = csv_line_quote_norm if out_file.endswith(".csv") else lambda x: x
    with open(out_file, "w", encoding="utf8") as out_f:
        if text_cls is not None: out_f.write("text\tlabel\n")
        for in_file in list_files(in_folder, suffix=in_ext):
            print("Combining ", in_file)
            for line in read_filelines(os.path.join(in_folder, in_file)):
                line = quote_norm(line.strip("\n").replace("\t", " "))
                out_f.write(line + out_plus)

# =============================================
#          Command line functions
# =============================================

def process_extract(args):
    process_dump(args.input, None, args.output, 1024**2, False, cpu_count() - 1, None)

def process_prepare(args):
    wiki_folder_prepare(args.input, args.output, language=args.l)

def process_normalize(args):
    wiki_folder_normalize(args.input, args.output, out_ext=args.o, tag_norm=args.t)

def process_combine(args):
    wiki_folder_combine(args.input, args.output, text_cls=args.l, in_ext=args.i)


parser = argparse.ArgumentParser(description="prepare wikipedia dataset")
subparsers = parser.add_subparsers(help="choose preparing process", required=True)

parser_extract = subparsers.add_parser("extract", help="extract text from wikipedia files")
parser_extract.add_argument("input", help="input wikipedia dump")
parser_extract.add_argument("output", help="output extracted text files")
parser_extract.set_defaults(func=process_extract)

parser_prepare = subparsers.add_parser("prepare", help="prepare wikipedia files: tokenization and filtering tags")
parser_prepare.add_argument("-l", help="language for nltk toknizer", default="english")
parser_prepare.add_argument("input", help="input folder containing text files")
parser_prepare.add_argument("output", help="output folder containing prepared files")
parser_prepare.set_defaults(func=process_prepare)

parser_normalize = subparsers.add_parser("normalize", help="normalize texts")
parser_normalize.add_argument("-o", help="output files extension", default="_norm.txt")
parser_normalize.add_argument("-t", help="normalize tags", default=False, action=argparse.BooleanOptionalAction)
parser_normalize.add_argument("input", help="input folder containing text files")
parser_normalize.add_argument("output", help="output folder normalized text files")
parser_normalize.set_defaults(func=process_normalize)


parser_combine = subparsers.add_parser("combine", help="normalize texts")
parser_combine.add_argument("-l", help="label of the text", default=None)
parser_combine.add_argument("-i", help="input files extension", default="_norm.txt")
parser_combine.add_argument("input", help="input folder containing text files")
parser_combine.add_argument("output", help="output folder normalized text files")
parser_combine.set_defaults(func=process_combine)

if __name__ == "__main__":

    argv = sys.argv[1:]

    L = "eng"
    C = "EN"
    W = "/home/karim/Data/DZDT/collect/wikipedia"

    # argv = [
    #     "extract",
    #     f"{W}/1_raw_xml/kabwiki.xml.bz2", #frawiki.tar.bz2",
    #     f"{W}/2_text/{L}wiki_text"
    # ]

    # argv = [
    #     "prepare",
    #     "-l", "french",
    #     f"{W}/2_text/{L}wiki_text",
    #     f"{W}/3_pp/{L}wiki_pp"
    # ]

    # argv = [
    #     "normalize",
    #     "-o", "_norm.txt",
    #     # "-t",
    #     f"{W}/3_pp/{L}wiki_pp",
    #     f"{W}/4_norm/{L}wiki_norm",
    # ]


    argv = [
        "combine",
        "-i", "_norm.txt",
        "-l", f"{C}",
        f"{W}/4_norm/{L}wiki_norm",
        f"{W}/{L}.wiki_norm.csv",
    ]

    args = parser.parse_args(argv)
    args.func(args)
