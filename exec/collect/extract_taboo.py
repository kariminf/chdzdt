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
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.tools.sec import decode_csv, encode_csv

SUB = {
    'i': ['1', '!'],
    'a': ['@', '4'],
    's': ['$', '5'],
    'o': ['0'],
    'e': ['3'],
    't': ['7', '+']
    }

SUBDZ = {
    # 'j': ['dj'],
    '7': ['h'],
    'kh': ["7'", '5'],
    'ch': ['sh'],
    's': ['9'],
    't': ['6'],
    'dh': ["6'"],
    'gh': ["3'"],
    'q': ['g'],
    'ou': ['o', 'oo'],
    'y': ['i'],
}

# =============================================

def extract(clusters: Dict[str, Set[str]], out_url:str, min_length: int = 2, form: str = "all"):

    save_key = form != "fnouns"

    print("saving ...")
    out_url = os.path.expanduser(out_url)
    i = -1
    out_f2 = open(out_url.replace(".csv", "_unic.csv"), "w", encoding="utf8")
    with open(out_url, "w", encoding="utf8") as out_f:
        
        out_f.write("word\tcluster\n")
        for root, words in clusters.items():
            if (not root ) or (len(words) < min_length): continue
            i += 1
            out_f2.write(f"{root}\n")
            if save_key:
                out_f.write(f"{root}\t{i}\n")
            for word in words:
                if not word: continue
                out_f.write(f"{word}\t{i}\n")
    out_f2.close()
    print("extraction finished")


def deobfuscate_latin_word(word: str) -> str:
    word = word.lower() 
    word = re.sub(r'[1!î]', 'i', word) 
    word = re.sub(r'[@â4]', 'a', word) 
    word = re.sub(r'[$5]', 's', word)
    word = re.sub(r'[0]', 'o', word)
    word = re.sub(r'[3éè]', 'e', word)
    word = re.sub(r'[7+]', 't', word)
    return word.strip()

def txtlist_extract(in_url: str, deobfus: bool= True) -> Dict[str, Set[str]]:
    clusters = {}
    in_url = os.path.expanduser(in_url)

    with open(in_url, "r", encoding="utf8") as f:
        for obfus in f:
            obfus = obfus.strip()
            if not obfus: continue
            word = obfus
            if deobfus:
                word = deobfuscate_latin_word(obfus)
            if word not in clusters:
                clusters[word] = set()
            if word != obfus:
                clusters[word].add(obfus)
     
    return clusters


def generate_variants(word):
        # For each character, get possible substitutions (including itself)
        chars = []
        for c in word:
            if c in SUB:
                chars.append([c] + SUB[c])
            else:
                chars.append([c])

        # Generate all combinations except the original word
        variants = set()
        for combo in product(*chars):
            variant = ''.join(combo)
            if "5" in variant and "$" in variant:
                continue
            if "1" in variant and "!" in variant:
                continue
            if "7" in variant and "+" in variant:
                continue
            if "4" in variant and "@" in variant:
                continue

            if variant != word:
                variants.add(variant)
        return variants

def augment_latin(clusters: Dict[str, Set[str]]) -> None:
    for word, noise in clusters.items():
        variants = generate_variants(word)
        noise.update(variants)

def generate_dz_variants(word: str) -> Set[str]:
    variants = set()
    for c, subs in SUBDZ.items():
        if c in word:
            for sub in subs:
                # replace only the first occurrence of c
                variant = word.replace(c, sub, 1)
                variants.add(variant)
    return variants

def augment_dz(clusters: Dict[str, Set[str]]) -> None:
    
    for word, noise in clusters.items():
        noise.update(generate_dz_variants(word))

    for word, noise in clusters.items():
        changed = True
        while changed:
            added = set()
            changed = False
            for noisyword in noise:
                if noisyword == word: continue
                # generate variants for the noisy word
                variants = generate_dz_variants(noisyword)
                added = variants - noise  
                noise.update(added)
                changed = changed or bool(added)


# =============================================
#          Command line functions
# =============================================

def process_extract(args):

    clusters = txtlist_extract(args.input, deobfus=args.d) 

    if args.a in ["fr", "en"]:
        augment_latin(clusters)
    elif args.a == "dz":
        augment_dz(clusters)

    extract(clusters, args.output, min_length=args.n, form="all")

def process_extract(args):

    clusters = txtlist_extract(args.input, deobfus=args.d) 

    if args.a in ["fr", "en"]:
        augment_latin(clusters)
    elif args.a == "dz":
        augment_dz(clusters)

    extract(clusters, args.output, min_length=args.n, form="all")

def process_protect(args):
    url = os.path.expanduser(args.input)
    if args.e:
        encode_csv(url, url.replace(".csv", "_enc.csv"))
    else:
        decode_csv(url, url.replace(".csv", "_dec.csv"))



parser = argparse.ArgumentParser(description="extract taboo/obfuscated lists ")
subparsers = parser.add_subparsers(help="choose preparing process", required=True)

parser_extract = subparsers.add_parser("extract", help="Extract clusters of obfuscations")
parser_extract.add_argument("-d", help="deobfuscate: if the input contains obfuscations", default=False, action=argparse.BooleanOptionalAction)
parser_extract.add_argument("-a", help="augment data language", default=None, choices=["ar", "fr", "en", "dz"])
parser_extract.add_argument("-n", help="min number of cluster elements", default=1, type=int)
parser_extract.add_argument("input", help="input the source file containing obfuscated words")
parser_extract.add_argument("output", help="output csv file containing words and their clusters")
parser_extract.set_defaults(func=process_extract)

parser_protect = subparsers.add_parser("protect", help="Encode/Decode csv files")
parser_protect.add_argument("-e", help="encode the file; if absent it will decode", default=False, action=argparse.BooleanOptionalAction)
parser_protect.add_argument("input", help="input the source file containing obfuscated words")
parser_protect.set_defaults(func=process_protect)



if __name__ == "__main__":

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args.func(args)