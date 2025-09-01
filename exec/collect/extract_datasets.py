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

import re
import argparse
import json
import csv
import os
import sys
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit



UNIMORPH_LABELS = {
    "0": "person", 
    "1": "person", 
    "2": "person", 
    "3": "person", 
    "4": "person", 
    "1DAY": "tense", 
    "AB": "comparison", 
    "ABL": "case", 
    "ABS": "case", 
    "ABV": "deixis", 
    "ACC": "case", 
    "ACCMP": "aktionsart", 
    "ACFOC": "voice", 
    "ACH": "aktionsart", 
    "ACT": "voice", 
    "ACTY": "aktionsart", 
    "ADJ": "part of speech", 
    "ADM": "mood", 
    "ADP": "part of speech", 
    "ADV": "part of speech", 
    "AGFOC": "voice", 
    "ALL": "case", 
    "ALN": "possession", 
    "ANIM": "animacy", 
    "ANTE": "case", 
    "ANTIP": "voice", 
    "APPL": "valency", 
    "APPRX": "case", 
    "APUD": "case", 
    "ARGAC3S": "argument marking", 
    "ART": "part of speech", 
    "ASSUM": "evidentiality", 
    "AT": "case", 
    "ATEL": "aktionsart", 
    "AUD": "evidentiality", 
    "AUNPRP": "mood", 
    "AUPRP": "mood", 
    "AUX": "part of speech", 
    "AVOID": "politeness", 
    "AVR": "case", 
    "BANTU1-23": "gender", 
    "BEL": "deixis", 
    "BEN": "case", 
    "BFOC": "voice", 
    "BYWAY": "case", 
    "CAUS": "valency", 
    "CFOC": "voice", 
    "CIRC": "case", 
    "CLF": "part of speech", 
    "CMPR": "comparison", 
    "CN R MN": "switch-reference", 
    "COL": "politeness", 
    "COM": "case", 
    "COMP": "part of speech", 
    "COMPV": "case", 
    "COND": "mood", 
    "CONJ": "part of speech", 
    "DAT": "case", 
    "DEB": "mood", 
    "DECL": "interrogativity", 
    "DED": "mood", 
    "DEF": "definiteness", 
    "DET": "part of speech", 
    "DIR": "voice", 
    "DITR": "valency", 
    "DRCT": "evidentiality", 
    "DS": "switch-reference", 
    "DSADV": "switch-reference", 
    "DU": "number", 
    "DUR": "aktionsart", 
    "DYN": "aktionsart", 
    "ELEV": "politeness", 
    "EQT": "comparison", 
    "EQTV": "case", 
    "ERG": "case", 
    "ESS": "case", 
    "EVEN": "deixis", 
    "EXCL": "person", 
    "FEM": "gender", 
    "FH": "evidentiality", 
    "FIN": "finiteness", 
    "FOC": "information structure", 
    "FOREG": "politeness", 
    "FORM": "politeness", 
    "FRML": "case", 
    "FUT": "tense", 
    "GEN": "case", 
    "GPAUC": "number", 
    "GRPL": "number", 
    "HAB": "aspect", 
    "HIGH": "politeness", 
    "HOD": "tense", 
    "HRSY": "evidentiality", 
    "HUM": "animacy", 
    "HUMB": "politeness", 
    "IFOC": "voice", 
    "IMMED": "tense", 
    "IMP": "mood", 
    "IMP+SBJV": "mood",
    "IMPRS": "valency", 
    "IN": "case", 
    "INAN": "animacy", 
    "INCL": "person", 
    "IND": "mood", 
    "INDF": "definiteness", 
    "INFER": "evidentiality", 
    "INFM": "politeness", 
    "INS": "case", 
    "INT": "interrogativity", 
    "INTEN": "mood", 
    "INTER": "case", 
    "INTJ": "part of speech", 
    "INTR": "valency", 
    "INV": "voice", 
    "INVN": "number", 
    "IPFV": "aspect", 
    "IRR": "mood", 
    "ITER": "aspect",
    "JUS": "mood", 
    "LFOC": "voice", 
    "LGSPEC1": "language-specific features", 
    "LGSPEC2": "language-specific features", 
    "LIT": "politeness", 
    "LKLY": "mood", 
    "LOG": "switch-reference", 
    "LOW": "politeness", 
    "MASC": "gender", 
    "MED": "deixis", 
    "MID": "voice", 
    "N": "part of speech", 
    "NAKH1-8": "gender", 
    "NALN": "possession", 
    "NEG": "polarity", 
    "NEUT": "gender", 
    "NFH": "evidentiality", 
    "NFIN": "finiteness", 
    "NHUM": "animacy", 
    "NOM": "case", 
    "NOMS": "case", 
    "NOREF": "deixis", 
    "NSPEC": "definiteness", 
    "NUM": "part of speech", 
    "NVIS": "deixis", 
    "NVSEN": "evidentiality", 
    "OBLIG": "mood", 
    "OBV": "person", 
    "ON": "case", 
    "ONHR": "case", 
    "ONVR": "case", 
    "OPT": "mood", 
    "OR": "switch-reference", 
    "PART": "part of speech", 
    "PASS": "voice", 
    "PAUC": "number", 
    "PCT": "aktionsart", 
    "PERM": "mood", 
    "PFOC": "voice", 
    "PFV": "aspect", 
    "PHOR": "deixis", 
    "PL": "number", 
    "POL": "politeness", 
    "POS": "person", 
    "POS": "polarity", 
    "POST": "case", 
    "POT": "mood", 
    "PRF": "aspect", 
    "PRIV": "case", 
    "PRO": "part of speech", 
    "PROG": "aspect", 
    "PROL": "case", 
    "PROPN": "part of speech", 
    "PROPR": "case", 
    "PROSP": "aspect", 
    "PROX": "case", 
    "PROX": "deixis", 
    "PRP": "case", 
    "PRS": "tense", 
    "PRT": "case", 
    "PRX": "person", 
    "PSS1D": "possession", 
    "PSS1DE": "possession", 
    "PSS1DI": "possession", 
    "PSS1P": "possession", 
    "PSS1PE": "possession", 
    "PSS1PI": "possession", 
    "PSS1S": "possession", 
    "PSS2D": "possession", 
    "PSS2DF": "possession", 
    "PSS2DM": "possession", 
    "PSS2P": "possession", 
    "PSS2PF": "possession", 
    "PSS2PM": "possession", 
    "PSS2S": "possession", 
    "PSS2SF": "possession", 
    "PSS2SFORM": "possession", 
    "PSS2SINFM": "possession", 
    "PSS2SM": "possession", 
    "PSS3D": "possession", 
    "PSS3DF": "possession", 
    "PSS3DM": "possession", 
    "PSS3P": "possession", 
    "PSS3PF": "possession", 
    "PSS3PM": "possession", 
    "PSS3S": "possession", 
    "PSS3SF": "possession", 
    "PSS3SM": "possession", 
    "PSSD": "possession", 
    "PST": "tense", 
    "PURP": "mood", 
    "QUOT": "evidentiality", 
    "RCT": "tense", 
    "REAL": "mood", 
    "RECP": "valency", 
    "REF1": "deixis", 
    "REF2": "deixis", 
    "REFL": "valency", 
    "REL": "case", 
    "REM": "case", 
    "REMT": "deixis", 
    "RL": "comparison", 
    "RMT": "tense", 
    "RPRT": "evidentiality", 
    "SBJV": "mood", 
    "SEMEL": "aktionsart", 
    "SEN": "evidentiality", 
    "SEQMA": "switch-reference", 
    "SG": "number", 
    "SIM": "mood", 
    "SIMMA": "switch-reference", 
    "SPEC": "definiteness", 
    "SPRL": "comparison", 
    "SS": "switch-reference", 
    "SSADV": "switch-reference", 
    "STAT": "aktionsart", 
    "STELEV": "politeness", 
    "STSUPR": "politeness", 
    "SUB": "case", 
    "TEL": "aktionsart", 
    "TERM": "case", 
    "TOP": "information structure", 
    "TR": "valency", 
    "TRANS": "case", 
    "TRI": "number", 
    "V": "part of speech", 
    "V.CVB": "part of speech", 
    "V.MSDR": "part of speech", 
    "V.PTCP": "part of speech", 
    "VERS": "case", 
    "VIS": "deixis", 
    "VOC": "case"
}

def jsonl_to_csv(url):
    """mapping semeval2016_task4 jsonl to csv"""
    # Label mapping
    label_map = {
        "0": "negative",
        "1": "neutral",
        "2": "positive"
    }

    csv_url = url.replace(".jsonl", ".csv")

    with open(url, "r", encoding="utf-8") as infile, \
         open(csv_url, "w", encoding="utf-8", newline="") as outfile:

        writer = csv.writer(outfile, delimiter="\t")
        # Write header
        writer.writerow(["text", "class"])

        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get("text", "").strip()
            label = label_map.get(str(data.get("label", "")), "unknown")
            writer.writerow([text, label])


def unimorph2csv(url):
    conj_features, dec_features = set(), set()
    conj_values, dec_values = [], []
    with open(url, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            lemma, word, values = line.split("\t")
            values = values.split(";")

            if values[0] not in ["V", "ADJ", "N"]: 
                continue
            
            feat_value = conj_values if (values[0] == "V") else dec_values
            features = conj_features if (values[0] == "V") else dec_features
            
            feat_value.append({"lemma": lemma, "word": word})

            for value in values[1:]:
                feature = UNIMORPH_LABELS[value]
                features.add(feature)
                feat_value[-1][feature] = value

    with open(url + "_conj.csv", "w", encoding="utf-8") as f:
        conj_features = list(conj_features)
        f.write("word\tlemma\t" + "\t".join(conj_features) + "\n")
        for line in conj_values:
            f.write(line["word"] + "\t" + line["lemma"])
            for feat in conj_features:
                v = line[feat] if feat in line else "NA"
                f.write("\t" + v)
            f.write("\n")

    with open(url + "_dec.csv", "w", encoding="utf-8") as f:
        dec_features = list(dec_features)
        f.write("word\tlemma\t" + "\t".join(dec_features) + "\n")
        for line in dec_values:
            f.write(line["word"] + "\t" + line["lemma"])
            for feat in dec_features:
                v = line[feat] if feat in line else "NA"
                f.write("\t" + v)
            f.write("\n")


def split(url):
    # Load TSV file
    data = pd.read_csv(url, sep='\t', encoding="utf-8", keep_default_na=False, na_values=[])

    # Start with the word column
    data2 = pd.DataFrame(data['word'], columns=['word'])

    # Build multilabel DataFrame: binary features kept as-is, others one-hot encoded
    for feature in data.columns:
        if feature in ["word", "lemma"]:
            continue
        nbr = data[feature].nunique()
        if nbr == 2:
            data2[feature] = data[feature]
        else:
            one_hot = pd.get_dummies(data[feature], prefix=feature)
            data2 = pd.concat([data2, one_hot], axis=1)

    # Features for stratification (exclude 'word')
    y = data2.drop(columns=['word'])

    # Stratified split
    msss = MultilabelStratifiedShuffleSplit(test_size=0.4, random_state=42)
    for train_idx, test_idx in msss.split(data2['word'], y):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

    # Save splits
    train_data.to_csv(url.replace(".csv", "_train.csv"), index=False, sep='\t', encoding="utf-8")
    test_data.to_csv(url.replace(".csv", "_test.csv"), index=False, sep='\t', encoding="utf-8")


def extract_word_tag(conll_file, output_file):
    with open(conll_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        words, tags = [], []
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                if words:
                    f_out.write(" ".join(words) + "\t" + " ".join(tags) + "\n")
                    words, tags = [], []
                continue

            parts = line.split("\t")
            words.append(parts[1])
            tags.append(parts[3])
        if words:
            f_out.write(" ".join(words) + "\t" + " ".join(tags) + "\n")

def clean_tweet(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove mentions (@username)
    text = re.sub(r"@\w+", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =======================================================================
#    Main function 
# ======================================================================= 


def main_func(args):
    pass
    



# =======================================================================
#    Command line parser 
# =======================================================================  

cmd_list = [
    "unimorph"
]      
 
parser = argparse.ArgumentParser(description="Some scripts to extract datasets")
parser.add_argument("cmd", help="command", choices=cmd_list)
parser.add_argument("url", help="output txt file containing the results")
parser.set_defaults(func=main_func)


if __name__ == "__main__":
    argv = sys.argv[1:]

    argv = [
        "encode_csv",
        "/home/karim/Data/DZDT/test/lexicon/french/fr_taboo_noise.csv"
    ]

    args = parser.parse_args(argv)
    args.func(args)