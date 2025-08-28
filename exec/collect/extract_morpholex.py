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
# PROB

import pandas as pd
import numpy as np
DTST = '~/Data/DZDT/test/morpholex/morpholex_en2.csv'
data = pd.read_csv(DTST, sep='\t')

data.head()

# read affixes and split them by ; and show unique affixes
d = data["affixes"].str.split(";")
affixes = set()
for affix_list in d:
    if isinstance(affix_list, list):
        for affix in affix_list:
            affixes.add(affix.strip())  

affixes

for affix in affixes:
    data[affix] = data["affixes"].str.contains(affix, na=False).astype(int)

data.head()

affix_list, freq_list = [], []
for affix in affixes:
    affix_list.append(affix)
    freq_list.append(data[affix].sum())

list(zip(affix_list, freq_list))

affix_list, freq_list = np.array(affix_list), np.array(freq_list)

idx = freq_list >= 200
list(zip(affix_list[idx], freq_list[idx]))

kept_affixes = affix_list[idx]

for affix in set(affixes) - set(kept_affixes):
    del data[affix]

data.to_csv(DTST.replace(".csv", "_final.csv"), index=False, sep='\t')

Xd = data.iloc[:, 0]
yd = data.iloc[:, 3:]

yd.head()

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

msss = MultilabelStratifiedShuffleSplit(test_size=0.4, random_state=42)

X = Xd.values
y = yd.values

for train_idx, test_idx in msss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# fuse the train and save it
train_data = pd.DataFrame(X_train, columns=['word'])
train_data = pd.concat([train_data, pd.DataFrame(y_train, columns=yd.columns)], axis=1)
train_data.to_csv(DTST.replace(".csv", "_train.csv"), index=False, sep='\t')

# fuse the test and save it
test_data = pd.DataFrame(X_test, columns=['word'])      
test_data = pd.concat([test_data, pd.DataFrame(y_test, columns=yd.columns)], axis=1)
test_data.to_csv(DTST.replace(".csv", "_test.csv"), index=False, sep='\t')


# =============================================
# COMP

import pandas as pd
import numpy as np
DTST = '~/Data/DZDT/test/morpholex/morpholex111_fr.csv'
data = pd.read_csv(DTST, sep='\t')

data.head()
# read affixes and split them by ; and show unique affixes
d = data["affixes"].str.split(";")
affixes = set()
for affix_list in d:
    if isinstance(affix_list, list):
        for affix in affix_list:
            affixes.add(affix.strip())  

affixes
for affix in affixes:
    data[affix] = data["affixes"].str.contains(affix, na=False).astype(int)

data.head()

Xd = data.iloc[:, 0:2]
yd = data.iloc[:, 3:]

yd.head()
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

msss = MultilabelStratifiedShuffleSplit(test_size=0.4, random_state=42)

X = Xd.values
y = yd.values
for train_idx, test_idx in msss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    # y_train, y_test = y[train_idx], y[test_idx]
    affix_train = data.loc[train_idx, "affixes"].str.split(";")
    affix_test = data.loc[test_idx, "affixes"].str.split(";")

affix_train = np.array(affix_train.to_list())

affix_train = np.char.replace(affix_train, '-', '')

affix_test = np.array(affix_test.to_list())

affix_test = np.char.replace(affix_test, '-', '')



affix_test
# reconstruct the dataset: word prefix root suffix
train_data = pd.DataFrame(X_train[:, 0], columns=['word'])
train_data = pd.concat([train_data, pd.DataFrame(affix_train[:, 0], columns=['prefix'])], axis=1)
train_data = pd.concat([train_data, pd.DataFrame(X_train[:, 1], columns=['root'])], axis=1)
train_data = pd.concat([train_data, pd.DataFrame(affix_train[:, 1], columns=['suffix'])], axis=1)

train_data.to_csv(DTST.replace(".csv", "_train.csv"), index=False, sep='\t')

test_data = pd.DataFrame(X_test[:, 0], columns=['word'])
test_data = pd.concat([test_data, pd.DataFrame(affix_test[:, 0], columns=['prefix'])], axis=1)
test_data = pd.concat([test_data, pd.DataFrame(X_test[:, 1], columns=['root'])], axis=1)
test_data = pd.concat([test_data, pd.DataFrame(affix_test[:, 1], columns=['suffix'])], axis=1)

test_data.to_csv(DTST.replace(".csv", "_test.csv"), index=False, sep='\t')
    


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
