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

import os
# shut tensorflow
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # kill TF runtime logs
# os.environ["TRANSFORMERS_NO_TF"] = "1"          # stop transformers from importing TF
# os.environ["TRANSFORMERS_NO_FLAX"] = "1"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # (optional) disables oneDNN warnings

import argparse
import sys

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib 
from torch.utils.data import DataLoader



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.extra.data import get_csv_string_data
from dzdt.model.classif import FTMultipleOutputClassifier
from dzdt.extra.plms import load_model, get_embedding_size
from dzdt.tools.struct import ObjectDict
from dzdt.pipeline.recorders import BatchPrinter, FTMultiWritePrintRecorder
from dzdt.pipeline.ptdatasets import MultipleOutputDataset
from dzdt.pipeline.preprocessor import ClsTokEmbedder
from dzdt.pipeline.pttrainers import FTMultiTrainerConfig, FTMultiTrainer
from dzdt.pipeline.pttesters import FTMultiTesterConfig, FTMultiTester


# =======================================================================
#    Testing 
# =======================================================================

def train(params):

    print("Loading the data ...")
    data = get_csv_string_data(params.input_url)

    label_encoders = {}
    
    outputs = []
    outputs_info = []
    output_classes = []
    output_features = []
    for feature in data.columns:
            if feature in ["word", "lemma"]:
                continue
            le = LabelEncoder()
            label_encoders[feature] = le
            Y = le.fit_transform(data[feature])
            if len(le.classes_) == 2:
                Y = torch.tensor(Y, dtype=torch.float)
            else:
                Y = torch.tensor(Y, dtype=torch.long)
                Y = torch.tensor(Y, dtype=torch.long)
            outputs.append(Y)
            output_classes.append((feature, le.classes_))
            outputs_info.append((feature, len(le.classes_)))
            output_features.append(feature)

    words = data["word"]
    del data

    shared_params = dict(
        input_dim=None,
        hidden_dim=768,
        hid_layers=1,
        dropout=0.2
    )

    model = FTMultipleOutputClassifier(shared_params, output_classes, params.p)

    data_loader = DataLoader(MultipleOutputDataset(words, outputs), batch_size=params.batch_size, shuffle=True)

    config = FTMultiTrainerConfig(
        model = model,
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3),
        data_loader = data_loader,
        output_features = output_features,
        recorder = FTMultiWritePrintRecorder(os.path.join(params.mdl_url, "logs")),
        stream  = True,
    )

    print("Training ...")
    trainer = FTMultiTrainer(config)
    trainer.train(epochs=params.epochs, gamma=params.gamma)

    print("Saving ...")
    model.save(params.mdl_url)

    print("Saving label encoders ...")
    joblib.dump(label_encoders, os.path.join(params.mdl_url, "label_encoders.pkl"))


def test(params):
    print("testing the classification model ...")

    print("Loading the data ...")
    data = get_csv_string_data(params.input_url)

    label_encoders = joblib.load(os.path.join(params.mdl_url, "label_encoders.pkl"))
    outputs = []
    outputs_info = []
    output_classes = []
    output_features = []
    for feature in data.columns:
            if feature in ["word", "lemma"]:
                continue
            le = label_encoders[feature]
            Y = le.transform(data[feature])
            outputs.append(Y)
            output_classes.append((feature, le.classes_))
            outputs_info.append((feature, len(le.classes_)))
            output_features.append(feature)

    words = data["word"]
    del data


    model = FTMultipleOutputClassifier.load(params.mdl_url)

    data_loader = DataLoader(MultipleOutputDataset(words, outputs), batch_size=params.batch_size, shuffle=False)


    def classif_report(Y_true, Y_pred):
        
        report = {}
        for feature, classes  in output_classes:

            report[feature] = classification_report(Y_true[feature], 
                                                    Y_pred[feature], 
                                                    target_names=classes, 
                                                    zero_division=0,
                                                    digits=4)

        return report

    config = FTMultiTesterConfig(
        model=model,
        data_loader=data_loader,
        stream=True,
        recorder=BatchPrinter(),
        criteria=[classif_report],
        output_features=output_features
    )

    tester = FTMultiTester(config)

    results = tester.test()[0] # only one criterion

    os.makedirs(params.mdl_url, exist_ok=True)
    with open(os.path.join(params.mdl_url, "test_results.txt"), "w", encoding="utf8") as f:
        for feature in output_features:
            f.write(feature + " report:\n")
            f.write(results[feature])


# =============================================
#          Command line parser
# =============================================  
# 


def test_main(args):

    output_url: str = os.path.expanduser(args.output)
    input_url: str = os.path.expanduser(args.input)
    mdl_url: str = os.path.join(output_url, args.m)


    params = ObjectDict()
    params.mdl_url = mdl_url
    params.input_url = input_url
    
    params.batch_size = 1000
    params.stream=True
    
    if "train" in args.input:
        params.p = os.path.expanduser(args.p)
        params.epochs = 20
        params.gamma = 0.1
        train(params)
    else:
        test(params)


        
parser = argparse.ArgumentParser(description="test morphological clustering using a pre-trained model")
parser.add_argument("-m", help="model label")
parser.add_argument("-p", help="model name/path")
parser.add_argument("input", help="input clusters' file")
parser.add_argument("output", help="output txt file containing the results")
parser.set_defaults(func=test_main)
# parser.set_defaults(func=test_other)


if __name__ == "__main__":

    # argv = sys.argv[1:]
    # args = parser.parse_args(argv)
    # # print(args)
    # # parser.print_help()
    # args.func(args)

    # src = "~/Data/DZDT/test/morph-tag/ara_conj_train.csv"
    # src = "~/Data/DZDT/test/morph-tag/ara_conj_test.csv"
    # dst = "~/Data/DZDT/results/morph-tag/ft_arabic_conj/"

    # src = "~/Data/DZDT/test/morph-tag/eng_conj_train.csv"
    # src = "~/Data/DZDT/test/morph-tag/eng_conj_test.csv"
    # dst = "~/Data/DZDT/results/morph-tag/ft_english_conj/"

    src = "~/Data/DZDT/test/morph-tag/fra_conj_train.csv"
    src = "~/Data/DZDT/test/morph-tag/fra_conj_test.csv"
    dst = "~/Data/DZDT/results/morph-tag/ft_french_conj/"

    mdls = [
        ("chdzdt_5x4x128_20it", "~/Data/DZDT/models/chdzdt_5x4x128_20it"),
        ("chdzdt_4x4x64_20it", "~/Data/DZDT/models/chdzdt_4x4x64_20it"),
        ("chdzdt_4x4x32_20it", "~/Data/DZDT/models/chdzdt_4x4x32_20it"),
        # ("chdzdt_3x2x16_20it", "~/Data/DZDT/models/chdzdt_3x2x16_20it"),
        # ("chdzdt_2x1x16_20it", "~/Data/DZDT/models/chdzdt_2x1x16_20it"),
        # ("chdzdt_2x4x16_20it", "~/Data/DZDT/models/chdzdt_2x4x16_20it"),
        # ("chdzdt_2x2x32_20it", "~/Data/DZDT/models/chdzdt_2x2x32_20it"),
        # ("chdzdt_2x2x16_20it", "~/Data/DZDT/models/chdzdt_2x2x16_20it"),
        # ("chdzdt_2x2x8_20it", "~/Data/DZDT/models/chdzdt_2x2x8_20it"),
        # ("chdzdt_1x2x16_20it", "~/Data/DZDT/models/chdzdt_1x2x16_20it"),
    ]

    for mdl in mdls:

        print(f"Testing model {mdl[0]} ...")
        argv = [
            "-m", mdl[0],
            "-p", mdl[1],
            src,
            dst
            ]
        args = parser.parse_args(argv)
        args.func(args)