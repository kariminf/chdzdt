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
from dzdt.model.classif import MultipleOutputClassifier
from dzdt.extra.plms import load_model, get_embedding_size
from dzdt.tools.struct import ObjectDict
from dzdt.pipeline.recorders import BatchPrinter, MultiWritePrintRecorder, FTMultiWritePrintRecorder
from dzdt.pipeline.ptdatasets import MultipleOutputDataset
from dzdt.pipeline.preprocessor import ClsTokEmbedder, DzDTEmbedder
from dzdt.pipeline.pttrainers import ClsTokMultiTrainerConfig, ClsTokMultiTrainer, MultiTrainerConfig, MultiTrainer
from dzdt.pipeline.pttesters import ClsTokMultiTesterConfig, ClsTokMultiTester, MultiTester, MultiTesterConfig


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

    emb_d = get_embedding_size(params.embedder.encoder)

    shared_params = dict(
        input_dim=emb_d,
        hidden_dim=768,
        hid_layers=1,
        dropout=0.2
    )
    cls_model = MultipleOutputClassifier(shared_params, output_classes)
    if params.tok:
        tok_model = MultipleOutputClassifier(shared_params, output_classes)

    data_loader = DataLoader(MultipleOutputDataset(words, outputs), batch_size=params.batch_size, shuffle=True)

    if params.tok:
        config = ClsTokMultiTrainerConfig(
            cls_model = cls_model,
            tok_model = tok_model,
            cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=1e-3),
            tok_optimizer = torch.optim.Adam(tok_model.parameters(), lr=1e-3),
            data_loader = data_loader,
            output_features = output_features,
            embedder = params.embedder,
            recorder = MultiWritePrintRecorder(os.path.join(params.mdl_url, "logs")),
            stream  = True,
        )

        trainer = ClsTokMultiTrainer(config)
    else:
        config = MultiTrainerConfig(
            model = cls_model,
            optimizer = torch.optim.Adam(cls_model.parameters(), lr=1e-3),
            data_loader = data_loader,
            output_features = output_features,
            embedder = params.embedder,
            recorder = FTMultiWritePrintRecorder(os.path.join(params.mdl_url, "logs")),
            stream  = True,
        )

        trainer = MultiTrainer(config)

    trainer.train(epochs=params.epochs, gamma=params.gamma)

    print("saving models ...")
    cls_model.save(os.path.join(params.mdl_url, "cls_model.pt"))
    if params.tok:
        tok_model.save(os.path.join(params.mdl_url, "tok_model.pt"))

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


    cls_model = MultipleOutputClassifier.load(os.path.join(params.mdl_url, "cls_model.pt"))
    if params.tok:
        tok_model = MultipleOutputClassifier.load(os.path.join(params.mdl_url, "tok_model.pt"))

    data_loader = DataLoader(MultipleOutputDataset(words, outputs), batch_size=params.batch_size, shuffle=False)

    if params.tok:
        def classif_report(Y_true, Y_pred):
            
            all_preds_cls, all_preds_tok = Y_pred
            cls_report, tok_report = {}, {}
            for feature, classes  in output_classes:

                cls_report[feature] = classification_report(Y_true[feature], 
                                                            all_preds_cls[feature], 
                                                            target_names=classes, 
                                                            zero_division=0,
                                                            digits=4)
                tok_report[feature] = classification_report(Y_true[feature], 
                                                            all_preds_tok[feature], 
                                                            target_names=classes, 
                                                            zero_division=0,
                                                            digits=4)

            return (cls_report, tok_report)

        config = ClsTokMultiTesterConfig(
            cls_model=cls_model,
            tok_model=tok_model,
            data_loader=data_loader,
            embedder=params.embedder,
            stream=True,
            recorder=BatchPrinter(),
            criteria=[classif_report],
            output_features=output_features
        )

        tester = ClsTokMultiTester(config)
    else:
        def classif_report(Y_true, Y_pred):
            report = {}
            for feature, classes  in output_classes:

                report[feature] = classification_report(Y_true[feature], 
                                                        Y_pred[feature], 
                                                        target_names=classes, 
                                                        zero_division=0,
                                                        digits=4)

            return report

        config = MultiTesterConfig(
            model=cls_model,
            data_loader=data_loader,
            embedder=params.embedder,
            stream=True,
            recorder=BatchPrinter(),
            criteria=[classif_report],
            output_features=output_features
        )

        tester = MultiTester(config)

    results = tester.test()[0] # only one criterion

    os.makedirs(params.mdl_url, exist_ok=True)
    with open(os.path.join(params.mdl_url, "test_results.txt"), "w", encoding="utf8") as f:
        for feature in output_features:
            if params.tok:
                f.write(feature + " CLS report:\n")
                f.write(results[0][feature])
                f.write(feature + " TOK report:\n")
                f.write(results[1][feature])
            else:
                f.write(feature + " CLS report:\n")
                f.write(results[feature])



# =============================================
#          Command line parser
# =============================================  
# 


def test_main(args):

    tok = False

    tokenizer, encoder = load_model(args.p)

    output_url: str = os.path.expanduser(args.output)
    input_url: str = os.path.expanduser(args.input)

    mdl_url: str = os.path.join(output_url, args.m)

    if tok:
        embedder = ClsTokEmbedder(tokenizer, encoder)
    else:
        embedder = DzDTEmbedder(tokenizer, encoder, pooling = "cls")

    params = ObjectDict()
    params.mdl_url = mdl_url
    params.input_url = input_url
    params.embedder = embedder
    params.tok = tok
    
    params.batch_size = 2000
    params.stream=True
    
    if "train" in args.input:
        params.epochs = 100
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
    # dst = "~/Data/DZDT/results/morph-tag/arabic_conj/"

    # src = "~/Data/DZDT/test/morph-tag/eng_conj_train.csv"
    # src = "~/Data/DZDT/test/morph-tag/eng_conj_test.csv"
    # dst = "~/Data/DZDT/results/morph-tag/english_conj/"

    src = "~/Data/DZDT/test/morph-tag/fra_conj_train.csv"
    # src = "~/Data/DZDT/test/morph-tag/fra_conj_test.csv"
    dst = "~/Data/DZDT/results/morph-tag/french_conj/"

    mdls = [
        # ("chdzdt_5x4x128_20it", "~/Data/DZDT/models/chdzdt_5x4x128_20it"),
        # ("chdzdt_4x4x64_20it", "~/Data/DZDT/models/chdzdt_4x4x64_20it"),
        # ("chdzdt_4x4x32_20it", "~/Data/DZDT/models/chdzdt_4x4x32_20it"),
        ("chdzdt_3x2x16_20it", "~/Data/DZDT/models/chdzdt_3x2x16_20it"),
        ("chdzdt_2x1x16_20it", "~/Data/DZDT/models/chdzdt_2x1x16_20it"),
        ("chdzdt_2x4x16_20it", "~/Data/DZDT/models/chdzdt_2x4x16_20it"),
        ("chdzdt_2x2x32_20it", "~/Data/DZDT/models/chdzdt_2x2x32_20it"),
        ("chdzdt_2x2x16_20it", "~/Data/DZDT/models/chdzdt_2x2x16_20it"),
        ("chdzdt_2x2x8_20it", "~/Data/DZDT/models/chdzdt_2x2x8_20it"),
        ("chdzdt_1x2x16_20it", "~/Data/DZDT/models/chdzdt_1x2x16_20it"),
        # ("arabert", "aubmindlab/bert-base-arabertv02-twitter"),
        # ("bert", "google-bert/bert-base-uncased"),
        # ("flaubert", "flaubert/flaubert_base_uncased"),
        # ("dziribert", "alger-ia/dziribert"),
        # ("caninec", "google/canine-c"),
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