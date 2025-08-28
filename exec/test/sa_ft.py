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
import sys
import os
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib 
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.text.pp import pad_truncate
from dzdt.extra.data import get_csv_string_data
from dzdt.model.classif import FTTokenSimpleClassifier
from dzdt.pipeline.ptdatasets import SimpleDataset
from dzdt.pipeline.pttesters import FTSimpleTester, SimpleTesterConfig
from dzdt.pipeline.recorders import BatchPrinter, WritePrintRecorder
from dzdt.tools.struct import ObjectDict
from dzdt.pipeline.pttrainers import FTSimpleTrainer, SimpleTrainerConfig

# =======================================================================
#    Training 
# =======================================================================


def train_model(params):

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(params.Y)
    Y = torch.tensor(Y, dtype=torch.long)


    def collate_batch(batch):
        sentences, labels = zip(*batch)   # unzip
        return list(sentences), torch.tensor(labels, dtype=torch.long)
    
    data_loader = DataLoader(SimpleDataset(params.sent_words, Y), batch_size=params.batch_size, shuffle=True, collate_fn=collate_batch)
    
    # if not params.stream:
    #     sentences = params.embedder.encode(sentences)
    #     data_loader = DataLoader(SimpleDataset(sentences, Y), batch_size=params.batch_size, shuffle=True)

    classifier_params = dict(
        input_dim=768,
        hidden_dim=768,
        output_dim=3,
        hid_layers=1,
        dropout=0.2
    )

    encoder_params = dict(
        input_dim=None,
        hidden_dim=384,
        num_layers=1,
        dropout= 0.2,
    )

    model = FTTokenSimpleClassifier(encoder_params, classifier_params, params.p)

    config = SimpleTrainerConfig(
        model=model,
        data_loader=data_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4), #1e-3
        criterion=nn.CrossEntropyLoss(),
        recorder=WritePrintRecorder(os.path.join(params.mdl_url, "logs")),
        stream=True,
    )
    
    print("Training Classif model ...")
    trainer = FTSimpleTrainer(config)

    trainer.train(epochs=params.epochs, gamma=params.gamma)
    

    print("Training complete, saving ...")
    # Save model & label encoder
    model.save(params.mdl_url)
    joblib.dump(label_encoder, os.path.join(params.mdl_url, "label_encoder.pkl"))

# =======================================================================
#    Testing
# =======================================================================

def test_model(params):
    print("testing the classification model ...")

    label_encoder = joblib.load(os.path.join(params.mdl_url, "label_encoder.pkl"))
    model = model = FTTokenSimpleClassifier.load(params.mdl_url, map_location="cpu")

    params.Y = label_encoder.transform(params.Y).tolist()


    def collate_batch(batch):
        sentences, labels = zip(*batch)   # unzip
        return list(sentences), torch.tensor(labels, dtype=torch.long)
    
    data_loader = DataLoader(SimpleDataset(params.sent_words, params.Y), batch_size=params.batch_size, shuffle=False, collate_fn=collate_batch)
    
    # if not params.stream:
    #     sentences = params.embedder.encode(sentences)
    #     data_loader = DataLoader(SimpleDataset(sentences, Y), batch_size=params.batch_size, shuffle=False)

    
    def classif_report(Y_true, Y_pred):
        return classification_report(Y_true, Y_pred, target_names=label_encoder.classes_, zero_division=0, digits=4)

    config = SimpleTesterConfig(
        model=model,
        data_loader=data_loader,
        stream=True,
        recorder=BatchPrinter(),
        criteria=[classif_report],
    )

    tester = FTSimpleTester(config)

    results = tester.test()

    with open(os.path.join(params.mdl_url, "test_results.txt"), "w", encoding="utf8") as f:
        f.write(results[0])


# =======================================================================
#    Main function 
# =======================================================================


def main_func(args):

    # if torch.cuda.is_available():
    #     print("cuda exists")
    # print("bye")
    # exit()
    params = ObjectDict()
    params.max_words = 30
    params.batch_size = 100
    params.stream=True

    params.p = os.path.expanduser(args.p)
    params.mdl_url = os.path.join(os.path.expanduser(args.output), args.m)

    data = get_csv_string_data(os.path.expanduser(args.input))

    params.sent_words = pad_truncate(data["text"].tolist(), params.max_words)

    params.Y = data["class"].tolist()

    del data

    try:
        if "train" in args.input:
            print("training the classification model ...")
            params.epochs=50
            params.gamma=0.1
            train_model(params)
        else:
            test_model(params)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user! Cleaning up CUDA...")
        torch.cuda.empty_cache()
        sys.exit(0) 

# =============================================
#          Command line parser
# ============================================= 

parser = argparse.ArgumentParser(description="test morphological clustering using a pre-trained model")
parser.add_argument("-m", help="model label")
parser.add_argument("-p", help="model name/path")
parser.add_argument("input", help="input clusters' file")
parser.add_argument("output", help="output txt file containing the results")
parser.set_defaults(func=main_func)
# parser.set_defaults(func=test_other)


if __name__ == "__main__":

    # argv = sys.argv[1:]
    # args = parser.parse_args(argv)
    # # print(args)
    # # parser.print_help()
    # args.func(args)

    # src = "~/Data/DZDT/test/text_classif/SemEval2017-task4-train.subtask-A.arabic_flt.csv"
    # src = "~/Data/DZDT/test/text_classif/SemEval2017-task4-test.subtask-A.english_flt.csv"
    # dst = "~/Data/DZDT/results/classif/ft_SemEval2017ar_flt/"

    # src = "~/Data/DZDT/test/text_classif/twifil_train_flt.csv"
    # src = "~/Data/DZDT/test/text_classif/twifil_test_flt.csv"
    # dst = "~/Data/DZDT/results/classif/ft_Twifil_dz_flt/"

    src = "~/Data/DZDT/test/text_classif/twitter-2016train-A_flt.csv"
    src = "~/Data/DZDT/test/text_classif/SemEval2017-task4-test.subtask-A.english_flt.csv"
    dst = "~/Data/DZDT/results/classif/ft_SemEval2017en_flt/"

    # src = "~/Data/DZDT/test/text_classif/fr_train.csv"
    # src = "~/Data/DZDT/test/text_classif/fr_test.csv"
    # dst = "~/Data/DZDT/results/classif/ft_Cardiffnlp_fr/"
    

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