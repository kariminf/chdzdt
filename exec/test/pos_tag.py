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
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib 
from torch import nn
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.extra.data import get_tagging_data
from dzdt.model.classif import SimpleClassifier, TokenSeqClassifier
from dzdt.extra.plms import get_embedding_size, load_model
from dzdt.extra.encode import encode_tags
from dzdt.tools.struct import ObjectDict
from dzdt.pipeline.recorders import BatchPrinter, WritePrintRecorder
from dzdt.pipeline.ptdatasets import SimpleDataset
from dzdt.pipeline.preprocessor import DzDTEmbedder, BertEmbedder
from dzdt.pipeline.pttrainers import SimpleTrainerConfig, SimpleTrainer, MaskedSimpleTrainer
from dzdt.pipeline.pttesters import SimpleTesterConfig, SimpleTester, MaskedSimpleTester


# =======================================================================
#    Training 
# =======================================================================

def collate_batch(batch):
        sentences, labels = zip(*batch)   # unzip
        return list(sentences), torch.tensor(labels, dtype=torch.long)

def train_model(params):
    print("Encoding PoS labels ...")
    label_encoder = LabelEncoder()

    Y = encode_tags(label_encoder, params.tags, train=True)

    # print(np.array(tags).shape, Y.shape)
    # exit(0)

    data_loader = DataLoader(SimpleDataset(params.sent_words, Y), 
                             batch_size=params.batch_size, 
                             shuffle=True, collate_fn=collate_batch)

    PAD_IDX = label_encoder.transform(["<PAD>"])[0]

    decoder_params = dict(
        input_dim=768,
        hidden_dim=768,
        output_dim=len(label_encoder.classes_),
        hid_layers=1,
        dropout=0.2
    )

    if params.seq:

        emb_d = get_embedding_size(params.embedder.encoder)
        encoder_params = dict(
            input_dim=emb_d,
            hidden_dim=384,
            num_layers=1,
            dropout=0.2,
        )
        pos_decoder = TokenSeqClassifier(encoder_params, decoder_params)

        loss_func = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        def my_loss(logits, targets):
            logits = logits.view(-1, len(label_encoder.classes_))
            targets = targets.view(-1)
            return loss_func(logits, targets)
        
        config = SimpleTrainerConfig(
            model = pos_decoder,
            optimizer = torch.optim.Adam(pos_decoder.parameters(), lr=1e-3),
            criterion = my_loss,
            data_loader=data_loader,
            embedder=params.embedder,
            recorder=WritePrintRecorder(os.path.join(params.mdl_url, "logs")),
            stream=True,
        )
        trainer = SimpleTrainer(config)
    
    else:
        pos_decoder = SimpleClassifier(**decoder_params)

        loss_func = nn.CrossEntropyLoss()
        def my_loss(logits_mask, targets):
            logits, mask = logits_mask
            # flatten
            logits = logits.view(-1, logits.size(-1))   # [B*T, N]
            targets = targets.view(-1)                  # [B*T]
            mask = mask.view(-1)                        # [B*T]
            mask2 = targets != PAD_IDX

            return loss_func(logits[mask], targets[mask2])
        
        config = SimpleTrainerConfig(
            model = pos_decoder,
            optimizer = torch.optim.Adam(pos_decoder.parameters(), lr=1e-3),
            criterion = my_loss,
            data_loader=data_loader,
            embedder=params.embedder,
            recorder=WritePrintRecorder(os.path.join(params.mdl_url, "logs")),
            stream=True,
        )
        trainer = MaskedSimpleTrainer(config)

    print("Training PoS model ...")

    trainer.train(epochs=params.epochs, gamma=params.gamma)

    print("saving models ...")
    pos_decoder.save(os.path.join(params.mdl_url, "pos_model.pt"))

    print("Saving label encoders ...")
    joblib.dump(label_encoder, os.path.join(params.mdl_url, "label_encoder.pkl"))
        
# =======================================================================
#    Testing 
# =======================================================================

def test_model(params):
    print("testing the classification model ...")

    label_encoder: LabelEncoder = joblib.load(os.path.join(params.mdl_url, "label_encoder.pkl"))

    Y = encode_tags(label_encoder, params.tags)

    data_loader = DataLoader(SimpleDataset(params.sent_words, Y), 
                             batch_size=params.batch_size, 
                             shuffle=False, collate_fn=collate_batch)
    PAD_IDX = label_encoder.transform(["<PAD>"])[0]

    if params.seq:
        pos_decoder = TokenSeqClassifier.load(os.path.join(params.mdl_url, "pos_model.pt"))
        def classif_report(Y_true, Y_pred):
            Y_true_fltn = np.array(Y_true).flatten()
            Y_pred_fltn = np.array(Y_pred).flatten()
            mask = Y_true_fltn != PAD_IDX
            # print("PAD_IDX", PAD_IDX, label_encoder.classes_, np.unique(Y_true_fltn), np.unique(Y_pred_fltn))
            return classification_report(Y_true_fltn[mask], Y_pred_fltn[mask], 
                                        target_names=label_encoder.classes_, 
                                        labels=range(len(label_encoder.classes_)),
                                        zero_division=0, digits=4)
        config = SimpleTesterConfig(
            model=pos_decoder,
            data_loader=data_loader,
            embedder=params.embedder,
            stream=True,
            recorder=BatchPrinter(),
            criteria=[classif_report],
        )
        tester = SimpleTester(config)
    else:
        pos_decoder = SimpleClassifier.load(os.path.join(params.mdl_url, "pos_model.pt"))
        def classif_report(Y_true, Y_pred):
            Y_true_fltn = np.array(Y_true).flatten()
            mask2 = Y_true_fltn != PAD_IDX
            Y_true_fltn = Y_true_fltn[mask2].tolist()
            return classification_report(Y_true_fltn, Y_pred, 
                                        target_names=label_encoder.classes_, 
                                        labels=range(len(label_encoder.classes_)),
                                        zero_division=0, digits=4)
        config = SimpleTesterConfig(
            model=pos_decoder,
            data_loader=data_loader,
            embedder=params.embedder,
            stream=True,
            recorder=BatchPrinter(),
            criteria=[classif_report],
        )
        tester = MaskedSimpleTester(config)
        

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

    tokenizer, encoder = load_model(args.p, pretokenized=True)

    seq = False
    if "chdzdt" in args.p:
        seq = True
        embedder = DzDTEmbedder(tokenizer, encoder, pooling="cls", one_word=True)
    else:
        word_mask = "fast"
        if "flaubert" in args.p:
            word_mask = "cmp"
        embedder = BertEmbedder(tokenizer, encoder, pretokenized=True, word_mask=word_mask)

    # sentences = [
    #     ["Je", "l'", "aimerais"],
    #     ["il", "est", "definitivement", "beau"],
    #     ["cela", "dit"]
    # ]

    # _, mask = embedder.encode(sentences)
    # print(mask)
    # sentences = [
    # "Les commotions cérébrales sont devenu si courantes dans ce sport qu' on les considére presque comme la routine .".split(),
    # "L' œuvre est située dans la galerie des de les batailles , dans le château de Versailles .".split()
    # ]

    # _, mask = embedder.encode(sentences)
    # print(mask)


    # exit(0)


    output_url: str = os.path.expanduser(args.output)
    input_url: str  = os.path.expanduser(args.input)

    print("Loading the data ...")
    sent_words, tags = get_tagging_data(input_url, max_words=60)


    mdl_url = os.path.join(output_url, args.m)

    params = ObjectDict()
    params.seq = seq
    params.mdl_url = mdl_url
    params.sent_words = sent_words
    params.tags = tags
    params.embedder = embedder

    params.batch_size = 500
    params.stream=True

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


# =======================================================================
#    Command line parser 
# =======================================================================        

 
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

    src = "~/Data/DZDT/test/pos-tag/arabic/ar_padt-ud-train.txt"
    src = "~/Data/DZDT/test/pos-tag/arabic/ar_padt-ud-test.txt"
    dst = "~/Data/DZDT/results/pos-tag/arabic"

    # src = "~/Data/DZDT/test/pos-tag/arabizi/qaf_arabizi-ud-train.txt"
    # src = "~/Data/DZDT/test/pos-tag/arabizi/qaf_arabizi-ud-test.txt"
    # dst = "~/Data/DZDT/results/pos-tag/arabizi"

    # src = "~/Data/DZDT/test/pos-tag/english/en_gum-ud-train.txt"
    # src = "~/Data/DZDT/test/pos-tag/english/en_gum-ud-test.txt"
    # dst = "~/Data/DZDT/results/pos-tag/english"

    # src = "~/Data/DZDT/test/pos-tag/french/fr_gsd-ud-train.txt"
    # src = "~/Data/DZDT/test/pos-tag/french/fr_gsd-ud-test.txt"
    # dst = "~/Data/DZDT/results/pos-tag/french"

    mdls = [
        # ("chdzdt_5x4x128_20it", "~/Data/DZDT/models/chdzdt_5x4x128_20it"),        # ("chdzdt_4x4x64_20it", "~/Data/DZDT/models/chdzdt_4x4x64_20it"),
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