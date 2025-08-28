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

from dzdt.extra.data import get_csv_string_data
from dzdt.model.classif import SimpleClassifier, TokenSimpleClassifier
from dzdt.extra.plms import get_embedding_size, load_model
from dzdt.pipeline.preprocessor import Embedder
from dzdt.pipeline.ptdatasets import SimpleDataset
from dzdt.pipeline.pttesters import SimpleTester, SimpleTesterConfig
from dzdt.pipeline.recorders import BatchPrinter, WritePrintRecorder
from dzdt.tools.struct import ObjectDict
from dzdt.pipeline.pttrainers import SimpleTrainer, SimpleTrainerConfig

# =======================================================================
#    Training 
# =======================================================================

def load_classif_model(url: str, seq=False):
    label_encoder: LabelEncoder = joblib.load(os.path.join(url, "label_encoder.pkl"))

    if seq:
        model = TokenSimpleClassifier.load(os.path.join(url, "model.pt"), map_location="cpu")
    else:
        model = SimpleClassifier.load(os.path.join(url, "model.pt"), map_location="cpu")

    return model, label_encoder


def train_model(params):

    data = get_csv_string_data(params.input_url)
    sentences = data["text"].tolist()

    if params.seq:
        sentence_words = []
        for sentence in sentences:
            sentence = sentence.split()
            l = len(sentence)
            if params.max_words != l:
                if l > params.max_words:
                    sentence = sentence[:params.max_words]
                else:
                    sentence = sentence + [""] * (params.max_words - l)
            sentence_words.append(sentence)
        sentences = sentence_words


    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(data["class"])
    Y = torch.tensor(Y, dtype=torch.long)


    def collate_batch(batch):
        sentences, labels = zip(*batch)   # unzip
        return list(sentences), torch.tensor(labels, dtype=torch.long)
    
    data_loader = DataLoader(SimpleDataset(sentences, Y), batch_size=params.batch_size, shuffle=True, collate_fn=collate_batch)
    
    if not params.stream:
        sentences = params.embedder.encode(sentences)
        data_loader = DataLoader(SimpleDataset(sentences, Y), batch_size=params.batch_size, shuffle=True)

    emb_d = get_embedding_size(params.embedder.encoder)
    classifier_params = dict(
        input_dim=768,
        hidden_dim=768,
        output_dim=3,
        hid_layers=1,
        dropout=0.2
    )

    if params.seq:
        encoder_params = dict(
            input_dim=emb_d,
            hidden_dim=384,
            num_layers=1,
            dropout=0.2,
        )
        model = TokenSimpleClassifier(encoder_params, classifier_params)
    else:
        model = SimpleClassifier(**classifier_params)

    config = SimpleTrainerConfig(
        model=model,
        data_loader=data_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        embedder=params.embedder,
        recorder=WritePrintRecorder(os.path.join(params.mdl_url, "logs")),
        stream=True,
    )
    
    print("Training Classif model ...")
    trainer = SimpleTrainer(config)

    trainer.train(epochs=params.epochs, gamma=params.gamma)
    

    print("Training complete, saving ...")
    # Save model & label encoder
    model.save(os.path.join(params.mdl_url, "model.pt"))
    joblib.dump(label_encoder, os.path.join(params.mdl_url, "label_encoder.pkl"))

# =======================================================================
#    Testing
# =======================================================================

def test_model(params):
    print("testing the classification model ...")

    model, label_encoder = load_classif_model(params.mdl_url, seq=params.seq)

    data = get_csv_string_data(params.input_url)
    sentences = data["text"].tolist()

    if params.seq:
        sentence_words = []
        for sentence in sentences:
            sentence = sentence.split()
            l = len(sentence)
            if params.max_words != l:
                if l > params.max_words:
                    sentence = sentence[:params.max_words]
                else:
                    sentence = sentence + [""] * (params.max_words - l)
            sentence_words.append(sentence)
        sentences = sentence_words



    Y = label_encoder.transform(data["class"]).tolist()


    def collate_batch(batch):
        sentences, labels = zip(*batch)   # unzip
        return list(sentences), torch.tensor(labels, dtype=torch.long)
    
    data_loader = DataLoader(SimpleDataset(sentences, Y), batch_size=params.batch_size, shuffle=False, collate_fn=collate_batch)
    
    if not params.stream:
        sentences = params.embedder.encode(sentences)
        data_loader = DataLoader(SimpleDataset(sentences, Y), batch_size=params.batch_size, shuffle=False)

    
    def classif_report(Y_true, Y_pred):
        return classification_report(Y_true, Y_pred, target_names=label_encoder.classes_, zero_division=0, digits=4)

    config = SimpleTesterConfig(
        model=model,
        data_loader=data_loader,
        embedder=params.embedder,
        stream=True,
        recorder=BatchPrinter(),
        criteria=[classif_report],
    )

    tester = SimpleTester(config)

    results = tester.test()

    with open(os.path.join(params.mdl_url, "test_results.txt"), "w", encoding="utf8") as f:
        f.write(results[0])



# def test_fusion(args):
#     print("fusion of chdzdt and best performer")
#     trn_url = os.path.expanduser("~/Data/DZDT/test/text_classif/fr_train.csv")
#     tst_url = os.path.expanduser("~/Data/DZDT/test/text_classif/fr_test.csv")
#     mdl_url = os.path.expanduser("~/Data/DZDT/results/classif/Cardiffnlp_fr/")
#     lr = 0.001
#     epochs = 100
#     batch_size = 1000 #train
#     batch = 1000 # test
#     gamma = 0.1

#     chdzdt_inf = ("chdzdt_4x4x32_20it", os.path.expanduser("~/Data/DZDT/models/chdzdt_4x4x32_20it"))
#     best_inf = ("dziribert", "alger-ia/dziribert")

#     tokenizer_chdzdt, encoder_chdzdt = load_model(chdzdt_inf[1])
#     tokenizer_best, encoder_best = load_model(best_inf[1])

#     model, label_encoder = load_classif_model(os.path.join(mdl_url, chdzdt_inf[0]), seq=True)

#     seq_sent_encoder = model.encoder

#     out_url = os.path.join(mdl_url, f"{best_inf[0]}_{chdzdt_inf[0]}")

#     model, label_encoder = load_classif_model(out_url, seq=False)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # model = model.to(device)
    

#     print("evaluating testing")
#     encoder_best.to(device)
#     model = model.to(device)
#     model.eval()
#     encoder_best.eval()
#     encoder_chdzdt.eval()

#     data = get_csv_string_data(tst_url)
#     sentences = data["text"].tolist()
#     Y = data["class"].tolist()
#     del data
#     all_preds = []

#     for i in range(0, len(sentences), batch):
#         print(f"batch {i} is being tested ...")
#         batch_sentences = sentences[i:i + batch]

#         with torch.no_grad():
#             bert_embeddings = get_sent_embeddings_cuda(batch_sentences, tokenizer_best, encoder_best, device=device)
#             chdzdt_embeddings = get_sent_seq_embeddings(batch_sentences, tokenizer_chdzdt, encoder_chdzdt)
#             chdzdt_embeddings = seq_sent_encoder(chdzdt_embeddings)
#             bert_embeddings = bert_embeddings.cpu()
#             embeddings = bert_embeddings + chdzdt_embeddings
#             del bert_embeddings, chdzdt_embeddings
#             embeddings = embeddings.to(device)
#             pred = model(embeddings).detach().cpu().numpy()

#         pred = np.argmax(pred, axis=1)
#         batch_preds = label_encoder.inverse_transform(pred)
#         all_preds.extend(batch_preds)

#     # Classification report
    
#     report_cls = classification_report(Y, all_preds, target_names=label_encoder.classes_, zero_division=0)

#     os.makedirs(out_url, exist_ok=True)
#     with open(os.path.join(out_url, "test_results.txt"), "w", encoding="utf8") as f:
#         f.write(report_cls)


# =======================================================================
#    Main function 
# =======================================================================


def main_func(args):

    # if torch.cuda.is_available():
    #     print("cuda exists")
    # print("bye")
    # exit()

    tokenizer, encoder = load_model(args.p)

    seq = False
    if "chdzdt" in args.p:
        seq = True
        embedder = Embedder(tokenizer, encoder, pooling="cls", one_word=True)
    else:
        embedder = Embedder(tokenizer, encoder, pooling="cls")


    output_url: str = os.path.expanduser(args.output)
    input_url: str = os.path.expanduser(args.input)
    mdl_url: str    = os.path.join(output_url, args.m)


    params = ObjectDict()
    params.seq = seq
    params.mdl_url = mdl_url
    params.input_url = input_url
    params.embedder = embedder
    params.max_words = 30 # if seq 30

    params.batch_size = 2000
    params.stream=True

    try:
        if "train" in args.input:
            print("training the classification model ...")
            params.epochs=10
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

    # src = "~/Data/DZDT/test/text_classif/fr_train.csv"
    # src = "~/Data/DZDT/test/text_classif/SemEval2017-task4-test.subtask-A.arabic_flt.csv"
    # dst = "~/Data/DZDT/results/classif/SemEval2017ar_flt/"

    # src = "~/Data/DZDT/test/text_classif/SemEval2017-task4-test.subtask-A.english_flt.csv"
    # dst = "~/Data/DZDT/results/classif/SemEval2017en_flt/"

    # src = "~/Data/DZDT/test/text_classif/twifil_test_flt.csv"
    # dst = "~/Data/DZDT/results/classif/Twifil_dz_flt/"

    # src = "~/Data/DZDT/test/text_classif/twifil_train_flt.csv"
    src = "~/Data/DZDT/test/text_classif/twifil_test_flt.csv"
    dst = "~/Data/DZDT/results/classif/testing/"
    

    mdls = [
        # ("chdzdt_5x4x128_20it", "~/Data/DZDT/models/chdzdt_5x4x128_20it"),
        # ("chdzdt_4x4x64_20it", "~/Data/DZDT/models/chdzdt_4x4x64_20it"),
        ("chdzdt_4x4x32_20it", "~/Data/DZDT/models/chdzdt_4x4x32_20it"),
        # ("chdzdt_3x2x16_20it", "~/Data/DZDT/models/chdzdt_3x2x16_20it"),
        # ("chdzdt_2x1x16_20it", "~/Data/DZDT/models/chdzdt_2x1x16_20it"),
        # ("chdzdt_2x4x16_20it", "~/Data/DZDT/models/chdzdt_2x4x16_20it"),
        # ("chdzdt_2x2x32_20it", "~/Data/DZDT/models/chdzdt_2x2x32_20it"),
        # ("chdzdt_2x2x16_20it", "~/Data/DZDT/models/chdzdt_2x2x16_20it"),
        # ("chdzdt_2x2x8_20it", "~/Data/DZDT/models/chdzdt_2x2x8_20it"),
        # ("chdzdt_1x2x16_20it", "~/Data/DZDT/models/chdzdt_1x2x16_20it"),
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