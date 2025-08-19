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
import pandas as pd
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import numpy as np
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib 
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.extra.data import get_csv_string_data
from dzdt.model.classif import MultipleOutputClassifier, MultipleOutputDataset
from dzdt.extra.plms import load_model, get_oneword_embeddings, get_oneword_embeddings_cuda

# =======================================================================
#    Testing 
# =======================================================================


def verify_input_output(embeddings, 
                outputs: List[np.ndarray], 
                outputs_info: List[Tuple[str, int]]):
    if not torch.is_tensor(embeddings):
        embeddings = torch.tensor(embeddings, dtype=torch.float)

    new_outputs = []

    for output, output_info in zip(outputs, outputs_info):
        if not torch.is_tensor(output):
            if output_info[1] == 2:
                output = torch.tensor(output, dtype=torch.float)
            else:
                output = torch.tensor(output, dtype=torch.long)
        new_outputs.append(output)
    return embeddings, new_outputs



def train_model(model: MultipleOutputClassifier, 
                embeddings, 
                outputs: List[np.ndarray], 
                outputs_info: List[Tuple[str, int]], 
                log_url,
                epochs=100, 
                batch_size=1000, 
                lr=1e-3, 
                gamma=0.1,
                device=None
                ):
    

    # Default to GPU if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    embeddings, new_outputs = verify_input_output(embeddings, outputs, outputs_info)

    # Create DataLoader
    dataset = MultipleOutputDataset(embeddings, new_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define loss and optimizer
    multiclass_criterion = nn.CrossEntropyLoss()
    binary_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    os.makedirs(log_url, exist_ok=True)
    writer = SummaryWriter(log_url)

    nb_outputs = len(outputs)

    # Training loop
    # Move model to device
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        losses = [0.0] * nb_outputs
        for data in dataloader:
            data = [d.to(device) for d in data]

            optimizer.zero_grad()
            pred = model(data[0])
            loss = 0.0
            batch_size_actual = data[0].size(0)
            for i in range(nb_outputs):
                if outputs_info[i][1] == 2:
                    lossi = binary_criterion(pred[outputs_info[i][0]].squeeze(), data[i+1])
                else:
                    lossi = multiclass_criterion(pred[outputs_info[i][0]], data[i+1])
                losses[i] = losses[i] + (lossi.item() * batch_size_actual)
                loss += lossi
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size_actual

        epoch_loss = total_loss / len(dataset)
        writer.add_scalar("Total loss/train", epoch_loss, epoch)
        printloss = f"Epoch [{epoch+1}/{epochs}], Total Loss: {epoch_loss:.4f}"
        for i, (name, _) in enumerate(outputs_info):
             lossi = losses[i] / len(dataset)
             printloss += f", {name} loss: {lossi:.4f}"
             writer.add_scalar(f"{name} loss/train", epoch_loss, epoch)
        
        print(printloss)

        if gamma is not None and epoch_loss <= gamma:
            break

    writer.close()



def train_model_streaming(models: Tuple[MultipleOutputClassifier, MultipleOutputClassifier],
                          tokenizer,
                          encoder,
                          words: np.array, 
                          outputs: List[np.ndarray], 
                          outputs_info: List[Tuple[str, int]], 
                          log_url,
                          epochs=100, 
                          batch_size=1000, 
                          lr=1e-3, 
                          gamma=0.1,
                          device=None
                          ):
    

    # Default to GPU if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    new_outputs = []

    for output, output_info in zip(outputs, outputs_info):
        if not torch.is_tensor(output):
            if output_info[1] == 2:
                output = torch.tensor(output, dtype=torch.float)
            else:
                output = torch.tensor(output, dtype=torch.long)
        new_outputs.append(output)

    # Create DataLoader
    dataset = MultipleOutputDataset(words, new_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Define loss and optimizer
    multiclass_criterion = nn.CrossEntropyLoss()
    binary_criterion = nn.BCEWithLogitsLoss()
    optimizer0 = optim.Adam(models[0].parameters(), lr=lr)
    optimizer1 = optim.Adam(models[1].parameters(), lr=lr)


    os.makedirs(log_url, exist_ok=True)
    writer = SummaryWriter(log_url)

    nb_outputs = len(outputs)

    # Training loop
    # Move model to device
    models[0].to(device)
    models[1].to(device)
    models[0].train()
    models[1].train()
    for epoch in range(epochs):
        total_loss0 = 0.0
        total_loss1 = 0.0
        losses0 = [0.0] * nb_outputs
        losses1 = [0.0] * nb_outputs
        for data in dataloader:

            emb_cls, emb_tok = get_oneword_embeddings_cuda(data[0], tokenizer, encoder, device=device)
            
            emb_cls, emb_tok = emb_cls.detach().to(device), emb_tok.detach().to(device)
            Y = [d.to(device) for d in data[1:]]
            batch_size_actual = emb_cls.size(0)

            optimizer0.zero_grad()
            pred0 = models[0](emb_cls)
            loss0 = 0.0
            for i in range(nb_outputs):
                if outputs_info[i][1] == 2:
                    lossi0 = binary_criterion(pred0[outputs_info[i][0]].squeeze(), Y[i])
                else:
                    lossi0 = multiclass_criterion(pred0[outputs_info[i][0]], Y[i])
                losses0[i] = losses0[i] + (lossi0.item() * batch_size_actual)
                loss0 += lossi0
            loss0.backward()
            optimizer0.step()


            optimizer1.zero_grad()
            pred1 = models[1](emb_tok)
            loss1 = 0.0
            for i in range(nb_outputs):
                if outputs_info[i][1] == 2:
                    lossi1 = binary_criterion(pred1[outputs_info[i][0]].squeeze(), Y[i])
                else:
                    lossi1 = multiclass_criterion(pred1[outputs_info[i][0]], Y[i])
                losses1[i] = losses1[i] + (lossi1.item() * batch_size_actual)
                loss1 += lossi1
            loss1.backward()
            optimizer1.step()

            total_loss0 += loss0.item() * batch_size_actual
            total_loss1 += loss1.item() * batch_size_actual

        epoch_loss0 = total_loss0 / len(dataset)
        epoch_loss1 = total_loss1 / len(dataset)
        writer.add_scalar("Total loss/ cls train", epoch_loss0, epoch)
        writer.add_scalar("Total loss/ tok train", epoch_loss1, epoch)

        for i, (name, _) in enumerate(outputs_info):
             lossi0 = losses0[i] / len(dataset)
             lossi1 = losses1[i] / len(dataset)
             writer.add_scalar(f"{name} loss/ cls train", lossi0, epoch)
             writer.add_scalar(f"{name} loss/ tok train", lossi1, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}], Total cls Loss: {epoch_loss0:.4f}, Total tok Loss: {epoch_loss1:.4f}")

        if gamma is not None and (epoch_loss0 <= gamma) and (epoch_loss1 <= gamma):
            break

    writer.close()


def test_model(models: Tuple[MultipleOutputClassifier, MultipleOutputClassifier],
               tokenizer,
               encoder,
               words: np.array, 
               outputs: List[np.ndarray],
               outputs_info: List[Tuple[str, int]],
               outputs_labels: Dict[str, LabelEncoder],
               out_url: str,
               batch: int = None, 
               device=None
               ):
    
    # Default to GPU if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    models[0].to(device)
    models[1].to(device)
    models[0].eval()
    models[1].eval()

    all_preds_cls, all_preds_tok = [], []

    for i in range(len(outputs_info)):
        all_preds_cls.append([])
        all_preds_tok.append([])

    if batch is None:
        pass
        # Process all at once
        # if isinstance(model, TokenSimpleClassifier):
        #     X = get_sent_seq_embeddings(sentences, tokenizer, encoder, 30)
        # else:
        #     X = get_sent_embeddings(sentences, tokenizer, encoder)

        # # Move embeddings to device
        # X = torch.tensor(X, dtype=torch.float32).to(device)

        # with torch.no_grad():
        #     pred = model(X).detach().cpu().numpy()

        # pred = np.argmax(pred, axis=1)
        # all_preds = label_encoder.inverse_transform(pred)

    else:
        # Process in batches
        for i in range(0, len(words), batch):
            print(f"batch {i} is being tested ...")
            batch_words= words[i:i + batch]

            emb_cls, emb_tok = get_oneword_embeddings_cuda(batch_words, tokenizer, encoder, device=device)
            
            emb_cls, emb_tok = emb_cls.detach().to(device), emb_tok.detach().to(device)

            with torch.no_grad():
                pred_cls = models[0](emb_cls)
                pred_tok = models[1](emb_tok)

            for i in range(len(outputs_info)):
                name = outputs_info[i][0]
                pred_clsi = pred_cls[name].detach().cpu().numpy()
                pred_toki = pred_tok[name].detach().cpu().numpy()

                pred_clsi = np.argmax(pred_clsi, axis=1)
                pred_toki = np.argmax(pred_toki, axis=1)

            #     pred_clsi = outputs_labels[name].inverse_transform(pred_clsi)
            #     pred_toki = outputs_labels[name].inverse_transform(pred_toki)

                all_preds_cls[i].extend(pred_clsi)
                all_preds_tok[i].extend(pred_toki)

    # Classification report
    os.makedirs(out_url, exist_ok=True)
    with open(os.path.join(out_url, "test_results.txt"), "w", encoding="utf8") as f:
        for i in range(len(outputs_info)):
            
            name = outputs_info[i][0]
            # print("classification report " + name)

            report = classification_report(outputs[i], all_preds_cls[i], target_names=outputs_labels[name].classes_, zero_division=0)
            f.write(name + " CLS report:\n")
            f.write(report)

            report = classification_report(outputs[i], all_preds_tok[i], target_names=outputs_labels[name].classes_, zero_division=0)
            f.write(name + " TOK report:\n")
            f.write(report)


# =============================================
#          Command line parser
# =============================================  
# 


def test_main(args):

    tokenizer, encoder = load_model(args.p)

    if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
        emb_d = encoder.config.hidden_size
    elif hasattr(encoder, "hidden_size"):
        emb_d = encoder.hidden_size
    elif hasattr(encoder, "d_model"):
        emb_d = encoder.d_model
    else:
        raise AttributeError("Cannot determine embedding dimension from encoder/model.")

    output_url: str = os.path.expanduser(args.output)
    input_url: str = os.path.expanduser(args.input)

    print("Loading the data ...")
    data = get_csv_string_data(input_url)

    print("Encoding the data ...")
    # emb_cls, emb_tok = get_oneword_embeddings(data["word"], tokenizer, encoder)


    outputs = []
    outputs_info = []
    output_classes = []
    output_label_encoders = {}


    for i, feature in enumerate(data.columns):
        if feature in ["word", "lemma"]:
            continue
        le = LabelEncoder()
        Y = le.fit_transform(data[feature])
        outputs.append(Y)
        output_classes.append((feature, le.classes_))
        outputs_info.append((feature, len(le.classes_)))
        output_label_encoders[feature] = le

    words = data["word"].to_list()

    del data

    mdl_url = os.path.join(output_url, args.m)
    
    if "train" in args.input:

        shared_params = dict(
            input_dim=emb_d,
            hidden_dim=768,
            hid_layers=1,
            dropout=0.2
        )
        cls_model = MultipleOutputClassifier(shared_params, output_classes)
        tok_model = MultipleOutputClassifier(shared_params, output_classes)

        # print("training the cls classification model ...")
        # train_model(cls_model, emb_cls, outputs, outputs_info, os.path.join(mdl_url, "cls_logs"))
        # print("training the tok classification model ...")
        # train_model(tok_model, emb_tok, outputs, outputs_info, os.path.join(mdl_url, "tok_logs"))

        train_model_streaming((cls_model, tok_model), 
                              tokenizer, 
                              encoder, 
                              words, 
                              outputs, 
                              outputs_info, 
                              os.path.join(mdl_url, "logs"))

        print("saving models ...")
        cls_model.save(os.path.join(mdl_url, "cls_model.pt"))
        tok_model.save(os.path.join(mdl_url, "tok_model.pt"))

        print("Saving label encoders ...")
        joblib.dump(output_label_encoders, os.path.join(mdl_url, "label_encoders.pkl"))

    else:
        print("testing the classification model ...")

        cls_model = MultipleOutputClassifier.load(os.path.join(mdl_url, "cls_model.pt"))
        tok_model = MultipleOutputClassifier.load(os.path.join(mdl_url, "tok_model.pt"))

        outputs_labels = joblib.load(os.path.join(mdl_url, "label_encoders.pkl"))

        test_model((cls_model, tok_model),
               tokenizer,
               encoder,
               words, 
               outputs,
               outputs_info,
               outputs_labels,
               mdl_url,
               batch=1000, 
               device=None
               )
        
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

    # src = "~/Data/DZDT/test/morph-tag/fra_conj_train.csv"
    src = "~/Data/DZDT/test/morph-tag/fra_conj_test.csv"
    dst = "~/Data/DZDT/results/morph-tag/french_conj/"

    mdls = [
        # ("chdzdt_5x4x128_20it", "~/Data/DZDT/models/chdzdt_5x4x128_20it"),
        # ("chdzdt_4x4x64_20it", "~/Data/DZDT/models/chdzdt_4x4x64_20it"),
        # ("chdzdt_4x4x32_20it", "~/Data/DZDT/models/chdzdt_4x4x32_20it"),
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
        ("caninec", "google/canine-c"),
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