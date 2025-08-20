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
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.decomposition import PCA
import joblib 
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.extra.data import get_csv_string_data
from dzdt.model.classif import SeqSentEncoder, SimpleClassifier, TokenSimpleClassifier
from dzdt.extra.plms import get_sent_seq_embeddings, get_sent_embeddings, load_model, get_sent_embeddings_cuda

# =======================================================================
#    Testing 
# =======================================================================


def train_model(data_url: str, out_url: str, tokenizer, encoder, seq= False, epochs=20, batch_size=1000, lr=1e-4, gamma=None):
    data = get_csv_string_data(data_url)
    # Get the output embedding dimension of the encoder/model
    if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
        emb_d = encoder.config.hidden_size
    elif hasattr(encoder, "hidden_size"):
        emb_d = encoder.hidden_size
    elif hasattr(encoder, "d_model"):
        emb_d = encoder.d_model
    else:
        raise AttributeError("Cannot determine embedding dimension from encoder/model.")
    
    sentences = data["text"].tolist()

    classifier_params = dict(
        input_dim=768,
        hidden_dim=768,
        output_dim=3,
        hid_layers=1,
        dropout=0.2
    )

    if seq:
        encoder_params = dict(
            input_dim=emb_d,
            hidden_dim=384,
            num_layers=1,
            dropout=0.2,
        )
        model = TokenSimpleClassifier(encoder_params, classifier_params)
        X = get_sent_seq_embeddings(sentences, tokenizer, encoder, max_words=30)
    else:
        model = SimpleClassifier(**classifier_params)
        X = get_sent_embeddings(sentences, tokenizer, encoder)

    # Convert labels to tensor
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(data["class"])
    Y = torch.tensor(Y, dtype=torch.long)

    # Convert embeddings to tensor
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float)

    # Create DataLoader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(out_url, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_url, "logs"))

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / len(dataset)
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acc: {accuracy:.2f}%")
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        if gamma is not None and epoch_loss <= gamma:
            break

    writer.close()

    print("Training complete, saving ...")
    # Save model & label encoder
    model.save(os.path.join(out_url, "model.pt"))
    joblib.dump(label_encoder, os.path.join(out_url, "label_encoder.pkl"))

    model.eval()
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(labels.cpu().numpy())
                          
    
    # Classification report
    report_cls = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)

    os.makedirs(out_url, exist_ok=True)
    with open(os.path.join(out_url, "train_results.txt"), "w", encoding="utf8") as f:
        f.write(report_cls)
        f.write("\n\n")
        f.write(f"Training epochs: {epoch+1}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Embedding dimension: {emb_d}\n")
        f.write(f"Loss: {epoch_loss}\n")
        

    return model, label_encoder

def load_classif_model(url: str, seq=False):
    label_encoder: LabelEncoder = joblib.load(os.path.join(url, "label_encoder.pkl"))

    if seq:
        model = TokenSimpleClassifier.load(os.path.join(url, "model.pt"))
    else:
        model = SimpleClassifier.load(os.path.join(url, "model.pt"))

    return model, label_encoder

# ---------- Dataset ----------
class SentenceDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
        
def train_model_streaming(
    data_url: str,
    out_url: str,
    tokenizer,
    encoder,
    seq=False,
    epochs=20,
    batch_size=32,
    lr=1e-4,
    gamma=None,
    max_len=30,
    device=None
):

    # ---------- Load CSV ----------
    data = get_csv_string_data(data_url)
    sentences = data["text"].tolist()
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["class"].tolist())
    labels = torch.tensor(labels, dtype=torch.long)

    # ---------- Model ----------
    if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
        emb_d = encoder.config.hidden_size
    elif hasattr(encoder, "hidden_size"):
        emb_d = encoder.hidden_size
    elif hasattr(encoder, "d_model"):
        emb_d = encoder.d_model
    else:
        raise AttributeError("Cannot determine embedding dimension from encoder/model.")

    classifier_params = dict(
        input_dim=768,
        hidden_dim=768,
        output_dim=len(label_encoder.classes_),
        hid_layers=1,
        dropout=0.2
    )

    if seq:
        encoder_params = dict(
            input_dim=emb_d,
            hidden_dim=384,
            num_layers=1,
            dropout=0.2,
        )
        model = TokenSimpleClassifier(encoder_params, classifier_params)
    else:
        model = SimpleClassifier(**classifier_params)

    # ---------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    encoder = encoder.to(device)

    # ---------- DataLoader ----------
    dataset = SentenceDataset(sentences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------- Loss & Optim ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(out_url, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_url, "logs"))

    # ---------- Training ----------
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for batch_sentences, batch_labels in dataloader:
            batch_labels = batch_labels.to(device)

            # Encode with BERT inside the loop
            inputs = tokenizer(
                list(batch_sentences),
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                bert_out = encoder(**inputs)
                if hasattr(bert_out, "last_hidden_state"):
                    embeddings = bert_out.last_hidden_state[:, 0, :]  # CLS token
                else:
                    embeddings = bert_out[0][:, 0, :]

            # if seq:
            #     embeddings = get_sent_seq_embeddings(batch_sentences, tokenizer, encoder, max_words=30)
            # else:
            #     model = SimpleClassifier(**classifier_params)
            #     embeddings = get_sent_embeddings(batch_sentences, tokenizer, encoder)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        epoch_loss = total_loss / len(dataset)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acc: {accuracy:.2f}%")
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        if gamma is not None and epoch_loss <= gamma:
            break

    writer.close()

    # ---------- Save ----------
    print("Training complete, saving ...")
    model.save(os.path.join(out_url, "model.pt"))
    joblib.dump(label_encoder, os.path.join(out_url, "label_encoder.pkl"))

    # ---------- Evaluation ----------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_sentences, batch_labels in DataLoader(dataset, batch_size=batch_size):
            batch_labels = batch_labels.to(device)
            inputs = tokenizer(
                list(batch_sentences),
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)

            bert_out = encoder(**inputs)
            if hasattr(bert_out, "last_hidden_state"):
                embeddings = bert_out.last_hidden_state[:, 0, :]
            else:
                embeddings = bert_out[0][:, 0, :]

            outputs = model(embeddings)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    report_cls = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)
    with open(os.path.join(out_url, "train_results.txt"), "w", encoding="utf8") as f:
        f.write(report_cls)

    return model, label_encoder


def test_model(data_url: str, out_url: str, label_encoder, model, tokenizer, encoder, batch: int = None, device=None):
    # Default to GPU if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model.to(device)
    model.eval()

    data = get_csv_string_data(data_url)
    sentences = data["text"].tolist()
    all_preds = []

    if batch is None:
        # Process all at once
        if isinstance(model, TokenSimpleClassifier):
            X = get_sent_seq_embeddings(sentences, tokenizer, encoder, 30)
        else:
            X = get_sent_embeddings(sentences, tokenizer, encoder)

        # Move embeddings to device
        X = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred = model(X).detach().cpu().numpy()

        pred = np.argmax(pred, axis=1)
        all_preds = label_encoder.inverse_transform(pred)

    else:
        # Process in batches
        for i in range(0, len(sentences), batch):
            print(f"batch {i} is being tested ...")
            batch_sentences = sentences[i:i + batch]

            if isinstance(model, TokenSimpleClassifier):
                X = get_sent_seq_embeddings(batch_sentences, tokenizer, encoder, 30)
            else:
                X = get_sent_embeddings(batch_sentences, tokenizer, encoder)

            if X.device != device: # ensure X is on the right device
                X = X.to(device)

            with torch.no_grad():
                pred = model(X).detach().cpu().numpy()

            del X
            torch.cuda.empty_cache()

            pred = np.argmax(pred, axis=1)
            batch_preds = label_encoder.inverse_transform(pred)
            all_preds.extend(batch_preds)

    # Classification report
    Y = data["class"].tolist()
    report_cls = classification_report(Y, all_preds, target_names=label_encoder.classes_, zero_division=0)

    os.makedirs(out_url, exist_ok=True)
    with open(os.path.join(out_url, "test_results.txt"), "w", encoding="utf8") as f:
        f.write(report_cls)







# =============================================
#          Command line parser
# =============================================  
# 

def test_other(args):
    print("fusion of chdzdt and best performer")
    trn_url = os.path.expanduser("~/Data/DZDT/test/text_classif/fr_train.csv")
    tst_url = os.path.expanduser("~/Data/DZDT/test/text_classif/fr_test.csv")
    mdl_url = os.path.expanduser("~/Data/DZDT/results/classif/Cardiffnlp_fr/")
    lr = 0.001
    epochs = 100
    batch_size = 1000 #train
    batch = 1000 # test
    gamma = 0.1

    chdzdt_inf = ("chdzdt_4x4x32_20it", os.path.expanduser("~/Data/DZDT/models/chdzdt_4x4x32_20it"))
    best_inf = ("dziribert", "alger-ia/dziribert")

    tokenizer_chdzdt, encoder_chdzdt = load_model(chdzdt_inf[1])
    tokenizer_best, encoder_best = load_model(best_inf[1])

    model, label_encoder = load_classif_model(os.path.join(mdl_url, chdzdt_inf[0]), seq=True)

    seq_sent_encoder = model.encoder

    out_url = os.path.join(mdl_url, f"{best_inf[0]}_{chdzdt_inf[0]}")

    model, label_encoder = load_classif_model(out_url, seq=False)

    # print("training ....")

    # classifier_params = dict(
    #     input_dim=768,
    #     hidden_dim=768,
    #     output_dim=len(label_encoder.classes_),
    #     hid_layers=1,
    #     dropout=0.2
    # )

    # model = SimpleClassifier(**classifier_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)


    # # ---------- Load CSV ----------
    # data = get_csv_string_data(trn_url)
    # sentences = data["text"].tolist()
    # labels = label_encoder.transform(data["class"].tolist())
    # labels = torch.tensor(labels, dtype=torch.long)

    # del data

    # # ---------- DataLoader ----------
    # dataset = SentenceDataset(sentences, labels)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # ---------- Loss & Optim ----------
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    # out_url = os.path.join(mdl_url, f"{best_inf[0]}_{chdzdt_inf[0]}")

    # os.makedirs(out_url, exist_ok=True)
    # writer = SummaryWriter(log_dir=os.path.join(out_url, "logs"))

    # # ---------- Training ----------
    # encoder_best.to(device)
    # model.train()
    # for epoch in range(epochs):
    #     total_loss, correct, total = 0.0, 0, 0
    #     for batch_sentences, batch_labels in dataloader:

    #         batch_sentences = list(batch_sentences)

    #         with torch.no_grad():
    #             # bert_embeddings = get_sent_embeddings(batch_sentences, tokenizer_best, encoder_best)
    #             bert_embeddings = get_sent_embeddings_cuda(batch_sentences, tokenizer_best, encoder_best, device=device)
    #             chdzdt_embeddings = get_sent_seq_embeddings(batch_sentences, tokenizer_chdzdt, encoder_chdzdt)
    #             chdzdt_embeddings = seq_sent_encoder(chdzdt_embeddings)
    #             bert_embeddings = bert_embeddings.cpu()
    #             embeddings = bert_embeddings + chdzdt_embeddings
    #             del bert_embeddings, chdzdt_embeddings

    #         batch_labels = batch_labels.to(device)
    #         embeddings = embeddings.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(embeddings)
    #         loss = criterion(outputs, batch_labels)
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item() * batch_labels.size(0)
    #         _, predicted = torch.max(outputs, 1)
    #         total += batch_labels.size(0)
    #         correct += (predicted == batch_labels).sum().item()

    #     epoch_loss = total_loss / len(dataset)
    #     accuracy = 100 * correct / total
    #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acc: {accuracy:.2f}%")
    #     writer.add_scalar("Loss/train", epoch_loss, epoch)
    #     writer.add_scalar("Accuracy/train", accuracy, epoch)

    #     if gamma is not None and epoch_loss <= gamma:
    #         break

    # writer.close()

    # # ---------- Save ----------
    # print("Training complete, saving ...")
    # model.save(os.path.join(out_url, "model.pt"))
    # joblib.dump(label_encoder, os.path.join(out_url, "label_encoder.pkl"))

    # # ---------- Evaluation ----------
    # print("evaluating training")
    # model.eval()
    # all_preds, all_labels = [], []
    # with torch.no_grad():
    #     for batch_sentences, batch_labels in DataLoader(dataset, batch_size=batch_size):
    #         batch_sentences = list(batch_sentences)

    #         bert_embeddings = get_sent_embeddings_cuda(batch_sentences, tokenizer_best, encoder_best, device=device)
    #         chdzdt_embeddings = get_sent_seq_embeddings(batch_sentences, tokenizer_chdzdt, encoder_chdzdt)
    #         chdzdt_embeddings = seq_sent_encoder(chdzdt_embeddings)
    #         bert_embeddings = bert_embeddings.cpu()
    #         embeddings = bert_embeddings + chdzdt_embeddings
    #         del bert_embeddings, chdzdt_embeddings
            
    #         batch_labels = batch_labels.to(device)
    #         embeddings = embeddings.to(device)

    #         outputs = model(embeddings)
    #         preds = torch.argmax(outputs, dim=1)
    #         all_preds.extend(preds.cpu().numpy())
    #         all_labels.extend(batch_labels.cpu().numpy())

    # report_cls = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)
    # with open(os.path.join(out_url, "train_results.txt"), "w", encoding="utf8") as f:
    #     f.write(report_cls)


    

    print("evaluating testing")
    encoder_best.to(device)
    model = model.to(device)
    model.eval()
    encoder_best.eval()
    encoder_chdzdt.eval()

    data = get_csv_string_data(tst_url)
    sentences = data["text"].tolist()
    Y = data["class"].tolist()
    del data
    all_preds = []

    for i in range(0, len(sentences), batch):
        print(f"batch {i} is being tested ...")
        batch_sentences = sentences[i:i + batch]

        with torch.no_grad():
            bert_embeddings = get_sent_embeddings_cuda(batch_sentences, tokenizer_best, encoder_best, device=device)
            chdzdt_embeddings = get_sent_seq_embeddings(batch_sentences, tokenizer_chdzdt, encoder_chdzdt)
            chdzdt_embeddings = seq_sent_encoder(chdzdt_embeddings)
            bert_embeddings = bert_embeddings.cpu()
            embeddings = bert_embeddings + chdzdt_embeddings
            del bert_embeddings, chdzdt_embeddings
            embeddings = embeddings.to(device)
            pred = model(embeddings).detach().cpu().numpy()

        pred = np.argmax(pred, axis=1)
        batch_preds = label_encoder.inverse_transform(pred)
        all_preds.extend(batch_preds)

    # Classification report
    
    report_cls = classification_report(Y, all_preds, target_names=label_encoder.classes_, zero_division=0)

    os.makedirs(out_url, exist_ok=True)
    with open(os.path.join(out_url, "test_results.txt"), "w", encoding="utf8") as f:
        f.write(report_cls)

    

def test_main(args):

    # if torch.cuda.is_available():
    #     print("cuda exists")
    # print("bye")
    # exit()

    seq = False
    if "chdzdt" in args.p:
        seq = True

    tokenizer, encoder = load_model(args.p)

    output: str = os.path.expanduser(args.output)
    input: str = os.path.expanduser(args.input)

    if "train" in args.input:
        print("training the classification model ...")
        model, label_encoder = train_model(args.input, 
                                           os.path.join(output, args.m), 
                                           tokenizer, 
                                           encoder, 
                                           epochs=100,
                                           batch_size=2000,
                                           seq=seq, 
                                           gamma=0.1,
                                           lr=0.001)
        
        # model, label_encoder = train_model_streaming(
        #                                 args.input, 
        #                                 os.path.join(output, args.m), 
        #                                 tokenizer,
        #                                 encoder,
        #                                 seq=seq,
        #                                 epochs=100,
        #                                 batch_size=1000,
        #                                 lr=0.001,
        #                                 gamma=0.1,
        #                                 )
    else:
        model, label_encoder = load_classif_model(os.path.join(output, args.m), seq=seq)

        test_model(input, os.path.join(output, args.m), label_encoder, model, tokenizer, encoder, 
                   batch=1000)

    

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

    # src = "~/Data/DZDT/test/text_classif/fr_train.csv"
    src = "~/Data/DZDT/test/text_classif/SemEval2017-task4-test.subtask-A.english.csv"
    dst = "~/Data/DZDT/results/classif/SemEval2017en/"

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