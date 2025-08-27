import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch.utils.data import Dataset
import numpy as np
import os

from transformers import configuration_utils, modeling_utils
from dzdt.extra.plms import get_embedding_size, load_chdzdt_model, load_model
from dzdt.model.chdzdt_mdl import MLMLMBertModel
from dzdt.model.chdzdt_tok import CharTokenizer
from dzdt.tools.const import char_tokenizer_config


class SaveableModel:
    """Mixin that adds self-contained save/load to PyTorch modules."""

    def save(self, path: str):
        checkpoint = {
            "class_name": self.__class__.__name__,
            "params": self.params,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        obj = cls(**checkpoint["params"])
        obj.load_state_dict(checkpoint["state_dict"])
        return obj


class SimpleClassifier(SaveableModel, nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hid_layers=1, dropout=0.2):
        super().__init__()
        self.params = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hid_layers=hid_layers,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] +
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hid_layers - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = F.relu(layer(out))
            out = self.dropout(out)
        out = self.fc2(out)  # raw logits (let loss handle softmax)
        return out
    
    def predict(self, x):
        return F.softmax(self.forward(x))


class SeqSentEncoder(SaveableModel, nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.params = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)
    
class SeqSeqEncoder(SaveableModel, nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.params = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class TokenSimpleClassifier(SaveableModel, nn.Module):
    def __init__(self, encoder_params, classifier_params):
        super().__init__()
        self.params = dict(
            encoder_params=encoder_params,
            classifier_params=classifier_params,
        )
        self.encoder = SeqSentEncoder(**encoder_params)
        self.classifier = SimpleClassifier(**classifier_params)

    def forward(self, x):
        enc = self.encoder(x)
        out = self.classifier(enc)
        return out
    
    def predict(self, x):
        return F.softmax(self.forward(x))
    

class FTTokenSimpleClassifier(TokenSimpleClassifier):
    def __init__(self, encoder_params, classifier_params, encoder_url: str):

        tokenizer, char_encoder = load_chdzdt_model(encoder_url)

        if encoder_params["input_dim"] is None:
            encoder_params["input_dim"] = get_embedding_size(char_encoder)
        super().__init__(encoder_params, classifier_params)
        self.char_encoder = char_encoder
        self.tokenizer = tokenizer
        # Saves ~50% memory at the cost of slower training (recomputes activations during backprop).
        self.char_encoder.gradient_checkpointing_enable()

    def save(self, path: str):
        super().save(os.path.join(path, "classif_model.pt"))
        char_tokenizer_config()
        self.char_encoder.save_pretrained(path, safe_serialization=False)
        self.tokenizer.save(os.path.join(path, "char_tokenizer.pkl"))

    @classmethod
    def load(cls, path: str, map_location=None):
        mdl_url = os.path.join(path, "classif_model.pt")
        checkpoint = torch.load(mdl_url, map_location=map_location)
        obj = cls(**checkpoint["params"], encoder_url=path)
        obj.load_state_dict(checkpoint["state_dict"])
        return obj    

    def tokenize_encode(self, text, device):
        batch_size, seq_size = len(text), len(text[0])
        text = np.array(text).flatten()
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        ).to(device)
        # This avoids storing extra hidden states.
        emb = self.char_encoder(**tokens, output_hidden_states=False, output_attentions=False)
        return emb.last_hidden_state[:, 0, :].view(batch_size, seq_size, -1)
    
class TokenSeqClassifier(SaveableModel, nn.Module):
    def __init__(self, encoder_params, classifier_params):
        super().__init__()
        self.params = dict(
            encoder_params=encoder_params,
            classifier_params=classifier_params,
        )
        self.encoder = SeqSeqEncoder(**encoder_params)
        self.classifier = SimpleClassifier(**classifier_params)

    def forward(self, x):
        enc = self.encoder(x)
        out = self.classifier(enc)
        return out
    
    def predict(self, x):
        return F.softmax(self.forward(x))
    

class FTTokenSeqClassifier(TokenSeqClassifier):
    def __init__(self, encoder_params, classifier_params, encoder_url: str):

        tokenizer, char_encoder = load_chdzdt_model(encoder_url)

        if encoder_params["input_dim"] is None:
            encoder_params["input_dim"] = get_embedding_size(char_encoder)
        super().__init__(encoder_params, classifier_params)
        self.char_encoder = char_encoder
        self.tokenizer = tokenizer
        # Saves ~50% memory at the cost of slower training (recomputes activations during backprop).
        self.char_encoder.gradient_checkpointing_enable()

    def save(self, path: str):
        super().save(os.path.join(path, "classif_model.pt"))
        char_tokenizer_config()
        self.char_encoder.save_pretrained(path, safe_serialization=False)
        self.tokenizer.save(os.path.join(path, "char_tokenizer.pkl"))

    @classmethod
    def load(cls, path: str, map_location=None):
        mdl_url = os.path.join(path, "classif_model.pt")
        checkpoint = torch.load(mdl_url, map_location=map_location)
        obj = cls(**checkpoint["params"], encoder_url=path)
        obj.load_state_dict(checkpoint["state_dict"])
        return obj    

    def tokenize_encode(self, text, device):
        batch_size, seq_size = len(text), len(text[0])
        text = np.array(text).flatten()
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        ).to(device)
        # This avoids storing extra hidden states.
        emb = self.char_encoder(**tokens, output_hidden_states=False, output_attentions=False)
        return emb.last_hidden_state[:, 0, :].view(batch_size, seq_size, -1)

    
class MultipleOutputClassifier(SaveableModel, nn.Module):
    def __init__(self, shared_params: List[Tuple[str, List[str]]], output_classes):
        super().__init__()
        self.params = dict(
            shared_params=shared_params,
            output_classes=output_classes
        )

        self.shared = nn.ModuleList(
            [nn.Linear(shared_params["input_dim"], shared_params["hidden_dim"])] +
            [nn.Linear(shared_params["hidden_dim"], shared_params["hidden_dim"]) for _ in range(shared_params["hid_layers"] - 1)]
        )
        self.dropout = nn.Dropout(shared_params["dropout"])

        outputs = []

        for _, out_values in output_classes:
            out_nbr = len(out_values)
            out_nbr = 1 if (out_nbr == 2) else out_nbr
            outputs.append(nn.Linear(shared_params["hidden_dim"], out_nbr))
            

        self.outputs = nn.ModuleList(outputs)

    def forward(self, x):
        out = x
        for layer in self.shared:
            out = F.relu(layer(out))
            out = self.dropout(out)
        output = {}
        for out_layer, (name, _) in zip(self.outputs, self.params["output_classes"]):
            output[name] = out_layer(out)

        return output
    

class FTMultipleOutputClassifier(MultipleOutputClassifier):
    def __init__(self, shared_params, output_classes, encoder_url: str):

        tokenizer, encoder = load_chdzdt_model(encoder_url)

        if shared_params["input_dim"] is None:
            shared_params["input_dim"] = get_embedding_size(encoder)
        super().__init__(shared_params, output_classes)
        self.encoder = encoder
        self.tokenizer = tokenizer


    def save(self, path: str):
        super().save(os.path.join(path, "classif_model.pt"))
        char_tokenizer_config()
        self.encoder.save_pretrained(path, safe_serialization=False)
        self.tokenizer.save(os.path.join(path, "char_tokenizer.pkl"))

    @classmethod
    def load(cls, path: str, map_location=None):
        mdl_url = os.path.join(path, "classif_model.pt")
        checkpoint = torch.load(mdl_url, map_location=map_location)
        obj = cls(**checkpoint["params"], encoder_url=path)
        obj.load_state_dict(checkpoint["state_dict"])
        return obj    

    def tokenize(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )

    def forward(self, tokens):
        x = self.encoder(**tokens).last_hidden_state[:, 0, :]
        return super().forward(x)
    

