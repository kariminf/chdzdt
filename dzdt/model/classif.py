import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch.utils.data import Dataset
import numpy as np


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
    
# ---------- Dataset ----------

    
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
    


# --- Example usage ---
if __name__ == "__main__":
    model = TokenSimpleClassifier(
        encoder_params={"input_dim": 100, "hidden_dim": 50, "num_layers": 2},
        classifier_params={"input_dim": 50, "hidden_dim": 20, "output_dim": 3},
    )

    # Save
    model.save("token_classifier.pt")

    # Load
    restored = TokenSimpleClassifier.load("token_classifier.pt")
    print(restored.params)
