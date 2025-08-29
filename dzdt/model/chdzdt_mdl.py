#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023-2025	Abdelkrime Aries <kariminfo0@gmail.com>
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

from typing import List, Optional, Tuple, Union

import torch
# import torch.utils.checkpoint
from torch import nn
from dataclasses import dataclass
import os
import pickle
import json

from transformers.models.bert import BertModel, BertPreTrainedModel, BertConfig
from transformers.models.bert.modeling_bert import ModelOutput, BertLMPredictionHead

from dzdt.tools.io import process_hub_path, HubMixin


@dataclass
class BertForMLMLMPreTrainingOutput(ModelOutput):
    pooler_output: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_labels_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    multi_label_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None

# class OneLabelPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.num_labels)
#         self.decoder = nn.Linear(config.num_labels, 1)

#     def forward(self, pooled_output):
#         hidden_states = self.dense(pooled_output)
#         hidden_states = self.decoder(hidden_states)
#         return hidden_states
    
# class MultiLabelPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.layer = nn.ModuleList([OneLabelPredictionHead(config) for _ in range(config.num_labels)])
#         self.num_labels = config.num_labels

#     def forward(self, pooled_output):
#         hidden_states = []
#         for i, layer_module in enumerate(self.layer):
#             hidden_states.append(torch.flatten(layer_module(pooled_output)))
#         hidden_states = torch.stack(hidden_states, dim=1)
#         return hidden_states
    
class MLMLMBertPreTrainingHeads(nn.Module):
    """
    MLMLMBertPreTrainingHeads is a neural network module designed for BERT-based pretraining tasks.
    It combines a masked language modeling (MLM) prediction head and a sequence classification head.

    Args:
        config: Configuration object containing model hyperparameters such as hidden size and number of labels.

    Attributes:
        predictions (BertLMPredictionHead): Head for masked language modeling predictions.
        seq_labels (nn.Linear): Linear layer for sequence-level classification or labeling.

    Forward Args:
        sequence_output (torch.Tensor): Output tensor from the BERT encoder for each token in the sequence.
        pooled_output (torch.Tensor): Pooled output tensor representing the entire sequence (e.g., [CLS] token).

    Returns:
        prediction_scores (torch.Tensor): Scores for masked language modeling for each token.
        seq_labels_score (torch.Tensor): Scores for sequence-level classification or labeling.
    """
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_labels = nn.Linear(config.hidden_size, config.num_labels)
        # self.seq_labels = MultiLabelPredictionHead(config)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_labels_score = self.seq_labels(pooled_output)
        return prediction_scores, seq_labels_score


class MLMLMBertModel(BertPreTrainedModel, HubMixin):
    """MLMLMBertModel is a custom BERT-based model designed for joint masked language modeling (MLM) and multi-label classification tasks.
    Args:
        config (BertConfig): Model configuration object containing hyperparameters and settings.
    Attributes:
        bert (BertModel): The underlying BERT model for encoding input sequences.
        cls (MLMLMBertPreTrainingHeads): Head module for MLM and multi-label classification.
        mlmloss (nn.CrossEntropyLoss): Loss function for masked language modeling.
        clsloss (nn.BCEWithLogitsLoss): Loss function for multi-label classification.
    Methods:
        get_output_embeddings():
            Returns the output embedding layer used for MLM predictions.
        set_output_embeddings(new_embeddings):
            Sets the output embedding layer for MLM predictions.
        forward(
            Performs a forward pass through the model.
            Args:
                input_ids (torch.Tensor, optional): Token IDs for input sequences.
                attention_mask (torch.Tensor, optional): Mask to avoid attention on padding tokens.
                token_type_ids (torch.Tensor, optional): Segment token indices.
                position_ids (torch.Tensor, optional): Position indices for input tokens.
                head_mask (torch.Tensor, optional): Mask for attention heads.
                inputs_embeds (torch.Tensor, optional): Precomputed input embeddings.
                labels (torch.Tensor, optional): Labels for MLM loss computation.
                multi_labels (torch.Tensor, optional): Labels for multi-label classification loss.
                output_attentions (bool, optional): Whether to return attention weights.
                output_hidden_states (bool, optional): Whether to return hidden states.
                return_dict (bool, optional): Whether to return a dict or tuple.
                BertForMLMLMPreTrainingOutput or tuple:
                    - prediction_logits: MLM prediction scores.
                    - seq_labels_logits: Multi-label classification scores.
                    - masked_lm_loss: MLM loss (if labels provided).
                    - multi_label_loss: Multi-label classification loss (if multi_labels provided).
                    - loss: Total loss (if both labels and multi_labels provided).
                    - pooler_output: Pooled output from BERT.
                    - last_hidden_state: Last hidden state from BERT.
                    - hidden_states: Hidden states from BERT (if requested).
                    - attentions: Attention weights from BERT (if requested).
        """


    files = ["char_config.json", "char_pytorch_model.bin"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"token_type_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = MLMLMBertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlmloss = nn.CrossEntropyLoss()
        self.clsloss = nn.BCEWithLogitsLoss()#reduction="sum"

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings


    # def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
    #     return super().__call__(*args, **kwds)

    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        multi_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForMLMLMPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            multi_labels (`torch.LongTensor` of shape `(batch_size, labels_num)`, *optional*):
                Labels for computing the multilabel prediction (classification) loss.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from dzdt import DzDtCharTokenizer, MLMLMBertModel
        >>> import torch

        >>> tokenizer = DzDtCharTokenizer.from_pretrained("dzdt-chartok")
        >>> model = MLMLMBertModel.from_pretrained("dzdt-charbert")

        >>> inputs = tokenizer("wording", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            last_hidden_state, pooler_output = outputs["last_hidden_state"], outputs["pooler_output"]
        else:
            last_hidden_state, pooler_output = outputs[:2]

        prediction_scores, seq_labels_score = self.cls(last_hidden_state, pooler_output)
        
    
        masked_lm_loss = None
        multi_label_loss = None
        total_loss = None
        if labels is not None and multi_labels is not None:
            # M = labels.shape[0]
            masked_lm_loss = self.mlmloss(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # print(seq_relationship_score.dtype, multi_labels.dtype)
            multi_label_loss = self.clsloss(seq_labels_score.view(-1), multi_labels.view(-1)) #/M
            total_loss = masked_lm_loss + multi_label_loss

        if not return_dict:
            output = (prediction_scores, seq_labels_score, last_hidden_state, pooler_output)
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForMLMLMPreTrainingOutput(
            pooler_output = pooler_output,
            last_hidden_state = last_hidden_state,
            prediction_logits=prediction_scores,
            seq_labels_logits=seq_labels_score,
            masked_lm_loss= masked_lm_loss,
            multi_label_loss= multi_label_loss,
            loss=total_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> "MLMLMBertModel":
        """Loads a MLMLMBertModel instance from a pretrained path.

        This method attempts to load a tokenizer from a local file if the given path exists.
        If the path does not correspond to a local file, it loads the tokenizer from a remote hub,
        using HuggingFace-style arguments.

        Args:
            pretrained_path (str): Path to the pretrained tokenizer file or hub identifier.
            **kwargs: Additional keyword arguments passed to the hub loading function.

        Returns:
            CharTokeMLMLMBertModelnizer: An instance of MLMLMBertModel loaded from the specified source.
        """

        if os.path.isdir(pretrained_path):
            return cls.load(pretrained_path)

        return cls.load_from_hub(**process_hub_path(pretrained_path), **kwargs)
    
    @classmethod
    def _load_from_files(cls, local_files, **kwargs):
        
        with open(local_files[cls.files[0]], "r") as f:
            config = json.load(f)
        with open(local_files[cls.files[1]], "rb") as f:
            state = torch.load(f)
        config = BertConfig.from_dict(config)
        mdl = cls(config)
        mdl.load_state_dict(state)
        return mdl
    
    @classmethod
    def load(cls, path: str) -> "MLMLMBertModel":
        with open(os.path.join(path, cls.files[0]), "r") as f:
            config = json.load(f)
        with open(os.path.join(path, cls.files[1]), "rb") as f:
            state = torch.load(f)
        config = BertConfig.from_dict(config)
        mdl = cls(config)
        mdl.load_state_dict(state)
        return mdl

