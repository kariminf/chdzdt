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

from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import ModelOutput, BertLMPredictionHead


@dataclass
class BertForMLMLMPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

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
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_labels = nn.Linear(config.hidden_size, config.num_labels)
        # self.seq_labels = MultiLabelPredictionHead(config)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_labels_score = self.seq_labels(pooled_output)
        return prediction_scores, seq_labels_score


class MLMLMBertModel(BertPreTrainedModel):
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

