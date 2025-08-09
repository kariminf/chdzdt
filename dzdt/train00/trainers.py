#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2024 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2024	Abdelkrime Aries <kariminfo0@gmail.com>
# This class is a modified version of transformers.data.data_collator.DataCollatorForLanguageModeling
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

# from torch.optim.optimizer import Optimizer as Optimizer
from transformers import Trainer
from torch.optim import AdamW

class CharBertTrainer(Trainer): 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = {}

    def training_step(self, model, inputs):
        self.inputs = inputs
        return super().training_step(model, inputs)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            outputs = model(**self.inputs, return_dict=True)
            self.log({
                "multi_label_loss": float(outputs.multi_label_loss),
                "masked_lm_loss": float(outputs.masked_lm_loss)
                })
            self.control.should_log = True
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
    
