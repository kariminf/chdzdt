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

import os

class Recorder:
    def start(self, *args, **kwargs):
        pass
    def finish(self, *args, **kwargs):
        pass
    def record(self, *args, **kwargs):
        pass

class WritePrintRecorder(Recorder):
    def __init__(self, url):
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(url, exist_ok=True)
        self.writer = SummaryWriter(url)
    def finish(self):
        self.writer.close()
    def record(self, nbr, total_nbr, loss):
        print(f"Epoch [{nbr+1}/{total_nbr}], Loss: {loss:.4f}")
        self.writer.add_scalar("Loss/train", loss, nbr)

class MultiWritePrintRecorder(Recorder):
    def __init__(self, url):
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(url, exist_ok=True)
        self.writer = SummaryWriter(url)
    def finish(self):
        self.writer.close()
    def record(self, nbr, total_nbr, loss):
        output_features, cls_epoch_losses, tok_epoch_losses = loss

        cls_loss, tok_loss = 0.0, 0.0
        for i, feature in enumerate(output_features):
            cls_lossi, tok_lossi = cls_epoch_losses[i], tok_epoch_losses[i]
            self.writer.add_scalar(f"{feature} loss/ cls train", cls_lossi, nbr)
            self.writer.add_scalar(f"{feature} loss/ tok train", tok_lossi, nbr)
            cls_loss += cls_lossi
            tok_loss += tok_lossi

        self.writer.add_scalar("Total loss/ cls train", cls_loss, nbr)
        self.writer.add_scalar("Total loss/ tok train", tok_loss, nbr)
        
        print(f"Epoch [{nbr+1}/{total_nbr}], Total cls Loss: {cls_loss:.4f}, Total tok Loss: {tok_loss:.4f}")

class BatchPrinter(Recorder):
    def record(self, nbr):
        print(f"Batch {nbr} ...")