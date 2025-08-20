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

class BatchPrinter(Recorder):
    def record(self, nbr):
        print(f"Batch {nbr} ...")