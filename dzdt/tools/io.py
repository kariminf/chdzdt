#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2024 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2024	Abdelkrime Aries <kariminfo0@gmail.com>
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
from typing import Iterator


def read_filelines(url: str) -> Iterator[str]:
    if not os.path.isfile(url):
        raise FileNotFoundError(f"The file {url} does not exist!")
    with open(url, "r", encoding="utf8") as reader:
        for line in reader:
            yield line
    
def list_subfolders(url: str) -> Iterator[str]:
    if not os.path.isdir(url):
        raise FileNotFoundError(f"The folder {url} does not exist!")
    for subfolder in os.listdir(url):
        if os.path.isdir(os.path.join(url, subfolder)):
            yield subfolder


def list_files(url: str, suffix=None) -> Iterator[str]:
    if not os.path.isdir(url):
        raise FileNotFoundError(f"The folder {url} does not exist!")
    for file in os.listdir(url):
        if (
            os.path.isfile(os.path.join(url, file))
            and
            ((suffix is None) or (file.endswith(suffix)))
            ):
            yield file
