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
import requests
from pathlib import Path

CACHE_DIR = Path.home() / ".dzdt_cache"

SUPPLIERS = {
    "hagginface": "https://huggingface.co/{model}/blob/main/{file}"
}


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


class HubMixin:

    @classmethod
    def load_from_hub(cls, supplier: str, model: str, variant: str, force_download=False, **kwargs):
        """
        Generic loader that checks local cache or downloads.
        """
        cache_path = cls.CACHE_DIR / supplier / model.replace("/", "_") / variant
        cache_path.mkdir(parents=True, exist_ok=True)

        # expected files
        required_files = cls.required_files()
        local_files = {fname: cache_path / fname for fname in required_files}

        # download if missing or forced
        if force_download or not all(p.exists() for p in local_files.values()):
            print(f"Downloading {cls.__name__} from {supplier}/{model}/{variant} ...")
            cls._download_files(supplier, model, variant, local_files)

        print(f"Loaded {cls.__name__} from cache:", cache_path)
        return cls._load_from_files(local_files, **kwargs)

    # ---- methods subclasses must implement ----
    @classmethod
    def required_files(cls):
        """Return list of required filenames for this resource."""
        raise NotImplementedError

    @classmethod
    def _load_from_files(cls, local_files, **kwargs):
        """Load object(s) from cached files."""
        raise NotImplementedError

    # ---- generic download method (can be overridden) ----
    @classmethod
    def _download_files(cls, supplier, model, local_files):
        url_pattern = cls.SUPPLIERS[supplier]

        for fname, path in local_files.items():
            url = url_pattern.format(model=model, file=fname)
            print(f"Downloading {url} → {path}")
            r = requests.get(url)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)