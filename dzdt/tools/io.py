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
from typing import Dict, List


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

    # added here in case the class defines another cache location
    CACHE_DIR = Path.home() / ".cache" / "dzdt_hub"

    # added here in case the class adds new providers
    PROVIDERS = {
        "hagginface": "https://huggingface.co/{model}/resolve/main/{file}?download=true",
        "github": "https://github.com/{model}/releases/download/{file}", # file = tag/file_name
    }

    # the class must define this: provider: list of files
    variants: Dict[str, List[str]] = {}

    @classmethod
    def load_from_hub(cls, model: str, variant: str, provider: str="hagginface", force_download=False, **kwargs):
        """
        Generic loader that checks local cache or downloads.
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(f"the provider '{provider}' not found. Available: {list(cls.PROVIDERS.keys())}")
        if variant not in cls.variants:
            raise ValueError(f"Variant '{variant}' not found. Available: {list(cls.variants.keys())}")

        cache_path = cls.CACHE_DIR / provider / model.replace("/", "_") / variant
        cache_path.mkdir(parents=True, exist_ok=True)

        # expected files
        local_files = {fname: cache_path / fname for fname in cls.variants[variant]}

        # download if missing or forced
        if force_download or not all(p.exists() for p in local_files.values()):
            print(f"Downloading {provider}/{model}/{variant} ...")
            cls._download_files(provider, model, local_files)

        print(f"Loaded {cls.__name__} from cache:", cache_path)
        return cls._load_from_files(local_files, **kwargs)

    @classmethod
    def _load_from_files(cls, local_files, **kwargs):
        """Load object(s) from cached files."""
        raise NotImplementedError

    # ---- generic download method (can be overridden) ----
    @classmethod
    def _download_files(cls, provider, model, local_files):
        url_pattern = cls.PROVIDERS[provider]

        for fname, path in local_files.items():
            url = url_pattern.format(model=model, file=fname)
            print(f"Downloading {url} ...")
            r = requests.get(url)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)