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
import re
from typing import Iterator
import requests
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


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

hub_path_style = re.compile("^([^:]+):([^:]+)(.*)?$")
def process_hub_path(hub_path):
    result = {
        "provider": None,
        "model": None,
        "variant": None
    }
    m = hub_path_style.match(hub_path)
    if m:
        result["provider"] = m.group(1)
        result["model"] = m.group(2)
        if m.group(3):
            result["variant"] = m.group(3)[1:]

    return result



class HubMixin:

    # added here in case the class defines another cache location
    CACHE_DIR = Path.home() / ".cache" / "dzdt_hub"

    # added here in case the class adds new providers
    PROVIDERS = {
        "huggingface": "https://huggingface.co/{model}/resolve/main/{file}?download=true",
        "github": "https://github.com/{model}/releases/download/{file}", 
    }

    files: List[str] = []

    @classmethod
    def load_from_hub(cls, model: str, variant: str = None, provider: str="huggingface", **kwargs):
        """
        Generic loader that checks local cache or downloads.
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(f"the provider '{provider}' not found. Available: {list(cls.PROVIDERS.keys())}")
        
        cache_path = cls.CACHE_DIR / provider / model.replace("/", "_")
        cache_path.mkdir(parents=True, exist_ok=True)

        # expected files
        if variant is None:
            local_files = {fname: cache_path / fname for fname in cls.files}
        else:
            local_files = {fname: cache_path / f"{variant}_{fname}" for fname in cls.files}

        
        force_download = kwargs.get("force_download", False)
        # download if missing or forced
        if force_download or not all(p.exists() for p in local_files.values()):
            print(f"Downloading {provider}/{model}/{variant} ...")
            cls._download_files(provider, model, variant, local_files, **kwargs)

        print(f"Loaded {cls.__name__} from cache:", cache_path)
        return cls._load_from_files(local_files, **kwargs)

    @classmethod
    def _download_files(cls, provider, model, variant, local_files, **kwargs):
        url_pattern = cls.PROVIDERS[provider]

        headers = {}
        if "token" in kwargs:
            headers["Authorization"] = f"Bearer {kwargs['token']}"

        for fname, path in local_files.items():
            if variant is not None:
                fname = f"{variant}_{fname}"
            url = url_pattern.format(model=model, file=fname)
            print(f"Downloading {url} ...")
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=fname
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            # r = requests.get(url, headers=headers)
            # r.raise_for_status()
            # with open(path, "wb") as f:
            #     f.write(r.content)
    
    @classmethod
    def _load_from_files(cls, local_files: Dict[str, str], **kwargs):
        """Load object(s) from cached files."""
        raise NotImplementedError("Subclasses must implement _load_from_files.")