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

from pathlib import Path
import getopt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.collect.collecting_youtube import YouTubeProcessor

SCRIPT = os.path.basename(__file__)

def download_youtube_comments(out_folder: str, api_key_url: str, list_url: str) -> None:
    """Download all comments from a list of channels

    Args:
        out_folder (str): The output folder (where to store files containing comments)
        api_key_url (str): A file containing Youtube API's key
        list_url (str): A file containing a list of channels to download.
                        Each line contains the channels name, ID and an indicator to download or not (NEW means download).
                        These components are separated by a comma (,)
    """

    with open(api_key_url) as f:
        api_key = f.readline().strip("\n")
    
    youtube = YouTubeProcessor(api_key)

    with open(list_url, "r") as f:
        for l in f:
            ch = l.strip("\n").split(",")
            if len(ch) == 3 and ch[2] == "NEW":
                print("processing channel " + ch[0])
                youtube.process_channel(ch[0], ch[1], out_folder)


# =============================================
#          Command line functions
# =============================================

def help():
    print("=========== HELP! ===========")
    print("Extract comments from youtube channels")
    print("COMMAND:")
    print("\t collect_youtube [OPTIONS]")

    print("OPTIONS:")

    print("\t -h, --help \t Show this help")

    print("\t -k, --key \t URL of the file containing Youtube API key")
    print("\t\t\t By default the URL is '[HOME FOLDER]/.youtube_api_key'")
    print("\t\t\t Get the key here: https://developers.google.com/youtube/v3/getting-started")

    print("\t -l, --list \t URL of the file containing channels list")
    print("\t\t\t By default the URL is 'channels_list.csv'")
    print("\t\t\t Each line of this file is like: 'channel_name,channel_id,NEW'")
    print("\t\t\t If you change the indicator NEW, the channel will not be processed")

    print("\t -o, --out \t output folder")
    print("\t\t\t By default the URL is 'data/'")



def main(argv):
    out_folder  = "data/"
    api_key_url = Path().home() / ".youtube_api_key"
    list_url    = "channels_list.csv"

    opts, args = getopt.getopt(argv,"hk:l:o:",["help", "key=", "list=", "out="])
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            help()
            exit()
        elif opt in ("-k", "--key"):
            api_key_url = arg
        elif opt in ("-l", "--list"):
            list_url = arg
        elif opt in ("-o", "--out"):
            out_folder = arg

    download_youtube_comments(out_folder, api_key_url, list_url)


if __name__ == "__main__":
    main(sys.argv[1:])
