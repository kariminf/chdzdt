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

from typing import List, Union
import getopt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dzdt.collect.collecting_wikipdia import (
    SPECIAL_PAGES, 
    WIKI_EN_FEAT_URL, 
    WIKI_EN_GOOD_URL, 
    WIKI_EXPORT,
    down_wikipedia_articles, 
    extract_wikipedia_titles, 
    get_wikicat_pages_names
    ) 

MAX_PAGES = 35
SCRIPT = os.path.basename(__file__)

def extract_save_wikipedia_titles(in_url: str, out_url: str) -> None:
    """Extract Wikipedia links from a given page and save them in a file.

    Args:
        in_url (str): The URL of the page containing the links (pages' titles).
        out_url (str): The URL of the output file.
    """

    l = extract_wikipedia_titles(in_url)
    with open(out_url, "w", encoding="utf8") as f:
        f.write("\n".join(l))

def save_wikipedia_pagenames(export_url: str, category: str, out_url: str, depth: int = 8) -> None:
    """Save all pages' names belonging to a category into a file.

    Args:
        export_url (str): The URL of the API.
        category (str): The category.
        out_url (str): _description_
        depth (int, optional): The depth of exploration. Defaults to 8.
    """

    l = get_wikicat_pages_names(export_url, category, depth=depth)
    with open(out_url, "w", encoding="utf8") as f:
        f.write("\n".join(l))

def special_titles_extract(ptype: str = "featured", lang: str = "en", out_url: str = None) -> None:
    out_url = out_url or (lang + "wiki_" + ptype + "_urls.txt")
    if lang == "en":
        if ptype == "featured":
            extract_save_wikipedia_titles(WIKI_EN_FEAT_URL, out_url)
        elif ptype == "good":
            extract_save_wikipedia_titles(WIKI_EN_GOOD_URL, out_url)
    else:
        export_url = WIKI_EXPORT.format(lang=lang)
        category = SPECIAL_PAGES[lang][ptype]
        save_wikipedia_pagenames(export_url, category, out_url, depth=1)

def save_wikipedia_articles(export_url: str, out_url: str, articles: Union[str, List[str]], offset: int=0) -> None:
    """Download articles and store them into many XML files. 
       The output files have a sequential number.
       The articles cannot be stored into one file since there is a maximum number of articles for each request.

    Args:
        export_url (str): The URL of the API.
        out_url (str): The output URL. It can be a folder and a prefix of the files.
        articles (Union[str, List[str]]): Either a URL of a file containing the list of pages' titles; or a list of pages' titles.
        offset (int, optional): The number which be used to start numbering files. Defaults to 0.

    Raises:
        Exception: This exception happens when either the name of the file nor the list of articles are afforded.
    """

    i = 0

    if isinstance(articles, str):
        with open(articles, "r") as f:
            article_names = [l.strip("\n") for l in f ] #f.readlines()
    elif isinstance(articles, list):
        article_names = articles
    else:
        raise Exception("The articles must be a url to a file or a list of strings")

    while True:
        start, end = i*MAX_PAGES, (i+1)*MAX_PAGES
        l = article_names[start:end]
        if len(l) == 0:
            break
        i += 1
        # print(l)
        out_url_i = out_url + str(i+offset) + ".xml"
        r = down_wikipedia_articles(export_url, l)
        print("processing ", out_url_i, "...")
        with open(out_url_i, "w", encoding="utf8") as f:
            f.write(r.text) 

def help():
    print("=========== HELP! ===========")
    print("Extract articles from Wikipedia")
    print("COMMAND:")
    print(f"\t {SCRIPT} [OPTIONS]")

    print("OPTIONS:")

    print("\t -h \t Show this help")

    print("\t -l <lang> \t Wikipedia language. By default: lang='en'")

    print("\t -p <process> \t The process type: [titles, featured, good, category, import]. By default: process='titles'")

    print("PROCESS:")

    print("\t -p titles \t Given a Wikipedia page, extract all links in it")
    print("\t\t -i <input> \t The URL of Wikipedia page containing the links")
    print("\t\t -o <output> \t The output where the pages titles are stored, each in a line")

    print("\t -p featured \t Extract featured articles titles")
    print("\t\t -o <output> \t The output where the pages titles are stored, each in a line")

    print("\t -p good \t Extract good articles titles")
    print("\t\t -o <output> \t The output where the pages titles are stored, each in a line")

    print("\t -p category \t Extract all members of a Wikipedia category")
    print("\t\t -d <depth> \t The depth of exploration. By default depth=1 (do not check sub-categories)")
    print("\t\t -c <category> \t The category. For example -c Category:Algeria")
    print("\t\t -o <output> \t The output where the pages titles are stored, each in a line")

    print("\t -p import \t Import Wikipedia pages as XML files and store them into a folder")
    print("\t\t -i <input> \t The URL of a file containing pages titles (one per line)")
    print("\t\t -o <output> \t The folder where the pages are extracted. You can add a file prefix after the folder URL")
    print("\t\t -s <start> \t The offset (the files are stored using a sequential numer). By default start=0")


def main(argv):

    ptype    = "titles"
    in_url   = None
    out_url  = None
    lang     = "en"
    depth    = 1
    category = None
    offset   = 0


    opts, args = getopt.getopt(argv,"hp:i:o:l:d:c:s:")
    for opt, arg in opts:
        if   opt == "-h":
            help()
            sys.exit()
        elif opt == "-p":
            ptype    = arg
        elif opt == "-i":
            in_url   = arg
        elif opt == "-o":
            out_url  = arg
        elif opt == "-l":
            lang     = arg
        elif opt == "-d":
            depth    = int(arg)
        elif opt == "-c":
            category = arg
        elif opt == "-s":
            offset   = int(arg)
    
    if ptype == "titles":
        extract_save_wikipedia_titles(in_url, out_url)
    elif ptype in ("featured", "good"):
        special_titles_extract(ptype, lang, out_url)
    elif ptype == "category":
        save_wikipedia_pagenames(WIKI_EXPORT.format(lang=lang), category, out_url, depth=depth)
    elif ptype == "import":
        save_wikipedia_articles(WIKI_EXPORT.format(lang=lang), out_url, in_url, offset=offset)
    else:
        print("command not supported, check help")
        help()

    

if __name__ == "__main__":
    main(sys.argv[1:])

