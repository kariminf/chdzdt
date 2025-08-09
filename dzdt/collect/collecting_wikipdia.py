#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023	Abdelkrime Aries <kariminfo0@gmail.com>
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

from typing  import List, Any, Union
# from pathlib import Path
import requests
import re


WIKI_EN_FEAT_URL = "https://en.wikipedia.org/wiki/Wikipedia:Featured_articles"
WIKI_EN_GOOD_URL = "https://en.wikipedia.org/wiki/Wikipedia:Good_articles/all"


WIKI_EXPORT   = "https://{lang}.wikipedia.org/w/index.php?title=Special:Export"
WIKI          = "https://{lang}.wikipedia.org/wiki/"


SPECIAL_PAGES = {
    "en": {"featured": "Category:Featured_articles", "good": "Category:Good_articles"},
    "fr": {"featured": "Catégorie:Article_de_qualité", "good": "Catégorie:Bon_article"},
    "ar": {"featured": "تصنيف:مقالات_مختارة", "good": "تصنيف:مقالات_ويكيبيديا_الجيدة"}
}

wiki_art_founder = re.compile('href="/wiki/([^:"]+)"')

wiki_category_founder = re.compile("<textarea[^>]*>([^<]*)")



# ===========================================
# Links extraction from a Wikipedia page
# ===========================================

def extract_wikipedia_titles(url: str) -> List[str]:
    """Extract wikipedia links from a given page. 
       It does not extract those having a colon (:)

    Args:
        url (str): The pages URL.

    Returns:
        List[str]: A list of wikipedia pages' titles.
    """

    resp = requests.get(url)
    return [requests.utils.unquote(u) for u in wiki_art_founder.findall(resp.text)]


# ===========================================
# Downloading Wikipedia pages
# ===========================================

def down_wikipedia_articles(export_url: str, pages_urls: List[str]) -> requests.Response:
    """Get Wikipedia articles using Wikipedia Extract API.

    Args:
        export_url (str): The URL of the API.
        pages_urls (List[str]): List of pages' titles.

    Returns:
        requests.Response: A response containing the articles as a compressed XML file.
    """

    data = {
        "action" : "submit", # Unused; set to "submit" in the export form.
        "pages"  : "\n".join(pages_urls), #A list of page titles, separated by linefeed (%0A) characters. Maximum of 35 pages.
        "curonly": True, # Include only the current revision (default for GET requests).
        }
    headers = {"Accept-Encoding": "gzip,deflate"}
    return requests.post(export_url, data=data, headers=headers)


# ===========================================
# Wikipedia category management
# ===========================================

def expand_wikipedia_category(export_url: str, category: str) -> List[str]:
    """Given a category, use Wikipedia Export API to get the list of its members.
       The API is limited to 5000 members, otherwise use other tools to extract 
       the members from the category's page.

    Args:
        export_url (str): The URL of the API.
        category (str): The category.

    Returns:
        List[str]: A list of members.
    """

    req = export_url + "&addcat&catname=" + category
    resp = requests.post(req)
    resp = wiki_category_founder.findall(resp.text)
    if not resp:
        return []
    resp = resp[0]
    resp = requests.utils.unquote(resp)
    return resp.split("\n")


def get_wikicat_pages_names(export_url: str, category: str, depth: int = 8) -> List[str]:
    """Get non category page's names from a categoy. 
       The program explores the categories seqquentially; given a categoy, it stores its non category pages
       and explore the sub-categories. The exploration process is limited by a depth parameter.

    Args:
        export_url (str): The URL of the API.
        category (str): The category.
        depth (int, optional): The depth of exploration. Defaults to 8.

    Raises:
        Exception: A category must be a category label, followed by a colon, then the name of the categoy.

    Returns:
        List[str]: A list of non category pages. It may contain categories whith the max depth.
    """

    if ":" not in category:
        raise Exception("A category must be a category label, followed by a colon, then the name of the categoy")
    
    cat_label = category[:category.index(":")+1]

    remained_categories = [category.strip()] # A set is more logical, but cannot draw elements from a set without transforming it
    remained_depths = [0]
    visited_categories = []
    added_articles = []

    while remained_categories:
        category = remained_categories.pop(0)
        current_depth = remained_depths.pop(0) + 1
        print("checking", category, "... still:", len(remained_categories), "articles:", len(added_articles))
        found = expand_wikipedia_category(export_url, category)
        visited_categories.append(category)
        for page in found:
            page = page.strip()
            # If the category is very deep, add it to the added articles (the user can delete or use it somehow)
            if page.startswith(cat_label) and current_depth < depth:
                if (page not in visited_categories) and (page not in remained_categories):
                    remained_categories.append(page)
                    remained_depths.append(current_depth)
            elif page not in added_articles:
                added_articles.append(page)
    
    return added_articles


# def save_all_wikipedia_articles():
#     # save_wikipedia_articles(WIKI_EN_EXPORT, "enwiki/enwiki_featured", "enwiki_featured_urls.txt")
#     # save_wikipedia_articles(WIKI_EN_EXPORT, "enwiki/enwiki_good", "enwiki_good_urls.txt")
#     save_wikipedia_articles(WIKI_EXPORT, "enwiki/enwiki_algeria", "enwiki_algeria_urls_remain.txt", offset=53)
