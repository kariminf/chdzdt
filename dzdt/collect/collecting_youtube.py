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

from googleapiclient.discovery import build
from googleapiclient.errors    import HttpError
from pathlib                   import Path
from typing                    import Tuple, List
from html.parser               import HTMLParser
import sys, getopt
import re


MIN_CMT     = 20   # Minimum number of comments of a video to be considered
MAX_MSG_LEN = 1000 # Maximum allowed message length (in term of characters)
MIN_MSG_LEN = 10   # Minimum allowed message length (in term of characters)


class HTMLFilter(HTMLParser):
    text = ""
    # is_link = False
    def handle_data(self, data):
        # if not self.is_link:
        self.text += data

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.text += "\n"


def html2text(text:str) -> str:
    """Tranform HTML into text

    Args:
        text (str): HTML representation

    Returns:
        str: A text without HTML tags.
    """
    html = HTMLFilter()
    html.feed(text)
    return re.sub(r"\n+", "\n", html.text)


def text2quoted_csv(text:str) -> str:
    """Prepare a text to be quoted for CSV storage.

    Args:
        text (str): Non quoted text

    Returns:
        str: Quoted text.
    """
    # since tabulation is used in csv, we will sacrifice it and replace it by a blank
    # a quotation mark is escaped by double quotation mark
    return '"' + text.replace('"', '""').replace("\t", " ").replace("\r", "") + '"'


class YouTubeProcessor:
    def __init__(self, api_key: str):
        self.api = build("youtube", "v3", developerKey=api_key)

    def get_channel_videos(self, playlist_id: str) -> Tuple[List[object], bool]:
        """Extract a list of videos from a youtube playlist

        Args:
            api (object): An object created using "googleapiclient.discovery.build"
            playlist_id (str): The ID of the playlist 

        Returns:
            Tuple[List[object], bool]: A list of objects {"id": the video's ID, "title": the video's title}; a boolean indicating that there were no problems
        """
    
        request = self.api.playlistItems().list(
            part       = "snippet,contentDetails",
            maxResults = 50,
            playlistId = playlist_id
        )

        videos = []

        while request:
            try:
                response = request.execute()
            except HttpError:
                return videos, False

            for res in response["items"]:
                snippet = res["snippet"]
                videos.append({
                    "id"   : snippet["resourceId"]["videoId"],
                    "title": snippet["title"],
                })
            

            request = self.api.playlistItems().list_next(request, response)
        
        return videos, True


    def get_video_comments(self, video_id: str) -> Tuple[List[str], bool]:
        """Extract comments from a youtube video

        Args:
            api (object): An object created using "googleapiclient.discovery.build"
            video_id (str): The video's ID

        Returns:
            Tuple[List[str], bool]: A list of comments; a boolean indicating that there were no problems
        """
        
        # retrieve youtube video comments
        request = self.api.commentThreads().list(
            part      = "snippet",
            videoId   = video_id,
            maxResults= 100
        )

        comments = []
        while request:
            try:
                response = request.execute()
            except HttpError as err:
                next = False
                if err.resp.status == 403 and err.reason == "commentsDisabled":
                    next = True
                return comments, next
            
            for item in response["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(snippet)

            request = self.api.commentThreads().list_next(request, response)

        return comments, True

    def process_channel(self, channel_name: str, channel_id: str, out_folder: str, id: int=0) -> None:
        """Given a channel, Extract all comments of all videos and save them in two files
        having the same name as the channel:
        <ul>
        <li>[channel_name]_idx.csv: contains a sequential ID of the video, its URL and its title</li>
        <li>[channel_name]_cmt.csv: contains the ID of the comment, an empty language type, an empty polarity and the comment's text</li>
        </ul>

        Args:
            api (object): An object created using "googleapiclient.discovery.build"
            channel_name (str): The name of the youtube channel (the one that follows @)
            channel_id (str): The channel's ID
            out_folder (str): The output folder where the files will be stored
            id (int, optional): The start ID for videos. Defaults to 0.
        """

        f_idx = open(out_folder + channel_name + "_idx.csv", "w", encoding="utf8")
        f_cmt = open(out_folder + channel_name + "_cmt.csv", "w", encoding="utf8")

        f_idx.write("vid\tchannel\turl\ttitle\n")
        f_cmt.write("vid\tlang_type\tpolarity\ttext\n")
        

        playlist_id    = list(channel_id)
        playlist_id[1] = "U"
        playlist_id    = "".join(playlist_id)
        videos, next = self.get_channel_videos(playlist_id)
        # If the quotats are done, no need to recover messages
        if next:
            for video in videos:
                comments, next = self.get_video_comments(video["id"])

                if len(comments) >= MIN_CMT:
                    id += 1
                    f_idx.write(str(id) + "\t" + channel_name + "\t"+ video["id"] + "\t" + text2quoted_csv(video["title"]) + "\n")

                    for comment in comments:
                        msg = comment["textDisplay"]
                        if MIN_MSG_LEN <= len(msg) <= MAX_MSG_LEN:
                            f_cmt.write(str(id) + "\t\t\t" + text2quoted_csv(html2text(msg)) + "\n")

                if not next: 
                    break

        f_idx.close()
        f_cmt.close()
