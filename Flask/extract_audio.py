"""
    Downloads YouTube videos and saves the audio as mp3 files in /mp3.
"""

import os
from apiclient.discovery import build
from apiclient.errors import HttpError
from pytube import YouTube
from oauth2client.tools import argparser
import subprocess

DEVELOPER_KEY = "AIzaSyCAKrxbdTZ4QHokFIW4UsV0iYMiN81wn1o"
API_SERVICE = "youtube"
API_VERSION = "v3"

MP4_DIR = "./mp4/"
MP3_DIR = "./mp3/"

def search(search_term):
    #reference: developers.google.com/api-client-library/python/apis
    #parameters: api_name, api_version
    youtube = build(API_SERVICE, API_VERSION, developerKey = DEVELOPER_KEY)
    
    search_response = youtube.search().list(q=search_term, maxResults=1).execute()

def download_yt_video(url):
    if not os.path.isdir(MP4_DIR):
        os.mkdir(MP4_DIR)
    if not os.path.isdir(MP3_DIR):
        os.mkdir(MP3_DIR)

    yt = YouTube(url)
    yt.filename = ''.join([i if ord(i) < 128 else '' for i in yt.filename])

    mp4_output = os.path.join(MP4_DIR, yt.filename + ".mp4")
    if os.path.exists(mp4_output):
        os.remove(mp4_output)

    vid = yt.filter('mp4')[-1]
    vid.download(MP4_DIR)

    mp3_output = os.path.join(MP3_DIR, yt.filename + ".mp3")
    if os.path.exists(mp3_output):
        os.remove(mp3_output)
    command = "ffmpeg -i \"%s\" -ab 160k -ac 2 -ar 44100 -vn \"%s\"" % (mp4_output, mp3_output)
    print(command)
    subprocess.call(command, shell=True)
    return mp3_output

def download_by_id(vid_id):
    """
    Download a YouTube video given the video's ID, and return the path to the
    video. If the video has already been downloaded, then return the path to
    that file without redownloading.
    """
    if not os.path.isdir(MP4_DIR):
        os.mkdir(MP4_DIR)
    if not os.path.isdir(MP3_DIR):
        os.mkdir(MP3_DIR)

    url = "https://www.youtube.com/watch?v=%s" % vid_id
    yt = YouTube(url)
    yt.filename = vid_id

    # Check if the mp3 already exists
    mp3_output = os.path.join(MP3_DIR, yt.filename + ".mp3")
    if os.path.exists(mp3_output):
        return mp3_output

    # Download full video
    mp4_output = os.path.join(MP4_DIR, yt.filename + ".mp4")
    if not os.path.exists(mp4_output):
        vid = yt.filter('mp4')[-1]
        vid.download(MP4_DIR)

    # Extract audio from video
    mp3_output = os.path.join(MP3_DIR, vid_id + ".mp3")
    if not os.path.exists(mp3_output):
        command = "ffmpeg -i \"%s\" -ab 160k -ac 2 -ar 44100 -vn \"%s\"" % (mp4_output, mp3_output)
        subprocess.call(command, shell=True)

    return mp3_output

if __name__ == "__main__":
    argparser.add_argument("--url", help="URL to YouTube video")
    args = argparser.parse_args()
    download_yt_video(args.url)
