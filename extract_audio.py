from apiclient.discovery import build
from apiclient.errors import HttpError
from pytube import YouTube
from oauth2client.tools import argparser
import subprocess

DEVELOPER_KEY = "AIzaSyCAKrxbdTZ4QHokFIW4UsV0iYMiN81wn1o"
API_SERVICE = "youtube"
API_VERSION = "v3"

def search(search_term):
	#reference: developers.google.com/api-client-library/python/apis
	#parameters: api_name, api_version
	youtube = build(API_SERVICE, API_VERSION, developerKey = DEVELOPER_KEY)
	
	#
	search_response = youtube.search().list(q=search_term, maxResults=1).execute()

def download_yt_video(args):
	url = args.url
	yt = YouTube(url)
	vid = yt.filter('mp4')[-1]
	vid.download('./mp4/')

	command = "ffmpeg -i ./mp4/\"" + yt.filename + ".mp4\" -ab 160k -ac 2 -ar 44100 -vn ./mp3/\"" + yt.filename + ".mp3\""
	print(command)
	subprocess.call(command, shell=True)

if __name__ == "__main__":
	argparser.add_argument("--url", help="URL to YouTube video")
	args = argparser.parse_args()
	download_yt_video(args)
