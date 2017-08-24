import extract_audio
import parse_audio
from flask import Flask, request
app = Flask(__name__)

#@app.route("/", methods=['POST'])
@app.route("/")
def hello():
    #print(request.form['foo'])
    return "Hello World!"

@app.route("/hello")
def hello2():
    return "Moo"

@app.route("/hello/<name>")
def hello3(name):
    return "Hello %s!" % name

@app.route("/dl/<vid_id>")
def download(vid_id):
    extract_audio.download_yt_video("https://www.youtube.com/watch?v=%s" % vid_id)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")
