import extract_audio
import parse_audio
from flask import Flask, request
import base64
import json
app = Flask(__name__)

#@app.route("/", methods=['POST'])
@app.route("/")
def hello():
    #print(request.form['foo'])
    return "Hello World!"

@app.route("/hello")
def hello2():
    return json.dumps([[1,2,3],[4,5,6]])

@app.route("/hello/<name>")
def hello3(name):
    return "Hello %s!" % name

@app.route("/dl/<vid_id>")
def download(vid_id):
    file_name = extract_audio.download_yt_video("https://www.youtube.com/watch?v=%s" % vid_id)
    chroma, sample_rate = parse_audio.compute_chroma(file_name)
    return json.dumps(chroma.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")
