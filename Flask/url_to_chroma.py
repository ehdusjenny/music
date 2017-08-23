import extract_audio
import parse_audio
from flask import Flask, request
app = Flask(__name__)

#@app.route("/", methods=['POST'])
@app.route("/")
def hello():
    #print(request.form['foo'])
    return "Hello World!"

if __name__ = "__main__":
    app.run(host="0.0.0.0", port="5000")
