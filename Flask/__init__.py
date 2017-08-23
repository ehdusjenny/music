from flask import Flask, request
app = Flask(__name__)
@app.route("/hi", methods=['POST'])
def hello():
	return request.form['foo']
	#return "hi"
if __name__ == "__main__":
	app.run()
