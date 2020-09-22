from flask import Flask,request,jsonify
import urllib.request
from model.buildscript import populate

app = Flask(__name__)

@app.route('/')
def index():
    return "Index Page"

@app.route('/populateScript',methods=['POST'])
def handle_script():
    content = request.get_json()
    req_script = content.get("req_script")
    temp = content.get("temp")
    return populate(temp,req_script)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)


