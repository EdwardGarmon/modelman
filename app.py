from flask import Flask,request,jsonify
import urllib.request
from model.buildscript import populate
from model.buildscript import load
import urllib.request

app = Flask(__name__)

files = ["MasterTokenizer.pkl","jerry_model.h5","elaine_model.h5","kramer_model.h5","george_model.h5"]
burl="https://nothingforevermodels.s3.us-east-2.amazonaws.com/"

def download():
    for name in files:
        urllib.request.urlretrieve(burl+name,"./model/"+name)


downloaded = False

@app.route('/')
def index():
    return "Index Page"

@app.route('/loadModel',methods=['GET'])
def l():
    download()
    load()
    global downloaded
    downloaded = True
    return("<p>Loaded model</p>")


@app.route('/populateScript',methods=['POST'])
def handle_script():
    global downloaded
    if(downloaded):
        content = request.get_json()
        req_script = content.get("req_script")
        temp = content.get("temp")
        return populate(temp,req_script)
    else:
        return("<p>Please run route /loadModel</p>")

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)


