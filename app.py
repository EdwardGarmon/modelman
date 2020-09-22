from flask import Flask,request,jsonify
import urllib.request
from model.buildscript import populate

burl = "https://nothingforevermodels.s3.us-east-2.amazonaws.com/"
app = Flask(__name__)
modelFiles = ["MasterTokenizer.pkl" ,"jerry_model.h5","elaine_model.h5","kramer_model.h5"]

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

for filename in modelFiles :
    print("downloading file : " + filename)
    urllib.request.urlretrieve(burl + filename, './model/' + filename)
print("done with model downloads")