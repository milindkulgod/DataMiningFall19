import os
import time
from flask import Flask, render_template, request

from textsearch import backend as b
from image import backend_img as bim
app = Flask(__name__)
backend = b()
backend_img = bim()
@app.before_first_request
def tdm_generator():
  if(os.path.isfile('wordbanknb.pickle') == False):
    print(" \n No word bank \n ")
    backend.InvInd()
  if(os.path.isfile('imgwordbankdoc.pickle') == False):
    backend_img.InvIndim()
    
@app.route('/')
def first():
    return render_template("home.html")
 
@app.route('/home')
def home():
    return render_template("home.html")
    
@app.route('/search', methods = ['POST','GET'])
def search():
    return render_template("index.html")
    
@app.route('/classify', methods = ['POST','GET'])
def classify():
    return render_template("review_index.html")

@app.route('/image', methods = ['POST','GET'])
def image():
    return render_template("image_index.html")

@app.route('/class_result', methods = ['POST','GET'])
def class_result():
  global backend 
  if request.method == 'POST':
    review = request.form['review']
    start = time.time()
    result = backend.classify(str(review))
    end = time.time()
    print("\n Classification result retrived in ", end-start," \n")
    return render_template("display_class.html", review=review, result=dict(result)) 
    
@app.route('/result', methods = ['POST','GET'])
def result():
  global backend
  if request.method == 'POST':
    query = request.form['search']
    start = time.time()
    result = backend.topk(str(query))
    end = time.time()
    if type(result) != str:
      
      print("Query retrive time result:",end - start,"\n")
      return render_template("display_result.html", query=query, result = result.to_html())
    else:
      
      print("Query retrive time  no :",end - start)
      return render_template("display_result.html", query=query, result=result)
      
@app.route('/image_result', methods = ['POST','GET'])
def image_result():
    global backend_img
    if request.method == 'POST':
        query = request.form['search']
        start = time.time()
        result = backend_img.topk_img(str(query))
        end = time.time()
        if type(result) != str: 
            return render_template("image_result.html", query=query, result = result.to_dict(orient='records'))
        else:
      
            print("Query retrive time  no :",end - start)
            return render_template("display_result.html", query=query, result=result)  
if __name__ == "__main__":
    app.run()