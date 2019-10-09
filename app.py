import os
import time
from flask import Flask, render_template, request
import textsearch as txt
from textsearch import backend as b
app = Flask(__name__)
backend = b()
@app.before_first_request
def tdm_generator():
  if(os.path.isfile('wordbankdoc.pickle') == False):
    print(" \n No word bank \n ")
    txt.InvInd()
    
@app.route('/')
def first():
  return render_template("home.html")
 
@app.route('/home')
def home():
    return render_template("home.html")
    
@app.route('/search', methods = ['POST','GET'])
def search():
    return render_template("index.html")

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
  
if __name__ == "__main__":
    app.run()