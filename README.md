# PharmEazy

## A Text Search Module using Python

The dataset used for this has been taken from [Kaggle](https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018/home).
This dataset consists of patient reviews for numerous drugs that have been prescribed for the symptoms that they have experienced, and how effective they have been.

The code was coded initially on Jupyter Notebook to have a stepwise visual response of the code flow.
The code was finalized on Notepad++.

Any other text editor will do just fine as long as they support the Python libraries that are being used.

For the application, Flask was used.

**Steps to run the flask app locally**
<body>
<pre>
1.Install flask in your local environment.

2.Set the FLASK_APP environment variable to app.py and run command **flask run** </b> 

or </b>

Directly run the app.py file by the command **python app.py**

3.Open **localhost:5000** in the web browser to see and interact with the app.
</pre>
</body>

To host the website, ngrok.exe was used.

To use ngrok:

<body>
 <pre>
 1. Download the application from www.ngrok.com
 
 2. Install the .exe file in your local folder and launch the application.
 
 3. A command prompt is opened. Type **ngrok http 5000** as 5000 is the port number.
 
 4. The local folder has been hosted, and the web app can be accessed through the url generated on running the .exe file.
 </pre>
 </body>

There is a fault with ngrok but, although it allows you to create a tunnel from the localhost to the server, data access becomes really slow over the time, which is unlikely with other hosting services like PythonAnywhere. The reason why ngrok was selected was that my dataset file size is big, thus exceeding the free upload linit in hosting services.
**Data Preprocessing:**


In order to create a text search module, first we need to have the appropriate dataset that we intend to work on.
On obtaining the data set, the dataset has to be processed, and brought to a comprehendable form.

Preprocessing of data of a large data set is a task, especially when it contains entries more than 100k. In the dataset that I have used, there are html codes that are present in place of special characters. For this, the unescape function of the html library has been used.

Text search will be performed  on the reviews column of the dataset, which contains long length of words. These words are separated from the string with the help of Python's split function.

In order to make the search more generalized, all characters have been converted into lowercase letters.

Next, we use the nltk library, which is probably one of the most useful libraries for natural language processing in Python. We import the stopwords function from the corpus of the module. We can utilize this to eliminate the stopwords that are present in the data. On completing this, we use the lemmatizer funtion in order to identify the root of the words, broadening the spectrum for the search. 


**Word Bank Creation:**


In order to make sure the search engine covers all the words that are present in the dataset, we have to create a word bank, which contains all the unique words that are present. The data structure that can be used for this is a python dictionary. This makes use of hash indexing, which is quick.

For each review, i.e document, we have to create a posting list, and this is possible by calculating the TF-IDF of the document.
(Term Frequency - Inverse Document Frequency).

**Query Analysis and Processing:**


We take an input from the user through a web application and process the query by calculating the word weight. On doing so, we retrieve the top k results that are required. The cosine similarity is calculated.

 If a review is not there in the first k elements, we will utilize weight in the kth element as the upper-bound on weight in the vector. thus finding the upper-bound score.
 
 ## Naive Bayes Classifier
 
 The task of classification is the process of predicting a class when a set of data points have been given. This classifier uses a training data to understand how the variables that are given as input, relate to a certain class. In this project, since we are carrying out classification on the basis of text, the Naive Bayes Classifier is the most appropriate classifier for the sccenario. Text classification plays a major role in Natural Language Processing as language does not exactly have a fixed schema and there is continuous learning taking place. The Naive Bayes Classifier is a classifier that is probabilistic. It is based on the Bayes Theorem, where an assumption is made, that the attributes are conditionally independent. The classification is carried out by calculating the probability of each given class, and displaying the classes with the highest probabilities as the output.
 
 In order to carry out the evaluation, the data can be divided into training data, testing data and validation data. In the given dataset, there are two .csv files that are separated as training and testing data. The validation dataset can be created by extracting it from the bigger training dataset in order to carry out the validation tests. Here, we will be displaying the accuracy of the classifier.

First, we need to calculate the prior probabilities, and this is done by creating a dictionary which contains the list of all documents, unique words and total number of words occuring in each class. Next, we create another dictionary within, for every class, in which words are keys, and store the frequency value of the word appearing and the number of reviews the word occured in the entire class. There is just one problem though, suppose if, the conditional probability turns out to be zero, that is, when the word in the query is absent in the vocabulary, then the entire probability becomes zero, that is, which isn't really helpful in terms of classification. Hence, we use something called the Laplace smoothing, which is used to regularize Naive Bayes. It is denoted as Alpha(the hyperparameter), done by adding 1 to the numerator, which helps in overcoming this problem. 

**Evaluation and Hyperparameter Tuning**

The Laplace smoothing is the hyperparameter. In case of hyperparameter tuning, the approach around the default was to change the value of the hyperparameter. Observations show that by decreasing the value of the hyperparameter, the accuracy increased.

## Imaage Caption Search

mage captioning works in the realms of Deep Learining. It is the process of generating a textual description of a given image. Deep Learning functions on two major fundamentals, Natural Language Processing and Computer Vision. This model undergoes two phases, namely training and testing. The training phase is where the image captioning model is trained with a set of images, and a testing set of images is used to test the trained model. The image will be linked to the corresponding output captions.

The image captioning model uses an attention-based model, which allows us to see the focus area of the image as the caption is generated. This is where Tensorflow's Keras and eager execution comes into picture. The dataset used for the image captioning model is the MS-COCO Dataset, which is a dataset of images easily accessible by everyone.

After the dataset is downloaded, a subset of the images are selected to train the model. For the preprocessing of images, Inception for classification.

Captions are tokenized to create a vocabulary of all unique words and create a word-index mapping.

The model is then trained. The model architecture is inspired by the Show, Attend and Tell paper.

After training the images, testing images were given as input in order to generate captions for freshly given images as input.

Most of the image captioning was done by the captioning model that was available on the tensorflow github repository, the next task was to use the captions generated as input for a search engine and display images containing the keywords in the caption.

After training the model, a csv file was opened, and, with the help of a for loop, the testing part of the model was made to run for the number of iterations that were equal to the number of images that were being tested. The caption, along with the url of the image were recorded in a .csv file and the captions were tokenized, bringing it into a suitable format to run it through the search engine, following the same procedure of calculating the TF-IDF scores and displaying the top-k results along with the image.

[Video Link](https://youtu.be/MXC41mWMOWY)
