import numpy as np
import pandas as pd
import pickle
import string 
import html
import ast
import re
import nltk
import time
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
stpwrds = stopwords.words("english")
from nltk.stem import WordNetLemmatizer
from itertools import chain
from sklearn.model_selection import train_test_split
from ast import literal_eval

class backend():
    def __init__(self):
        self.dataset = pd.read_csv("full1.csv")
        try:
            with open("wordbankdoc.pickle", "rb") as pic:
                self.wordbank = pickle.load(pic)
            print("\n dict ldd\n")
        except: 
            start = time.time()
            print("\nCreating word bank for the full data\n")
            self.wordbank = self.InvInd(self.dataset)
            with open('wordbankdoc.pickle',"wb") as p:
                pickle.dump(self.wordbank, p)
        try:
            with open("drugwordbanknb.pickle", "rb") as cip:
                self.naivecp = pickle.load(cip)
        except:
            start = time.time()
            print("\nCalculating priops for the NBC\n")
            self.naivecp = self.MultiNaiveBayes()
            with open('drugwordbanknb.pickle',"wb") as p:
                pickle.dump(self.naivecp, p)
            end = time.time()
            print("Done...!", end-start," time\n")
    def encode_the_reviews(self,review):
        return html.unescape(review)
    
    def elim_stopword(self,r):
        r_n = " ".join([i for i in r if i not in stpwrds])
        return r_n
        
    def lem(self,tokens):
        l = WordNetLemmatizer()
        out = [l.lemmatize(word) for word in tokens]
        return out

    def InvInd(self,dataset,n=0):
        dataset['review']=dataset['review'].apply(str)
        dataset['review']=dataset['review'].apply(self.encode_the_reviews)
        response = dataset['review'].str.replace("[^a-zA-Z]", " ")
        response = response.apply(lambda r: " ".join([w for w in r.split() if len(w)>2]))
        response = [self.elim_stopword(r.split()) for r in response]
        response = [r.lower() for r in response]
        response = pd.Series(response)
        word_tokens = response.apply(lambda r: r.split())
        response = word_tokens.apply(self.lem)
        print("\n start vocab")
        wordbank = {}
                
        for i,r in enumerate(response, start=0):
            for j,w in enumerate(r , start=0):
                if w not in wordbank:
                    wordbank[w] = [1,{i:[j]}] #seeing word for first time. First value is 1, document count. second element is Document id and position at which word occurs.
                else:
                    if i not in wordbank[w][1]: #if word exists, go to second element. Same word, different document, increment the document count and add to dictionary.
                        wordbank[w][0] += 1
                        wordbank[w][1][i] = [j] #one document, word can occur many times.
                    else:
                        if j not in wordbank[w][1][i]: #both word and document id exists. Append to the list. Create wordbank
                            wordbank[w][1][i].append(j)

        N = np.float64(dataset.shape[0])                    

        for w in wordbank.keys(): #every word in dictionary
            plist = {} #Creation of posting list.
            for i in wordbank[w][1].keys(): #Keys of document dictionary. For every word, for every document the word occurs in.
                tf = (len(wordbank[w][1][i])/len(response[i])) #number of times the term occurs in the document by number of all terms in document.
                weight_i = (1 + np.log10(tf)) * np.log10(N/wordbank[w][0]) #weight calculation. 
                plist[i] = weight_i
            wordbank[w].append(plist) #every document has weight. Append to list. plist is formed. Appended to give third element.
        if n == 1:
          return wordbank  
        else:
           with open('wordbanknb.pickle',"wb") as p:
                pickle.dump(wordbank,p)#save the file

    def ttv_splitter(self,data, cat):
        train = pd.DataFrame()
        validation = pd.DataFrame()
        test = pd.DataFrame()

        categories = cat#list(set(y_all))
    

        for c in categories:
            s = data.loc[data['condition'] == c]
            if s.shape[0] < 4 :
              train = train.append(s)
            else:
              tr, intr = train_test_split( s , test_size=0.5, random_state=666 )
              tst, val = train_test_split( intr , test_size=0.5, random_state=777 )
              train = train.append(tr)
              validation = validation.append(val)
              test = test.append(tst)    
        print("data:",data.shape," train:",train.shape," test:",test.shape," validation:", validation.shape )
        return train, validation, test
       
    def MultiNaiveBayes(self):
        data = self.dataset
        categories = list(set(data.condition))
        train, validation, test = self.ttv_splitter(data, categories)
    
        try:
            with open("drugwordbanknb.pickle", "rb") as pic:
                vocabulary = pickle.load(pic)
        except:
            start = time.time()
            print("\nCreating vocab for the train data\n")
            vocabulary = self.InvInd(train,1)
            with open('drugwordbanknb.pickle',"wb") as p:
                pickle.dump(vocabulary,p)
            end = time.time()
            print("Done...!", end-start," time\n")
        
    

        print("\nLength of classes:", len(categories), "\n")
        classp = {}
        for c in categories:
            s = train.loc[train['condition'] == c]

            reviews = s.vector.apply(literal_eval)##['review'].apply(lambda r: r.split())
            uniqwrds = list(set(chain(*reviews)))
            unw = []

                                             
            for w in uniqwrds:
                if w in vocabulary.keys():
                    unw.append(w)
            wc = len(list(chain(*reviews)))
            classp[c] = [ s.revID.tolist(), unw, wc]
    
        print("\n\n.........................................STAGE 1 COMPLETED!!.............................................\n\n")
        naivecps = {}
        for c in classp.keys():
            naivecps[c] = [len(classp[c][0])/train.shape[0], classp[c][2], len(classp[c][0])]
            wrdict = {}
            for w in classp[c][1]:
                lent = 0
                count = 0
                for i in classp[c][0]:
                    if i in vocabulary[w][1].keys():
                        lent += len(vocabulary[w][1][i])
                        count += 1
                wrdict[w] = [lent, count]
            naivecps[c].append(wrdict)
      
      
        print("\n\n.........................................STAGE 2 COMPLETED!!.............................................\n\n", len(vocabulary.keys()))
        return  naivecps
    
    def topk(self,query):
        with open("wordbanknb.pickle", "rb") as pic:
            self.wordbank = pickle.load(pic)

        q = query.replace("[^a-zA-Z]", " ").lower()
        q_vec = self.elim_stopword(q.split())  #Query preprocessing
        q_vect = self.lem(q_vec.split())
        
        srtdplist = {}
        qw = {}
        x = 0
        for w in q_vect:    #going through all words in query. 
            
            print("\n loop in\n")        
            if w in self.wordbank.keys(): #if word exists
                if w not in srtdplist.keys():
                    print("\n in if ")
                    srtdplist[w] = sorted(self.wordbank[w][2].items(), key=lambda x:x[1], reverse=True)[:10] #top 10 results are retrieved.
                print(x,"\n  q words in vocab \n")
            if w not in qw:  #weight of the word occured in the query
                qw[w] = [1,(1/len(q_vect))] #no. of times word occurs by the query length. If word is seen for first time, count = 1
            elif w in qw: #if word already exists in query, increment count by 1 
                qw[w][0] += 1 
                qw[w][1] = (qw[w][0]/len(q_vect)) #calculates new weight with new count. returns tuples.
        if srtdplist == {}: 
            return "No results found"
        
        topk = [] #empty list to store (document id, weight) tuples.
        N = self.dataset.shape[0]  
        for i in range(N): #iterates through all document ids
            count = 0 #current doc id
            sd = 0 #weight
            for w in q_vect: 
                for (di,wt) in srtdplist[w]: #for every word, (document id, weight) tuple  
                    if di == i: count += 1 #if document id is the current document, increment the count.
            if count > 0 and count == len(q_vect):  #count of number of times a document has occured
                for w in q_vect:
                    l = [x for x in srtdplist[w] if x[0] == i] 
                    sd += l[0][1] * qw[w][1] #calculation of weight from the plist. Final weight calculated.
                topk.append((i,sd)) 
            elif count > 0 and count < len(q_vec): 
                
                for w in q_vect:
                    l = srtdplist[w][9] #document that hasnt occured.
                    sd += l[1] * qw[w][1] 
                topk.append((i,sd))  
                
        
        show = [x for x in sorted(topk, key=lambda i:i[1], reverse=True)]        
        out = []
        for (ind,s) in show:
             out.append( [self.dataset.loc[self.dataset.index[ind], 'drugName'], self.dataset.loc[self.dataset.index[ind], 'usefulCount'], self.dataset.loc[self.dataset.index[ind], 'condition'], self.dataset.loc[self.dataset.index[ind], 'rating'], self.dataset.loc[self.dataset.index[ind], 'review'], s])
        
        
        pd.set_option('display.max_columns', -1)  
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', -1)
        out =  pd.DataFrame(out, columns=['Drug Name','Useful count','Condition','Rating(/10)','Review','Score'])
      
        return out  
    
    def classify(self, review, a=0.00000001):
        naivecps = self.naivecp
        #naivevocablen = self.naivecp['vocablen']
        newr = review.replace("[^a-zA-Z]", " ").lower()
        newr = self.elim_stopword(newr.split())

        new_rev_vec = self.lem(newr.split())
        #print(new_rev_vec, "\n")
        
        out = {}
        
        for c in naivecps.keys():
         
            prc = np.log10(naivecps[c][0]) 
            for w in new_rev_vec:
                
                if w in naivecps[c][3].keys():
                    prc +=  np.log10((naivecps[c][3][w][0] + a)/(naivecps[c][1] + 21167 ))
                else:
                    prc +=  np.log10((a)/(naivecps[c][1] + 21167 ))
                
            out[c] = prc      
        
                
        sortout = sorted(out.items(), key=lambda x:x[1], reverse=True)[:10]
        #print("res:\n",sortout,"\n")
        return sortout
