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

class backend():
    def __init__(self):
        self.dataset = pd.read_csv("full1.csv")
        try:
            with open("wordbankdoc.pickle", "rb") as pic:
                self.wordbank = pickle.load(pic)
        except: 
            start = time.time()
            print("\nCreating vocab for the full data\n")
            self.wordbank = scr.InvInd(self.dataset)
            with open('wordbankdoc.pickle',"wb") as p:
                pickle.dump(self.wordbank, p)

    def encode_the_reviews(self,review):
        return html.unescape(review)
	
    def elim_stopword(self,r):
        r_n = " ".join([i for i in r if i not in stpwrds])
        return r_n
        
    def lem(self,tokens):
        l = WordNetLemmatizer()
        out = [l.lemmatize(word) for word in tokens]
        return out

    def InvInd(self,dataset):
        dataset['review']=dataset['review'].apply(str)
        dataset['review']=dataset['review'].apply(self.encode_the_reviews)
        response = dataset['review'].str.replace("[^a-zA-Z]", " ")
        response = response.apply(lambda r: " ".join([w for w in r.split() if len(w)>2]))
        response = [self.elim_stopword(r.split()) for r in response]
        response = [r.lower() for r in response]
        response = pd.Series(response)
        word_tokens = response.apply(lambda r: r.split())
        response = word_tokens.apply(self.lem)
                
        wordbank = {}
                
        for i,r in enumerate(response, start=0):
                for j,w in enumerate(r , start=0):
                        if w not in wordbank:
                            wordbank[w] = [1,{i:[j]}]
                        else:
                            if i not in wordbank[w][1]:
                                    wordbank[w][0] += 1
                                    wordbank[w][1][i] = [j]
                            else:
                                    if j not in wordbank[w][1][i]:
                                        wordbank[w][1][i].append(j)

        N = np.float64(dataset.shape[0])                    

        for w in wordbank.keys():
            plist = {}
            for i in wordbank[w][1].keys():
                tf = (len(wordbank[w][1][i])/len(response[i]))
                weight_i = (1 + np.log10(tf)) * np.log10(N/wordbank[w][0])
                plist[i] = weight_i
            wordbank[w].append(plist)
        p = open('wordbankdoc.pickle',"wb")
        pickle.dump(wordbank,p)


    def topk(self,query):

        q = query.replace("[^a-zA-Z]", " ").lower()
        q_vec = self.elim_stopword(q.split())
        q_vect = self.lem(q_vec.split())
        
        srtdplist = {}
        qw = {}
        for w in q_vect:
            if w in self.wordbank.keys():
                if w not in srtdplist.keys():
                    srtdplist[w] = sorted(self.wordbank[w][2].items(), key=lambda x:x[1], reverse=True)[:10]
            if w not in qw:
                qw[w] = [1,(1/len(q_vect))]
            elif w in qw:
                qw[w][0] += 1
                qw[w][1] = (qw[w][0]/len(q_vect))
        if srtdplist == {}:
            return "No results found"
        
        topk = []
        N = self.dataset.shape[0]
        for i in range(N):
            count = 0
            sd = 0
            for w in q_vect:
                for (di,wt) in srtdplist[w]:
                    if di == i: count += 1
            if count > 0 and count == len(q_vect):
                for w in q_vect:
                    l = [x for x in srtdplist[w] if x[0] == i]
                    sd += l[0][1] * qw[w][1]
                topk.append((i,sd))
            elif count > 0 and count < len(q_vec):
                
                for w in q_vect:
                    l = srtdplist[w][9]
                    sd += l[1] * qw[w][1]
                topk.append((i,sd))  
                
        
        show = [x for x in sorted(topk, key=lambda i:i[1], reverse=True)]        
        out = []
        for (ind,s) in show:
             out.append( [self.dataset.loc[self.dataset.index[ind], 'drugName'], self.dataset.loc[self.dataset.index[ind], 'usefulCount'], self.dataset.loc[self.dataset.index[ind], 'condition'], self.dataset.loc[self.dataset.index[ind], 'rating'], self.dataset.loc[self.dataset.index[ind], 'review'], s])
        
        
        pd.set_option('display.max_columns', -1)  
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', -1)
        out =  pd.DataFrame(out, columns=['Drug Name','Useful count','Condition','Rating(/10)','Review','Similarity%'])
      
        return out  

