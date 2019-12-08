import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
stpwrds = stopwords.words("english")
from nltk.stem import WordNetLemmatizer
from ast import literal_eval
import numpy as np
import pandas as pd
import pickle


class backend_img():
    def elim_stopword(self,r):
            r_n = [i for i in r if i not in stpwrds]
            return r_n
        
    def lem(self,tokens):
            l = WordNetLemmatizer()
            out = [l.lemmatize(word) for word in tokens]
            return out
            
    def InvIndim(self):    
        data = pd.read_csv("images_n1.csv")
        data["caption"] = data["caption"].apply(literal_eval)
        capitons = data["caption"].apply(lambda r: [w for w in r if len(w)>2])
        capitons = capitons.apply(self.elim_stopword)
        capitons = capitons.apply(self.lem)

        wordbank = {}
            
        for i,r in enumerate(capitons, start=0):
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
                        

        N = np.float64(data.shape[0])                    

        for w in wordbank.keys(): #every word in dictionary
            plist = {} #Creation of posting list.
            for i in wordbank[w][1].keys(): #Keys of document dictionary. For every word, for every document the word occurs in.
                        tf = (len(wordbank[w][1][i])/len(capitons[i])) #number of times the term occurs in the document by number of all terms in document.
                        weight_i = (1 + np.log10(tf)) * np.log10(N/wordbank[w][0]) #weight calculation. 
                        plist[i] = weight_i
            wordbank[w].append(plist) #every document has weight. Append to list. plist is formed. Appended to give third element.

        p = open('imgwordbankdoc.pickle',"wb")
        pickle.dump(wordbank,p) #save the file

    def topk_img(self,query):
            data = pd.read_csv("images_n12.csv")
            q = query.replace("[^a-zA-Z]", " ").lower()
            q_vec = self.elim_stopword(q.split())  #Query preprocessing
            q_vect = self.lem(q_vec)
        
            with open("imgwordbankdoc.pickle", "rb") as pic:
                    wordbank = pickle.load(pic)
        
            srtdplist = {}
            qw = {}
            x = 0
            for w in q_vect:    #going through all words in query. 
            
                print("\n loop in\n")        
                if w in wordbank.keys(): #if word exists
                    if w not in srtdplist.keys():
                        print("\n in if ")
                        srtdplist[w] = sorted(wordbank[w][2].items(), key=lambda x:x[1], reverse=True)[:10] #top 10 results are retrieved.
                    print(x,"\n  q words in vocab \n")
                if w not in qw:  #weight of the word occured in the query
                    qw[w] = [1,(1/len(q_vect))] #no. of times word occurs by the query length. If word is seen for first time, count = 1
                elif w in qw: #if word already exists in query, increment count by 1 
                    qw[w][0] += 1 
                    qw[w][1] = (qw[w][0]/len(q_vect)) #calculates new weight with new count. returns tuples.
            if srtdplist == {}: 
                return "No results found"
        
            topk = [] #empty list to store (document id, weight) tuples.
            N = data.shape[0]  
            for i in range(N): #iterates through all document ids
                count = 0 #current doc id
                sd = 0 #weight
                for w in srtdplist.keys(): 
                    for (di,wt) in srtdplist[w]: #for every word, (document id, weight) tuple  
                        if di == i: count += 1 #if document id is the current document, increment the count.
                if count > 0 and count == len(q_vect):  #count of number of times a document has occured
                    for w in srtdplist.keys():
                        l = [x for x in srtdplist[w] if x[0] == i] 
                        sd += l[0][1] * qw[w][1] #calculation of weight from the plist. Final weight calculated.
                    topk.append((i,sd)) 
                elif count > 0 and count < len(q_vec): 
                
                    for w in srtdplist.keys():
                        l = srtdplist[w][-1] #document that hasnt occured.
                        sd += l[1] * qw[w][1] 
                    topk.append((i,sd))  
                
        
            show = [x for x in sorted(topk, key=lambda i:i[1], reverse=True)]  
            out = []
            for r in show:
                   cap = literal_eval(data.loc[r[0]].caption )
                   cap = " ".join([ w for w in cap])
                   out.append([ data.loc[r[0]].url, cap] )
            out = pd.DataFrame(out, columns=['Url', 'Caption'])                
            print(out['Caption'])
            return out