#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
import pickle
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


# In[2]:


input_df = pd.read_csv('Sample_Test_Data.csv',header=None)
length = input_df.shape[0]


# In[3]:


f2 = open('dict_nlp_eve.pkl','rb')
dict_nlp_eve = pickle.load(f2)
f2.close()


# In[4]:


all_employees=[]


# In[5]:


df = pd.read_csv('Employee_Data.csv')
copydf = df.copy()


# In[6]:


inst = open('EXTRACTOR_DOM.pkl','rb')                
Domain_extract = pickle.load(inst)
inst2 = open("Event_extract_file.pkl","rb")
Event_extract = pickle.load(inst2)


# In[7]:


event_list=[]
for x in range(0,length):
    event_list.append((input_df.iloc[x,0]).upper())
print(event_list)


# In[8]:


def domain_extractor(t):
    words=[]
    flag_domain = 0
    word = t.split(',')
    for m in word:
        tmp = m.split()
        for x in tmp:
            words.append(x)
    DE= []
    for x in words:
        try:
            DE.append(Domain_extract[x])
            flag_domain = 1
        except:
            pass
        
    if True:
        for x in words:
            for y in words:
                try:
                    dummy = x+' '+y
                    DE.append(Domain_extract[dummy])
                except:
                    pass  
    DE = list(dict.fromkeys(DE))
    return DE


# In[9]:


def domain_extractor_ml(t):
    DE_ml=[]
    nlp_df = DataFrame(list(Domain_extract.items()),columns = ['txt','Domain'])
    fx = open('Vectorizer.pkl','rb')
    vectorizer = pickle.load(fx)
    fx.close()
    train_vectors = vectorizer.transform(nlp_df.txt)
    temp1 = vectorizer.transform([t])
    
    f1 = open('clf_svm.pkl','rb')
    clf_svm = pickle.load(f1)
    f1.close()
    pred = clf_svm.predict(temp1)[0]
    DE_ml.append(pred)

    f2 = open('clf_tree.pkl','rb')
    clf_tree = pickle.load(f2)
    f2.close()
    pred = clf_tree.predict(temp1)[0]
    DE_ml.append(pred)
    

    f3 = open('clf_bay.pkl','rb')
    clf_bay = pickle.load(f3)
    f3.close()
    pred = clf_bay.predict(temp1.toarray())[0]
    DE_ml.append(pred)
    
    DE_ml = list(dict.fromkeys(DE_ml))
    return DE_ml


# In[10]:


def event_extractor(t):
    words=t.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    words = [re_punc.sub('', x) for x in words]
    EE= []
    for x in words:
        try:            
            EE.append(Event_extract[x])
        except:
            pass

    return EE


# In[11]:


def event_extractor_ml(t):
    EE_ml=[]
    nlp_df = DataFrame(list(dict_nlp_eve.items()),columns = ['Event','txt']) 

    fx = open('Vectorizer_event.pkl','rb')
    vectorizer = pickle.load(fx)
    fx.close()
    train_vectors = vectorizer.transform(nlp_df.txt)
    temp1 = vectorizer.transform([t])

    f1 = open('clf_bayes_event.pkl','rb')
    clf_bay = pickle.load(f1)
    f1.close()
    pred = clf_bay.predict(temp1.toarray())[0]
    EE_ml.append(pred)
    
    return EE_ml


# In[12]:


def employees_event(DE,EE):
    employees=[]
    for x in DE:
        temp1 = copydf[copydf['Domain']==x].copy()
        for y in EE:
            temp2 = temp1[temp1['Event1']==y].copy()
            temp3 = temp1[temp1['Event2']==y].copy()
            
            try:
                for z in temp2.Name:
                    employees.append(z)
                for z in temp3.Name:
                    employees.append(z)
            except:
                pass

    return employees


# In[13]:



for t in event_list:
    DE = domain_extractor(t)
    DE_ml = domain_extractor_ml(t)
    EE =  event_extractor(t)
    EE_ml = event_extractor_ml(t)
    for x in DE_ml:
        if x not in DE:
            DE.append(x)
    DE = list(dict.fromkeys(DE))
    employees = employees_event(DE,EE)
    if not employees:
          employees.append('NONE')
    all_employees.append(employees)


# In[14]:


dict1={}
for totalppl,event in zip(all_employees,event_list):
    strx = str(totalppl)[1:-1]
    strxx = strx.replace("'", "")
    dict1[event]=strxx


# In[15]:


dfx = DataFrame(list(dict1.items()),columns = ['Event Name','Employee Names']) 
dfx


# In[ ]:


dfx.to_excel('output.xlsx', index = None, header=True)
print('Thank you for using Cloud events!\nCheck Your Excel Sheet for recommendations')

