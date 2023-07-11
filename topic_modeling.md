<img src="images/DSL_logo1.png" width=40% height=40%>

## Topic modeling of tweets on the blockchain approach for carbon credit markets.

[Dynamic Sustainability Lab](https://www.dynamicslab.org/)

Here, I expand my research to examine the topics people were talking about when they mentioned carbon credits or Net Zero. I used [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) on the cleaned training data. Latent Dirichlet Allocation (LDA) is a Bayesian network that explains a set of observations through unobserved groups, and each group explains why some parts of the data are similar.

### Required libraries 
```
#Base and Cleaning 
import json
import requests
import pandas as pd
import numpy as np
import emoji
import regex
import re
import string
from collections import Counter

#Visualizations
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt 
#import pyLDAvis.gensim
import chart_studio
import chart_studio.plotly as py 
import chart_studio.tools as tls

#Natural Language Processing (NLP)
import spacy
import gensim
from spacy.tokenizer import Tokenizer
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
```

### Cleaning and pre-processing 

Import necessary tools:
```
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
  
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
```
Write the function to clean the tweet and tokenize them:
```
def clean_tweet(tweet):
    if type(tweet) == float:
            return ""

    # turn text into lower
    test = tweet.lower()
    # remove all mentions and hashtags 
    test = re.sub("@[A-Za-z0-9_]+","", test)
    test = re.sub("#[A-Za-z0-9_]+","", test)
    #remove links
    test = re.sub(r"http\S+", "", test)
    test = re.sub(r"www.\S+", "", test)
    #remove punctuation
    test = re.sub('[()!?]', ' ', test)
    test = re.sub('\[.*?\]',' ', test)
    #remove non alphabetical characters
    test = re.sub("[^a-z0-9]"," ", test)
    #remove extra spaces
    test = re.sub(' +', ' ', test)
    # remove single letter words
    test = ' '.join( [w for w in test.split() if len(w)>1] )
    
    test = ' '.join( [lemma.lemmatize(x) for x in nltk.wordpunct_tokenize(test) if x not in stop_words])
    test =[lemma.lemmatize(x, nltk.corpus.reader.wordnet.VERB) for x in nltk.wordpunct_tokenize(test) if x not in stop_words]

    return test

# clean the tweets and create two columns: tokenized tweet and whole tweet
data21["clean_tweet"]=data21["Tweet"].apply(lambda x:clean_tweet(x))
```

### LDA method

Create a id2word dictionary:
```
id2word = Dictionary(data21["clean_tweet"])
print(len(id2word))

# https://towardsdatascience.com/twitter-topic-modeling-e0e3315b12e2
```
Filter Extremes:
```
id2word.filter_extremes(no_below=2, no_above=.99)
print(len(id2word))

# https://towardsdatascience.com/twitter-topic-modeling-e0e3315b12e2
```
Create a corpus object:
```
corpus = [id2word.doc2bow(d) for d in data21["clean_tweet"]]

# https://towardsdatascience.com/twitter-topic-modeling-e0e3315b12e2
```
Instantiate a Base LDA model:
```
base_model = LdaMulticore(corpus=corpus, num_topics=5, id2word=id2word, workers=12, passes=5)
```
Filter for words:
```
words = [re.findall(r'"([^"]*)"',t[1]) for t in base_model.print_topics()]
```


