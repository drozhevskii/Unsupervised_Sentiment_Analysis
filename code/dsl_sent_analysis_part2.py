#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import spacy
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import re
from wordcloud import WordCloud
import itertools
import collections
import nltk
import string
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gensim
import multiprocessing
from gensim.models import Word2Vec
from multiprocessing import Process
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import sklearn
from sklearn.cluster import KMeans


# In[2]:


data18 = pd.read_csv('carbondata_labeled_custom22.csv')
#data18 = pd.read_csv('carbondata_21_221031.csv')


# In[3]:


data22 = pd.read_csv('carbondata_labeled_custom22.csv')
data21 = pd.read_csv('carbondata_labeled_custom21.csv')


# In[2]:


data21_withSA = pd.read_csv('data21_3algos.csv')


# In[4]:


data22_withSA = pd.read_csv('data22_3algos.csv')


# In[5]:


data22_withSA.head()


# In[4]:


df18_location = data18.loc[data18['Place'].notnull()]
df18_location


# In[20]:


df18_location[df18_location['Place'].str.contains('GB')]['Place']


# In[21]:


((1661+3149)/6739)*100


# In[13]:


df18_location.Place.value_counts()


# In[27]:


data22


# In[28]:


data22.sentiments_val.value_counts()


# In[26]:


(27292/125298)*100


# In[ ]:


data22.sentiments_val.value_counts()


# In[31]:


(74721/99799)*100


# In[32]:


data21.groupby('month').sentiments_val.value_counts()


# ### Pre-processsing

# In[24]:


data18['Tweet'].iloc[3]


# In[25]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
  
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# In[26]:


nltk.download('wordnet')


# In[27]:


# the function to clean the tweet and tokenize them
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
    
    #remove many abbreviations
    test = re.sub(r"there's", "there is", test)
    test = re.sub(r"i'm", "i am", test)
    test = re.sub(r"he's", "he is", test)
    test = re.sub(r"she's", "she is", test)
    test = re.sub(r"it's", "it is", test)
    test = re.sub(r"that's", "that is", test)
    test = re.sub(r"what's", "that is", test)
    test = re.sub(r"where's", "where is", test)
    test = re.sub(r"how's", "how is", test)
    test = re.sub(r"\'ll", " will", test)
    test = re.sub(r"\'ve", " have", test)
    test = re.sub(r"\'re", " are", test)
    test = re.sub(r"\'d", " would", test)
    test = re.sub(r"\'re", " are", test)
    test = re.sub(r"won't", "will not", test)
    test = re.sub(r"can't", "cannot", test)
    test = re.sub(r"n't", " not", test)
    test = re.sub(r"n'", "ng", test)
    test = re.sub(r"'bout", "about", test)
    test = re.sub(r"'til", "until", test)
    test = re.sub(r"\"", "", test)
    test = re.sub(r"\'", "", test)
    test = re.sub(r' s ', "", test)
    test = re.sub(r"[\[\]\\0-9()\"$#%/@;:<>{}`+=~|.!?,-]", "", test)
    test = re.sub(r"&", "", test)
    test = re.sub(r"\\n", "", test)
    
    # remove single letter words
    test = ' '.join( [w for w in test.split() if len(w)>1] )
    
    test = ' '.join( [lemma.lemmatize(x) for x in nltk.wordpunct_tokenize(test) if x not in stop_words])
    test =[lemma.lemmatize(x, nltk.corpus.reader.wordnet.VERB) for x in nltk.wordpunct_tokenize(test) if x not in stop_words]

    return test


# In[28]:


clean_tweet(data18['Tweet'].iloc[4])


# In[29]:


# clean the tweets and create two columns: tokenized tweet and whole tweet
data18["clean_tweet"]=data18["Tweet"].apply(lambda x:clean_tweet(x))
data18["cleaned_tweet"]=data18["clean_tweet"].apply(lambda x:' '.join(x))


# In[30]:


data18['clean_tweet'].iloc[4]


# In[13]:


data18['cleaned_tweet'].iloc[4]


# In[14]:


data18.shape


# In[15]:


tweets = data18['clean_tweet']
tweets[:10]


# ### Hashtags

# In[16]:


# define a function to clean the Hashtags.
def clean_hashtags(hashtags):
    '''
    hashtags: String
              Input Data
    hashtags: String
              Output Data
           
    func: Convert hashtags to lower case
          Replace ticker symbols with space. The ticker symbols are any stock symbol that starts with $.
          Replace everything not a letter or apostrophe with space
          Removes any spaces or specified characters at the start and end of hashtags.
          
    '''
    if hashtags:
        hashtags = hashtags.lower()
        hashtags = re.sub('\$[a-zA-Z0-9]*', ' ', hashtags)
        hashtags = re.sub('[^a-zA-Z]', ' ', hashtags)
        hashtags=hashtags.strip() 
    return hashtags


# In[17]:


# clean the hashtags
data18["Hashtags"]=data18["Hashtags"].astype(str)
data18["Hashtags"]=data18["Hashtags"].apply(lambda x:clean_hashtags(x))


# In[18]:


data18.head()


# ### DateColumns: + month, year columns

# In[19]:


data18['date'] = pd.to_datetime(data18['Date_Tweet'], format='%Y-%m-%d')
data18['month'] = data18['date'].dt.month
data18['year'] = data18['date'].dt.year


# In[20]:


data18.tail()


# In[21]:


list(data18['cleaned_tweet'][(data18['year']==2022)&(data18['month']==9)][:10])


# ## Kmeans algorithm

# ### Turn tweets into embedding vectors 

# In[ ]:


#Converting the "clean_tweet" column in the format supported by embeddings.
sent = [row for row in data18["clean_tweet"]]

#use Gensim Phrases package to automatically detect common phrases (bigrams) from a list of sentences.
phrases = Phrases(sent, min_count=1, progress_per=50000)
bigram = gensim.models.phrases.Phraser(phrases)
sentences = bigram[sent]
sentences[1]

# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial 


# In[31]:


sent = [row for row in data18["clean_tweet"]]
sent


# ## VADER algorithm

# In[4]:


# import the sentiment analyzer 
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


# In[5]:


cl_tweet = data18['cleaned_tweet'].iloc[5]
cl_tweet


# In[6]:


data18 = data18[data18['cleaned_tweet'].notna()]


# In[7]:


len(data18)


# In[8]:


sid.polarity_scores(cl_tweet)


# In[9]:


data18['sentiments_val2'] = data18['cleaned_tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

data18.head()


# In[10]:


data18['compound']  = data18['sentiments_val2'].apply(lambda score_dict: score_dict['compound'])


# In[11]:


data18.head()


# In[12]:


def sentimentPredict(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05: 
        return "negative"
    else:
        return "neutral"


# In[13]:


data18['sentiments_val2'] =data18['compound'].apply(lambda x: sentimentPredict(x))
data18.head(5)


# ## BERT algorithm

# In[23]:


# installing the library 'transformers' which contains BERT implementation
get_ipython().system('pip install transformers')
 
# installing the library tensorflow
get_ipython().system('pip install tensorflow')


# In[14]:


# importing the pipeline module
from transformers import pipeline
 
# Downloading the sentiment analysis model
SentimentClassifier = pipeline("sentiment-analysis")


# In[15]:


def FunctionBERTSentiment(inpText):
  return(SentimentClassifier(inpText)[0]['label'])


# In[16]:


data18['cleaned_tweet'].iloc[3]


# In[17]:


data18['Tweet'].iloc[3]


# In[18]:


FunctionBERTSentiment(data18['cleaned_tweet'].iloc[3])


# In[19]:


# Calling BERT based sentiment score function for every tweet
data18['sentiments_val3']=data18['cleaned_tweet'].apply(FunctionBERTSentiment)
data18.head(10)


# In[20]:


data18['sentiments_val3'] = data18['sentiments_val3'].str.lower()


# In[6]:


data18 = pd.read_csv('data22_3algos_avg.csv')
data18


# In[7]:


data19 = pd.read_csv('data21_3algos_avg.csv')
data19


# In[8]:


# turn all the sentiments bavk to numbers
def sentimentBack(score):
    if score == 1:
        return 'positive'
    elif score == -1: 
        return 'negative'
    else:
        return 'neutral'


# In[9]:


data18['sentiments_val'] = data18['sentiments_val'].apply(lambda x: sentimentBack(x))
data18['sentiments_val2'] =data18['sentiments_val2'].apply(lambda x: sentimentBack(x))
data18['sentiments_val3'] =data18['sentiments_val3'].apply(lambda x: sentimentBack(x))
data18.head()


# In[10]:


data19['sentiments_val'] = data19['sentiments_val'].apply(lambda x: sentimentBack(x))
data19['sentiments_val2'] =data19['sentiments_val2'].apply(lambda x: sentimentBack(x))
data19['sentiments_val3'] =data19['sentiments_val3'].apply(lambda x: sentimentBack(x))
data19.head()


# In[11]:


data2122 = pd.concat([data19, data18], axis=0)
len(data2122)


# In[12]:


data2122.year.value_counts()


# In[53]:


#data21.to_csv('data21_3algos.csv', index=False)


# In[6]:


data_pie=data2122["sentiments_val"].value_counts().reset_index()
fig = plt.gcf()
fig.set_size_inches(7,7)
colors = ["yellow","cyan","pink"]
plt.pie(data_pie["sentiments_val"],labels=data_pie["index"],radius=2,autopct="%1.1f%%", colors=colors)
plt.axis('equal')
plt.title("KMeans: Sentiment Distribution of Tweets 2022", fontsize=20)
#plt.savefig("images/Sentiment_Distribution.png")
plt.show()
data_pie

#plt.savefig("sent_dist_tweets21.png")


# In[7]:


data_pie=data18["sentiments_val2"].value_counts().reset_index()
fig = plt.gcf()
fig.set_size_inches(7,7)
colors = ["yellow","cyan","pink"]
plt.pie(data_pie["sentiments_val2"],labels=data_pie["index"],radius=2,autopct="%1.1f%%", colors=colors)
plt.axis('equal')
plt.title("VADER: Sentiment Distribution of Tweets 2022", fontsize=20)
#plt.savefig("images/Sentiment_Distribution.png")
plt.show()
data_pie

#plt.savefig("sent_vader_tweets21.png")


# In[8]:


data_pie=data18["sentiments_val3"].value_counts().reset_index()
fig = plt.gcf()
fig.set_size_inches(7,7)
colors = ["yellow","cyan","pink"]
plt.pie(data_pie["sentiments_val3"],labels=data_pie["index"],radius=2,autopct="%1.1f%%", colors=colors)
plt.axis('equal')
plt.title("Pre-trained BERT: Sentiment Distribution of Tweets 2022", fontsize=20)
#plt.savefig("images/Sentiment_Distribution.png")
plt.show()
data_pie

#plt.savefig("sent_bert_tweets21.png")


# In[32]:


data_pie=data2122["sent_avg"].value_counts().reset_index()
fig = plt.gcf()
fig.set_size_inches(7,7)
colors = ["yellow","cyan","pink"]
plt.pie(data_pie["sent_avg"],labels=data_pie["index"],radius=2,autopct="%1.1f%%", colors=colors, textprops={'fontsize': 15})
plt.axis('equal')
plt.title("Average Opinion on Carbon Credits in 2021-2022", fontsize=15, pad=15)
#plt.savefig("images/Sentiment_Distribution.png")
plt.show()
data_pie

plt.savefig("avg_tweets2122.png")


# In[30]:


# plotting Tweets Sentiments for each year
plt.subplots(figsize = (10,8))
chart = sns.countplot(x="month",data=data18, palette="Set2",hue="sent_avg");
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.setp(chart.get_legend().get_texts(), fontsize='15') # for legend text
chart.set_xlabel("Month",fontsize=15)
chart.set_ylabel("Total Tweets",fontsize=15)

plt.title("Average: Tweets Sentiment per month in 2022", fontsize=20)
plt.savefig("Tweets_avg22.png")
plt.show();


# In[58]:


# plotting Tweets Sentiments for each year
plt.subplots(figsize = (10,8))
chart = sns.countplot(x="month",data=data18, palette="Set2",hue="sentiments_val2");
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("VADER: Tweets Sentiment per month in 2021", fontsize=20)
plt.savefig("Tweets_per_year_vader21.png")
plt.show();


# In[59]:


# plotting Tweets Sentiments for each year
plt.subplots(figsize = (10,8))
chart = sns.countplot(x="month",data=data18, palette="Set2",hue="sentiments_val3");
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("BERT: Tweets Sentiment per month in 2021", fontsize=20)
plt.savefig("Tweets_per_yearBERT21.png")
plt.show();


# In[60]:


# plotting Tweets Sentiments for each year
plt.subplots(figsize = (10,8))
chart = sns.countplot(x="month",data=data18, palette="Set2",hue="sentiments_val");
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("BERT: Tweets Sentiment per month in 2021", fontsize=20)
plt.savefig("Tweets_per_year21.png")
plt.show();


# ## Average over all algos

# In[62]:


data18.tail()


# In[63]:


# turn all the sentiments bavk to numbers
def sentimentNum(score):
    if score == 'positive':
        return 1
    elif score == 'negative': 
        return -1
    else:
        return 0


# In[64]:


data18['sentiments_val'] =data18['sentiments_val'].apply(lambda x: sentimentNum(x))
data18['sentiments_val2'] =data18['sentiments_val2'].apply(lambda x: sentimentNum(x))
data18['sentiments_val3'] =data18['sentiments_val3'].apply(lambda x: sentimentNum(x))
data18.head(5)


# In[65]:


def sentimentAvg(val1, val2, val3):
    summ = val1 + val2 + val3
    if summ >= 1:
        return 'positive'
    elif summ < 0: 
        return 'negative'
    else:
        return 'neutral'


# In[66]:


data18['sent_avg'] = data18.apply(lambda x: sentimentAvg(val1 = x['sentiments_val'], val2 = x['sentiments_val2'], val3 = x['sentiments_val3']), axis=1)
data18.head(5)


# In[67]:


data21_withSA = data21_withSA.drop(['sentiment', 'compound'], axis=1)
data21_withSA


# In[68]:


data_pie=data18["sent_avg"].value_counts().reset_index()
fig = plt.gcf()
fig.set_size_inches(7,7)
colors = ["yellow","cyan","pink"]
plt.pie(data_pie["sent_avg"],labels=data_pie["index"],radius=2,autopct="%1.1f%%", colors=colors)
plt.axis('equal')
plt.title("On Average: Sentiment Distribution of Tweets 2021", fontsize=20)
#plt.savefig("images/Sentiment_Distribution.png")
plt.show()
data_pie

plt.savefig("sent_dist_tweets21avg.png")


# In[43]:


# plotting Tweets Sentiments for each year
plt.subplots(figsize = (10,8))
chart = sns.countplot(x="month",data=data18, palette="Set2",hue="sent_avg");
plt.setp(chart.get_legend().get_texts(), fontsize='25') # for legend text
chart.set_xlabel("Month",fontsize=25)
chart.set_ylabel("Total Tweets",fontsize=25)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, fontsize='15')
chart.tick_params(labelsize=20)

plt.title("Tweets' Average Sentiment per month in 2022", fontsize=30, pad=30)
plt.savefig("Tweets_per_year22avg.png")
plt.show();


# In[48]:


data18.to_csv('data22_3algos_avg.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




