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


#data18 = pd.read_csv('carbonMarket2018_2022.csv')
data18 = pd.read_csv('carbondata_21_221031.csv')


# In[4]:


data18.head()


# In[9]:


len(data18.columns.tolist())


# In[178]:


data18['TweetC'] = data18['Tweet']
data18.head()


# In[45]:


# for plotting missing values

def return_missing_values(data_frame):
    missing_values = data_frame.isnull().sum()/len(data_frame)
    missing_values = missing_values[missing_values>0]
    missing_values.sort_values(inplace=True)
    return missing_values

def plot_missing_values(data_frame):
    missing_values = return_missing_values(data_frame)
    missing_values = missing_values.to_frame()
    missing_values.columns = ['count']
    missing_values.index.names = ['Name']
    missing_values['Name'] = missing_values.index
    sns.set(style='darkgrid')
    sns.barplot(x='Name', y='count', data=missing_values)
    plt.xticks(rotation=90)
    plt.title('Missing Values Fraction for Columns')
    plt.show()
    
    
#https://github.com/ShilpiParikh/EDA-on-COVID-19-tweets/blob/main/Covid19_tweets_EDA%20.ipynb


# In[46]:


return_missing_values(data18)


# In[47]:


plot_missing_values(data18)


# In[38]:


# unique values from data
def return_unique_values(data_frame):
    unique_dataframe = pd.DataFrame()
    unique_dataframe['Features'] = data_frame.columns
    uniques = []
    for col in data_frame.columns:
        u = data_frame[col].nunique()
        uniques.append(u)
    unique_dataframe['Uniques'] = uniques
    return unique_dataframe

#https://github.com/ShilpiParikh/EDA-on-COVID-19-tweets/blob/main/Covid19_tweets_EDA%20.ipynb


# In[40]:


unidf = return_unique_values(data18)
print(unidf)


# In[41]:


f, ax = plt.subplots(1,1, figsize=(10,5))

sns.barplot(x=unidf['Features'], y=unidf['Uniques'], alpha=0.7)
plt.title('Bar plot for Unique Values in each column')
plt.ylabel('Unique values', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.xticks(rotation=90)
plt.show()

#https://github.com/ShilpiParikh/EDA-on-COVID-19-tweets/blob/main/Covid19_tweets_EDA%20.ipynb


# In[89]:


places = data18['Place'].loc[data18['Place'].notnull()].tolist()
places[1]


# In[108]:


places = data18['Place'].value_counts(ascending=False).rename_axis('unique_values').to_frame('counts')
places


# In[81]:


data18['Place'].loc[data18['Place'].notnull()]


# In[85]:


data18['Place'].iloc[793]


# In[ ]:


s.split(':')[0]


# In[70]:


sns.barplot(x= data18.Place.value_counts()[:10].index,y=data18.Place.value_counts()[:10]).set(title='Tweets by Location')
plt.xticks(rotation=90)


# ### Pre-processsing

# In[49]:


data18['Tweet'].iloc[3]


# In[50]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
  
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# In[51]:


nltk.download('wordnet')


# In[52]:


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
    # remove single letter words
    test = ' '.join( [w for w in test.split() if len(w)>1] )
    
    test = ' '.join( [lemma.lemmatize(x) for x in nltk.wordpunct_tokenize(test) if x not in stop_words])
    test =[lemma.lemmatize(x, nltk.corpus.reader.wordnet.VERB) for x in nltk.wordpunct_tokenize(test) if x not in stop_words]

    return test


# In[151]:


# define a function to clean the tweet.
def clean_tweet2(tweet):

    tweet = tweet.lower()
    tweet = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', tweet)
    tweet = re.sub('\$[a-zA-Z0-9]*', ' ', tweet)
    tweet = re.sub('\@[a-zA-Z0-9]*', ' ', tweet)
    tweet = re.sub('[^a-zA-Z\']', ' ', tweet)
    tweet = ' '.join( [w for w in tweet.split() if len(w)>1] )
    
    tweet=' '.join([lemma.lemmatize(x) for x in nltk.wordpunct_tokenize(tweet) if x not in stop_words])
    tweet=[lemma.lemmatize(x,nltk.corpus.reader.wordnet.VERB) for x in nltk.wordpunct_tokenize(tweet) if x not in stop_words]
    return tweet


# In[53]:


clean_tweet(data18['Tweet'].iloc[3])


# In[54]:


# clean the tweets and create two columns: tokenized tweet and whole tweet
data18["clean_tweet"]=data18["Tweet"].apply(lambda x:clean_tweet(x))
data18["cleaned_tweet"]=data18["clean_tweet"].apply(lambda x:' '.join(x))


# In[55]:


# we choose tweets in English and with at least 1 like
data18 = data18[data18['Language'] == 'en']
data18 = data18[data18['Number_of_Likes'] >= 1]


# In[187]:


data18['clean_tweet'].iloc[4]


# In[188]:


data18['cleaned_tweet'].iloc[4]


# In[189]:


data18.shape


# In[190]:


tweets = data18['clean_tweet']
tweets[:10]


# ### Hashtags

# In[110]:


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


# In[111]:


# clean the hashtags
data18["Hashtags"]=data18["Hashtags"].astype(str)
data18["Hashtags"]=data18["Hashtags"].apply(lambda x:clean_hashtags(x))


# In[112]:


data18.head()


# ### DateColumns: + month, year columns

# In[194]:


data18['date'] = pd.to_datetime(data18['Date_Tweet'], format='%Y-%m-%d')
data18['month'] = data18['date'].dt.month
data18['year'] = data18['date'].dt.year


# In[195]:


data18.tail()


# In[113]:


list(data18['cleaned_tweet'][(data18['year']==2022)&(data18['month']==9)][:10])


# ### Turn tweets into embedding vectors 

# In[44]:


get_ipython().system('pip install -U gensim')


# In[196]:


#Converting the "clean_tweet" column in the format supported by embeddings.
sent = [row for row in data18["clean_tweet"]]
#use Gensim Phrases package to automatically detect common phrases (bigrams) from a list of sentences.
phrases = Phrases(sent, min_count=1, progress_per=50000)
bigram = gensim.models.phrases.Phraser(phrases)
sentences = bigram[sent]
sentences[1]

# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial 


# In[197]:


len(sentences)


# ### Word2Vec model

# In[198]:


#Initializing the word2vec model
w2v_model = Word2Vec(min_count=4,
                     window=5,
                     vector_size =300,
                     sample=1e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     seed= 42,
                     workers=multiprocessing.cpu_count()-1)


#building vocab of the word2vec model from the custom data
w2v_model.build_vocab(sentences, progress_per=50000)

# https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483


# In[199]:


#training the word2vec model
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=60, report_delay=1)


# In[201]:


w2v_model.wv.most_similar(positive =["carbon"])


# In[202]:


#saving the word2vec model
w2v_model.save("word2vec.model")


# In[203]:


#Loading the word2vec model
word_vectors = Word2Vec.load("word2vec.model").wv


# ### Clustering

# In[204]:


#Feeding the embeddings to a KMeans model to cluster words into positive, negative, and neutral clusters
model = KMeans(n_clusters=3, max_iter=1000, random_state=42, n_init=50).fit(X=word_vectors.vectors.astype('double'))


# In[207]:


# check what we have in each cluster to label the clusters
word_vectors.similar_by_vector(model.cluster_centers_[0], topn=200, restrict_vocab=None)


# In[209]:


# Labelling the clusters based on the type of words they carry
positive_cluster_center = model.cluster_centers_[1]
negative_cluster_center = model.cluster_centers_[0]
neutral_cluster_center= model.cluster_centers_[2]


# In[222]:


#Creating a DataFrame of words with their embeddings and cluster values
words = pd.DataFrame(word_vectors.index_to_key)
words.columns = ['words']
words['vectors'] = words.words.apply(lambda x: word_vectors[f'{x}'])
words['cluster'] = words.vectors.apply(lambda x: model.predict([np.array(x)]))
words.cluster = words.cluster.apply(lambda x: x[0])

# https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483


# In[223]:


words


# In[224]:


#Assigning 1 to positive values, 0 to neutral and -1 for negative values
words['cluster_value'] = [1 if i==1 else 0 if i==2 else -1 for i in words.cluster]
words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)


# In[241]:


with pd.option_context('display.max_rows', None,):
    print(words[words["cluster_value"]==-1][:300].sort_values("closeness_score"))


# In[246]:


positive = ['good','better','clean','fantastic','right',"hope", "improve","save", "innovation", "delight", "great"]
neutral = ['nuclear','india','australia','play','data','scotland','canada','job',"race","happens","grocery","person",
          'heat','house','may',"national","state"]
negative= ['risk','waste','carbon_footprint']
for i in positive:
    words.loc[words["words"]==i,"cluster_value"]=1
    
for i in neutral:
    words.loc[words["words"]==i,"cluster_value"]=0
    
for i in negative:
    words.loc[words["words"]==i,"cluster_value"]=-1


# In[248]:


words[words["words"]=="dangerous"]


# ### Sentiment analysis of words

# In[426]:


# Plotting pie chart of Sentiment Distribution of words
emotion = {0: "neutral",
           1: "positive",
          -1: "negative"}

words["sentiments"]=words["cluster_value"].map(emotion)


fig = plt.gcf()
fig.set_size_inches(7,7)
colors = ["cyan","pink","yellow"]
df_pie=words["sentiments"].value_counts().reset_index()
plt.pie(df_pie["sentiments"],labels=df_pie["index"],radius=2,colors=colors,autopct="%1.1f%%")
plt.axis('equal')
plt.title("Sentiment Distribution of Words ", fontsize=20)
plt.show()
df_pie

plt.savefig("Downloads/sent_dist_words.png")


# #### Out of 19911 unique words and bigram from the dataset:
# 
# 11621 (33.61%) are Neutral sentiments
# 14786 (42.76%) are Positive sentiments
# 8170 (23.63%) are Negative sentiments
# 
# #### It shows that the Neutral and Positive words have larger domination in the dataset

# ### Custom sentiment analysis of tweets

# In[254]:


# creating a dictionary of the word and its cluster value
words_dict = dict(zip(words.words, words.cluster_value))


# In[396]:


# define a function to get the sentiment for the entire tweet
def get_sentiments(x,words_dict):
    '''
    x:         List
               Input data: Row of a DataFrame
    sent_dict: Dictionary
               Input: Dictionary of Words: Sentiments
    sentiment: String
               Output: Sentiment of the whole sentence
    
    Function: Getting sentiments of the entire sentence by averaging out the sentiments of individual words
    '''
    total=0
    count=0
    test=x["clean_tweet"]
    #print(test)
    for t in test:
        if words_dict.get(t):
            total+=int(words_dict.get(t))
            #print('adding', int(words_dict.get(t)))
        count+=1
    if count == 0:
        sentiment = 'no data'
    else:
        avg=total/count
        sentiment=-1 if avg<-0.15 else 1 if avg >0.15 else 0
    return sentiment


# In[395]:


#x = data18.iloc[20]

total=0
count=0
#test=data18.iloc[2431]["clean_tweet"]
test=data18.iloc[0]["clean_tweet"]
print(test)
for t in test:
    if words_dict.get(t):
        total+=int(words_dict.get(t))
        print('adding', int(words_dict.get(t)))
    count+=1
if count == 0:
    print('ZERO ERROR')
    sentiment = 'no data'
else:
    avg=total/count
    sentiment=-1 if avg<-0.15 else 1 if avg >0.15 else 0
    
print('total:', total)
print('count:', count)
print('average:', avg)
print('sentiment:', sentiment)


# In[399]:


for i in range(len(data18)):
    x = data18.iloc[i]
    data18['sentiment'][i] = get_sentiments(x, words_dict)


# In[400]:


data18.head()


# In[380]:


counts = 0
for i in range(len(data18)):
    test = type(get_sentiments(data18.iloc[i], words_dict))
    if test is str:
        counts+=1
print(counts)
    #print('sentiment for', i, ':', get_sentiments(data18.iloc[i], words_dict))


# In[401]:


# checking the value counts of each sentiment
data18["sentiment"].value_counts()


# In[427]:


# Plotting pie chart of Sentiment Distribution of tweets
emotion = {0: "neutral",
           1: "positive",
          -1: "negative"}

data18["sentiments_val"]=data18["sentiment"].map(emotion)
data_pie=data18["sentiments_val"].value_counts().reset_index()
fig = plt.gcf()
fig.set_size_inches(7,7)
colors = ["yellow","cyan","pink"]
plt.pie(data_pie["sentiments_val"],labels=data_pie["index"],radius=2,autopct="%1.1f%%", colors=colors)
plt.axis('equal')
plt.title("Sentiment Distribution of Tweets ", fontsize=20)
#plt.savefig("images/Sentiment_Distribution.png")
plt.show()
data_pie

plt.savefig("Downloads/sent_dist_tweets.png")


# #### Out of 77467 tweets from the dataset:
# 
# 102577(45.6%) are Negative sentiments
# 102577(9.1%) are Neutral sentiments
# 20412(45.3%) are Positive sentiments

# In[405]:


#data18.to_csv('carbondata_labeled_custom1.csv', index=False)


# In[406]:


data_negative = data18[data18["sentiment"]==-1]


# In[473]:


# checking the cause of negative tweets in 2019
list(data18['cleaned_tweet'][(data18['year']==2021)&(data18['month'].isin([10,11]))])


# In[409]:


# some positive tweets
list(data18[data18["sentiment"]==1]["cleaned_tweet"][300:330])


# ### Data Visualization

# In[424]:


#data_list=["carbon","offsets","credit","blockchain"]
data_list=["carbon","offsets","credit","john oliver","oliver"]
pattern="|".join(data_list)
data18_2_sent=data18[(data18["cleaned_tweet"].str.contains(pattern))]
sns.countplot(x=data18_2_sent["sentiments_val"]);
plt.title("Sentiment Distribution of Tweets ", fontsize=20)
#plt.savefig("Downloads/johnoliver_sent2122.png")


# In[423]:


# plot Tweets count
plt.subplots(figsize = (10,8))
data22=data18[data18["year"]==2022]
chart = sns.countplot(x="month",data=data22, palette="Set2");
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("Tweets per month in 2022 ", fontsize=20)
plt.savefig("Downloads/num_tweets2022.png")
plt.show();


# In[422]:


# plot Tweets count
plt.subplots(figsize = (10,8))
data21=data18[data18["year"]==2021]
chart = sns.countplot(x="month",data=data21, palette="Set2");
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("Tweets per month in 2021 ", fontsize=20)
plt.savefig("Downloads/num_tweets2021.png")
plt.show();


# In[420]:


# plotting Tweets Sentiments for each year
plt.subplots(figsize = (10,8))
chart = sns.countplot(x="year",data=data18, palette="Set2",hue="sentiments_val");
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("Tweets Sentiments' per year ", fontsize=20)
plt.savefig("Downloads/Tweets_per_year.png")
plt.show();


# In[164]:


# plotting Tweets Sentiments for each year
plt.subplots(figsize = (10,8))
chart = sns.countplot(x="month",data=data21, palette="Set2",hue="sentiments_val");
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("Tweets Sentiment per month in 2021 ", fontsize=20)
#plt.savefig("Downloads/Tweets_per_year.png")
plt.show();


# In[165]:


# plotting Tweets Sentiments for each year
plt.subplots(figsize = (10,8))
chart = sns.countplot(x="month",data=data22, palette="Set2",hue="sentiments_val");
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("Tweets Sentiment per month in 2022 ", fontsize=20)
#plt.savefig("Downloads/Tweets_per_year.png")
plt.show();


# In[429]:


# Top 10 highest tweeting usernames
plt.subplots(figsize = (10,8))
plt.title("Top 10 highest tweeting usernames", fontsize=20)
chart=sns.countplot(x="User",hue="sentiments_val",data=data18,palette="Set2",
                    order= data18["User"].value_counts().iloc[:10].index);
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');

plt.savefig("Downloads/top10_usernames_dist.png")


# In[430]:


# plotting Top 10 hashtag
plt.subplots(figsize = (15,10))
plt.title("Top 10 hashtags", fontsize=20)
chart=sns.countplot(x="Hashtags",hue="sentiments_val",data=data18,palette="Set2",
                    order= data18["Hashtags"].value_counts().iloc[1:10].index);
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');

plt.savefig("Downloads/top10_hashtags_dist.png")


# ### Wordcloud

# In[24]:


def create_wordcloud(text):
    words=' '.join([words for words in text])
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


# In[434]:


#wordcloud for all tweets in 2022
create_wordcloud(data22["cleaned_tweet"].values)

plt.savefig("Downloads/wordcloud_22_1.png")


# In[435]:


#wordcloud for positive tweets
create_wordcloud(data22[data22["sentiment"]==1]["cleaned_tweet"].values)

plt.savefig("Downloads/wordcloud_22_pos1.png")


# In[436]:


#wordcloud for positive tweets
create_wordcloud(data22[data22["sentiment"]==-1]["cleaned_tweet"].values)

plt.savefig("Downloads/wordcloud_22_neg1.png")


# ### Sentiment Curve for 2022

# In[437]:


data_list=["carbon","offsets","credit","blockchain","carbon credits", "carbon offsets"]
pattern="|".join(data_list)
data22_sent=data18[(data18["cleaned_tweet"].str.contains(pattern))]

data22_sent=data22_sent[data22_sent["year"]==2022]


# In[444]:


len(data22_sent)


# In[443]:


data22_sent = data22_sent[data22_sent["sentiment"]!='no data']


# In[445]:


data22_sent_gp=data22_sent.groupby(['month'])["sentiment"].sum()


# In[446]:


data22_sent_gp=data22_sent_gp.reset_index()
data22_sent_gp


# In[448]:


from sklearn import preprocessing

X = preprocessing.MinMaxScaler()
scaled_sent22= pd.DataFrame(X.fit_transform(data22_sent_gp.iloc[:,1:]),columns=data22_sent_gp.columns[1:])
scaled_sent22["month"]=data22_sent_gp["month"]


# In[452]:


scaled_sent22.set_index('month').plot();

plt.savefig("Downloads/22_month_sent.png")


# In[455]:


data22_sent=data18[(data18["cleaned_tweet"].str.contains(pattern))]
data21_sent=data22_sent[data22_sent["year"]==2021]


# In[456]:


len(data21_sent)


# In[457]:


data21_sent = data21_sent[data21_sent["sentiment"]!='no data']
data21_sent_gp=data21_sent.groupby(['month'])["sentiment"].sum()

data21_sent_gp=data21_sent_gp.reset_index()
data21_sent_gp


# In[458]:


from sklearn import preprocessing

X = preprocessing.MinMaxScaler()
scaled_sent21= pd.DataFrame(X.fit_transform(data21_sent_gp.iloc[:,1:]),columns=data21_sent_gp.columns[1:])
scaled_sent21["month"]=data21_sent_gp["month"]


# In[460]:


scaled_sent21.set_index('month').plot();

plt.savefig("Downloads/21_month_sent.png")


# In[464]:


data22


# ### EDA of results

# In[3]:


#data22.to_csv('carbondata_labeled_custom22.csv', index=False)
#data21.to_csv('carbondata_labeled_custom21.csv', index=False)

data22 = pd.read_csv('carbondata_labeled_custom22.csv')
data21 = pd.read_csv('carbondata_labeled_custom21.csv')


# In[4]:


data22


# In[4]:


plt.figure(figsize=(15, 8)) 

plt.subplot(221) 
sns.histplot(x=data21.month,stat='count',binwidth=1,kde='true',discrete=True) 
plt.title('2021 Monthly Tweets Counts')
plt.xticks(np.arange(1,13,1)) 
plt.grid() 

plt.subplot(222) 
sns.histplot(x=data22.month,stat='count',binwidth=1,kde='true',discrete=True) 
plt.title('2022 Monthly Tweets Counts')
plt.xticks(np.arange(1,13,1)) 
plt.grid() 


# In[63]:


#ax=plt.subplot(221) 
sns.lineplot(data21.month.value_counts()) 
ax.set_xlabel("Month") 
ax.set_ylabel('Count') 


# In[5]:


print(data21['TweetC'][data21['month']==9][:10])


# In[6]:


print(data21['clean_tweet'][data21['month']==9][:10])


# In[5]:


data22 = data22[data22['cleaned_tweet'].notna()]


# In[18]:


data21 = data21[data21['cleaned_tweet'].notna()]


# In[ ]:


len(data22)


# In[19]:


len(data21)


# In[9]:


data_list=["john oliver","oliver"]
pattern="|".join(data_list)

data22_oliver = data22[data22['month'].isin([8,9])]
data22_oliver=data22_oliver[(data22_oliver["cleaned_tweet"].str.contains(pattern))]
sns.countplot(x=data22_oliver["sentiments_val"]);

plt.title("Sentiment Distribution of Oliver Tweets", fontsize=20)


# In[10]:


data22_fall = data22[data22['month'].isin([8,9])]
sns.countplot(x=data22_fall["sentiments_val"]);
plt.title("Sentiment Distribution on Aug/Nov", fontsize=20)


# In[29]:


data21_1112 = data21[['Date_Tweet', 'Retweet_Count', 'TweetC',"cleaned_tweet",'sentiments_val']][(data21['month'].isin([10,11]))]
data21_1112 = data21_1112.sort_values(by=['Retweet_Count'], ascending=False)
data21_1112


# In[32]:


for i in range(len(data21_1112)):
    #if data21['sentiments_val'][i] == "positive":
    print("Date:", data21_1112['Date_Tweet'].iloc[i], data21_1112['sentiments_val'].iloc[i], data21_1112['TweetC'].iloc[i])


# In[25]:


#wordcloud for positive tweets
create_wordcloud(data21_1112[data21_1112['sentiments_val']=="positive"]["cleaned_tweet"].values)


# In[ ]:





# In[ ]:




