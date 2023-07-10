# Sentiment analysis of public opinions on the blockchain approach for carbon credit markets on Twitter.
![DSL logo](images/dsl_prjstructure.png)

[Dynamic Sustainability Lab]([images/dsl_prjstructure.png](https://www.dynamicslab.org/))

I'm working on a solo project of sentiment analysis of public opinions on the blockchain approach for carbon credit markets on Twitter. 

### Porject Description 

The project's goals were to analyze tweets in English that can help understand public opinion on blockchain and its role in the transition to a net-zero economy around the world and find possible correlations between the geo-political and demographical background of those Tweets. For that purpose, various unsupervised approaches were used and evaluated.

![Project structure diagram](images/dsl_prjstructure.png)

The project's poster is available through [this link](reports/DSL_poster_v2.pdf). The project's paper is [here](reports/DSL_paper_v1.pdf).
The full code is available here.

### Project Overview:

#### Data
In this paper, the project collects and analyzes the contents of around 225,098 English tweets that discussed either the current general perception of carbon credits or the blockchain role in the transition to a net-zero carbon economy in the past 2 years period (1/1/2021 – 10/31/2022).

The library used for data scraping is [snscrape](https://github.com/JustAnotherArchivist/snscrape)

#### Methods
-**K-Means**. First, I wanted to explore one of the most popular techniques for unsupervised sentiment analysis, K-Means clustering. The size of the dataset made it possible to create a large enough dictionary of words for the Word2vec model. I decided to work with 3 clusters: positive, negative, and neutral. The neutral cluster is supposed to collect possible spam tweets or tweets with not enough information for humans to determine the sentiment.

-**VADER** (Valence Aware Dictionary and sEntiment Reasoner). Second, I imported and applied the VADER algorithm on the same per-processed text as that given to K-Means. VADER is a key-based algorithm for sentiment analysis, which means it has its own dictionary of words for sentiment classification 

-**BERT** (Bidirectional Encoder Representations for Transformers). Finally, I decided to run BERT, which is a model with pre-trained language representations that has an internal library for sentiment analysis (6). BERT is able to identify sentiment based on common keywords, sentence structure, as well as the context of each tweet based on the generated embeddings. By design, BERT is able to identify either positive or negative tweets. It is one of the most advanced unsupervised methods for sentiment analysis yet and I wanted to see how similar its results are to the K-Means model

#### Pre-processing

Import necessary libraries and functions.
```
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
  
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
```

Write function to clean and tokenize the data:
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
```

#### Methods
