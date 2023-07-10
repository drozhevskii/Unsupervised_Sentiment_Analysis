# Sentiment analysis of public opinions on the blockchain approach for carbon credit markets on Twitter.
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

#### Pre-processing

Import necessary libraries and functions.
```
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
  
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
```

#### Methods
