# Stock-sentiment-analysis-using-news-headlines
## Table of contents
   - [Overview](#overview)
   - [Motivation](#motivation)
   - [Format](#format)
   - [Technical Aspect](#technical-aspect)
   - [Technologies Used](#technologies-used)
   - [Credits](#credits)

### Overview
This is a simple stock sentiment analysis using news headlines based on Natural Language Processing in Python. It is implemented using simple bag-of-words model which predicts whether the stock price has an increasing trend (label as 1) or has decreasing or constant trend (label as 0).
### Motivation
The price of stock of a company depends on different events associated with the company. Based on the top news headlines for a particular day, we would like to predict the type of trend of stock price.    
### Format
The dataset in consideration is a combination of the world news and stock price shifts available on Kaggle. The Stock Sentiment analysis (text file: stock_predict) has a total 4101 entries of top 25 headlines between the year 2008 and 2016 among which 1935 have label 0 (47.18%) and 2166 have label 1(52.82%). Each line in the dataset is composed by 27 columns: date, label (0 or 1) and 25 others as top headlines of the day.![Capture](https://user-images.githubusercontent.com/74978788/130363522-e2bcb02e-2317-4e04-8f98-a88dca63da08.JPG)

### Technical Aspect
This project is divided into three parts:
1. Cleaning and preprocessing of collected data.
2. Training various machine learning and deep learning models.
3. Comparing the accuracy of different models on the test data.
### Technologies Used
The Code is written in Python 3.9. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. I have used Google Colaboratory, or "Colab" for short which allows us to write and execute Python in browser. I have used various open-source Python packages in the solution, for example:
1. [PANDAS](#pandas)
2. [RE](#re)
3. [NLTK](#re)
4. [SCIKIT-LEARN](#scikit-learn)
5. [KERAS](#keras)
6. [NUMPY](#numpy)
#### PANDAS
PANDAS is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language and made available through the pandas module.
#### RE
Regular expressions (called REs) are essentially a tiny, highly specialized programming language embedded inside Python and made available through the re module.
#### NLTK
NLTK is a leading platform for building Python programs to work with human language data and made available through the nltk module. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers, stopwords for industrial-strength NLP libraries, and an active discussion forum.
#### SCIKIT-LEARN
SCIKIT-LEARN is made available through the sklearn module which is a simple and efficient tool for predictive data analysis built on NumPy, SciPy, and matplotlib. I have used CountVectorizer, different metrics for testing accuracy and various ML models.
#### KERAS
From Tensorflow we imported Keras which is one of the most used deep learning frameworks as it makes it easier to run new experiments. Keras API is designed for human beings, which follows best practices for reducing cognitive load as it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. It also has extensive documentation and developer guides.
#### NUMPY
Numpy is the fundamental package for scientific computing with Python which offers powerful multi-dimensional arrays and various numerical computing tools for free.
### Credits
The dataset has been collected from Kaggle's stock_sentiment_analysis.  
