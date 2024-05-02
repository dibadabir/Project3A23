## Project3A23

The purpose of this project is to successfully develop a sentiment analysis system to analyse public opinions toward artificial inteligence (AI) from various resources such as websites, podcasts, videos, etc. The goal is to understand general sentiment (positive, negative, or neutral) towards AI-realted topics. This README file has been created to make it easier to use this repository. You will find information here about the structure and what files to use to create the final product.

## Team members

* Justyna Dobersztajn
* Diba Dabiransari
* Aleeza Azad

## Development Methodology
For this assignment we used the Cross Industry Standard Process for Data Mining (CRISP DM) to move on with our project. There are various steps to this model which are:
* Business Understanding
* Data Understanding
* Data Preparation                                                                                                     
* Modelling
* Evaluation
* Deployment
                                                                            ![CRISP-DM](https://github.com/dibadabir/Project3A23/assets/152966994/3ceeae0c-6898-4fb1-892b-ce43fb3af03b)

## 1. What does the business need?
As mentioned earlier, the aim of this project is to analyse public opinion and gain a better understanding of their thoughts about AI. We identified categories in which AI is utilised and did some in-depth research about them.
Moreover, this project is supposed to be a good practice for us to improve our skills, both technical and communicative skills. We were able to explore and develop a machine learning model whilst being exposed to diffrent platforms suh as google colab and pycharm.

## 2. What data do we need?
For this analysis, we chose six categories representing sectors impacted by artificial intelligence:
* AI in Healthcare
* AI in Travel Industry
* AI in Fashion
* AI in E-Commerce / Marketing
* AI in Education
* Generative AI
  
Our aim was to gather data that both describes and critiques the utilisation of artificial systems within these domains.

## 3. Cleaning, Processing, and labeling
After doing some research and choosing our subject, we started looking for websites and scraping the text out of them. We used BeautifulSoup to extract text from websites and Deepgram to transcribe podcasts.

The data that were scraped from websites and audios, were raw data and we were not able to use it just on its own. We had to process the textual data in a way that would be understandable for the machine. We first cleaned the collected data, the punctuation marks, special characters and stop words were removed. Instead of completely deleting numeric numbers, we replaced them with their written forms. After that, split paragraphs into sentences so that the machine can process them better.

The next step was to label our data. We used to diffrent libraries to tag our data with Positive, Negative, or Neutral. The libraries used were TextBlob and Vader, both widely recognized in the field of Natural Language Processing (NLP). We used them to compute the polarity of the sentences and then label them based on that criteria.

### Example of **web scraping** implementation - Education

#### Using TextBlob
* [Web-Scraping Code File](https://github.com/dibadabir/Project3A23/blob/main/Web%20Scraping/Education/Education_webscrape_code%20file%20without%20numbers%20in%20the%20dataset.ipynb)
* [Scraped Results](https://github.com/dibadabir/Project3A23/blob/main/Web%20Scraping/Education/education%20dataset%20(no%20numbers).csv)

#### Using Vader
* [Web-Scraping Code File](https://github.com/dibadabir/Project3A23/blob/main/Web%20Scraping/Education/Education_webscrape_(without_numbers)_Vader_ver_.ipynb)
* [Scraped Results](https://github.com/dibadabir/Project3A23/blob/main/Web%20Scraping/Education/Education%20dataset%20(no%20numbers)%20-%20Vader%20ver.csv)

### Example of **audio scraping** implementation - Education
* [Audio-Scraping Code File](https://github.com/dibadabir/Project3A23/blob/main/Speech%20to%20Text/Education/Audio_Scraping_Using_DEEPGRAM(Education).ipynb)
* [Scraped Results](https://github.com/dibadabir/Project3A23/blob/main/Speech%20to%20Text/Fashion/Audio_Scraping_Using_DEEPGRAM(Fashion).ipynb)

## 4. Choosing the best model
Following the machine learning pipeline, the next phase would be choosing an appropriate model for sentiment analysis. We explored a variety of supervised learning algorithms such as **Naive Bayes**, **Logistic Regression**, **Random Forest**, and **Decision Tree** using two vectorizing methods, CountVectorizer and TF-IDF Vectorizer and then we compared the accuracy of these models and how diffrnt each model performs. We then demonstrated the results to gain a better understanding of them and make a better decision.

[Training Models using CountVectorizer](https://github.com/dibadabir/Project3A23/blob/main/Final%20(Everything%20combined!)/Compare_Models_(CountVectorizer).ipynb)

[Training Models using TF-IDF Vectorizer](https://github.com/dibadabir/Project3A23/blob/main/Final%20(Everything%20combined!)/Compare_Models_(TF_IDFVectorizer).ipynb)

## 5. Evaluating Models


### Libraries used in this project

```
import re
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import requests
import nltk
import os
import httpx
import json
import pickle
import logging, verboselogs
from bs4 import BeautifulSoup as bs
from num2words import num2words
from textblob import TextBlob
from dotenv import load_dotenv
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deepgram import ( DeepgramClient, DeepgramClientOptions, PrerecordedOptions, FileSource)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
```

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
