#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


#Importing Necessary Libraires
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load pre-trained model
model = joblib.load('model.joblib')

vectorizer = joblib.load('vectorizer.joblib')


# In[2]:


#Define a function to clean the tweet
def clean_tweet(tweet):
    # Remove URLs, mentions, and hashtags
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@[^\s]+[\s]?", "", tweet)
    tweet = re.sub(r"#([^\s]+[\s]?)+", "", tweet)

    # Remove stopwords and stem the remaining words
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    words = nltk.word_tokenize(tweet)
    words = [stemmer.stem(word) for word in words if word not in stopwords_english]

    return " ".join(words)


# In[4]:


#Define streamlit app

def app():
    # Set the title and page icon
    st.set_page_config(page_title='Sentiment Analysis', page_icon=':smiley:')

    # Add a title and subtitle
    st.title('Sentiment Analysis')
    st.write('Enter a tweet below to predict the sentiment.')
    

    # Add a text input for user input
    tweet_input = st.text_input('Enter a tweet:')
    analyze_button = st.button('Analyze')
    
    # Check if the user has entered anything
    if tweet_input:
        # Clean the tweet
        cleaned_tweet = clean_tweet(tweet_input)

        # Vectorize the cleaned tweet using your pre-trained vectorizer
        # This step depends on how you vectorized your data during training, so modify it accordingly
        tweet_vectorized = vectorizer.transform([cleaned_tweet])

        # Predict the sentiment of the tweet using your pre-trained model
        sentiment = model.predict(tweet_vectorized)[0]

        # Display the predicted sentiment
        if sentiment == 'positive':
            st.write('The sentiment is positive :smile:')
        elif sentiment == 'negative':
            st.write('The sentiment is negative :disappointed:')
        else:
            st.write('The sentiment is neutral :neutral_face:')


# In[5]:


if __name__ == '__main__':
    app()


# In[ ]:





# In[ ]:




