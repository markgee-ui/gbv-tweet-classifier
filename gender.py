#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# gbv_classifier_app.py

import pandas as pd
import numpy as np
import re
import string
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud

# 1. Load Data
@st.cache_data
def load_data():
    train_df = pd.read_csv("Train.csv")
    test_df = pd.read_csv("Test.csv")
    return train_df, test_df

train_df, test_df = load_data()

# 2. Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

train_df['clean_tweet'] = train_df['tweet'].apply(clean_text).apply(remove_stopwords)
test_df['clean_tweet'] = test_df['tweet'].apply(clean_text).apply(remove_stopwords)

# 3. Train Model
X = train_df['clean_tweet']
y = train_df['type']

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=300))
])

model.fit(X, y)

# 4. Predict on Test Data
test_preds = model.predict(test_df['clean_tweet'])

submission = pd.DataFrame({
    'Tweet_ID': test_df['Tweet_ID'],
    'type': test_preds
})

# 5. Streamlit UI
st.title("GBV Tweet Classifier")

st.markdown("""
This app classifies tweets into the following Gender-Based Violence categories:
- Sexual Violence
- Emotional Violence
- Harmful Traditional Practices
- Physical Violence
- Economic Violence
""")

# --- Sidebar ---
st.sidebar.title("📊 Visualizations")
selected_vis = st.sidebar.selectbox("Select a Visualization", [
    "GBV Class Distribution",
    "Tweet Length Distribution",
    "Most Frequent Words (WordCloud)",
])

# --- Visualizations ---
if selected_vis == "GBV Class Distribution":
    st.subheader("Distribution of GBV Classes")
    fig, ax = plt.subplots()
    sns.countplot(data=train_df, x='type', order=train_df['type'].value_counts().index, palette='Set2', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

elif selected_vis == "Tweet Length Distribution":
    st.subheader("Tweet Length Distribution")
    train_df['tweet_length'] = train_df['clean_tweet'].apply(lambda x: len(x.split()))
    fig, ax = plt.subplots()
    sns.histplot(train_df['tweet_length'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_xlabel("Number of Words")
    st.pyplot(fig)

elif selected_vis == "Most Frequent Words (WordCloud)":
    st.subheader("Most Common Words in Tweets")
    all_words = ' '.join(train_df['clean_tweet'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# --- Classification Interface ---
st.markdown("---")
example = st.text_area("Enter a tweet to classify", placeholder="e.g., She was beaten by her husband every night...")

if st.button("Classify"):
    cleaned = remove_stopwords(clean_text(example))
    pred = model.predict([cleaned])[0]
    st.success(f"Predicted GBV Category: **{pred.replace('_', ' ').title()}**")

# --- Download Section ---
st.markdown("---")
st.markdown("### Download Prediction Results")
st.dataframe(submission.head())

csv = submission.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='gbv_predictions.csv',
    mime='text/csv'
)


# In[2]:


get_ipython().system('jupyter nbconvert --to script your_notebook.ipynb')


# In[3]:


get_ipython().system('jupyter nbconvert --to script gender.ipynb')


# In[ ]:




