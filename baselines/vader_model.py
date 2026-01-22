import pandas as pd
import numpy as np
import string
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Loading dataset
FILE_PATH = r"../data/labelled_sentiments.csv"
df = pd.read_csv(FILE_PATH)
print("Dataset loaded.")
print(df.head())

df = df.rename(columns={'statement': 'comments', 'status': 'sentiment'})
print("\nNull values:\n", df.isnull().sum())

# Text Cleaning & Preprocessing
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = str(text).lower()
    text = [word.strip(string.punctuation) for word in text.split()]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and len(word) > 1]
    pos_tags = pos_tag(text)
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(text)

print("\nPreprocessing text..")
df['comments_clean'] = df['comments'].apply(clean_text)
print("\nSample after cleaning:\n", df[['comments', 'comments_clean']].head())


# VADER Sentiment Analysis
sid = SentimentIntensityAnalyzer()
df['sentiments'] = df['comments_clean'].apply(lambda x: sid.polarity_scores(x))

df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

df['nb_chars'] = df['comments'].fillna("").apply(lambda x: len(str(x)))
df['nb_words'] = df['comments'].fillna("").apply(lambda x: len(str(x).split()))

print("\nData with sentiment scores and text metrics:\n", df.head())

# Plotting sentiment distributions
plt.figure(figsize=(12, 5))
sns.histplot(df['compound'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of VADER Compound Sentiment Scores')
plt.xlabel('Compound Score')
plt.ylabel('Frequency')
plt.show()

df['vader_label'] = df['compound'].apply(lambda c: 'positive' if c > 0.05 else ('negative' if c < -0.05 else 'neutral'))
plt.figure(figsize=(6, 5))
sns.countplot(x='vader_label', data=df, palette='Set2')
plt.title('Count of VADER Sentiment Labels')
plt.show()