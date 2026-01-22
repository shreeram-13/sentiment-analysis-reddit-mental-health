import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

# Downloading NLTK resources
nltk.download('stopwords')

INPUT_FILE = "../data/sample_reddit_data.csv"
OUTPUT_FILE = "../data/reddit_clean.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f" Could not find '{INPUT_FILE}' in this folder.")

# Reading in chunks (for large files)
chunk_size = 50000
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower().strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(tokens)

# Removing old output file if exists
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

chunk_no = 1
for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size):
    print(f"Processing chunk {chunk_no}...")

    if 'post_title' in chunk.columns and 'comment_body' in chunk.columns:
        chunk['merged_text'] = chunk['post_title'].fillna('') + " " + chunk['comment_body'].fillna('')
    elif 'post_title' in chunk.columns:
        chunk['merged_text'] = chunk['post_title'].fillna('')
    elif 'comment_body' in chunk.columns:
        chunk['merged_text'] = chunk['comment_body'].fillna('')
    else:
        raise ValueError(" No suitable text columns found.")

    chunk['cleaned_text'] = chunk['merged_text'].apply(clean_text)

    chunk.to_csv(OUTPUT_FILE, mode='a', index=False, header=not os.path.exists(OUTPUT_FILE))
    chunk_no += 1

print(f" Cleaning complete! Saved to {OUTPUT_FILE}")
