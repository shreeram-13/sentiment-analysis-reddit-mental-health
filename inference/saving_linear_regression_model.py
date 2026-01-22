import numpy as np
import pandas as pd
import re
import nltk
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

train = False
# Set True to train & save model; False to only predict

train_file = r"../data/labelled_sentiments.csv"  # labelled data
predict_file = r"../data/reddit_clean.csv"  # New unlabeled data of reddit comments
model_file = '../saved-models/logistic_regression.pkl'
vectorizer_file = '../saved-models/vectorizer.pkl'
labelencoder_file = '../saved-models/label_encoder.pkl'

# NLTK
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Text Preprocessing
def preprocess_text(text):
    text = re.sub('<.*?>', ' ', str(text))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# TRAINING & SAVING MODELS
if train:
    print("\n Loading labeled dataset for training...")
    df = pd.read_csv(train_file)
    df = df.rename(columns={'statement': 'review', 'status': 'sentiment'})
    print("Dataset loaded. Sample:")
    print(df.head(3))

    le = LabelEncoder()
    df['sentiment'] = le.fit_transform(df['sentiment'])

    # Preprocess text
    df['review'] = df['review'].astype(str).apply(preprocess_text)

    X = df['review']
    y = df['sentiment']

    vectorizer = CountVectorizer(max_features=5000)
    X_counts = vectorizer.fit_transform(X)

    # Train Logistic Regression
    print("\n Training Logistic Regression model...")
    reg = LogisticRegression(max_iter=1000)
    reg.fit(X_counts, y)
    print(" Training completed.")

    # Save model, vectorizer, and label encoder
    joblib.dump(reg, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    joblib.dump(le, labelencoder_file)
    print(f" Model, vectorizer, and label encoder saved. ")


# PREDICTION MODE
else:
    print("\n Loading model and vectorizer for prediction...")
    # Loading model
    reg = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    le = joblib.load(labelencoder_file)

    # Load unlabeled dataset
    df_new = pd.read_csv(predict_file)
    if 'cleaned_text' not in df_new.columns:
        raise ValueError("The dataset must contain a 'cleaned_text' column.")
    df_new['clean_text'] = df_new['cleaned_text'].astype(str)

    # Transform using saved vectorizer
    X_new = vectorizer.transform(df_new['clean_text'])

    # Predictions
    preds = reg.predict(X_new)
    df_new['Predicted_Sentiment'] = le.inverse_transform(preds)

    output_file = r"../data/final_predictions.csv"
    df_new.to_csv(output_file, index=False)
    print(f" Predictions saved to {output_file}")