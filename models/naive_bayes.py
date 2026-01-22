import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import nltk
import re

FILE_PATH = r"../data/labelled_sentiments.csv"

df = pd.read_csv(FILE_PATH)
print("\n Dataset loaded.")
print("Sample data:\n", df.head(5))

df = df.rename(columns={'statement': 'review', 'status': 'sentiment'})

print("\nNull values:\n", df.isnull().sum())

# Encode sentiment
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])
print("\n After encoding:\n", df.head(5))

df = df.sample(frac=1).reset_index(drop=True)

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Text Preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub('<.*?>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

print("\n Preprocessing text...")
df = df.dropna(subset=['review'])
df['review'] = df['review'].astype(str).apply(preprocess_text)

print("\n After preprocessing:\n", df.head(2))

X = df['review']
y = df['sentiment']

vectorizer = CountVectorizer(max_features=5000)
X_counts = vectorizer.fit_transform(X)

# Training Naive Bayes Model
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.3, random_state=42)

print("\n Training Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nNaive Bayes Model")
print("Sample Predictions: ", y_pred[:10])

# Evaluation Metrics
nb_acc = accuracy_score(y_test, y_pred)
nb_pre = precision_score(y_test, y_pred, average='weighted')
nb_rec = recall_score(y_test, y_pred, average='weighted')
nb_f1 = f1_score(y_test, y_pred, average='weighted')
nb_cm = confusion_matrix(y_test, y_pred)

print("Model Performance")
print("Accuracy:", round(nb_acc, 3))
print("Precision:", round(nb_pre, 3))
print("Recall:", round(nb_rec, 3))
print("F1-Score:", round(nb_f1, 3))
print("Confusion Matrix:\n", nb_cm)

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
plt.imshow(nb_cm, cmap='Blues', interpolation='nearest')
plt.title('Naive Bayes Confusion Matrix')
plt.colorbar()

classes = le.classes_
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, ha='right')
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, nb_cm[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

print("\n Model training and evaluation done.")