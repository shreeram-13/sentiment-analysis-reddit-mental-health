import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

FILE_PATH = r"../data/labelled_sentiments.csv"

df = pd.read_csv(FILE_PATH)
print("\n Dataset loaded.")
print("Sample data:\n", df.head(5))

df = df.rename(columns={'statement': 'review', 'status': 'sentiment'})
print("\nNull values:\n", df.isnull().sum())

# Encoding target labels
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])
print("\n After encoding:\n", df.head(5))

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Downloading necessary NLTK data
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

print("\n Preprocessing text")
df['review'] = df['review'].astype(str).apply(preprocess_text)

print("\n After preprocessing:\n", df.head(2))

X = df['review']
y = df['sentiment']

# Training Logistic Regression Model and Predictions
vectorizer = CountVectorizer(max_features=5000)
X_counts = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_counts, y, test_size=0.3, random_state=42
)

reg = LogisticRegression(max_iter=1000)
reg.fit(X_train, y_train)

reg_pred = reg.predict(X_test)

# Evaluation Metrics
print("\nModel: Logistic Regression")
print("Sample Predictions: ", reg_pred[:10])

lr_acc = accuracy_score(y_test, reg_pred)
lr_pre = precision_score(y_test, reg_pred, average='weighted')
lr_rec = recall_score(y_test, reg_pred, average='weighted')
lr_f1 = f1_score(y_test, reg_pred, average='weighted')

lr_roc = roc_auc_score(y_test, reg.predict_proba(X_test), multi_class='ovr')
lr_cm = confusion_matrix(y_test, reg_pred)

print("Model Performance")
print("Accuracy:", round(lr_acc, 3))
print("Precision:", round(lr_pre, 3))
print("Recall:", round(lr_rec, 3))
print("F1-Score:", round(lr_f1, 3))
print("ROC-AUC:", round(lr_roc, 3))
print("Confusion Matrix:\n", lr_cm)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(lr_cm, cmap='Blues', interpolation='nearest')
plt.title('Logistic Regression Confusion Matrix')
plt.colorbar()

classes = le.classes_
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, ha='right')
plt.yticks(tick_marks, classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, lr_cm[i, j], ha='center', va='center', color='black')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

print("\n Model training and evaluation done.")