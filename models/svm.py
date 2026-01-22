import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

FILE_PATH = "../data/labelled_sentiments.csv"

df = pd.read_csv(FILE_PATH)
df = df.rename(columns={"statement": "review", "status": "sentiment"})

# Encode labels
le = LabelEncoder()
df["sentiment"] = le.fit_transform(df["sentiment"])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Text preprocessing & Feature extraction
def preprocess_text(text):
    text = re.sub("<.*?>", " ", str(text))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

df["review"] = df["review"].astype(str).apply(preprocess_text)

X = df["review"]
y = df["sentiment"]

vectorizer = CountVectorizer(max_features=5000)
X_counts = vectorizer.fit_transform(X)

# Training SVM
X_train, X_test, y_train, y_test = train_test_split(
    X_counts, y, test_size=0.3, random_state=42
)

svm = SVC(
    kernel="linear",
    probability=True,
    random_state=42
)
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

# Evaluation Metrics
svm_acc = accuracy_score(y_test, svm_pred)
svm_pre = precision_score(y_test, svm_pred, average="weighted")
svm_rec = recall_score(y_test, svm_pred, average="weighted")
svm_f1 = f1_score(y_test, svm_pred, average="weighted")
svm_roc = roc_auc_score(y_test, svm.predict_proba(X_test), multi_class="ovr")
svm_cm = confusion_matrix(y_test, svm_pred)

print("\nModel: Support Vector Machine")
print("Accuracy:", round(svm_acc, 3))
print("Precision:", round(svm_pre, 3))
print("Recall:", round(svm_rec, 3))
print("F1-Score:", round(svm_f1, 3))
print("ROC-AUC:", round(svm_roc, 3))
print("Confusion Matrix:\n", svm_cm)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(svm_cm, cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.colorbar()

classes = le.classes_
ticks = np.arange(len(classes))
plt.xticks(ticks, classes, rotation=45, ha="right")
plt.yticks(ticks, classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, svm_cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# NOTE:
# SVM with probability=True can be slow on large text datasets.
# This script is intended for model comparison and academic evaluation.