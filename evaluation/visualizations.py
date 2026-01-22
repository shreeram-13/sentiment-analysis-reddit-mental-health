import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = r"../data/final_predictions.csv"

df = pd.read_csv(FILE_PATH)
print("\n Dataset loaded.")
print("Sample data:\n", df.head(3))

if 'Predicted_Sentiment' not in df.columns:
    raise ValueError(" Required column 'Predicted_Sentiment' not found.")

df = df.dropna(subset=['Predicted_Sentiment'])
print(f"\n Total records used for analysis: {len(df)}")

# Sentiment Distribution (All Classes)
sentiment_counts_all = df['Predicted_Sentiment'].value_counts()
print("\n Sentiment distribution (all):\n", sentiment_counts_all)

plt.figure(figsize=(8, 5))
plt.bar(
    sentiment_counts_all.index,
    sentiment_counts_all.values,
    color=plt.cm.Set2.colors,
    edgecolor='black'
)

plt.title("Sentiment Distribution of Reddit Comments (All Classes)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.xticks(rotation=30, ha='right')

for i, v in enumerate(sentiment_counts_all.values):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Sentiment Distribution (Excluding 'normal')
df_filtered = df[df['Predicted_Sentiment'].str.lower() != 'normal']
sentiment_counts_filtered = df_filtered['Predicted_Sentiment'].value_counts()
print("\n Sentiment distribution (excluding 'normal'):\n", sentiment_counts_filtered)

plt.figure(figsize=(8, 5))
colors = plt.cm.Paired(range(len(sentiment_counts_filtered)))

bars = plt.bar(
    sentiment_counts_filtered.index,
    sentiment_counts_filtered.values,
    color=colors,
    edgecolor='black'
)

plt.title("Emotion Distribution in Reddit Comments (Excluding Normal)")
plt.xlabel("Emotion")
plt.ylabel("Number of Comments")
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(int(bar.get_height())),
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(7, 7))
plt.pie(
    sentiment_counts_all.values,
    labels=sentiment_counts_all.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Set3.colors
)
plt.title("Overall Sentiment Proportion in Reddit Data")
plt.show()

print("\n Visualization and analysis done!")