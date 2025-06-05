# Zomato Review Sentiment Analysis Project

import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle


# Load dataset
df = pd.read_csv("c:/Users/nikit/Downloads/zomato_reviews.csv")
print(df)

# Drop unnecessary columns and NaNs
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df = df.dropna(subset=['review'])
print("✅ Dropped rows with missing values in the 'review' column.")

# Basic stopwords list
simple_stopwords = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'yours',
    'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
    'what', 'which', 'who', 'whom', 'this', 'that', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'can', 'will', 'just', 'don', 'should', 'now'
])
print(f"✅ Created basic stopwords list with {len(simple_stopwords)} words.")


# Text cleaning function
def basic_clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in simple_stopwords]
    return ' '.join(tokens)
sample_text = "The food was absolutely amazing! I loved it."
cleaned_text = basic_clean_text(sample_text)
print(f"🔍 Original: {sample_text}")
print(f"🧹 Cleaned: {cleaned_text}")


# Clean the review text
df['cleaned_review'] = df['review'].apply(basic_clean_text)
print(df[['review', 'cleaned_review']].head(5))


# Label sentiment
def label_sentiment(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

df['sentiment'] = df['rating'].apply(label_sentiment)
print("✅ Sentiment labels assigned based on 'rating' column. New column 'sentiment' created.")


# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']
print("Sample TF-IDF features:", vectorizer.get_feature_names_out()[:10])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split into train and test sets.")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)
print(f"📈 Training accuracy: {train_acc:.4f}")


# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"✅ Model evaluation complete.")
print(f"Accuracy on test set: {accuracy:.4f}")
print("Classification Report:\n", report)


# Save model and vectorizer
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
    print("✅ Model saved to 'sentiment_model.pkl'.")
    print("✅ TF-IDF vectorizer saved to 'tfidf_vectorizer.pkl'.")
