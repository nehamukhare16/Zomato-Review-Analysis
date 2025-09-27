from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

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

def basic_clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in simple_stopwords]
    return ' '.join(tokens)

# Load model and vectorizer once when the app starts
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    review = ''
    if request.method == 'POST':
        review = request.form['review']
        cleaned_review = basic_clean_text(review)
        vectorized_review = vectorizer.transform([cleaned_review])
        prediction = model.predict(vectorized_review)[0]
        sentiment = prediction.capitalize()
    return render_template('index.html', sentiment=sentiment, review=review)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

