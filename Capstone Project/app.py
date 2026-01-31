from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load model & features
with open("sentiment_model_1.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("word_features_1.pkl", "rb") as f:
    word_features = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(words):
    words = [w.lower() for w in words if w.isalpha()]
    words = [w for w in words if w not in stop_words]
    return words

def extract_features_from_text(text):
    words = word_tokenize(text)
    words = set(clean_text(words))
    features = {}

    for word in word_features:
        features[word] = (word in words)

    return features

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None

    if request.method == "POST":
        text = request.form["text"]
        features = extract_features_from_text(text)
        sentiment = classifier.classify(features)

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
