import sys
import subprocess

# Function to install required packages
def install_packages():
    required_packages = [
        'Flask', 'torch', 'transformers', 'pandas', 'torch_directml',
        'nltk', 'emoji', 'contractions', 'tqdm'
    ]
    for package in required_packages:
        try:
            __import__(package.split('==')[0])  # Try to import package without version
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Install required packages if not already installed
install_packages()

# Now continue with the rest of your app.py imports
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, render_template, request, jsonify
import emoji
import contractions
import nltk

# Ensure NLTK data is downloaded (place this after imports)
def ensure_nltk_data():
    packages = ['wordnet', 'omw-1.4', 'stopwords']
    for package in packages:
        try:
            nltk.data.find(f'corpora/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

ensure_nltk_data()
app = Flask(__name__)

# Initialize NLTK tools
lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load the fine-tuned model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained('./final_model')
model.to(device)
tokenizer = AutoTokenizer.from_pretrained('./final_tokenizer')

# Data Cleaning Functions
def convert_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def expand_contractions(text):
    return contractions.fix(text)

def handle_negations(tweet):
    tweet = re.sub(r'\b(not|no)\s+(\w+)', r'\1_\2', tweet)
    return tweet

def clean_tweet(tweet):
    tweet = expand_contractions(tweet)
    tweet = convert_emojis(tweet)
    tweet = handle_negations(tweet)
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.lower()
    words = tweet.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Route to serve the main webpage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle sentiment analysis requests
@app.route('/predict', methods=['POST'])
def predict():
    message = request.json['message']
    cleaned_message = clean_tweet(message)
    inputs = tokenizer(
        cleaned_message,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = 'positive' if torch.argmax(probs) == 1 else 'negative'
    confidence = torch.max(probs).item()
    return jsonify({'sentiment': sentiment, 'confidence': f"{confidence:.2f}"})

# Route to handle random tweet generation (optional)
@app.route('/random_tweet', methods=['GET'])
def random_tweet():
    # Load a sample of tweets for random selection
    data_sample = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None, nrows=1000)
    data_sample.columns = ['Sentiment', 'ID', 'Date', 'Query', 'User', 'Tweet']
    random_tweet = data_sample['Tweet'].sample(n=1).values[0]
    return jsonify({'tweet': random_tweet})

if __name__ == '__main__':
    app.run(debug=True)
