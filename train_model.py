import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch_directml  # For AMD GPU support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import nltk
import logging
import emoji
import contractions
import multiprocessing as mp

# Suppress NLTK logging and output
logging.getLogger("nltk").setLevel(logging.CRITICAL)

# Ensure NLTK data is downloaded
def ensure_nltk_data():
    packages = ['wordnet', 'omw-1.4', 'stopwords']
    for package in packages:
        try:
            nltk.data.find(f'corpora/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

ensure_nltk_data()

# Data Cleaning Functions
def convert_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def expand_contractions(text):
    return contractions.fix(text)

def handle_negations(tweet):
    tweet = re.sub(r'\b(not|no)\s+(\w+)', r'\1_\2', tweet)
    return tweet

def clean_tweet(tweet):
    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
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

def process_tweet(args):
    index, tweet = args
    cleaned_tweet = clean_tweet(tweet)
    return index, cleaned_tweet

def main():
    # Set up the DirectML device
    device = torch_directml.device()

    # Load the data
    print("Loading data...")
    df = pd.read_csv(
        'sentiment140.csv',
        encoding='latin-1',
        header=None,
        names=['Sentiment', 'ID', 'Date', 'Query', 'User', 'Tweet']
    )

    # Simplify the dataset
    print("Processing data...")
    df = df[['Sentiment', 'Tweet']]
    df['Sentiment'] = df['Sentiment'].apply(lambda x: 1 if x == 4 else 0)

    # Optional: Use a subset of data for faster processing
    df = df.sample(frac=0.2, random_state=42)  # Use 10% of the data
    df.reset_index(drop=True, inplace=True)  # Reset index after sampling

    print(f"Total number of samples: {len(df)}")

    # Determine the number of processes based on CPU cores
    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes for parallel data cleaning.")

    # Prepare arguments for multiprocessing
    args = list(df['Tweet'].items())

    # Apply the cleaning function in parallel with progress bar
    print("Cleaning tweets in parallel...")
    with mp.Pool(processes=num_processes) as pool:
        cleaned_tweets = []
        with tqdm(total=len(args), desc="Cleaning tweets") as pbar:
            for result in pool.imap_unordered(process_tweet, args):
                cleaned_tweets.append(result)
                pbar.update()

    # Sort results by index to maintain order
    cleaned_tweets.sort(key=lambda x: x[0])
    cleaned_texts = [tweet for index, tweet in cleaned_tweets]
    df['Tweet'] = cleaned_texts

    # Verify data processing
    print("First 5 cleaned tweets:")
    print(df['Tweet'].head())

    # Split the data
    print("Splitting data...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Tweet'], df['Sentiment'], test_size=0.2, random_state=42
    )

    # Initialize the tokenizer and model
    print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    model = AutoModelForSequenceClassification.from_pretrained(
        'huawei-noah/TinyBERT_General_4L_312D', num_labels=2
    )
    model.to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # Tokenize the data with progress bar
    print("Tokenizing training data...")
    train_encodings = tokenizer(
        list(train_texts), truncation=True, padding=True, max_length=128
    )

    print("Tokenizing validation data...")
    val_encodings = tokenizer(
        list(val_texts), truncation=True, padding=True, max_length=128
    )

    # Create datasets
    class SentimentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels.tolist()

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    print("Creating datasets...")
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 1  # Set to 1 for faster training

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        # Training loop with progress bar
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        total_eval_loss = 0
        total_eval_accuracy = 0
        # Evaluation loop with progress bar
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_eval_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total_eval_accuracy += correct

        avg_val_loss = total_eval_loss / len(val_loader)
        accuracy = total_eval_accuracy / len(val_dataset)
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")

    # Save the model
    print("Saving the model...")
    model.cpu()  # Move model to CPU before saving
    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_tokenizer')
    print("Model saved.")

if __name__ == '__main__':
    main()
