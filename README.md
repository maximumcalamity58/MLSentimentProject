# Sentiment Analysis Web App with BERT

This repository contains a web application built using Flask and a fine-tuned BERT model for sentiment analysis. The app allows users to input a sentence or tweet and predicts whether the sentiment is positive or negative. It also includes an option to generate a random tweet for analysis.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Train the Model](#train-the-model)
4. [Run the Web Application](#run-the-web-application)
5. [How it Works](#how-it-works)
6. [Requirements](#requirements)

---

## Installation

Before using the application, you'll need to install the necessary dependencies and set up the environment. Follow these steps to get everything up and running.

### 1. Clone the Repository

```bash
git clone https://github.com/maximumcalamity58/MLSentimentProject.git
cd sentiment-analysis-app
```

### 2. Install Dependencies

There are two ways to install dependencies, depending on how you plan to use the project.

#### Install from `requirements.txt`

If you just want to run the app or train the model, install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

#### Alternative Installation (Automatic Dependency Handling)

If you plan to distribute the app and want the dependencies to auto-install when running the script, the dependencies will be installed the first time `app.py` or `train_model.py` runs. You can add automatic package installation code to `app.py` and `train_model.py`.

---

## Usage

There are two ways to use this project:

1. **Train the sentiment model from scratch** using `train_model.py`.
2. **Run the web application** using `app.py`.

---

## Train the Model

If you want to train your own BERT-based model for sentiment analysis on the Sentiment140 dataset, follow these steps:

### Step 1: Download the Sentiment140 Dataset

You can download the Sentiment140 dataset [here](https://www.kaggle.com/datasets/kazanova/sentiment140). You must extract it and insert the .csv file into the main directory.

### Step 2: Train the Model

Run the `train_model.py` file to start training. This script tokenizes the data, trains a BERT-based model on the dataset, and saves the final model.

To train the model, use:

```bash
python train_model.py
```

This will load the Sentiment140 dataset, tokenize the tweets, and fine-tune a BERT model. The model and tokenizer will be saved in the `./final_model` and `./final_tokenizer` directories.

---

## Run the Web Application

To run the web application, simply execute `app.py` and open a web browser to `http://127.0.0.1:5000`.

```bash
python app.py
```

Once the app is running, you can input a sentence or tweet into the text box, and the app will predict the sentiment (positive or negative) and display the confidence level. You can also generate a random tweet from the dataset for sentiment analysis.

---

## How it Works

- **Model**: The model is a fine-tuned BERT-based model (TinyBERT) for sequence classification, specifically trained to predict sentiment (positive or negative).
- **Web Application**: Flask is used to serve the web application, which communicates with the model to perform inference on user inputs.
- **Data**: The Sentiment140 dataset is used to train the model, which consists of tweets labeled as either positive or negative.

### Key Functions
- **Sentiment Prediction**: User input is tokenized using the BERT tokenizer and passed through the model to predict sentiment.
- **Random Tweet Generation**: A random tweet from the Sentiment140 dataset is displayed for sentiment analysis.

---

## Requirements

The following Python libraries are required:

- `torch`
- `torch_directml` (for AMD GPU support)
- `transformers`
- `pandas`
- `flask`
- `nltk`
- `tqdm`
- `emoji`
- `contractions`

To install all dependencies, you can run:

```bash
pip install -r requirements.txt
```

If you're using an AMD GPU, ensure that `torch_directml` is installed.

