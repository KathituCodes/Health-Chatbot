# 🩺 Health Assistant Chatbot

## Overview

This project is a **Health Assistant Chatbot** built using **Natural Language Processing (NLP)** and machine learning techniques. It is designed to answer basic health-related queries, simulating the behavior of a virtual assistant that helps users navigate general health information in a conversational format.

## Technologies Used

* Python 🐍
* Natural Language Toolkit (**NLTK**)
* Scikit-learn (**sklearn**)
* TensorFlow & Keras
* NumPy & Pandas
* Tkinter (GUI)
* JSON for intent management

## Features

* Chatbot can interpret and respond to common health-related queries.
* Trained on a predefined set of intents and responses stored in a JSON file.
* NLP preprocessing pipeline (tokenization, lemmatization, bag-of-words).
* Feedforward Neural Network for intent classification.
* GUI frontend using Tkinter for user interaction.

## Model Training

The chatbot uses a **Multi-Layer Perceptron (MLP)** model with the following architecture:

* **Input Layer:** Bag-of-words representation of user input
* **Hidden Layers:** Two dense layers with ReLU activation
* **Output Layer:** Softmax activation for intent prediction

The model was trained using categorical cross-entropy loss and Adam optimizer.

## 🧰 How It Works

1. **Data Preparation:**
   Intents (patterns and responses) are stored in a JSON file and used to train the model.

2. **Preprocessing:**

   * Text is tokenized and lemmatized.
   * Bag-of-words vectorization is applied to convert text into numerical input.

3. **Training the Model:**

   * Inputs: Processed user phrases
   * Outputs: One-hot encoded intents
   * The model is trained for multiple epochs to achieve high accuracy.

4. **Chat Flow:**

   * User enters a query in the GUI.
   * The model predicts the intent.
   * The chatbot responds with a relevant answer from the dataset.

## GUI Preview

The chatbot comes with a simple GUI interface built with Tkinter. It provides:

* A clean chat window
* Input box for user messages
* Scrollable chat history

## Getting Started

### 🛠 Requirements

Install dependencies:

```bash
pip install nltk numpy tensorflow sklearn
```

Download NLTK resources (only once):

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### ▶️ Run the App

```bash
python Health_Assistant_Chatbot.ipynb
```

Or open the notebook in Jupyter and run all cells.

## 📁 File Structure

```
├── intents.json              # Dataset of patterns and responses
├── chatbot_model.h5          # Trained model (after training)
├── words.pkl / labels.pkl    # Pickled data for inference
├── Health_Assistant_Chatbot.ipynb  # Jupyter notebook
```

## Future Improvements

* Add more intents for broader coverage
* Connect to real-time health APIs
* Deploy the chatbot as a web app or Telegram bot
* Add speech-to-text input and voice responses

## Disclaimer

Disclaimer - This chatbot is for **educational and informational purposes only**. It is **not** a substitute for professional medical advice or treatment.


