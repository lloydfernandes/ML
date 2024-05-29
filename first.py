import pathlib
import random
import shutil
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from gensim.utils import simple_preprocess
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import os

# Download the NLTK stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Step 1: Fetch dataset
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
!rm -r aclImdb/train/unsup

# Prepare directories
base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"
for category in ("neg", "pos"):
    os.makedirs(val_dir / category, exist_ok=True)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname, val_dir / category / fname)

# Load data using text_dataset_from_directory (test, train, and validation)
batch_size = 32
train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

# Step 2: Pre-processing
def preprocess_text(text):
    return ' '.join([word for word in simple_preprocess(text) if word not in stop_words])

def dataset_to_numpy(dataset):
    texts, labels = [], []
    for text_batch, label_batch in dataset:
        for text, label in zip(text_batch.numpy(), label_batch.numpy()):
            texts.append(text.decode('utf-8'))
            labels.append(label)
    return texts, labels

train_texts, train_labels = dataset_to_numpy(train_ds)
val_texts, val_labels = dataset_to_numpy(val_ds)
test_texts, test_labels = dataset_to_numpy(test_ds)

train_texts = [preprocess_text(text) for text in train_texts]
val_texts = [preprocess_text(text) for text in val_texts]
test_texts = [preprocess_text(text) for text in test_texts]

# Step 3: Feature Extraction (using LDA)
vectorizer = Tokenizer()
vectorizer.fit_on_texts(train_texts)
train_sequences = vectorizer.texts_to_sequences(train_texts)
val_sequences = vectorizer.texts_to_sequences(val_texts)
test_sequences = vectorizer.texts_to_sequences(test_texts)

# Padding sequences
maxlen = 200
X_train = pad_sequences(train_sequences, maxlen=maxlen)
X_val = pad_sequences(val_sequences, maxlen=maxlen)
X_test = pad_sequences(test_sequences, maxlen=maxlen)

# Step 4: Split into Aspect Base
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_features = lda.fit_transform(X_train)

# Step 5: Vectorization
vocab_size = len(vectorizer.word_index) + 1
embedding_dim = 50

# Step 6: Use RNN-LSTM model
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(X_train, np.array(train_labels), epochs=5, batch_size=batch_size, validation_data=(X_val, np.array(val_labels)))

# Step 7: Aspect driving factor
aspects = lda.components_
top_aspects = np.argsort(aspects, axis=1)[:, -1:-11:-1]

# Print top words for each aspect
terms = vectorizer.word_index
terms_inv = {index: word for word, index in terms.items()}
for idx, topic in enumerate(top_aspects):
    print(f"Aspect {idx + 1}: ", [terms_inv[i] for i in topic])

# Step 8: Sentiment Prediction
test_loss, test_acc = model.evaluate(X_test, np.array(test_labels))
print(f"Test Accuracy: {test_acc}")

# Prediction example
def predict_sentiment(text):
    text = preprocess_text(text)
    sequence = vectorizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded_sequence)
    return "Positive" if prediction >= 0.5 else "Negative"

# Example prediction
example_text = "The movie was fantastic and thrilling."
print(predict_sentiment(example_text))
