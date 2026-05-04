import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


vocab_size = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)   //IMDB movie reviews dataset(positive/negative)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

def build_model(model_type):
    model = Sequential()
    
    model.add(Embedding(vocab_size, 128, input_length=max_len))  //Embedding - Words ko numbers → vectors me convert karta hai
    
    if model_type == "RNN":
        model.add(SimpleRNN(32))
    elif model_type == "LSTM":
        model.add(LSTM(32))
    elif model_type == "GRU":
        model.add(GRU(32))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


models = ["RNN", "LSTM", "GRU"]
results = {}

for m in models:
    print(f"\nTraining {m} model...")
    
    model = build_model(m)
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    results[m] = {
        "accuracy": accuracy,
        "time": training_time,
        "history": history
    }


print("\n=== Model Comparison ===")
for m in results:
    print(f"{m} -> Accuracy: {results[m]['accuracy']:.4f}, Time: {results[m]['time']:.2f}s")

for m in results:
    plt.plot(results[m]['history'].history['val_accuracy'], label=m)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
