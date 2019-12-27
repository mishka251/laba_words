import numpy as np
import pandas as pd
import datetime
# NLP
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Preprocessing
from sklearn import preprocessing
from sklearn.utils import class_weight as cw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

# Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Evaluation Metrics
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

# Deep Learing Preprocessing - Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

# Deep Learning Model - Keras
from keras.models import Model
from keras.models import Sequential

# Deep Learning Model - Keras - RNN
from keras.layers import Embedding, LSTM, Bidirectional

# Deep Learning Model - Keras - General
from keras.layers import Input, Add, concatenate, Dense, Activation, BatchNormalization, Dropout, Flatten
from keras.layers import LeakyReLU, PReLU, Lambda, Multiply

# Deep Learning Parameters - Keras
from keras.optimizers import RMSprop, Adam

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from keras import backend as K

from timeit import default_timer as timer

print('done')

films = pd.read_csv("movie-review/movie_review.csv")

X = films["text"]
X = X.apply(lambda x: BeautifulSoup(x, "lxml").get_text())
X = X.apply(lambda x: x.lower())
X = X.apply(lambda x: re.sub("[^a-zA-Z]", " ", x))
X = X.apply(lambda x: re.sub("\s+", " ", x))
stopwords = set(stopwords.words('english'))
X = X.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
Y = films["tag"]

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
Y = to_categorical(Y)
#
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

max_words = 10000
max_len = 150

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

dropouts = [0.35, 0.35, 0.45]
lstm_cnts = [64, 32, 32]

times_f = open("times.txt", "w")

for (lstm, dropout) in zip(lstm_cnts, dropouts):
    times_f.write(f"sigmoid dropout={dropout}, lstm={lstm}")
    start = timer()
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_seq = sequence.pad_sequences(X_train_seq, maxlen=max_len)

    model = Sequential()
    model.add(Embedding(max_words, 100, input_length=max_len))
    model.add(LSTM(lstm))  # количество нейронов
    model.add(Dropout(dropout))  # вероятность
    model.add(Dense(2, activation=K.sigmoid))
    # model.add(Activation('softmax'))

    # learning_rate = 0.001
    optimizer = Adam(0.001)
    # Компиляция данных
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    verbose = 1
    epochs = 10
    batch_size = 128
    validation_split = 0.2

    history = model.fit(
        X_train_seq,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split
    )

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(history_dict['accuracy']) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.xlabel('Epohs')
    plt.ylabel('Loss')
    plt.title(f"sigmoid dropout={dropout} lstm={lstm}")
    plt.legend()
    plt.show()

    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.xlabel('Epohs')
    plt.ylabel('Accuracy')
    plt.title(f"sigmoid dropout={dropout} lstm={lstm}")
    plt.legend()
    plt.show()

    test_X_seq = tokenizer.texts_to_sequences(X_test)
    test_X_seq = sequence.pad_sequences(test_X_seq, maxlen=max_len)
    results = model.evaluate(test_X_seq, Y_test)
    times_f.write(f'test loss, test acc:{results}\n')

    ypreds = model.predict_classes(test_X_seq, verbose=1)

    times_f.write(f"ypreds = {ypreds}\n")

    model.save("model" + str(datetime.datetime.now().microsecond) + ".h5")
    times_f.write(f"time = {timer() - start}\n")
    times_f.write("\n\n")

print("end")
