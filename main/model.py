import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

#Imports
import nltk
import json
import random
import pickle as p
import numpy as np

from nltk.stem import WordNetLemmatizer
from keras.src.models import Sequential
from keras.src.layers.core.dense import Dense
from keras.src.layers.regularization.dropout import Dropout
from keras.src.optimizers.sgd import SGD

#Initialization
lemmatizer = WordNetLemmatizer()

words = []
tags = []
documents = []
ignore_words = ["?", "!"]

intents = json.loads(open("intents.json").read())

#Tokenization data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent["tag"]))
        if intent["tag"] not in tags:
            tags.append(intent["tag"])

#Lemmatize
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))
tags = sorted(list(set(tags)))

print(f"{len(documents)} documents")
print(f"{len(tags)} tags, {tags}")
print(f"{len(words)} unique words, {words}")

p.dump(words, open("main\\words.pkl", "wb"))
p.dump(tags, open("main\\tags.pkl", "wb"))

#Training and Bag_Of_Words
training = []
output_empty = [0]*len(tags)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    bag.extend([1 if word in pattern_words else 0 for word in words])

    output_row = list(output_empty)
    output_row[tags.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype = object)

#input and output training lists
train_x = np.array(list(training[:, 0]), dtype = np.float32)
train_y = np.array(list(training[:, 1]), dtype = np.float32)
print("training data created")


#Creating model
model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = "softmax"))

#Compile model
sgd = SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])
hist = model.fit(train_x, train_y, epochs = 200, batch_size = 5, verbose = 1)
model.save("main\\chatbot_model.h5", hist)

print("Model created")
