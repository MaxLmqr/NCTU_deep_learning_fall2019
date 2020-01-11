# -*- coding: utf-8 -*-
"""0845034.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DhOfekcKbTD4kK--G3uhJ8Cm9IaCaNyn

Importation des bibliothèques utiles au fonctionnement du programme
"""

import numpy as np
import io
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.preprocessing.sequence import pad_sequences
import keras
from keras import regularizers

"""Snippet pour importer les fichiers textes qui serviront au training de notre modèle. (nécessaire car je travaille sur google colab)"""

valid_data_URL = 'shakespeare_valid.txt'
data_URL = 'shakespeare_train.txt'
with io.open(data_URL, 'r', encoding = 'utf8') as f:
    text = f.read()

with io.open(valid_data_URL,'r',encoding = 'utf8') as f:
  valid_text = f.read()

vocab = set ( text )
vocab_to_int = {c : i for i, c in enumerate ( vocab )}
int_to_vocab = dict ( enumerate ( vocab ) )
train_data = np.array( [ vocab_to_int[c] for c in text ], dtype = np.int32 )
valid_data = np.array( [vocab_to_int[c] for c in valid_text], dtype=np.int32)

vocab_size  = len(vocab)
length = 10
sequences = list()
val_sequences = list()
for i in range(length, len(train_data),2):
	seq = train_data[i-length:i+1]
	sequences.append(seq)
for i in range(length, len(valid_data),2):
	valseq = valid_data[i-length:i+1]
	val_sequences.append(valseq)
print('Total Sequences: %d' % len(sequences))

sequences = np.array(sequences)
val_sequences = np.array(val_sequences)

X,y = sequences[:,:-1],sequences[:,-1]
X_val, y_val = val_sequences[:,:-1],val_sequences[:,-1]

sequences = np.empty((X.shape[0],X.shape[1],vocab_size))
val_sequences = np.empty((X_val.shape[0],X_val.shape[1],vocab_size))
for i in range(length):
  sequences[:,i,:] = to_categorical(X[:,i], num_classes=vocab_size)
  val_sequences[:,i,:] = to_categorical(X_val[:,i], num_classes=vocab_size)
X = sequences
X_val = val_sequences
y = to_categorical(y, num_classes=vocab_size)
y_val  = to_categorical(y_val, num_classes=vocab_size)

model = Sequential()
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2]),recurrent_initializer='glorot_uniform', kernel_regularizer=regularizers.L1L2(0.1,0.1)))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint('model{epoch:08d}.h5', period=4) 
results = model.fit(X, y,validation_data = (X_val,y_val),callbacks=[checkpoint], epochs=20, batch_size=1024)

model.save("model.h5")
from pickle import dump
dump(vocab_to_int, open('vocab_to_int.pkl', 'wb'))
dump(int_to_vocab, open('int_to_vocab.pkl','wb'))

import matplotlib.pyplot as plt
plt.plot(1-np.array(results.history['acc']))
plt.plot(1-np.array(results.history['val_acc']))
plt.title('model error rate')
plt.ylabel('error rate')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

import matplotlib.pyplot as plt
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Learning curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

"""## **Generate text**"""

def generate_seq(model, vocab_to_int,int_to_vocab, seq_length, seed_text, n_chars):
  in_text = seed_text
  model.reset_states()
  for i in range(n_chars):
    encoded = [vocab_to_int[char] for char in in_text]
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    encoded = to_categorical(encoded, num_classes=len(vocab_to_int))
    yhat = model.predict(encoded, verbose=0)
    yhat = np.random.choice(len(yhat[0]), p=yhat[0])
    out_char = int_to_vocab[yhat]
    in_text += out_char
  return in_text

text = generate_seq(model,vocab_to_int,int_to_vocab,10,'No more talking',200)
print(text)