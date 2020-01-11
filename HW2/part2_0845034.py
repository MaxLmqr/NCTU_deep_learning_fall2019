from tensorflow import keras
import tensorflow as tf
import numpy as np
import pickle
import sys
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization,Conv2D, Flatten, MaxPool2D, Dropout, Activation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import regularizers


"""
        ['airplane', 
        'automobile', 
        'bird', 
        'cat', 
        'deer', 
        'dog', 
        'frog', 
        'horse', 
        'ship', 
        'truck']
"""

with open('preprocess.p','rb') as fo:
    x_train,y_train = pickle.load(fo)

with open('preprocess_validation.p','rb') as fo:
    x_val,y_val = pickle.load(fo)

random_seed = 3

# Initialize model
model = Sequential()


# Add layers
model.add(Conv2D(32,kernel_size=3,strides=(2,2),activation='relu',input_shape=(32,32,3),kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('tanh'))
model.add(Conv2D(32, kernel_size=3, strides=(2,2), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('tanh'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(16, kernel_size=3, strides=(2,2), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Flatten())
model.add(Dense(128, activation = "relu",activity_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(256, activation = "relu",activity_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(512, activation = "relu",activity_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(10,activation='softmax',activity_regularizer=regularizers.l2(0.01)))

# Compile and fit model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
results = model.fit(x_train,y_train,validation_data=(x_val,y_val), epochs=50)
"""
first_layer_weights = model.layers[0].get_weights()[0]
first_layer_biases  = model.layers[0].get_weights()[1]
second_layer_weights = model.layers[1].get_weights()[0]
second_layer_biases  = model.layers[1].get_weights()[1]
third_layer_weights = model.layers[5].get_weights()[0]
third_layer_biases  = model.layers[5].get_weights()[1]
fourth_layer_weights = model.layers[7].get_weights()[0]
fourth_layer_biases = model.layers[7].get_weights()[1]

param1 = list(first_layer_weights.reshape(-1))+list(first_layer_biases)
param2 = list(second_layer_weights.reshape(-1))+list(second_layer_biases)
param3 = list(third_layer_weights.reshape(-1))+list(third_layer_biases)
param4 = list(fourth_layer_weights.reshape(-1))+list(fourth_layer_biases)


fig, ax = plt.subplots(1,2)
plt.tight_layout()

ax[0].plot(results.history['accuracy'])
ax[0].plot(results.history['val_accuracy'])
ax[0].set_title('model accuracy')
ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'val'], loc='upper left')

ax[1].plot(results.history['loss'])
ax[1].plot(results.history['val_loss'])
ax[1].set_title('model loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'val'], loc='upper left')

ax[1][0].hist(param1,70)
ax[1][0].set_title('Conv2D')
ax[1][0].set_ylabel('Number')
ax[1][0].set_xlabel('Value')

ax[1][1].hist(param2,80)
ax[1][1].set_title('Conv2D')
ax[1][1].set_ylabel('Number')
ax[1][1].set_xlabel('Value')

ax[2][0].hist(param3,100)
ax[2][0].set_title('Dense')
ax[2][0].set_ylabel('Number')
ax[2][0].set_xlabel('Value')

ax[2][1].hist(param4,100)
ax[2][1].set_title('Dense')
ax[2][1].set_ylabel('Number')
ax[2][1].set_xlabel('Value')

plt.show()
"""



###### FIND MISTAKES
# predicted = model.predict(x_train)
# result = np.absolute(y_train-predicted)
# for i in range(54000):
#     for j in range(10):
#         if result[i][j]>0.5:
#             print(i,j)

###### See the intermediate activations
# Fetch parameters to be able to plot the intermediate activation
# outputs = [layer.output for layer in model.layers]
# activation = Model(inputs=model.input, outputs=outputs)

# activations = activation.predict(x_train[7].reshape((1,32,32,3)))
# first_layer_activation = activations[0]
# second_layer_activation = activations[2]
# plt.imshow(first_layer_activation[0,:,:,30])
# plt.imshow(second_layer_activation[0,:,:,7])
