import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_train = X_train.astype('float32')
X_train /= 255

y_train = np_utils.to_categorical(y_train, 10)

model = Sequential()
model.add(Dense(100, input_shape=[10,], activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(784, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(y_train, X_train, epochs=10, batch_size=128)

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
