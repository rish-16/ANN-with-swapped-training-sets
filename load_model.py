import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(10000, 784)

X_test = X_test.astype('float32')
X_test /= 255
y_test = np_utils.to_categorical(y_test, 10)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


idx = 800

truth = X_test[idx]
truth = truth.reshape([28,28])

pred = loaded_model.predict(y_test, batch_size=128)
example = pred[idx].reshape([28,28])

plt.figure(1)
plt.subplot(1, 2, 1)
plt.title("Ground Truth")
plt.imshow(truth, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Fake")
plt.imshow(example, cmap='gray')
plt.show()
