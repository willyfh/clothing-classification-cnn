"""
	Clothing classification using CNN
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

### Load Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class name
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#### preprocessed data

train_images = train_images / 255.0 # scale to range [0,1]
test_images = test_images / 255.0 # scale to range [0,1]

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

### build the model

# configure model
model = keras.Sequential([
	keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)),
	keras.layers.MaxPooling2D(pool_size=2),
	keras.layers.Dropout(0.3),
	
	keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
	keras.layers.MaxPooling2D(pool_size=2),
	keras.layers.Dropout(0.3),
	
	keras.layers.Flatten(),
	keras.layers.Dense(256, activation='relu'),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(10, activation='softmax')
])

# Take a look at the model summary
model.summary()

# compile the model
model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

### train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, verbose=2)

### evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('\nTest accuracy:', test_acc)

### make prediction
# the model predict the label for each image in the testing set.
predictions = model.predict(test_images) # use argmax to retrieve the label from each predicton of predictions array


# plot an image with its predicted and true label
def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	min_dims = img.squeeze();
	plt.imshow(min_dims, cmap=plt.cm.binary)
	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'
	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
								100*np.max(predictions_array),
								class_names[true_label]),
								color=color)
								
# plot the prediction probabilities for an image
def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)
	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

### show sample of the prediction result
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
	plt.subplot(num_rows, 2*num_cols, 2*i+1)
	plot_image(i, predictions, test_labels, test_images)
	plt.subplot(num_rows, 2*num_cols, 2*i+2)
	plot_value_array(i, predictions, test_labels)
plt.show()







  
  
