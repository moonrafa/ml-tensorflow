import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

IMG_INDEX = 1
plt.imshow(train_images[IMG_INDEX], cmap='gray')
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

