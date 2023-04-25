import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# we have 60000 images, with 28x28 pixels (784 each picture)
print(train_images.shape)

# one pixel can be from 0 to 255, 0 is black and 255 is white
print(train_images[0, 23, 23])

# range from 0 to 9
print(test_labels[:10])

# this array will indicate which label refers to which picture
class_names = ['T-shirt/Top', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocessing (good to make values smaller and easier to feed the model)
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential(
    [keras.layers.Flatten(input_shape=(28, 28)),  # input layer 784 neurons
     keras.layers.Dense(128, activation='relu'),  # hidden layer
     keras.layers.Dense(10, activation='softmax')  # output layer
     ]
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

print('The accuracy:', test_acc)

predictions = model.predict(test_images)

print(predictions[0], np.argmax(predictions[0]), class_names[np.argmax(predictions[0])])

plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# FUNCTIONS

COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR


def predict(mdl, img, correct_label, names):
    prediction = mdl.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(img, names[correct_label], predicted_class)


def show_image(img, lbl, guess):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title("Expected: " + lbl)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_number():
    while True:
        number = input("Choose a number: ")
        if number.isdigit():
            number = int(number)
            if 0 <= number <= 1000:
                return int(number)
            else:
                print("Try again")


num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label, class_names)
