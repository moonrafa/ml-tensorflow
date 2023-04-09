from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
from input_function import input_function_classification

CSV_COLUMNS_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv",
                                     "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv",
                                    "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMNS_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMNS_NAMES, header=0)

print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')

fc = []
for key in train.keys():
    fc.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(feature_columns=fc, hidden_units=[30, 10], n_classes=3)

classifier.train(input_fn=lambda: input_function_classification(train, train_y, training=True), steps=5000)

eval_result = classifier.evaluate(input_fn=lambda: input_function_classification(test, test_y, training=False))

print(eval_result)


def convert_input_to_ds_wo_labels(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


predict = {}

FEATURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

print('input type numeric values\n')

for feature in FEATURES:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False
    predict[feature] = [float(val)]
predictions = classifier.predict(input_fn=lambda: convert_input_to_ds_wo_labels(predict))

for pre_dict in predictions:
    class_id = pre_dict['class_ids'][0]
    probability = pre_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))

