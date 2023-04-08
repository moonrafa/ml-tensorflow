from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow as tf
import tensorflow.python.feature_column as fc


# load data
df_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
df_eval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")  # testing data

y_train = df_train.pop('survived')
y_eval = df_eval.pop('survived')

print(df_train.head())  # five first rows

print(df_train.describe())  # statistics

print(df_train.shape)  # (627, 9)

df_train.age.hist(bins=20)

df_train.sex.value_counts().plot(kind='barh')

df_train['class'].value_counts().plot(kind='barh')

pd.concat([df_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = df_train[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float32))

print(feature_columns)
