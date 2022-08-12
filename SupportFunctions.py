import numpy as np
import os
import random
import csv

def BinConfigurations():
    # Defines the number and AUC of each bin array.
    # bin = [0]
    # bin = [-.431, .431]
    bin = [-.674, 0, .674]
    # bin = [-.842, -0.253, 0.253, .842]
    # bin = [-0.967, -0.431, 0, 0.431, 0.967]
    # bin = [-1.068, -0.566, -0.18, 0.18, 0.566, 1.068]
    # bin = [-1.15, -.674, -.319, 0, .319, .674, 1.15]

    return bin

def merge_Lists(ListA, ListB):
    ListA[:len(ListB)] = ListB
    return ListA

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=20):
  dataframe = dataframe.copy()
  # labels = dataframe.pop('target') becomes the two columns we don't want in training:
  labels = dataframe.pop('The_Category')
#   print(dict(dataframe))
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  # print(ds)
  return ds