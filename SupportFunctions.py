import numpy as np
import os
import random
import csv

from Questionnaire import createQuestionnaire, load_Subject_Questions
from Image import Questionnaire_Image, load_images, LabelImage
from MixedData_Label_with_Image import createmodel, train
from DimensionalityBinning import DimensionalBinChoice


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

def Create_test_Data(model, Questions, Subjects, width, height):
    from numpy import asarray
    import pandas as pd
    from PIL import Image

    TestSubjectsAnswers = createQuestionnaire(Subjects, Questions)
    # Write Subjects Questionnaires to file

    TestQuestionnaireDF = pd.DataFrame.from_records(TestSubjectsAnswers)

    # Our first input consists of our batch size and the 2 dimensional images (3x3).
    input_images = np.ones((Subjects, width, height))
    Testgreyscale_images = np.ones((Subjects, width, height))
    # Our second input consists of our batch size and the 1 dimensional numerical data.
    input_number = np.zeros((Subjects, Questions))
    # The output consists of our batch size and 1 dimensional numerical data.
    Testoutput_array = np.zeros((Subjects, 1))

    # Sum the scores of each of the questionnaires in the dataframe and place them in the dataframe as well.
    Sum_of_Scores = TestQuestionnaireDF.iloc[:, :].sum(axis=1)
    TestQuestionnaireDF["Sum_of_Scores"] = Sum_of_Scores
    # TestQuestionnaireDF = Standardize(TestQuestionnaireDF)
    Binsize = BinConfigurations()
    BinnedScore = DimensionalBinChoice(Sum_of_Scores, Binsize)
    # Bin == Label!

    TestQuestionnaireDF["Label"] = LabelImage(BinnedScore)

    for subject in range(len(TestSubjectsAnswers)):
        # Need to bin the answers for each subject. Then use this bin to be that which is predicted.
        LabelValue = TestQuestionnaireDF.iloc[subject]["Label"]
        # LabelValue is varying as it is the binned value.
        Testoutput_array[subject] = LabelValue

        # A --------------------------------------------------
        # This is the main code to use for inputting multiple feature values (questions)
        # However, I can  use section B below instead to overwrite and add images of the same color to test predictions.
        image = Questionnaire_Image(width, height, Item_vals=TestSubjectsAnswers[subject])
        image_arr = asarray(image)
        image_arr = image_arr / 255

        new_width = new_height = 300
        # large_image = image.resize((new_width, new_height), resample=Image.Resampling.NEAREST)
        # write_greyscaleImage_toFile(subject, large_image, new_width, new_height, scorefile_path)

        # Here the image arrays are put into an array
        Testgreyscale_images[subject] = image_arr
        input_number[subject] = TestSubjectsAnswers[subject]
        # ---------------------------------------------------

    # input_number = input_number - 1
    Testoutput_array = Testoutput_array - 1
    input_images = Testgreyscale_images.reshape(Subjects, width, height, 1)

    # Predict on `X_unseen`
    y_pred_unseen = model.predict(input_images)
    print(y_pred_unseen)
