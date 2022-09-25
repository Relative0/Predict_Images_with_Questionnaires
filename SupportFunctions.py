import numpy as np
import os
import random
import csv
import shap

from Questionnaire import createQuestionnaire, load_Subject_Questions
from Image import Questionnaire_Image, load_images, LabelImage
from MixedData_Label_with_Image import createmodel, train
from DimensionalityBinning import DimensionalBinChoice


def BinConfigurations():
    # Defines the number and AUC of each bin array.
    # bin = [0]
    # bin = [-.431, .431]
    # bin = [-.674, 0, .674]
    # bin = [-.842, -0.253, 0.253, .842]
    # bin = [-0.967, -0.431, 0, 0.431, 0.967]
    # bin = [-1.068, -0.566, -0.18, 0.18, 0.566, 1.068]
    bin = [-1.15, -.674, -.319, 0, .319, .674, 1.15]

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

def Create_test_Data(network_output, Questions, Subjects, width, height):
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
    test_number = np.zeros((Subjects, Questions))
    # The output consists of our batch size and 1 dimensional numerical data.
    Testoutput_array = np.zeros((Subjects, 1))

    # Sum the scores of each of the questionnaires in the dataframe and place them in the dataframe as well.
    Sum_of_Scores = TestQuestionnaireDF.iloc[:, :].sum(axis=1)
    TestQuestionnaireDF["Sum_of_Scores"] = Sum_of_Scores
    # TestQuestionnaireDF = Standardize(TestQuestionnaireDF)
    Binsize = BinConfigurations()
    BinnedScore = DimensionalBinChoice(Sum_of_Scores, Binsize)

    output_array = np.array(LabelImage(BinnedScore)) - 1
    TestQuestionnaireDF["Label"] = output_array

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
        test_number[subject] = TestSubjectsAnswers[subject]
        # ---------------------------------------------------

    # input_number = input_number - 1
    test_images = Testgreyscale_images.reshape(Subjects, width, height, 1)

    # predictions = network_output.predict(test_images)

    output = network_output(test_images, test_number)
    preds = np.argmax(output, axis=1)
    print(preds)

    # test_network_output = network_output(test_images, test_number)
    # predicted_number = np.predict(test_network_output)


    acc = 0

    for i in range(len(test_images)):
        if (preds[i] == Testoutput_array[i]):
            acc += 1

    print("Accuracy: ", acc / len(input_images) * 100, "%")


    # SHAP testing - DOESN'T work - but I think it possible that the reason it doesn't as, while the example here:
    # https://coderzcolumn.com/tutorials/artificial-intelligence/shap-values-for-image-classification-tasks-keras
    # works my model is a combined model: x = tf.concat([N, I], 1)  # Concatenate through axis #1 - whereas their
    # model is just of a flattened image. Maybe if I can somehow flatten my concatenation?

    # masker = shap.maskers.Image("inpaint_telea", test_images[0].shape)
    #
    # classes =  np.unique(output_array)
    #
    # class_labels = []
    #
    # for i in range(len(Binsize)+1):
    #         class_labels.append("ConfigurationLevel_" + str(i + 1))
    # mapping = dict(zip(classes, class_labels))
    #
    # explainer = shap.Explainer(network_output, masker, output_names=class_labels)
    #
    # # X_train, X_test = input_images.reshape(-1,28,28,1), input_images.reshape(-1,28,28,1)
    # # inputval = input_images[:4]
    # flipped_value = shap.Explanation.argsort.flip[:5]
    # shap_values = explainer([test_number[:4], test_images[:4]], outputs=flipped_value)
    #
    # print(shap_values.shape)

    # print("Actual Labels    : {}".format([mapping[i] for i in Y_test[:4]]))
    # probs = trained_network.predict(X_test[:4])
    # print("Predicted Labels : {}".format([mapping[i] for i in np.argmax(probs, axis=1)]))
    # print("Probabilities : {}".format(np.max(probs, axis=1)))
