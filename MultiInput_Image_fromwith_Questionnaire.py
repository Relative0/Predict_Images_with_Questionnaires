# Use locations of item along with their values to determine colors. I could then put them as unique colors at random locations.

# Maybe, for each subjects image each pixel location corresponds to a particular question and each color for each pixel corresponds
# to the answer.

# Or perhaps make some of the encoding/decoding outside the image e.g. each pixel location correpsonds to a prime number (so each location
# corresponds to a question) then each pixel color value corresponds to that prime number to the power of the answer. This will only
# work for images containing the number of questions less than the fifth root of 256^3 (for 1-5 answers). This will give a unique Godel number. Looks like
# I can get 27.85, so 27 questions out of this. Then making it a square image, I can do a 5x5 = 25 question image.
# - Seems that there ar problems with this, first as the 27th prime number is 103 and 103^5 is greater than 256^3, I had forgotten that
# there are gaps between prime numbers.
# I could instead just make the pixel locations the questions and make their colors not unique but all following a pattern of being
# one of 5 colors (one color for each of the answers). Or maybe make more colors by multiplying each of the 5 potential color values
# by pixel locations.
# We could also build a unique number by stacking each of the answers. So for example question 1,2,and 3 have a value of 3,5,4 respectively
# which would give the number 354 and as long as the questions only have values 0-9 we have a unique number. Now I could just give a color
# coding simply based on the answer or I could multiply the answer by the location and convert that in to RGB values. Perhaps I could
# also factor in the sum of scores on the set of questions as a sort of hash check for each image. That is, each of the images has a color
# coding decided in part by the overall sum of scores.

# Now to create a basic image from a column of questions with just 5 potential colors.

# for each column, need to translate that to a pixel value

import random
import numpy as np
from numpy import asarray
import pandas as pd
import shap
from pandas import DataFrame
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns
import os
import random
import uuid
import math
import csv
import sys
import cv2
import glob
from tensorflow import keras
from IPython.display import clear_output
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from Questionnaire import createQuestionnaire, load_Subject_Questions, write_questionnaires_to_file
from Image import Questionnaire_Image, load_images, LabelImage, write_greyscaleImage_toFile
from MixedData_Label_with_Image import createmodel, train
from DimensionalityBinning import DimensionalBinChoice
from SupportFunctions import BinConfigurations, Create_test_Data


# width = height = 3;  imageSize = (width, height)
ListofVals= []
Questions = 20; Subjects = 500
QuestionsandProbabilities = []
path = "./ImageWithQuestion"; subjectScores = "Synthetic_Questionnaire_Answers.txt"
scorefile_path = path + "/" + subjectScores

width = height = int(math.ceil(math.sqrt(Questions)))
AllSubjectsAnswers = createQuestionnaire(Subjects, Questions)
write_questionnaires_to_file(scorefile_path, AllSubjectsAnswers)
# Write Subjects Questionnaires to file

QuestionnaireDF = pd.DataFrame.from_records(AllSubjectsAnswers)

# Our first input consists of our batch size and the 2 dimensional images (3x3).
input_images = np.ones((Subjects, width, height))
greyscale_images = np.ones((Subjects, width, height))
# Our second input consists of our batch size and the 1 dimensional numerical data.
input_number = np.zeros((Subjects, Questions))
# The output consists of our batch size and 1 dimensional numerical data.
output_array = np.zeros((Subjects, 1))

# Sum the scores of each of the questionnaires in the dataframe and place them in the dataframe as well.
Sum_of_Scores = QuestionnaireDF.iloc[:, :].sum(axis=1)
QuestionnaireDF["Sum_of_Scores"] = Sum_of_Scores
# QuestionnaireDF = Standardize(QuestionnaireDF)
Binsize = BinConfigurations()
BinnedScore = DimensionalBinChoice(Sum_of_Scores,Binsize)
# Bin == Label!

QuestionnaireDF["Label"] = LabelImage(BinnedScore)

Images = []
Labels = []
for subject in range(len(AllSubjectsAnswers)):
    # Need to bin the answers for each subject. Then use this bin to be that which is predicted.
    LabelValue = QuestionnaireDF.iloc[subject]["Label"]
    # LabelValue is varying as it is the binned value.
    output_array[subject] = LabelValue

    # A --------------------------------------------------
    # This is the main code to use for inputting multiple feature values (questions)
    # However, I can  use section B below instead to overwrite and add images of the same color to test predictions.
    image = Questionnaire_Image(width, height, Item_vals=AllSubjectsAnswers[subject])
    image_arr = asarray(image)
    image_arr = image_arr / 255


    new_width = new_height = 300
    large_image = image.resize((new_width, new_height), resample=Image.Resampling.NEAREST)
    write_greyscaleImage_toFile(subject, large_image, new_width, new_height, path)

    # Here the image arrays are put into an array
    greyscale_images[subject] = image_arr
    input_number[subject] = AllSubjectsAnswers[subject]
    # ---------------------------------------------------

# input_number = input_number - 1
output_array = output_array - 1
input_images = greyscale_images.reshape(Subjects, width, height,1)

SubjectsQuestions_df = load_Subject_Questions(scorefile_path)
# Sum_of_Scores = SubjectsQuestions_df.iloc[:, :].sum(axis=1)
# SubjectsQuestions_df["Sum_of_Scores"] = Sum_of_Scores
SubjectsQuestions_df["Bin"] = output_array
# load_images(SubjectsQuestions_df,path)

network, optimizer, loss_function = createmodel()

trained_network = train(network,optimizer,loss_function,300,input_images,input_number,output_array)
print(trained_network.summary())
# network_output = trained_network(input_images, input_number)
# preds = np.argmax(network_output, axis=1)

# network_output = trained_network(input_images,input_number)


Create_test_Data(trained_network, Questions, Subjects, width, height)
#
# preds = np.argmax(network_output, axis=1)
# acc = 0
# for i in range(len(input_images)):
#     if (preds[i] == output_array[i]):
#         acc += 1
#
# print("Accuracy: ", acc / len(input_images) * 100, "%")



# preds = np.argmax(network_output, axis=1)
#
# predIdxs = model.predict(input_number)
#
# predIdxs = np.argmax(predIdxs, axis=1)
#
# test_labels = np.hstack(test_labels)
# Accuracy = accuracy_score(test_labels,predIdxs )
# Precision = precision_score(test_labels,predIdxs , average="macro")
# Recall = recall_score(test_labels,predIdxs , average="macro")
# Fscore = f1_score(test_labels,predIdxs , average="macro")
#
# print(Accuracy)
# print(Precision)
# print(Recall)
# print(Fscore)
#
#
# print()