import pandas as pd

def createQuestionnaire(Subjects, NumberofQuestions):
    AllAnswers = []
    for x in range(Subjects):
        a, b = findweights()
        # Change a and b multipliers to change graph thickness
        [DirichletProbabilities] = np.random.dirichlet((a, 2 * a, 2 * (a ** 2 + b ** 2), 2 * b, b), size=1).round(10)
        AnsweredQuestion = AnswerQuestions(NumberofQuestions, DirichletProbabilities)
        AllAnswers.append(AnsweredQuestion)
        # ListofVals.append(DirichletProbabilities.tolist())
        # QuestionsandProbabilities.append([AnsweredQuestion, DirichletProbabilities])

    return  AllAnswers
import numpy as np
import os
import random
import csv

def findweights():
    binarychoice = [0, 1]
    choice = random.choice(binarychoice)
    if choice == 0:
        tuple = (1,2)
    elif choice == 1:
        tuple = (2,1)
    else:
        print("Issues in findweights()")
    return tuple

    return Categories

def AnswerQuestions(numberofQuestions,DirichletProbs):
    test = np.random.choice([1, 2, 3, 4, 5], numberofQuestions, p=DirichletProbs)
    return test

def load_Subject_Questions(inputPath):
    SubjectQuestionnaires_df = pd.read_csv(inputPath, sep=",", header=None)
    return SubjectQuestionnaires_df

def write_questionnaires_to_file(inputPath,QuestionnaireArray):
    # basePath = os.path.sep.join([inputPath, "{0}".format('*')])
    pathname = os.path.dirname(inputPath)
    if not os.path.isdir(pathname):
        print("[INFO] 'creating {}' directory".format(pathname))
        os.makedirs(pathname)
    with open(inputPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(QuestionnaireArray)