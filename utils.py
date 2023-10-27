
import pandas as pd
import numpy as np
import os, sys, re, argparse, random, time, gc, datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix


'''
Function for cleaning the multilingual corpus texts. Removes numbers and most of the special symbols specials. Casing is preserved, since it represents a valuable feature on some of the languages. A more carefull text cleaning procedure could retain more valuable features for solving the task.
'''
def text_preprocess(text):
    # remove special symbols and numbers
    text = re.sub(r'[0-9!,"%@#$()^*?:;~`]', ' ', text)
    # fix consequetive 2+ spaces
    text = re.sub(r'\s{2,}', " ", text)
    return text

# Function for computing the accuracy based on predictions and labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function for representing elapsed time in nice hh:mm:ss format
def format_time(elapsed):
    rounded = int(round((elapsed))) # round in seconds
    # return as string in hh:mm:ss format
    return str(datetime.timedelta(seconds=rounded))



