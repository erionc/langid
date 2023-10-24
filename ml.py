
'''
Descript:	Script that uses machine learning models for identifying languages.
Author:		Erion Çano
Language: 	Python 3.11.6 
Reproduce:	Tested on Ubuntu 23.10 with Python 3.11.6
'''

import pandas as pd
import numpy as np
import os, sys, re, argparse
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# function for cleaning the dataset samples
def text_preprocess(text):
    # drop special symbols and numbers
    text = re.sub(r'[0-9!,"%@#$()^*?:;~`]', ' ', text)
    # fix double+ spaces
    text = re.sub(r'\s{2,}', " ", text)
    # lowercase everything
    text = text.lower()
    return text

# function for identifying the given text from the user
def identify(text):
    x = cv.transform([text]).toarray()
    pred = model.predict(x)
    lang = le.inverse_transform(pred)
    print(f"The given text is written in: {lang[0]}")

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classifier', choices=['lr', 'mnb', 'svm', 'dt', 'knn', 'stck', 'rf', 'ada', 'gb', 'xgb'], help='Classification Model', required=True)
args = parser.parse_args()

# Loading the dataset
data = pd.read_csv("./data/languages.csv")

# separating the independent and dependant features
X = data["Text"]
y = data["Language"]

# converting categorical variables to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# list with cleaned text samples
sample_lst = []

# iterating through all samples
for s in X:
    clean_text = text_preprocess(s)
    sample_lst.append(clean_text)

# creating bag of words using countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(sample_lst).toarray()

#train test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# basic classification models
lr_model = LogisticRegression()
svm_model = SVC()
mnb_model = MultinomialNB()
knn_model = KNeighborsClassifier()
dt_model = DecisionTreeClassifier()

# preparing the stack model
level0 = list() ; level0.append(('lr', lr_model)) ; level0.append(('svc', svm_model))
level0.append(('nb', mnb_model)) ; level0.append(('knn', knn_model)) ; level0.append(('dt', dt_model))
level1 = lr_model
stck_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# bagging ensemble model 
rf_model = RandomForestClassifier(random_state=7, n_jobs=-1)

# boosting ensemble models 
ada_model = AdaBoostClassifier(random_state=7)
gb_model = GradientBoostingClassifier(random_state=7)
xgb_model = xgb.XGBClassifier(random_state=7)

# selecting the classifier
if args.classifier.lower() == "lr":
    model = lr_model
elif args.classifier.lower() == "mnb":
    model = mnb_model
elif args.classifier.lower() == "svm":
    model = svm_model
elif args.classifier.lower() == "knn":
    model = knn_model
elif args.classifier.lower() == "dt":
    model = dt_model
elif args.classifier.lower() == "stck":
    model = stck_model
elif args.classifier.lower() == "rf":
    model = rf_model
elif args.classifier.lower() == "ada":
    model = ada_model
elif args.classifier.lower() == "gb":
    model = gb_model
elif args.classifier.lower() == "xgb":
    model = xgb_model
else:
    print("Wrong Classifier...")
    sys.exit()

# fit the model
model.fit(x_train, y_train)

# prediction 
y_pred = model.predict(x_test)

# model evaluation
ac = accuracy_score(y_test, y_pred)
print(f"\nThe accuracy of the selected model is {ac:.4f}\n")
target_text = input("Type or paste your text below:\n")

identify(target_text)
