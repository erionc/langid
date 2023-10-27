
'''
Description:    Script that uses machine learning models for identifying languages.

Author:         Erion Çano

Reproduce:	    Tested on Ubuntu 23.10 with python=3.11.6

Run:            python ml.py -c <classifier>
'''

from utils import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings ; warnings.simplefilter("ignore")

# function for identifying the given text from the user
def identify(text):
    x = cv.transform([text]).toarray()
    pred = model.predict(x)
    lang = le.inverse_transform(pred)
    print(f"\nThe given text is written in: {lang[0]}\n")

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classifier', choices=['lr', 'mnb', 'svm', 'dt', 'knn', 'stck', 'rf', 'ada', 'gb', 'xgb'], help='Classification Model', required=True)
args = parser.parse_args()

# Loading the dataset in a dataframe
data = pd.read_csv("./data/languages.csv")

# separating the texts from the language categories 
X = data["Text"]
y = data["Language"]

# converting language categories to numerical values
le = LabelEncoder()
y = le.fit_transform(y)

# iterating through all samples to clean them
for i, s in enumerate(X):
    X[i] = text_preprocess(s)

'''
Creating bag of words text representation with words used as features. A more carefull selection and combination (e.g., by including n-gram embeddings) of features can lead to better performance of the models.
'''
cv = CountVectorizer()
X = cv.fit_transform(X).toarray()

# train-test splitting of the samples
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# instantiating the basic classification models
lr_model = LogisticRegression()
svm_model = SVC()
mnb_model = MultinomialNB()
knn_model = KNeighborsClassifier()
dt_model = DecisionTreeClassifier()

# preparing the stack ansemble model
level0 = list() ; level0.append(('lr', lr_model)) ; level0.append(('svc', svm_model))
level0.append(('nb', mnb_model)) ; level0.append(('knn', knn_model)) ; level0.append(('dt', dt_model))
level1 = lr_model
stck_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# instantiating the bagging ensemble model 
rf_model = RandomForestClassifier(random_state=7, n_jobs=-1)

# instantiating boosting ensemble models 
ada_model = AdaBoostClassifier(random_state=7)
gb_model = GradientBoostingClassifier(random_state=7)
xgb_model = xgb.XGBClassifier(random_state=7)

# selecting the classifier to use in this run
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

# training the selected model
model.fit(x_train, y_train)

# getting prediction and accuracy
y_pred = model.predict(x_test)
ac = accuracy_score(y_test, y_pred)

# report accuracy and confusion matrix
print(f"\nOverall accuracy of the selected model is: {ac:.4f}\n")
print(f"\nConfusion matrix is:\n {confusion_matrix(y_test, y_pred)}\n")

# identifying language in text entered by the user
target_text = input("Type or paste your text below:\n\n")
identify(target_text)
