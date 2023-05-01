import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
]
sampler = [
    SMOTE(),
    RandomUnderSampler()
]
# load data
data = pd.read_csv('combined_min_binary.csv', encoding='latin1', sep=';')
X = data['Transcription']
Y = data['Dimension']

for s in sampler:
    steps = [
        ('vect', CountVectorizer(max_features=None, min_df=0, max_df=0.9)),
        ('e', s),
        ('model', SVC(gamma=1)),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, X, Y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=5, n_jobs=-1)

    # print("Bag of words + C-Support Vector Classification")
    print(s.__class__)
    print("accuracy %s" % (sum(scores["test_accuracy"]) / len(scores["test_accuracy"])))
    print("recall(macro) %s" % (sum(scores["test_recall_macro"]) / len(scores["test_recall_macro"])))
    print("precision(macro) %s" % (sum(scores["test_precision_macro"]) / len(scores["test_precision_macro"])))

steps = [
    ('vect', CountVectorizer(max_features=None, min_df=0, max_df=0.9)),
    ('model', SVC(gamma=1)),
]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
scores = cross_validate(pipeline, X, Y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=5, n_jobs=-1)

# print("Bag of words + C-Support Vector Classification")
print("none")
print("accuracy %s" % (sum(scores["test_accuracy"]) / len(scores["test_accuracy"])))
print("recall(macro) %s" % (sum(scores["test_recall_macro"]) / len(scores["test_recall_macro"])))
print("precision(macro) %s" % (sum(scores["test_precision_macro"]) / len(scores["test_precision_macro"])))