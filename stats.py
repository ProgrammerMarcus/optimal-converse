import pandas as pd
import numpy
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# load and manage data

conversation_management = ['Appreciation', 'Request Confirmation', 'Apologize', 'Listening', 'Suggest Action',
                           'Acknowledge', 'Accept and Confirm', 'Task', 'Request Focus Change', 'Summarize Information',
                           'Reject', 'Request Attention', 'Maintenance']

active_discussion = ['Encourage', 'Lead', 'Inform', 'Justify', 'Suggest', 'Request', 'Motivate', 'Elaboration',
                     'Elaborate', 'Clarification', 'Reinforce', 'Assert', 'Explain and Clarify', 'Illustration',
                     'Opinion', 'Justification', 'Information']

creative_conflict = ['Offer Alternative', 'Propose Exception', 'Doubt', 'Suppose', 'Mediate', 'Agree', 'Disagree',
                     'Argue', 'Infer', 'Conciliate', ]

unknown = ['Scoping', 'Technical Problem', 'Communication Decision', 'Design Principle', 'Functionality',
           'Technical Question', 'Implementation Decision', 'Use Case', 'Usability and User Interface', 'Awareness',
           'Technical Decision', 'Behavioural Decision', 'Notation Decision', 'Rephrase', 'Actor',
           'Structural Decision', 'Coordinate Group Process', 'Driver', 'Data Decision', 'Assumption']

df = pd.read_csv('combined.csv', encoding='latin1', sep=';')

for u in unknown:
    df = df.drop(df[df['Name'] == u].index)

df = df.replace(conversation_management, 'Conversation Management')
df = df.replace(active_discussion, 'Active Discussion')
df = df.replace(creative_conflict, 'Creative Conflict')

y = df['Name']
X = df['Coded Text']


def fold_performance():
    """
    Prints the statistics of each fold in Stratified 5-fold cross-validation in LaTex format.

    The function retrieves performance scores such as accuracy, precision, and recall from the dictionary 'scores'.
    It creates a pandas DataFrame using the scores and then prints the DataFrame in LaTeX format.

    """

    steps = [
        ('vect', CountVectorizer(max_features=None, min_df=0, max_df=1.0)),
        ('model', RandomForestClassifier()),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=5, n_jobs=-1)

    table = {'Accuracy': scores["test_accuracy"],
             'Precision': scores["test_precision_macro"],
             'Recall': scores["test_recall_macro"]}

    data_folds = pd.DataFrame(data=table)

    print(data_folds.to_latex(caption="Fold Performance",
                              index=False,
                              formatters={"name": str.upper},
                              float_format="{:.4f}".format))


fold_performance()

# classifiers

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="rbf", gamma="scale"),
    SVC(kernel="rbf", gamma=1),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
]

names = ["3-Neighbour", "Linear SVM", "RBF SVM (scale)", "RBF SVM (1)", "Decision Tree", "Random Forest", "MLP", "AdaBoost"]
accuracy = []
precision = []
recall = []

for c, n in zip(classifiers, names):
    steps = [
        ('vect', CountVectorizer(max_features=None, min_df=0, max_df=1.0)),
        ('model', c),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=5, n_jobs=-1)

    accuracy += [(sum(scores["test_accuracy"]) / len(scores["test_accuracy"]))]
    precision += [sum(scores["test_precision_macro"]) / len(scores["test_precision_macro"])]
    recall += [(sum(scores["test_recall_macro"]) / len(scores["test_recall_macro"]))]

table = {'Name': names,
         'Accuracy': accuracy,
         'Precision': precision,
         'Recall': recall}

data_folds = pd.DataFrame(data=table)

print(data_folds.to_latex(caption="Classifier Performance",
                          index=False,
                          formatters={"name": str.upper},
                          float_format="{:.4f}".format))

# vectorizers

vecotorizers = [
    CountVectorizer(max_features=None, min_df=0, max_df=1.0),
    CountVectorizer(max_features=None, min_df=0, max_df=1.0, stop_words=stopwords.words('english')),
    TfidfVectorizer(max_features=None, min_df=0, max_df=1.0),
    TfidfVectorizer(max_features=None, min_df=0, max_df=1.0, stop_words=stopwords.words('english')),
]

names = ["Bag-of-words", "Bag-of-words (Remove Stopwords)", "TF-IDF", "TF-IDF (Remove Stopwords)"]
accuracy = []
precision = []
recall = []

for v, n in zip(vecotorizers, names):
    steps = [
        ('vect', v),
        ('model', RandomForestClassifier()),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=5, n_jobs=-1)

    accuracy += [(sum(scores["test_accuracy"]) / len(scores["test_accuracy"]))]
    precision += [sum(scores["test_precision_macro"]) / len(scores["test_precision_macro"])]
    recall += [(sum(scores["test_recall_macro"]) / len(scores["test_recall_macro"]))]

table = {'Name': names,
         'Accuracy': accuracy,
         'Precision': precision,
         'Recall': recall}

data_folds = pd.DataFrame(data=table)

print(data_folds.to_latex(caption="Vectorizer Performance",
                          index=False,
                          formatters={"name": str.upper},
                          float_format="{:.4f}".format))

# samplers

samplers = [
    SMOTE(),
    RandomUnderSampler()
]

names = ["SMOTE oversampling", "Random Undersampling"]
accuracy = []
precision = []
recall = []

for s, n in zip(samplers, names):
    steps = [
        ('vect', CountVectorizer(max_features=None, min_df=0, max_df=1.0)),
        ('samper', s),
        ('model', RandomForestClassifier()),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=5, n_jobs=-1)

    accuracy += [(sum(scores["test_accuracy"]) / len(scores["test_accuracy"]))]
    precision += [sum(scores["test_precision_macro"]) / len(scores["test_precision_macro"])]
    recall += [(sum(scores["test_recall_macro"]) / len(scores["test_recall_macro"]))]

table = {'Name': names,
         'Accuracy': accuracy,
         'Precision': precision,
         'Recall': recall}

data_folds = pd.DataFrame(data=table)

print(data_folds.to_latex(caption="Sampler Performance",
                          index=False,
                          formatters={"name": str.upper},
                          float_format="{:.4f}".format))
