import re
import time
import tkinter as tk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize

start_time = time.time()

conversation_management = ['Appreciation', 'Request Confirmation', 'Apologize', 'Listening', 'Suggest Action',
                           'Acknowledge', 'Accept and Confirm', 'Task', 'Request Focus Change', 'Summarize Information',
                           'Reject', 'Request Attention', 'Maintenance']

active_discussion = ['Encourage', 'Lead', 'Inform', 'Justify', 'Suggest', 'Request', 'Motivate', 'Elaboration',
                     'Elaborate', 'Clarification', 'Reinforce', 'Assert', 'Explain and Clarify', 'Illustration',
                     'Opinion', 'Justification', 'Information']

creative_conflict = ['Offer Alternative', 'Propose Exception', 'Doubt', 'Suppose', 'Mediate', 'Agree', 'Disagree',
                     'Argue', 'Infer', ]

unknown = ['Scoping', 'Technical Problem', 'Communication Decision', 'Design Principle', 'Functionality',
           'Technical Question', 'Implementation Decision', 'Use Case', 'Usability and User Interface', 'Awareness',
           'Technical Decision', 'Behavioural Decision', 'Notation Decision', 'Conciliate', 'Rephrase', 'Actor',
           'Structural Decision', 'Coordinate Group Process', 'Driver', 'Data Decision', 'Assumption']

df = pd.read_csv('combined.csv', encoding='latin1', sep=';')

for u in unknown:
    df = df.drop(df[df['Name'] == u].index)

df = df.replace(conversation_management, 'Conversation Management')
df = df.replace(active_discussion, 'Active Discussion')
df = df.replace(creative_conflict, 'Creative Conflict')

y = df['Name']
X = df['Coded Text']

text = list(X)
words = []
for i in range(len(text)):
    r = word_tokenize(text[i])
    r = [WordNetLemmatizer().lemmatize(word) for word in r]
    r = ' '.join(r)
    words.append(r)
X = np.array(words)
y = np.array(y)
print(X)

# 5-fold cross-validation object
skf = StratifiedKFold(n_splits=5)

accuracies = []
precisions = []
recalls = []

for train_index, test_index in skf.split(X, y):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    nb = Pipeline(
        [('vect', CountVectorizer(max_features=1500, min_df=0, max_df=0.8)),
         ('clf', SVC(gamma=1, probability=False)),
         ])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    precision_recall_f_score = precision_recall_fscore_support(y_test, y_pred, average="macro")
    sep = precision_recall_fscore_support(y_test, y_pred,
                                          labels=["Conversation Management", "Active Discussion", "Creative Conflict"])
    precisions.append(precision_recall_f_score[0])
    recalls.append(precision_recall_f_score[1])

print("time %s seconds" % (time.time() - start_time))
print("accuracy %s" % round((sum(accuracies) / len(accuracies)), 4))
print("precision %s" % round((sum(precisions) / len(precisions)), 4))
print("recall %s" % round((sum(recalls) / len(recalls)), 4))
