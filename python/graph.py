import time
import tkinter as tk
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from nltk import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

"""
Loads three conversation management dimensions and shows a graph.
"""

# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

start_time = time.time()


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

df = pd.read_csv(Path(__file__).parent / '../data/combined.csv', encoding='latin1', sep=';')

for u in unknown:
    df = df.drop(df[df['Name'] == u].index)

df = df.replace(conversation_management, 'Conversation Management')
df = df.replace(active_discussion, 'Active Discussion')
df = df.replace(creative_conflict, 'Creative Conflict')

y = df['Name']
X = df['Coded Text']

X = np.array(X)
y = np.array(y)

# Lemmatization
lemmatized_X = []
for text in X:

    lemmatized_text = ' '.join([lemmatizer.lemmatize(word.lower()) for word in text.split()])
    lemmatized_X.append(lemmatized_text)

lemmatized_X = np.array(lemmatized_X)

# 5-fold cross-validation object
# increase in accuracy is due to split
# skf = StratifiedKFold(n_splits=5)
# example with lower accuracy
skf = StratifiedKFold(n_splits=10, random_state=66, shuffle=True)

accuracies = []
precisions = []
recalls = []
predictions = []
truth = []

for train_index, test_index in skf.split(lemmatized_X, y):
    # Split the data into training and testing sets
    X_train, X_test = lemmatized_X[train_index], lemmatized_X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    nb = Pipeline(
        [('vect', TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9)),
         ('model', RandomForestClassifier(random_state=66)),
         ])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    # Report should mention that "macro" average is used, also that score is different for each class.
    precision_recall_f_score = precision_recall_fscore_support(y_test, y_pred, average="macro")
    precisions.append(precision_recall_f_score[0])
    recalls.append(precision_recall_f_score[1])
    predictions += list(y_pred)
    truth += list(y_test)

table = {'Accuracy': accuracies,
         'Precision': precisions,
         'Recall': recalls}

data_folds = pd.DataFrame(data=table)

print(data_folds.to_latex(caption="Fold Performance",
                          formatters={"name": str.upper},
                          float_format="{:.4f}".format))

print("time %s seconds" % (time.time() - start_time))
print("accuracy %s" % round((sum(accuracies) / len(accuracies)), 4))
print("precision %s" % round((sum(precisions) / len(precisions)), 4))
print("recall %s" % round((sum(recalls) / len(recalls)), 4))
root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

colors = ['#FF6969', '#A6D0DD', '#DEDEA7']

dimension_labels = ["Conversation Management", "Creative Conflict", "Active Discussion"]
predictions = [list(predictions).count("Conversation Management"), list(predictions).count("Active Discussion"),
               list(predictions).count("Creative Conflict")]
figure1 = Figure()
tasty = figure1.add_subplot(111)
tasty.pie(predictions, radius=1, labels=dimension_labels, autopct='%0.2f%%', labeldistance=1.05,
          wedgeprops={'linewidth': 4, 'edgecolor': 'white'}, colors=colors)
tasty.set_title('Predicted Testing Values')
chart = FigureCanvasTkAgg(figure1, frame)
chart.get_tk_widget().pack(side=tk.TOP)

actual = [list(truth).count("Conversation Management"), list(truth).count("Active Discussion"),
          list(truth).count("Creative Conflict")]
figure2 = Figure()
delicious = figure2.add_subplot(111)
delicious.pie(actual, radius=1, labels=dimension_labels, autopct='%0.2f%%', labeldistance=1.05,
              wedgeprops={'linewidth': 4, 'edgecolor': 'white'}, colors=colors)
delicious.set_title('Actual Testing Values')
chart = FigureCanvasTkAgg(figure2, frame)
chart.get_tk_widget().pack(side=tk.BOTTOM)

root.mainloop()
