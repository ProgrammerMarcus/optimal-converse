import time
import tkinter as tk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

"""
Loads two conversation management dimensions and shows a graph.
"""

start_time = time.time()
# load data from csv
data = pd.read_csv('combined_min_binary.csv', encoding='latin1', sep=';')

X = data['Transcription']
y = data['Dimension']

# 5-fold cross-validation object
# increase in accuracy is due to split
skf = StratifiedKFold(n_splits=5)
# example with lower accuracy
# skf = StratifiedKFold(n_splits=5, random_state=99, shuffle=True)

accuracies = []
precisions = []
recalls = []
perl = []

for train_index, test_index in skf.split(X, y):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    nb = Pipeline(
        [('vect', CountVectorizer(max_features=1500, min_df=0, max_df=0.8)),
         ('clf', SVC(gamma=1, probability=False, class_weight={"Conversation Management": 1.4, "Other": 1})),
         ])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    # Report should mention that "macro" average is used, also that score is different for each class.
    precision_recall_f_score = precision_recall_fscore_support(y_test, y_pred, average="macro")
    sep = precision_recall_fscore_support(y_test, y_pred,
                                    labels=["Conversation Management", "Other"])
    perl.append(sep)
    precisions.append(precision_recall_f_score[0])
    recalls.append(precision_recall_f_score[1])

print("time %s seconds" % (time.time() - start_time))
print("accuracy %s" % (sum(accuracies) / len(accuracies)))
print("precision %s" % (sum(precisions) / len(precisions)))
print("recall %s" % (sum(recalls) / len(recalls)))

root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

dimension_labels = ["Conversation Management", "Other"]
predictions = [list(y_pred).count("Conversation Management"), list(y_pred).count("Other")]
figure1 = Figure()
tasty = figure1.add_subplot(111)
tasty.pie(predictions, radius=1, labels=dimension_labels, autopct='%0.2f%%')
tasty.set_title('Predicted Values')
chart = FigureCanvasTkAgg(figure1, frame)
chart.get_tk_widget().pack(side=tk.RIGHT)

actual = [list(y_test).count("Conversation Management"), list(y_test).count("Other")]
figure2 = Figure()
delicious = figure2.add_subplot(111)
delicious.pie(actual, radius=1, labels=dimension_labels, autopct='%0.2f%%')
delicious.set_title('Actual Values')
chart = FigureCanvasTkAgg(figure2, frame)
chart.get_tk_widget().pack(side=tk.LEFT)

root.mainloop()
