import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

# load data from csv
data = pd.read_csv('combined_min_binary.csv', encoding='latin1', sep=';')

X = data['Transcription']
y = data['Dimension']

# 5-fold cross-validation object
skf = StratifiedKFold(n_splits=5)

scores = []
for train_index, test_index in skf.split(X, y):
    # Split the data into training and teststing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    nb = Pipeline(
        [('vect', CountVectorizer(max_features=1500, min_df=0, max_df=0.8)),
         ('standardscaler', MaxAbsScaler()),
         ('clf', MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, solver='adam', max_iter=5000)),
         ])

    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

print("accuracy %s" % (sum(scores)/len(scores)))
