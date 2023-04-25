import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

# load data from csv
data = pd.read_csv('combined_min_merged.csv', encoding='latin1', sep=';')

X = data['Transcription']

y = data['Dimension']

# (should be stratified k-fold cross-validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Tfid and stopword removal seem to decrease accuracy slightly
nb = Pipeline(
    [('vect', CountVectorizer(max_features=1500, min_df=0, max_df=0.8)),
     ('standardscaler', MaxAbsScaler()),
     ('clf', MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, solver='adam', max_iter=5000)),
     ])

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))

d = {'Input': X_test, 'Prediction': y_pred, 'Actual': y_test}

df = pd.DataFrame(data=d)

df