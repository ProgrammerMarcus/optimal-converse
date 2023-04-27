import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC

data = pd.read_csv('combined_min_binary.csv', encoding='latin1', sep=';')
X = data['Transcription']
Y = data['Dimension']

# define pipeline
vect = CountVectorizer(max_features=1500, min_df=0, max_df=0.8)
model = SVC(gamma=2, C=2)
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [
    ('vect', vect),
    ('over', over),
    ('model', model),
]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)

print(scores)