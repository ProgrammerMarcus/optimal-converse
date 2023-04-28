import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = pd.read_csv('combined_min_binary.csv', encoding='latin1', sep=';')
X = data['Transcription']
Y = data['Dimension']

# define pipeline
vect = CountVectorizer(max_features=None, min_df=0, max_df=0.9)
model = SVC(gamma=1, probability=False, class_weight={"Conversation Management": 1.4, "Other": 1})
over = SMOTE(sampling_strategy=1)
under = RandomUnderSampler(sampling_strategy=1)
steps = [
    ('vect', vect),
    ('model', model),
]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
scores = cross_val_score(pipeline, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)

print(sum(scores) / len(scores))