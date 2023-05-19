import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
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

skf = StratifiedKFold(n_splits=5, random_state=66, shuffle=True)

def fold_performance():
    """
    Prints the statistics of each fold in Stratified 5-fold cross-validation in LaTex format.

    The function retrieves performance scores such as accuracy, precision, and recall from the dictionary 'scores'.
    It creates a pandas DataFrame using the scores and then prints the DataFrame in LaTeX format.

    """

    steps = [
        ('vect', TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9)),
        ('model', RandomForestClassifier()),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=skf, n_jobs=-1)

    table = {'Accuracy': scores["test_accuracy"],
             'Precision': scores["test_precision_macro"],
             'Recall': scores["test_recall_macro"]}

    data_folds = pd.DataFrame(data=table)

    print(data_folds.to_latex(caption="Fold Performance",
                              index=False,
                              formatters={"name": str.upper},
                              float_format="{:.4f}".format))


def classifier_performance():
    """
    Prints the statistics of each classifier in the list 'classifiers' in LaTex format.

    The function retrieves performance scores such as accuracy, precision, and recall from the dictionary 'scores'.
    It creates a pandas DataFrame using the scores and then prints the DataFrame in LaTeX format.
    """
    classifiers = [
        KNeighborsClassifier(),
        SVC(kernel="linear"),
        SVC(kernel="rbf", gamma="scale"),
        SVC(kernel="rbf", gamma=1),
        MultinomialNB(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        LogisticRegression(),
    ]

    names = ["5-Neighbour", "Linear SVM", "RBF SVM (scale)", "RBF SVM (1)", "Multinomial Naïve Bayes", "Decision Tree",
             "Random Forest", "MLP", "AdaBoost", "Logistic Regression"]
    accuracy = []
    precision = []
    recall = []

    for c, n in zip(classifiers, names):
        steps = [
            ('vect', TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9)),
            ('model', c),
        ]
        pipeline = Pipeline(steps=steps)

        # evaluate pipeline
        scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=skf,
                                n_jobs=-1)

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


def vectorizer_performance():
    """
    Prints the statistics of each vectorizer in the list 'vectorizers' in LaTex format.

    The function retrieves performance scores such as accuracy, precision, and recall from the dictionary 'scores'.
    It creates a pandas DataFrame using the scores and then prints the DataFrame in LaTeX format.
    """
    vecotorizers = [
        CountVectorizer(max_features=1000, min_df=0, max_df=0.9),
        CountVectorizer(max_features=1000, min_df=0, max_df=0.9, stop_words=stopwords.words('english')),
        TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9),
        TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9, stop_words=stopwords.words('english')),
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
        scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=skf,
                                n_jobs=-1)

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


def sampler_performance():
    """
    Prints the statistics of 'SMOTE oversampling' and 'Random Undersampling' in LaTex format.

    The function retrieves performance scores such as accuracy, precision, and recall from the dictionary 'scores'.
    It creates a pandas DataFrame using the scores and then prints the DataFrame in LaTeX format.
    """

    names = ["SAMPLER"]
    accuracy = []
    precision = []
    recall = []

    steps = [
        ('vect', TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9)),
        ('over', SMOTE()),
        ('under', RandomUnderSampler()),
        ('model', RandomForestClassifier()),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=skf,
                            n_jobs=-1)

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

def just_performance():
    """
    Prints the statistics of 'SMOTE oversampling' and 'Random Undersampling' in LaTex format.

    The function retrieves performance scores such as accuracy, precision, and recall from the dictionary 'scores'.
    It creates a pandas DataFrame using the scores and then prints the DataFrame in LaTeX format.
    """

    names = ["STANDARD"]
    accuracy = []
    precision = []
    recall = []

    steps = [
        ('vect', TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9)),
        ('model', RandomForestClassifier()),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=skf,
                            n_jobs=-1)

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


fold_performance()
classifier_performance()
vectorizer_performance()
# sampler_performance()
just_performance()
