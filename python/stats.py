from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
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

df = pd.read_csv(Path(__file__).parent / '../data/combined.csv', encoding='latin1', sep=';')

for u in unknown:
    df = df.drop(df[df['Name'] == u].index)

df = df.replace(conversation_management, 'Conversation Management')
df = df.replace(active_discussion, 'Active Discussion')
df = df.replace(creative_conflict, 'Creative Conflict')

y = df['Name']
X = df['Coded Text']

# Lemmatization
lemmatized_X = []
lemmatizer = WordNetLemmatizer()
for text in X:

    lemmatized_text = ' '.join([lemmatizer.lemmatize(word.lower()) for word in text.split()])
    lemmatized_X.append(lemmatized_text)

lemmatized_X = np.array(lemmatized_X)

X = np.array(lemmatized_X)
y = np.array(y)

skf = StratifiedKFold(n_splits=10, random_state=66, shuffle=True)


def fold_performance():
    """
    Prints the statistics of each fold in Stratified 5-fold cross-validation in LaTex format.

    The function retrieves performance scores such as accuracy, precision, and recall from the dictionary 'scores'.
    It creates a pandas DataFrame using the scores and then prints the DataFrame in LaTeX format.

    """

    steps = [
        ('vect', TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9)),
        ('model', RandomForestClassifier(random_state=66)),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=skf, n_jobs=-1)

    table = {'Accuracy': scores["test_accuracy"],
             'Precision': scores["test_precision_macro"],
             'Recall': scores["test_recall_macro"]}

    accuracy_average = (sum(scores["test_accuracy"]) / len(scores["test_accuracy"]))
    precision_average = sum(scores["test_precision_macro"]) / len(scores["test_precision_macro"])
    recall_average = (sum(scores["test_recall_macro"]) / len(scores["test_recall_macro"]))

    data_folds = pd.DataFrame(data=table)

    average_row = pd.DataFrame({'Accuracy': accuracy_average,
                                'Precision': precision_average,
                                'Recall': recall_average}, index=['Average'])

    data_folds = data_folds._append(average_row)

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
        KNeighborsClassifier(5),
        SVC(kernel="linear", random_state=66),
        SVC(kernel="rbf", gamma="scale", random_state=66),
        MultinomialNB(),
        DecisionTreeClassifier(random_state=66),
        RandomForestClassifier(random_state=66),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(random_state=66),
        LogisticRegression(random_state=66),
    ]

    names = ["5-Neighbour", "Linear SVM", "RBF SVM (scale)", "Multinomial Naïve Bayes", "Decision Tree",
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
            ('model', RandomForestClassifier(random_state=66)),
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
        ('model', RandomForestClassifier(random_state=66)),
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


def category_performance():
    """
    Prints the statistics of all categories in single pipeline in LaTex format.

    The function retrieves performance scores such as accuracy, precision, and recall from the dictionary 'scores'.
    It creates a pandas DataFrame using the scores and then prints the DataFrame in LaTeX format.
    """
    xe = np.array(X)
    ye = np.array(y)
    cm_recalls = []
    cm_precisions = []
    ad_recalls = []
    ad_precisions = []
    cc_recalls = []
    cc_precisions = []
    for train_index, test_index in skf.split(xe, ye):
        # Split the data into training and testing sets
        X_train, X_test = xe[train_index], xe[test_index]
        y_train, y_test = ye[train_index], ye[test_index]

        nb = Pipeline(
            [('vect', TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9)),
             ('model', RandomForestClassifier(random_state=66)),
             ])
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        precision_recall_f_score = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=["Conversation Management", "Active Discussion", "Creative Conflict"])
        cm_precisions.append(precision_recall_f_score[0][0])
        cm_recalls.append(precision_recall_f_score[1][0])
        ad_precisions.append(precision_recall_f_score[0][1])
        ad_recalls.append(precision_recall_f_score[1][1])
        cc_precisions.append(precision_recall_f_score[0][2])
        cc_recalls.append(precision_recall_f_score[1][2])

    table = {
        'Type': ["CM Precision", "CM Recall", "AD Precision", "AD Recall", "CC Precision", "CC Recall", "Average P",
                 "Average R"],
        'Scores': [sum(cm_precisions) / len(cm_precisions),
                   sum(cm_recalls) / len(cm_recalls),
                   sum(ad_precisions) / len(ad_precisions),
                   sum(ad_recalls) / len(ad_recalls),
                   sum(cc_precisions) / len(cc_precisions),
                   sum(cc_recalls) / len(cc_recalls),
                   sum(cm_precisions + ad_precisions + cc_precisions) / len(cm_precisions + ad_precisions + cc_precisions),
                   sum(cm_recalls + ad_recalls + cc_recalls) / len(cm_recalls + ad_recalls + cc_recalls)]
        }

    print(table)

    data_folds = pd.DataFrame(data=table)

    print(data_folds.to_latex(caption="Category Performance",
                              index=False,
                              formatters={"name": str.upper},
                              float_format="{:.4f}".format))


def no_lemma_performance():
    """
    Prints the statistics of a single pipeline with no lemmatization applied, in LaTex format.

    Applies no lemmatization to the text, before performing stratified K-fold cross validation and placing the results
    in a Pandas DataFrame, which is then printed as a LaTeX Table.
    """

    names = ["NO LEMMA"]
    accuracy = []
    precision = []
    recall = []

    steps = [
        ('vect', TfidfVectorizer(max_features=1000, min_df=0, max_df=0.9)),
        ('model', RandomForestClassifier(random_state=66)),
    ]
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    scores = cross_validate(pipeline, df['Coded Text'], df['Name'],
                            scoring=['accuracy', 'precision_macro', 'recall_macro'], cv=skf,
                            n_jobs=-1)

    accuracy += [(sum(scores["test_accuracy"]) / len(scores["test_accuracy"]))]
    precision += [sum(scores["test_precision_macro"]) / len(scores["test_precision_macro"])]
    recall += [(sum(scores["test_recall_macro"]) / len(scores["test_recall_macro"]))]

    table = {'Name': names,
             'Accuracy': accuracy,
             'Precision': precision,
             'Recall': recall}

    data_folds = pd.DataFrame(data=table)

    print(data_folds.to_latex(caption="No-Lemma Performance",
                              index=False,
                              formatters={"name": str.upper},
                              float_format="{:.4f}".format))


fold_performance()
classifier_performance()
vectorizer_performance()
sampler_performance()
category_performance()
no_lemma_performance()
