"""
Trains classification model. Saves best performing model.
"""

import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
import pdb
import pickle
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import re
import sqlite3
import sys
import time
from pip._internal import main
from joblib import dump, load
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sqlalchemy import create_engine
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from sklearn.utils import parallel_backend

# Make sure requisite packages are installed:
pkg='scikit-learn'
version='0.24.1'
main(['install', '{0}=={1}'.format(pkg, version)])

def load_data(database_filepath):
    """
    Loads data from SQLite database into dataframe,
    converts 'genre' to category data type, drops
    rows as appropriate

    Parameters:
    database_filepath

    Returns:
    X: independent variables
    Y: dependent variables
    category_names: column names
    """
    con = sqlite3.connect(database_filepath)

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("DisasterResponse.db",engine)
    # drop id col
    df.drop('id', axis=1, inplace=True)
    df['genre'] = df['genre'].astype('category')

    # extract values from X and y
    X = df[['message', 'genre']]
    Y = df.iloc[:, 3:]

    # TODO: Drop rows where Y['related']==2

    rowstodrop = Y[Y['related']==2].index

    Y.drop(rowstodrop, inplace=True)
    # do the same to X:
    X.drop(rowstodrop, inplace=True)

    category_names = list(df.columns[3:])

    return X, Y, category_names

def tokenize(text):
    """"
    Tokenization function for use in TfidfVectorizer in build_model()
    1. Normalizes text (converts to lowercase)
    2. Removes punctuation
    3. Removes stop_words
    4. Splits sentence into words (tokens)
    5. Performs POS tagging

    Parameters:
    text: string to be tokenized

    Returns:
    tagged_tokens: POS-tagged tokens
    """
    # normalize text
    text = text.lower()

    # Remove punctuation characters
    text = re.sub(r'[^(a-z)(A-Z)(0-9]', ' ', text)

    word_split = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    word_split = [word for word in word_split if word not in stop_words]

    # POS tagging
    tagged_tokens = pos_tag(word_split)
    return tagged_tokens

def build_model():
    """
    Builds ML pipeline
    1. Builds preprocessing pipeline (ColumnTransformer object)
    2. Append Multi-Output Classifier pipeline to preprocessor
       pipeline (Pipeline object)
    3. Implement Grid Search to optimize hyperparameter values

    Parameters:
    none

    Returns:
    cv: ML pipeline (Pipeline object comprising preprocessing & classification)
    """
    # build preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        # for text data
        ('tfidf_vec', TfidfVectorizer(tokenizer=tokenize),
         'message'),
        # for categorical data
        ('onehot_vec', OneHotEncoder(handle_unknown='ignore'), ['genre'])
    ])

    # build full prediction pipeline by appending classifier to preprocessor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # classifier
        ('clf', MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier())))
    ])

    # define parameter grid: e.g. varying the number of estimators in the
    # classifier between 5 and 10
    param_grid = {'clf__estimator__n_estimators': (5,10)}

    # finally, putting together the completed pipeline
    cv = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=4, verbose=2)

    return cv

def eval_metrics(array1, array2, category_names):
    """"
    Presents evaluation metrics from sklearn.metrics classification_report
    for array1 & array2 (test and predicted values of Y respectively).
    Also prints list of category names common to both arrays.

    Parameters:
    array1: typically, test values of Y
    array2: typically, predicted values of Y
    category_names: category names taken from Y

    Returns:
    none
    """
    metrics = []
    # Evaluate metrics for each set of labels
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(array1[:,i],array2[:,i]))

def evaluate_model(model, X_test, Y_test, category_names):
    """"
    1. Calculated predicted values of Y
    2. Calls eval_metrics() function with test and predicted values of Y

    Parameters:
    model: ML model
    X_test: test values of X
    Y_test: test values of Y
    category_names: category names taken from Y
    """
    Y_pred = model.predict(X_test)
    eval_metrics(Y_test.values, Y_pred, category_names)

def save_model(model, model_filepath):
    """"
    Dumps model with best hyperparameters (from GridSearchCV) into filepath
    given by model_filepath

    Parameters:
    model: ML model
    model_filepath: path to ML model (to save)
    """
    dump(model.best_estimator_, model_filepath+'filename.pkl')

def main():
    """
    If enough arguments are passed:
    1. Load data from database_filepath into X, Y, and category_names
    2. Split data into training and test sets
    3. Train, evaluate, and save model with best performance.

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        with parallel_backend('multiprocessing'):
            print('Building model...')
            model = build_model()

            print('Training model...')
            t0 = time.time()
            model.fit(X_train, Y_train)
            t1 = time.time()
            total = t1-t0
            print('Time taken to fit model: {}'.format(total))

            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
