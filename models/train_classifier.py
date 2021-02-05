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

    # build preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        # for text data
        ('tfidf_vec', TfidfVectorizer(tokenizer=tokenize),
         'message'),
         #TODO: Figure out how to incorporate 'genre' categorical column
        # for categorical data
        ('onehot_vec', OneHotEncoder(handle_unknown='ignore'), ['genre'])
    ])

    # append classifier
    # build full prediction pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # classifier
        ('clf', MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier())))
    ])

    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    #max_depth.append(None)
    # Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    #bootstrap = [True, False]

    param_grid = {'clf__estimator__n_estimators': (5 ,10)}

    cv = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=-1, verbose=2)

    return cv

def eval_metrics(array1, array2, category_names):
    metrics = []
    # Evaluate metrics for each set of labels
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(array1[:,i],array2[:,i]))

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    eval_metrics(Y_test.values, Y_pred, category_names)

def save_model(model, model_filepath):
    dump(model.best_estimator_, model_filepath+'filename.pkl')

def main():
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
