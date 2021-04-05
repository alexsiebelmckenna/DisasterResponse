import json
import plotly
import pandas as pd
import pdb
import nltk
import pickle
import os
import plotly.express as px

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

import re
import string
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
from nltk.corpus import stopwords
from sklearn.utils import parallel_backend

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



app = Flask(__name__)

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

def mini_tokenize(text):
    # normalize text
    text = text.lower()

    # Remove punctuation characters
    text = re.sub(r'[^(a-z)(A-Z)(0-9]', ' ', text)

    word_split = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    word_split = [word for word in word_split if word not in stop_words]

    return word_split

preprocessor = ColumnTransformer(transformers=[
    # for text data
    ('tfidf_vec', TfidfVectorizer(tokenizer=tokenize),
     'message'),
    # for categorical data
    ('onehot_vec', OneHotEncoder(handle_unknown='ignore'), ['genre'])])

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse.db', engine)

X = df[['message', 'genre']]
Y = df.drop(columns=['id', 'message', 'original', 'genre'])
# load model
model = joblib.load("models/classifier.pkl")

tokenized = df['message'].apply(lambda x: mini_tokenize(x))

# set here to explore data further (un-comment as necessary):
#import pdb; pdb.set_trace()
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df_wordcount = df['message'].map(len)
    df_wordcount_genre = pd.concat([df['genre'], df_wordcount], axis=1)
    df_wordcount_genre_mean = df_wordcount_genre.groupby('genre').mean()
    df_wordcount_genre_mean.rename(columns={'message':'length'}, inplace=True)
    df_wordcount_genre_mean.reset_index(level=0, inplace=True)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                x=df_wordcount_genre_mean['genre'],
                y=df_wordcount_genre_mean['length']
                )
            ],

            'layout': {
                'title': 'Average Message Length by Genre',
                'yaxis': {
                    'title': "Average Message Length (Characters)"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }




        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    message_query = request.args.get('message')

    # save user input in genre
    genre_query = request.args.get('genre')

    df_query = (pd.DataFrame(columns=['message', 'genre'], index=range(0,1))\
                    .assign(message=message_query, genre=genre_query))

    # use model to predict classification for query
    classification_labels = model.predict(df_query)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html. Please see that file.
    return render_template(
        'go.html',
        message=message_query,
        genre=genre_query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
