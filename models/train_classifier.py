import pandas as pd
import pickle
import sqlite3
import sys
from pip._internal import main
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sqlalchemy import create_engine

# Make sure requisite packages are installed:
pkg='scikit-learn'
version='0.24.0'
main(['install', '{0}=={1}'.format(pkg, version)])

def load_data(database_filepath):

    con = sqlite3.connect(database_filepath)
    #cursor = con.cursor()

    #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #print(cursor.fetchall())

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("DisasterResponse.db",engine)
    # drop id col
    df.drop('id', axis=1, inplace=True)
    df['genre'] = df['genre'].astype('category')

    # extract values from X and y
    X = df[['message', 'genre']]
    Y = df.iloc[:, 3:]

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
         selector(dtype_exclude='category')),
        # for categorical data
        ('onehot_vec', OneHotEncoder(), selector(dtype_include='category'))
    ])

    # append classifier
    # build full prediction pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # classifier
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    #Y_pred = model.predict(X_test, Y_test)
    print('Score: {}%'.format(model.score(X_test, Y_test)))


def save_model(model, model_filepath):
    pickle.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

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
