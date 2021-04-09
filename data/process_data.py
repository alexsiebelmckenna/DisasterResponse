""" Data Loading, Cleaning and Saving

This script accomplishes the following:
1. Loads data from csv file, loads into dataframe
2. Cleans data for ML Pipeline
3. Saves data into database

Sample usage:
python data/process_data.py data/disaster_messages.csv \
data/disaster_categories.csv data/DisasterResponse.db

"""

import sys
import pandas as pd
import matplotlib as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges data from 2 separate csv files (messages + categories)
    into Pandas dataframe.

    Parameters:
    messages_filepath: path to messages csv file
    categories_filepath: path to categories csv file

    Returns:
    df: merged dataframe

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Cleans data for classification:
        - creates columns for categories dataframe
        - converts data type category -> binary

    Parameters:
    df: dataframe to be cleaned

    Returns:
    df: cleaned dataframe

        """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # trim to get cleaned categories
    category_colnames = row.str[:-2]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # converting data into 1s and 0s
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop original categories column
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop rows where 'related' == 2
    # (https://knowledge.udacity.com/questions/113567):
    df = df.drop(df[df.related==2].index)
    # drop duplicates
    df.drop_duplicates(subset='message', inplace=True)
    return df

def save_data(df, database_filename):
    """
    Saves dataframe as SQLite database

    Parameters:
    df: dataframe to be saved
    database_filename: (relative) path to database

    Returns:
    n/a

    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, if_exists='replace', index=False)


def main():
    """
    If enough arguments are passed:
    1. Load data from messages_filepath and categories_filepath
    2. Clean and merge data
    3. Save data into database_filepath
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
