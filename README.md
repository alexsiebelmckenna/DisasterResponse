# Disaster Response Pipeline Project

### Motivation

This pipeline categorizes messages sent during disasters into one of 36 categories to enable more efficient disaster response processes on the ground. Data used to construct the classifier used in this pipeline is taken from Figure Eight's (now Appen, https://appen.com/) collection of "off-the-shelf" datasets for AI and ML.

### Python libraries used

- pandas
- numpy
- matplotlib
- sklearn
- nltk
- plotly
- json
- pdb
- pickle
- os
- joblib
- flask
- sqlalchemy
- re
- sys
- time
- pip

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

4. Input your message, followed by which of the 3 broader categories best describes it. Press the Enter button to classify your message.


### Summary of results

Your message will be classified according to at least one of 36 categories, as displayed below:
![Results](file:///Users/alexsiebelmckenna/Desktop/DR_results.png)


### Description of files

process_data.py: preprocessing data
train_classifier.py: trains and saves classifier
run.py: runs web app

### Acknowledgements

Big thank you to Appen for providing the data used for this project.
