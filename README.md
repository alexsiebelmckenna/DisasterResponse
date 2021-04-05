# Disaster Response Pipeline Project

This pipeline categorizes messages sent during disasters into one of 36 categories to enable more efficient disaster response processes on the ground.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. Enter your message, followed by which of the 3 broader categories best describes your message. This is used by the ML model to categorize your message. Click "Classify Message".
