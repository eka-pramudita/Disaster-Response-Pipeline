# Disaster Response Pipeline Project

## Table of Content
1. [Library Installation](#installation)
2. [Project Motivation](#motivation)
3. [Instructions](#instruction)
4. [Acknowledgements](#acknowledgement)

## Library Installation <a name="installation"></a>
All libraries used for this projects are documented in `requirements.txt`. To install all, run this command:
```
pip install -r requirements.txt
```

## Project Motivation <a name="motivation"></a>
This project is purposed to create an API based on the data from FigureEight to classify message related to disasters. The process is applying ETL Pipeline to clean the data and ML Pipeline to produce the model. Finally, a web app is created using Flask to perform the classification.

## Instructions <a name="instruction"></a> 
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgement <a name="acknowledgement"></a>
The data source is from the [FigureEight](https://appen.com/). 