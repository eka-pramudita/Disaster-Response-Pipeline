import sys
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pandas as pd
import pickle
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    Load data from the database created from the ETL process

        Parameters: 
            database_filepath (str): path to the database

        Return:
            X (DataFrame): message
            y (DataFrame): labels
            category_names (list): list of labels
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    y.replace({'related': 2}, 1, inplace = True)
    category_names = list(y.columns)

    return X, y, category_names


def tokenize(text):
    '''
    Tokenization function including url placeholder and lemmatization

        Parameters:
            text (str): input text

        Return:
            clean_tokens (list): tokenized and lemmatized words
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Model function using Pipeline and GridSearchCV

        Return:
            cv (object) = Grid Search Cross Validation model
    '''
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'text_pipeline__vect__max_features': (None, 2500, 5000),
        'text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 25, 50],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Model evaluation function using classification_report

        Parameter:
            model (object): model that is used to predict values
            X_test (DataFrame): message of test data
            y_tes (DataFrame): labels of test data
            category_names: list of labels

        Return:
            none - print the classification report result
    '''
    y_pred = model.predict(X_test)
    report = classification_report(Y_test, y_pred, target_names=category_names)
    print(report)


def save_model(model, model_filepath):
    '''
    Save model function to pickle file

        Parameter:
            model (object) - model that is used to predict values
            model_filepath - file path where the model will be saved

        Return:
            none - save model to desired file path
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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