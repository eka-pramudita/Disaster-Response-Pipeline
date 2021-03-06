import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories

        Parameters:
            messages_filepath (str): messages csv file path
            categories_filepath (str): categories csv file path

        Return:
            df (DataFrame): merged dataframe of messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how='left', on='id')

    categories = categories['categories'].str.split(";", expand=True)
    row = categories.iloc[1,:]

    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)

    return df

def clean_data(df):
    '''
    Clean merged data

        Parameters:
            df (DataFrame): merged DataFrame

        Return:
            df (DataFrame): cleaned DataFrame
    '''
    df = df.dropna()

    return df


def save_data(df, database_filename):
    '''
    Save data to database

        Parameters:
            df (DataFrame): cleaned DataFrame
            database_filename (str): database name

        Return:
            none - save DataFrame to database
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster', engine, if_exists='replace', index=False) 


def main():
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