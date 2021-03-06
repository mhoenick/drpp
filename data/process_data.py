import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load and merge messages and categories CSV files
    
    Parameters
    messages_filepath (str): path to message csv file, supplied via cmd line
    categories_filepath (str): path to categories csv file, supplied via cmd line
    
    Returns
    df (df): dataframe containing messages and categories    
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    '''
    Clean messages and categories data
    
    Category target values extracted and converted to binary. 
    Rows of duplicate messages and target values not 0 or 1 dropped.
    
    Parameters
    df (df): dataframe containing messages and categories from load_data() func
    
    Returns
    df (df): cleaned data containing messages and categories
    '''
    
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Rename columns
    col_names = categories.iloc[0,]
    col_names = col_names.apply(lambda x: x[:-2]).values
    categories.columns = col_names
    
    # Convert categories to given numbers (0, 1)
    for column in categories:
        categories[column] = categories[column].astype(str)\
                            .apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
    
    # Exchange df categories columns with new categories dataframe
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1)
    
    # Number of duplicates
    duplicates = df['message'].shape[0] - df['message'].value_counts().shape[0]
    if duplicates != 0:
        df = df.drop_duplicates(subset=['message'])
        d_left = df['message'].shape[0] - df['message'].value_counts().shape[0]
        print(f'    Duplicates removed: {duplicates}\n\
            Duplicates left: {d_left}')
  
    # Remove any rows where target label not in [0,1]
    drop_idxs = []
    for col in col_names:
        if df[col][(df[col] != 0) & (df[col] != 1)].any():
            drop_idxs += df[col][(df[col] != 0) & (df[col] !=1)]\
                        .index.values.tolist()
    
    df = df.drop(drop_idxs)
  
    return df

def save_data(df, database_filename):
    '''
    Save cleaned data in SQLite database
    
    Parameters
    df (df): cleaned dataset from clean_data() func
    database_filename (str): filename of database, supplied via cmd line
    
    Returns
    Writes a sqlite database to path database_filename
    '''
    
    sql_db = 'sqlite:///' + database_filename 
    engine = create_engine(sql_db)
    df.to_sql(database_filename.rstrip('.db'), 
                engine, 
                index=False, 
                if_exists='replace')


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