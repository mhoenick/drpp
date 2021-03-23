import nltk
nltk.download('punkt', 'stopwords', 'wordnet')
import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

stopwords_set = set(stopwords.words('english'))

def load_data(database_filepath):
    ''' Load data from sqlite database '''
    
    # Create sql engine
    engine = create_engine('sqlite:///' + database_filepath)
    # Read data; file name == table name
    sql_cmd = 'SELECT * FROM ' +\
                re.search('/\w+\.', database_filepath).group()[1:-1]
    df = pd.read_sql(sql_cmd, engine)
    
    # Split into data and target
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    return X, Y, Y.columns


def tokenize(text):
    ''' Tokenize the supplied text messages '''
    # Remove majority of urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_bitly = 'http\s[a-z\.]+\s[a-zA-Z0-9]+'
    
    text = re.sub(url_regex, 'urlplaceholder', text)
    text = re.sub(url_bitly, 'urlplaceholder', text)
    # Remove punctuation
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    
    # Remove stopwords and lemmatize
    text = [w.lower() for w in word_tokenize(text) if w not in stopwords_set]
    text = [WordNetLemmatizer().lemmatize(word) for word in text]
    text = [WordNetLemmatizer().lemmatize(word, pos='v') for word in text]
    
    return text


def build_model(X_train):
    ''' Model: Transformer and Estimator Pipeline '''

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier())
    ])
    
    # get vocabulary size
    vocab = CountVectorizer(tokenizer=tokenize)
    vocab.fit_transform(X_train)
    vocab_len = len(vocab.vocabulary_)
    
    parameters = {
            #'vect__max_df': [1.0], 
            #'vect__ngram_range': [(1,1)], 
            'vect__max_features': [None, int(vocab_len*0.7)], 
            'clf__max_features': [1000, 1500],
            'clf__min_samples_split': [2, 3, 4], 
            'clf__n_estimators': [100, 150]
            }
    
    model = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=10)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    ''' Print accuracy, precision, recall and f1-score.
     Summarised overall categories and per category.
    '''
    Y_pred = model.predict(X_test)
    
    print('\n\nClassification Report - Average over all Categories:')
    print(classification_report(Y_test.to_numpy().flatten(),
                                Y_pred.flatten()))

    print('\n\nClassification Report - Results per Categories:\n')
    for i in range(36):
        print(f'Category: {category_names[i]}')
        print(classification_report(Y_test.to_numpy()[:,i], Y_pred[:,i],
                                    zero_division=0),'\n')
    
    # Create classification_report df for frontend usage
    df_rep = pd.DataFrame()
    for i in range(36):
        clf_rep = classification_report(Y_test.to_numpy()[:,i],
                                        Y_pred[:,i],
                                        output_dict=True,
                                        zero_division=0)
        df = pd.DataFrame.from_dict(clf_rep)
        df = df.set_index([[category_names[i]] * 4, df.index])
        df_rep = pd.concat([df_rep, df])
    
    return df_rep


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as destination:
        pickle.dump(model, destination)


def save_report(df):
    with open('classification_report.pkl', 'wb') as destination:
        pickle.dump(df, destination)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Model params:\n', model.best_params_)
        
        print('Evaluating model...')
        df_rep = evaluate_model(model, X_test, Y_test, category_names)

        print('Saving report...\n   REPORT: classification_report.pkl')
        save_report(df_rep)
        
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