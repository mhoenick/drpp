import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    ''' Tokenize text messages '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
df = df.drop(df[df['related']==2].index)  #delete me

# load model
model = joblib.load("../models/classifier.pkl")

# load classification_report
df_rep = joblib.load("../models/classification_report.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    data_per_category = df.iloc[:,4:].sum().sort_values(ascending=False)
    category_names = data_per_category.index
    
    # Align df_rep with sorted data_per_category.index
    index = []
    for w in data_per_category.index:
        index.append((w, 'precision'))
        index.append((w, 'recall'))
        index.append((w, 'f1-score'))
        index.append((w, 'support'))

    df_rep_align = df_rep.reindex(index)
    
    # filter f1-score data out of df_rep
    df_rep_align = df_rep_align.xs('f1-score', axis=0, level=1)\
                                    .mul(100)\
                                    .round(1)\
                                    .iloc[:,:3]\
                                    .rename(columns={
                                        '0': 'f1-score: not this category',
                                        '1': 'f1-score: this category'}
                                        )
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=data_per_category
                )
            ],

            'layout': {
                'title': 'Messages per Category',
                'Autosize': False,
                'height': 600,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True
                }
            }
        },
                {
            'data': [
                Scatter(
                    x=category_names,
                    y=df_rep_align['accuracy'],
                    name='Accuracy',
                    type='scatter'
                ),
                Scatter(
                    x=category_names,
                    y=df_rep_align['f1-score: not this category'],
                    name='f1-score: not this category'
                ),
                Scatter(
                    x=category_names,
                    y=df_rep_align['f1-score: this category'],
                    name='f1-score: this category'
                )
            ],

            'layout': {
                'title': 'Prediction Accuracy and F1-Score on a Test Dataset',
                'height': 600,
                'yaxis': {
                    'title': "Percentage [%]"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True
                },
                'legend': {
                    'xanchor': "auto",
                    'yanchor': "auto",
                    'y': 0.8
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
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()