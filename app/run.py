import json
import pandas as pd

from flask import Flask
from flask import render_template, request

import plotly
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
import plotly.express as px

from sqlalchemy import create_engine

import dill as pickle

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///./MessageCategory.db')
df = pd.read_sql_table('MessageCategory', engine)

# load model
with open('./classifier.pkl', 'rb') as file:
    model = pickle.load(file)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Here we explore the data classes in conjunction to the 'request' category
    # It aims to better look at the  classes in terms of a request rather than a cause
    # of a disaster.

    feature_list_requested_items = ['medical_products', 'water', 'food', 'money', 'electricity']
    request_data = df.groupby('request').sum()[feature_list_requested_items].iloc[1]
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=feature_list_requested_items,
                    y=request_data
                )
            ],

            'layout': {
                'title': 'Distribution of resources requests',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Resources"
                }
            }
        }
    ]

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