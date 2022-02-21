import sys
import re
import dill as pickle
from functools import partial

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw'])

# performance measure
from sklearn.metrics import classification_report

# ML algorithms
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

# Pipeline & model selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath, table_name='MessageCategory'):
    '''
    INPUT:
    database_filepath (str) - file path of the database to be analyzed
    table_name (str) - table name for the main data

    OUTPUT:
    X - numpy array representing messages
    y - numpy array representing the value of each label(feature) for each message
    feature_list str[] - an array containing every feature in string format

    Description:
    This function retrieves the main data and splits it into messages, outcomes (features)
    and list of the features in string format
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name=table_name, con=engine)

    feature_list = ['related', 'request', 'offer',
                    'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                    'security', 'military', 'water', 'food', 'shelter',
                    'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                    'infrastructure_related', 'transport', 'buildings', 'electricity',
                    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                    'other_weather', 'direct_report']

    X = df.message.values
    y = np.c_[df[feature_list]]

    return X, y, feature_list


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    '''
    INPUT:
    test (str) - text to be tokenized

    OUTPUT:
    clean_tokens - array of processed tokens

    Description:
    This function returns an array of processed tokens extracted from the text parameter
    '''

    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize

    # remove parts of a url structure
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # remove special characters
    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    # tokenize the sentence
    tokens = word_tokenize(text)

    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        nonsense_flag = False

        # check if the token is part of the wordnet lexical database in order to avoid nonsensical words
        if wordnet.synsets(tok):
            nonsense_flag = False
        else:
            nonsense_flag = True

        if not nonsense_flag:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

def create_fd_fatal(X, y):
    '''
    INPUT:
    X - numpy array containing messages
    y - numpy array containing categories

    OUTPUT:
    fd_fatal - nltk.FreqDist object

    Description:
    This function creates a frequency distribution object. It is created using the messages which
    fall into 'died' category using the nltk module 'FreqDist'
    '''

    import nltk

    # get the index of 'died' category from y_train
    fatality_category_index = 16

    # get row corresponding to the 'died' category
    fatal_row_idx = (y[:, :][:, [fatality_category_index]] == 1)

    # convert X_train to df
    X_train_df = pd.DataFrame(X)
    # select rows which indicate fatalities
    X_train_fatal = np.array(X_train_df[fatal_row_idx])

    # tokenize each row and add tokens into a single list
    tokens_fatal = []

    for row in X_train_fatal:
        tokens = []
        tokens = tokenize(row[0])
        tokens_fatal = tokens_fatal + tokens

    # create word frequency object out of fatal row tokens
    fd_fatal = nltk.FreqDist(tokens)

    return fd_fatal

class HasFatalities(BaseEstimator, TransformerMixin):
    """
    A class representing a custom transformer to be used in our pipeline.
    Using a pretrained nltk.FreqDist object, it tags each document as True or False
    depending weather it contains tokens found in the test training dataset relating
    to messages which fall into the 'died' category

    Attributes
    ----------
    fd_fatal : nltk.FreqDist module pretrained

    Methods
    -------
    has_fatalities(text):
        Returns True if one of the tokens is found to be in the fd_fatal attribute
    """

    fd_fatal = None

    def __init__(self, fd_fatal=None):
        self.fd_fatal = fd_fatal

    def has_fatalities(self, text):
        tokens = tokenize(text)
        for token in tokens:
            return (self.fd_fatal[token] != 0)
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.has_fatalities)
        return pd.DataFrame(X_tagged)


def build_model(X, y):
    '''
    INPUT:
    X - numpy array containing messages
    y - numpy array containing categories

    OUTPUT:
    cv - GridSearchCV module containing our pipeline and parameters to be tuned

    Description:
    In this function the main Pipeline is configured and added to GridSearchCV along with
    parameters to be tuned for its transformers and classifier.
    '''

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    # train nltk.FreqDist
    fd_fatal = create_fd_fatal(X, y)

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=partial(tokenize))),
                ('tfidf', TfidfTransformer())
            ])),

            ('has_fatalities', HasFatalities(fd_fatal))
        ])),

        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])

    parameters = {
        # ngram size
        'features__text_pipeline__vect__ngram_range': [(1, 2), (1, 3)],
        # instead of counting the words, set 1 to a found feature
        # 'features__text_pipeline__vect__binary': [False, True],
        # ignore terms with frequency higher than the maximum
        'features__text_pipeline__vect__max_df': (0.9, 1.0),
        # ignore terms with frequency lower than the minimum
        'features__text_pipeline__vect__min_df': (0.01, 0.05),

        'features__text_pipeline__tfidf__use_idf': (True, False),
        
        # # use different types of regularization in order to check if overfitting is a problem in our case
        'clf__estimator__penalty': ['l2'],  # ,'l1', 'l2'
        # 'clf__estimator__max_iter': [500],
        # nu ne inteleg cu asta

        'clf__estimator__solver': ['liblinear', 'sag'],# ,'sag' sag only supports l2 or none / liblinear doesn't work with nan or does it?

        'features__transformer_weights': (
            {'text_pipeline': 1, 'has_fatalities': 0},
            {'text_pipeline': 0.5, 'has_fatalities': 0.5},
            {'text_pipeline': 0.9, 'has_fatalities': 0.1},
        )
    }

    print(pipeline.get_params())
    # todo: comment cv param
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, cv=2)

    return cv


def evaluate_model(model, X_test, y_test):
    '''
    INPUT:
    model - machine learning pipeline containing the best GridSearchCV estimator
    X_test - numpy array containing test messages
    y_test - numpy array containing test categories

    OUTPUT:
    reports - array of reports computed with classification_report module
    from sklearn.metrics

    Description:
    This function runs and print a classification report for the category predictions
    for each category against the testing data.
    '''

    y_pred = model.predict(X_test)

    # calculate predictions against y_test
    reports = []
    for i, col in enumerate(y_pred.T[:5]):
        report = classification_report(y_test.T[i], col, zero_division=0, output_dict=True)
        reports.append(report)

    return reports

def save_model(model, model_filepath):
    '''
    INPUT:
    model - machine learning pipeline containing the best GridSearchCV estimator
    model_filepath (str) - file path of our pickled classifier

    OUTPUT:
    file_paths (str) - file path for the newly pickled classifier

    Description:
    This function has the main task to pickle our ML pipeline in order to be used in
    an app
    '''

    file_path = pickle.dump(model, open(model_filepath, "wb"))
    return file_path


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...', X_train.shape, Y_train.shape)
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)

        if hasattr(model, 'best_estimator_'):
            print("BEST ESTIMATOR", model.best_estimator_)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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