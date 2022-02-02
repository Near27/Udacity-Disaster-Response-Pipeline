# Disaster Response Pipeline Project

Udacity project in collaboration for the Data Science nanodegree. Dataset is provided by Figure Eight.

## Project description

The project aims to process a feed of twitter messages, train a ML model using labels for the already existent data and output the same type of labels for new messages through a web app. 

Examples of labels are: 

`related`, `request`, `offer`, `aid_related`, 
`medical_help`, `medical_products`, `search_and_rescue`,
`security`, `military`, `water`, `food`, 
`shelter`, `clothing`, `money`, `missing_people`, 
`refugees`, `death`, `other_aid`, `infrastructure_related`, 
`transport`, `buildings`, `electricity`, `tools`, 
`hospitals`, `shops`, `aid_centers`, `other_infrastructure`,  
`earthquake`, `cold`, `other_weather`, `direct_report`

## Project dependencies

Main programming language and package manager
- Python 3.8+
- pip - package manager - https://pypi.org/project/pip/

Data Analysis
- Pandas - https://pandas.pydata.org/
- NumPy - https://numpy.org/

Database management
- SQLAlchemy - https://www.sqlalchemy.org/

Pickle alternative to produce the classifier.pkl file.
- Dill - https://pypi.org/project/dill/

Natural language processing library
- NLTK - https://www.nltk.org/

The script ./models/train_classifier.py automatically downloads the nltk data needed.

```python
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw'])
```


## Usage

1. Process raw data. 

Raw data can be found in ./data/disaster_messages.csv and ./data/disaster_categories.csv. The first file contains twitter messages. The second file contains categories in which every message falls.

Run the ETL pipeline to merge the two raw files into a single database:

```bash
python ./data/process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv MessageCategory.db
```

2. Train Machine learning model

The machine learning model uses the sklearn Logistic Regression algorithm and tunes the parameters using GridSearchCV
Sklearn Logistic Regression documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
GridSearchCV documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Run the ML pipeline to produce the message classifier:

```bash
python ./models/train_classifier.py ./MessageCategory.db classifier.pk
```

3. Run the web application

The web application is created using the Flask framework.
Flask documentation: https://flask.palletsprojects.com/en/2.0.x/

To create a local web server to see the application, run the following command:

```bash
python ./app/run.py
```

Then open your browser at http://0.0.0.0:3001/
