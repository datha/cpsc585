from sknn.mlp import Classifier, Layer

import logging
from sklearn.grid_search import GridSearchCV
from time import time
from pprint import pprint

import pandas as pd
import numpy as np

import re

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Display progress logs on stdout
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger()
print "Reading in data"
traindf = pd.read_json("train.json")
traindf['ingredients_clean_string'] = [
    ' , '.join(z).strip() for z in traindf['ingredients']]
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub(
    '[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]

traindf, testdf = train_test_split(traindf, test_size=0.2)
# used to submit to kaggle
# testdf = pd.read_json("test.json")
# testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
# testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]

print "Finished reading in data"
nn = Classifier(
    layers=[
        Layer("Maxout", units=1100, pieces=2),
        Layer("Maxout", units=450, pieces=2),
        Layer("Maxout", units=150, pieces=2),
        Layer("Softmax")],
    learning_rate=0.01,

    dropout_rate=.05,
    learning_momentum=0.9,
    valid_size=.2,
    batch_size=100,
    n_stable=50,
    f_stable=.001,
    n_iter=100)

vectorizertr = TfidfVectorizer(stop_words='english',
                               ngram_range=(1, 1), analyzer="word",
                               max_df=.57, binary=True, token_pattern=r'\w+', sublinear_tf=False)

pipeline = Pipeline([
    ('tfidf', vectorizertr),
    ('nn', nn),
])


parameters = {
    'nn__hidden0__units': [1100],
    'nn__hidden0__type': ["Maxout"],
    'nn__hidden0__pieces': [2],
    'nn__hidden1__units': [450],
    'nn__hidden1__type': ["Maxout"],
    'nn__hidden1__pieces': [2],
    'nn__hidden2__units': [150],
    'nn__hidden2__type': ["Maxout"],
    'nn__hidden2__pieces': [2],
    'nn__dropout_rate': [.05],
    'nn__loss_type': ["mae"],
    'tfidf__use_idf': [True],
    'tfidf__norm': ['l2'],
    'tfidf__binary': [True],

}

print "Performing grid search..."
print "pipeline:", [name for name, _ in pipeline.steps]
print "parameters:"
gs = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy', verbose=3)
pprint(parameters)
t0 = time()
gs.fit(traindf['ingredients_string'], traindf['cuisine'])
print "done in %0.3fs" % (time() - t0)
print

print "Best score: %0.4f" % gs.best_score_
print "Best parameters set:"

best_parameters = gs.best_estimator_.get_params()
pprint(best_parameters)

predictions = gs.predict(testdf['ingredients_string'])


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    cuisine_array = np.unique(traindf.as_matrix(columns=['cuisine']))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cuisine_array))
    plt.xticks(tick_marks, cuisine_array, rotation=45)
    plt.yticks(tick_marks, cuisine_array)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(testdf['cuisine'], predictions)
np.set_printoptions(precision=2)
print 'Confusion matrix, without normalization'
# print cm
plt.figure()
plot_confusion_matrix(cm)


cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
# print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
# Used to test submit to kaggle
#testdf['cuisine'] = predictions
#testdf = testdf.sort('id', ascending=True)
#testdf[['id' , 'cuisine' ]].to_csv("test.csv")
