# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:09:21 2015

@author: Dipayan
"""

from sknn.mlp import Classifier, Layer
#from sknn.platform import gpu32
import sys
import logging

from sklearn.grid_search import GridSearchCV
#from sklearn.metrics import classification_report
from time import time

#import csv
from pprint import pprint
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
import logging
# from sklearn.metrics import classification_report
# import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
# Display progress logs on stdout
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger()
print "Reading in data"
traindf = pd.read_json("train.json")
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf = pd.read_json("test.json") 
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

print "Finished reading in data"
nn = Classifier(
    layers=[
        Layer("Maxout", units=1000, pieces=2),
        Layer("Maxout", units=450, pieces=2),
        Layer("Maxout",units=150, pieces=2),
        Layer("Softmax")],
    learning_rate=0.01,
    # weight_decay=.0001,
    dropout_rate=.05,
    # random_state= ,
    learning_momentum=0.9,
    valid_size=.2,
    batch_size=100,
    n_stable=20,
    f_stable=.001,
    # valid_set=(X_valid,y_valid),
    n_iter=100)

vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)

pipeline = Pipeline([
    ('tfidf', vectorizertr),
    ('nn', nn),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'nn__hidden0__units': [1000,1100],
    'nn__hidden0__type': ["Maxout"],
    'nn__hidden0__pieces': [2,3],
    'nn__hidden1__units': [400,450],
    'nn__hidden1__type': ["Maxout"],
    'nn__hidden1__pieces': [2,3],
    'nn__hidden2__units': [100,150],
    'nn__hidden2__type': ["Maxout"],
    'nn__hidden2__pieces': [2,3],
    'nn__dropout_rate':[.05],
    'nn__loss_type' : ["mae"],
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1','l2'),
    
}

print "Performing grid search..."
print "pipeline:", [name for name, _ in pipeline.steps]
print "parameters:"
gs = GridSearchCV(pipeline,parameters, cv=5, scoring='accuracy',verbose=3)
pprint(parameters)
t0 = time()
gs.fit(traindf['ingredients_string'],traindf['cuisine'])
print "done in %0.3fs" % (time() - t0)
print

print "Best score: %0.4f" % gs.best_score_
print "Best parameters set:"

best_parameters = gs.best_estimator_.get_params()
pprint(best_parameters)

predictions=gs.predict(testdf['ingredients_string'])
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)
#testdf[['id' , 'cuisine' ]].to_csv("test.csv")