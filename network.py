import numpy as np
from sknn.mlp import Classifier, Layer
from sknn.platform import gpu32
import sys
import logging
import pickle
import argparse
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import csv

parser = argparse.ArgumentParser(description='Train network.')
parser.add_argument('action', metavar='N', type=str,
                    help='[train, predict]')
parser.add_argument('--loadfile', dest='loadfile', action='store',
                    type=str,
                    help='pkl file to load.')

args = parser.parse_args()

class_mapper = {'irish': 0, 'mexican': 1, 'chinese': 2,
                'filipino': 3, 'vietnamese': 4, 'spanish': 13, 'japanese': 7,
                'moroccan': 5, 'french': 12, 'greek': 9, 'indian': 10,
                'jamaican': 11, 'british': 8, 'brazilian': 6, 'russian':
                14, 'cajun_creole': 15, 'thai': 16, 'southern_us': 17,
                'korean': 18, 'italian': 19}

#reverse mapping from string to int
class_mapper = dict(zip(class_mapper.values(), class_mapper.keys()))



def main():
    NUMPY_INPUT_FILE_TEST = 'nn_test.npy'
    NUMPY_INPUT_FILE = 'nn_train_39774.npy'
    #NUMPY_INPUT_FILE = 'nn_train_140000.npy'
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        stream=sys.stdout)

    data = np.load(NUMPY_INPUT_FILE)
    data_test = np.load(NUMPY_INPUT_FILE_TEST)
    print "Total Train Samples %d" % len(data.transpose())
    print "Total Test Samples %d" % len(data_test.transpose())
    uid, y_train, X_train = data[
        0,:].transpose(), data[1,:].transpose(), data[2:,:].transpose()
    uid_test,  X_test = data_test[
        0,:].transpose(), data_test[1:,:].transpose()
    train_index = int(X_train.shape[0] * .8)
    print "Using %d samples to train" % train_index

    # X_train, y_train, X_valid, y_valid = X_train[:train_index],
    # y_train[:train_index], X_train[train_index:], y_train[train_index:]

    # The most basic method of hyper paramater search
    # learning reate, batch size
    nn = Classifier(
        layers=[
            Layer("Maxout", units=450, pieces=2),  # ,dropout=.1),
            # Layer("Tanh",units=300),#,dropout=.1),
            Layer("Softmax")],
        learning_rate=0.1,
        # weight_decay=.0001,
        dropout_rate=.05,
        # random_state= ,
        learning_momentum=.5,
        valid_size=.2,
        batch_size=100,
        n_stable=5,
        f_stable=.001,
        # valid_set=(X_valid,y_valid),
        n_iter=60)
    if args.loadfile:
        gs = pickle.load(open(args.loadfile, 'rb'))
    else:

        gs = GridSearchCV(nn, cv=3, scoring='accuracy', param_grid={
            'hidden0__units': [450,600],
            #'hidden0__units': [150,300,450],
            'hidden0__type': ["Maxout"],
            'hidden0__pieces': [2],
            'learning_rate': [0.05],
            'batch_size': [100],
            'dropout_rate': [.05]})
    if args.action == 'train':
        try:

            gs.fit(X_train, y_train)
            #nn.fit(X_train, y_train)

            print gs.grid_scores_
            print gs.best_score_
            print gs.best_params_
            
            nn_filename = 'nnout/nn_%s.pkl' % 1001
            gs_filename = 'nnout/gs_%s.pkl' % 1001

            pickle.dump(nn, open(gs_filename, 'wb'))
            pickle.dump(gs, open(nn_filename, 'wb'))

            print "Saving %s" % nn_filename
            print "Saving %s" % gs_filename

        except KeyboardInterrupt:
            pickle.dump(gs, open('gs_temp.pkl', 'wb'))
            pickle.dump(nn, open('nnout/nn_%s.pkl' % 'trained', 'wb'))
            pass
    elif args.action == 'predict':
        y_pred = gs.predict(X_test)
        y_pred_mapped = [class_mapper[y] for y in y_pred.flat]
        results = np.append([uid_test.astype(str)], [y_pred_mapped], axis=0)
        print "Saving test.csv"
        np.savetxt("test.csv", results.transpose(), "%s,%s", header="id,cuisine")



if __name__ == "__main__":
    main()
