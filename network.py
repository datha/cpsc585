import numpy as np
from sknn.mlp import Classifier, Layer
from sknn.platform import gpu32
import sys
import logging
import pickle



def main():

    logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)
    data = np.load('nn_input.npy')
    #print data

    uid, y_train, X_train = data[0,:].transpose(), data[1,:].transpose(), data[2:,:].transpose()
    nn = Classifier(
        layers=[
            Layer("Maxout", name="h1", units=200, pieces=2,dropout=.1),
            Layer("Maxout", name="h2", units=100, pieces=2,dropout=.1),
            Layer("Softmax")],
        learning_rate=0.001,
        n_iter=8)

    print "Trying to fit data"
    train_index = int(X_train.shape[0] * .8)
    try:
        nn.fit(X_train[:train_index], y_train[:train_index])
    except KeyboardInterrupt:
        pass
    #nn = pickle.load(open('nn.pkl', 'rb'))
    pickle.dump(nn, open('nn.pkl', 'wb'))

    print('score =', nn.score(X_train[train_index:], y_train[train_index:]))
    #np.savetxt('x_train.out', x_train[:10], delimiter=',',fmt='%i')
    #np.savetxt('y_train.out', y_train[:10], delimiter=',',fmt='%i')
if __name__ == "__main__":
    main()
