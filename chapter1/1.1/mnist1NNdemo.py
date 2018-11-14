import pickle, gzip

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

def get_dataset():
    # Taken from https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
    with gzip.open('../../datasets/mnist.pkl.gz', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()

        train_set_x = np.concatenate((train_set[0], valid_set[0]))
        train_set_y = np.concatenate((train_set[1], valid_set[1]))

        test_set_x, test_set_y = test_set

        return train_set_x, train_set_y, test_set_x, test_set_y

def train(train_set_x, train_set_y):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_set_x, train_set_y)
    return neigh

def shuffle_cols(train_set_x, test_set_x):
    sets = np.concatenate((train_set_x, test_set_x))
    sets = np.transpose(sets)

    np.random.shuffle(sets)

    sets = np.transpose(sets)

    return sets[:0 - len(test_set_x)], sets[0 - len(test_set_x):]


if __name__ == "__main__":
    train_set_x, train_set_y, test_set_x, test_set_y, = get_dataset()

    neigh = train(train_set_x, train_set_y)
    
    print("Non shuffled features")
    print("Misclassification rate on the first 1000 test cases: {}".format(1 - neigh.score(test_set_x[:1000], test_set_y[:1000])))
    print("Misclassification rate on all the test cases: {}".format(1 - neigh.score(test_set_x, test_set_y)))

    train_set_x, test_set_x = shuffle_cols(train_set_x, test_set_x)

    neigh = train(train_set_x, train_set_y)

    print("Shuffled features")
    print("Misclassification rate on the first 1000 test cases: {}".format(1 - neigh.score(test_set_x[:1000], test_set_y[:1000])))
    print("Misclassification rate on all the test cases: {}".format(1 - neigh.score(test_set_x, test_set_y)))
