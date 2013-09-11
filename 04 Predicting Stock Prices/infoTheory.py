import numpy as np
import itertools
# Taken from http://orange.biolab.si/blog/2012/06/15/joint-entropy-in-python/

# X MUST BE A NUMPY ARRAY!
def entropy(*X):
    return np.sum(-p * np.log2(p) if p > 0 else 0 for p in
        (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
            for classes in itertools.product(*[set(x) for x in X])))

def information(X,Y):
    return entropy(X) + entropy(Y) - entropy(X,Y)
