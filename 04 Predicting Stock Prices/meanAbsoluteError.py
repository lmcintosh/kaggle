import numpy as np

def maeFun(actual, pred):
    actual = np.ravel(actual)  #make 1d
    pred = np.ravel(pred)
    mae = sum(abs(actual - pred))
    mae = float(mae)/float(len(actual))
    return mae
