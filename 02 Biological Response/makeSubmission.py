from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import logloss
import numpy as np


def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])
    test = np.genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=1500, n_jobs=2)
    cv = cross_validation.KFold(len(train), n_folds=63, indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = rf.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() )
   
    #generate predictions and save to file
    predicted_probs = [x[1] for x in rf.predict_proba(test)]
    np.savetxt('Data/submission.csv', predicted_probs, delimiter=',', fmt='%f')

if __name__=="__main__":
    main()
