from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import logloss
import numpy as np
import matplotlib.pyplot as plt



#create the training & test sets, skipping the header row with [1:]
dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
target = np.array([x[0] for x in dataset])
train = np.array([x[1:] for x in dataset])
test = np.genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]

ITERATIONS = 50;
LEARNING_RATE = 10000;
N_TREES = [100,200];
N_CVS = [5,10];
loglosses_Trees = [0.5];
loglosses_CVs = [0.5];

for i in range(0,ITERATIONS):
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=N_TREES[-1], n_jobs=2)
    cv = cross_validation.KFold(len(train), n_folds=N_CVS[-2], indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = rf.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    newResult_Trees = np.array(results).mean()

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=N_TREES[-2], n_jobs=2)
    cv = cross_validation.KFold(len(train), n_folds=N_CVS[-1], indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = rf.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    newResult_CVs = np.array(results).mean()



    print "Results: " + str( [newResult_Trees, newResult_CVs] )
    print "Trees and Folds: " + str( [N_TREES[-1], N_CVS[-1]] )

    #keep track of log loss
    loglosses_Trees.append(newResult_Trees)
    loglosses_CVs.append(newResult_CVs)
    
    #compute gradient
    direction_Trees = loglosses_Trees[-2] - loglosses_Trees[-1]
    direction_CVs = loglosses_CVs[-2] - loglosses_CVs[-1]
    N_TREES.append( np.int(N_TREES[-1] + LEARNING_RATE*direction_Trees) )
    N_CVS.append( np.int(N_CVS[-1] + LEARNING_RATE*0.1*direction_CVs) )


    
plt.plot(trees,loglosses)
plt.show()

plt.plot(cvs,loglosses)
plt.show()



#generate predictions and save to file
#predicted_probs = [x[1] for x in rf.predict_proba(test)]
#np.savetxt('Data/submission.csv', predicted_probs, delimiter=',', fmt='%f')
     

#results seem to converge at
#N_TREES: 
#N_FOLDS: 

