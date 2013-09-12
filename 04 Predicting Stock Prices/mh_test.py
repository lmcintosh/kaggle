from linearMAE import LinearMAE
from numpy import array
import numpy as np
import meanAbsoluteError as err
import datetime

trainingDays = range(1,511) #200 days of training data, 510 total days
headers = np.genfromtxt(open('Data/data/1.csv','r'),delimiter=',',dtype='string')[0]
numRows = (np.array(np.genfromtxt(open('Data/data/1.csv','r'),delimiter=',',dtype='f8')[1:])).shape[0]
numCols = (np.array(np.genfromtxt(open('Data/data/1.csv','r'),delimiter=',',dtype='f8')[1:])).shape[1]
isOutput = [headers[x][0] == 'O' for x in range(0,numCols)]
isInput  = [headers[x][0] == 'I' for x in range(0,numCols)]
trainOutput = np.zeros((len(trainingDays),numRows,sum(isOutput)))
trainInput  = np.zeros((len(trainingDays),numRows,sum(isInput)))
for i in trainingDays:
    dataset = np.genfromtxt(open('Data/data/'+str(i)+'.csv','r'), delimiter=',', dtype='f8')[1:]
    dataset = np.array(dataset)  # (5minIncrement,stock/feature)
    for j in range(0,numCols):
        if headers[j][0] == 'O':
            trainOutput[i-1,:,j] = dataset[:,j]   # (day,5minIncrement,stock)
        elif headers[j][0] == 'I':
            trainInput[i-1,:,(j-sum(isOutput))] = dataset[:,j]    # (day,5minIncrement,feature)

#target prices 2 hours later (only outputs, no inputs) 
target = np.array(np.genfromtxt(open('Data/trainLabels.csv','r'), delimiter=',', dtype='f8')[1:])
target = target[:,1:]  # (day,price2HrsLater)


#X = array([[0,1,2,4], [2,1,0,5]])
#y = array([[0,1], [2,3]])

X = trainOutput[0:200,:,:]
X = X.reshape(X.shape[0]*X.shape[2],X.shape[1])
y = target.reshape(target.shape[0]*target.shape[1],1)


lin = LinearMAE(l1=1.0, l2=0.0, verbose=True, opt='cg', maxiter=1000)
lin.fit(X,y)

test = trainOutput[200:511,:,:]
testPrep = test.reshape(test.shape[0]*test.shape[2],test.shape[1])
pred = lin.predict(testPrep)
pred = pred.reshape(test.shape[0],test.shape[2])

np.savetxt('Data/submission'+str(datetime.date.today())+'_mhTest.csv', pred, delimiter=',', fmt='%f')  #predictions for file 201 to 510

#print
#print 'Prediction' 
#print lin.predict(X)
#print 'Target'
#print y
