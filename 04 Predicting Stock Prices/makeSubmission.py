from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import meanAbsoluteError as err
import numpy as np
import datetime
import random

def main():
    #DATA PROCESSING
    #create the training & test sets, skipping the header row with [1:]
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


    #DATA EXPLORATION
    #use exponential moving average and grad descent to find discountRate
    iterations = 20
    learningRate = 1
    discountRate = [0.552329, 0.5523]
    error = [0.43811969431]
    for i in range(0,iterations):
        pred = np.zeros((len(trainingDays),sum(isOutput)))
        for day in range(0,len(trainingDays)):
            for stock in range(0,sum(isOutput)):
                for j in range(0,numRows):
                    pred[day,stock] = discountRate[-1]*trainOutput[day,j,stock] + (1-discountRate[-1])*pred[day,stock]

        error.append(err.maeFun(target,pred[0:200,:])) #maeFun(actual,pred)
        print "Results: " + str(error[-1]) + " Discount Rate: " + str(discountRate[-1])
        discountRate.append(discountRate[-1] - learningRate*(error[-2] - error[-1])/(discountRate[-2] - discountRate[-1]))


    #generate predictions and save to file
    np.savetxt('Data/submission'+str(datetime.date.today())+'.csv', pred[200:510,:], delimiter=',', fmt='%f')  #predictions for file 201 to 510

if __name__=="__main__":
    main()
