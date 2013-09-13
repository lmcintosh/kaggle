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
    #use stoch grad descent to fit linear models of the last observed value for each stock, using all t, t+2hr data
    iterations   = 700
    learningRate = 0.01
    h            = (2e-15)**(1./3)
    coeff        = np.zeros((sum(isOutput),3))
    coeff[:,1]   = np.squeeze(np.ones((sum(isOutput),1)))
    error        = []
    gradient     = [0,0,0]
    twoHrs = 120/5
    maxLag = numRows - (twoHrs+1)  # last datapoint x where 2hrs from x is recorded
    pred0P = np.zeros((len(trainingDays),sum(isOutput)))
    pred0M = np.zeros((len(trainingDays),sum(isOutput)))
    pred1P = np.zeros((len(trainingDays),sum(isOutput)))
    pred1M = np.zeros((len(trainingDays),sum(isOutput)))
    pred2P = np.zeros((len(trainingDays),sum(isOutput)))
    pred2M = np.zeros((len(trainingDays),sum(isOutput)))
    pred   = np.zeros((len(trainingDays),sum(isOutput)))
    for stock in range(0,sum(isOutput)):
        for day in range(0,len(trainingDays)):
            pred[day,stock]   = (coeff[stock,1])*trainOutput[day,-1,stock] + (coeff[stock,0])

    for stock in range(0,sum(isOutput)):
        for i in range(0,iterations):
            for day in range(0,len(trainingDays)):
                for row in range(1,maxLag):
                    deriv = trainOutput[day,row,stock] - trainOutput[day,row-1,stock]
                    pred0Pi = (coeff[stock,1])*trainOutput[day,row,stock] + (coeff[stock,2])*deriv + (coeff[stock,0] + h)
                    pred0Mi = (coeff[stock,1])*trainOutput[day,row,stock] + (coeff[stock,2])*deriv + (coeff[stock,0] - h)
                    pred1Pi = (coeff[stock,1]+h)*trainOutput[day,row,stock] +  (coeff[stock,2])*deriv + (coeff[stock,0])
                    pred1Mi = (coeff[stock,1]-h)*trainOutput[day,row,stock] +  (coeff[stock,2])*deriv +(coeff[stock,0])
                    pred2Pi = (coeff[stock,1])*trainOutput[day,row,stock] +  (coeff[stock,2]+h)*deriv + (coeff[stock,0])
                    pred2Mi = (coeff[stock,1])*trainOutput[day,row,stock] +  (coeff[stock,2]-h)*deriv +(coeff[stock,0])
                    #pred   = (coeff[stock,1])*trainOutput[day,row,stock] + (coeff[stock,2])*deriv +(coeff[stock,0])
                    gradient[0] = (err.maeFun(trainOutput[day,row+maxLag,stock],pred0Pi) - err.maeFun(trainOutput[day,row+maxLag,stock],pred0Mi))/(2*h)
                    gradient[1] = (err.maeFun(trainOutput[day,row+maxLag,stock],pred1Pi) - err.maeFun(trainOutput[day,row+maxLag,stock],pred1Mi))/(2*h)
                    gradient[2] = (err.maeFun(trainOutput[day,row+maxLag,stock],pred2Pi) - err.maeFun(trainOutput[day,row+maxLag,stock],pred2Mi))/(2*h)
                    coeff[stock,:] = [coeff[stock,x] - learningRate*gradient[x] for x in range(0,len(gradient))]

                deriv = trainOutput[day,-1,stock] - trainOutput[day,-2,stock]
                pred0P[day,stock] = (coeff[stock,1])*trainOutput[day,-1,stock] + (coeff[stock,2])*deriv + (coeff[stock,0] + h)
                pred0M[day,stock] = (coeff[stock,1])*trainOutput[day,-1,stock] + (coeff[stock,2])*deriv + (coeff[stock,0] - h)
                pred1P[day,stock] = (coeff[stock,1]+h)*trainOutput[day,-1,stock] +  (coeff[stock,2])*deriv + (coeff[stock,0])
                pred1M[day,stock] = (coeff[stock,1]-h)*trainOutput[day,-1,stock] +  (coeff[stock,2])*deriv +(coeff[stock,0])
                pred2P[day,stock] = (coeff[stock,1])*trainOutput[day,-1,stock] +  (coeff[stock,2]+h)*deriv + (coeff[stock,0])
                pred2M[day,stock] = (coeff[stock,1])*trainOutput[day,-1,stock] +  (coeff[stock,2]-h)*deriv +(coeff[stock,0])
                pred[day,stock]   = (coeff[stock,1])*trainOutput[day,-1,stock] + (coeff[stock,2])*deriv +(coeff[stock,0])

            gradient[0] = (err.maeFun(target,pred0P[0:200,:]) - err.maeFun(target,pred0M[0:200,:]))/(2*h)
            gradient[1] = (err.maeFun(target,pred1P[0:200,:]) - err.maeFun(target,pred1M[0:200,:]))/(2*h)
            gradient[2] = (err.maeFun(target,pred2P[0:200,:]) - err.maeFun(target,pred2M[0:200,:]))/(2*h)
            coeff[stock,:] = [coeff[stock,x] - learningRate*gradient[x] for x in range(0,len(gradient))]
            print "Stock: " + str(stock) + " Error: " + str(err.maeFun(target[:,stock],pred[0:200,stock])) + " Coeff: " + str(coeff[stock,:])
    
        error.append(err.maeFun(target,pred[0:200,:])) #maeFun(actual,pred)
        print "Results: " + str(error[-1]) + " Variance of coefficients: " + str([np.var(coeff[:,0]), np.var(coeff[:,1])])


    #generate predictions and save to file
    np.savetxt('Data/submission'+str(datetime.date.today())+'.csv', pred[200:510,:], delimiter=',', fmt='%f')  #predictions for file 201 to 510

if __name__=="__main__":
    main()

