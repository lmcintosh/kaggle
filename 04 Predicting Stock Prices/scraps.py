#scraps


#use last price as prediction
    #pred = np.zeros((len(trainingDays),sum(isOutput)))
    #for day in range(0,len(trainingDays)):
    #    for stock in range(0,sum(isOutput)):
    #        pred[day,stock] = trainOutput[day,-1,stock]



    #use exponential moving average and grad descent to find discountRate
    #iterations = 5
    #learningRate = 1
    #discountRate = [0.552329, 0.5523]
    #error = [0.43811969431]
    #for i in range(0,iterations):
    #    pred = np.zeros((len(trainingDays),sum(isOutput)))
    #    for day in range(0,len(trainingDays)):
    #        for stock in range(0,sum(isOutput)):
    #            for j in range(0,numRows):
    #                pred[day,stock] = discountRate[-1]*trainOutput[day,j,stock] + (1-discountRate[-1])*pred[day,stock]

    #    error.append(err.maeFun(target,pred[0:200,:])) #maeFun(actual,pred)
    #    print "Results: " + str(error[-1]) + " Discount Rate: " + str(discountRate[-1])
    #    discountRate.append(discountRate[-1] - learningRate*(error[-2] - error[-1])/(discountRate[-2] - discountRate[-1]))


    #discountRate = [0.9]
    #pred = np.zeros((len(trainingDays),sum(isOutput)))
    #for day in range(0,len(trainingDays)):
    #        for stock in range(0,sum(isOutput)):
    #            for j in range(0,numRows):
    #                pred[day,stock] = discountRate[-1]*trainOutput[day,j,stock] + (1-discountRate[-1])*pred[day,stock]
    #
    #error = err.maeFun(target,pred[0:200,:])
    #print "Results: " + str(error) + " Discount Rate: " + str(discountRate[-1])


    iterations   = 100
    learningRate = 0.000001
    h            = (2e-15)**(1./3)
    coeff        = [0,1,0]
    error        = []
    gradient     = [0,0,0]
    for i in range(0,iterations):
        pred0P = np.zeros((len(trainingDays),sum(isOutput)))
        pred0M = np.zeros((len(trainingDays),sum(isOutput)))
        pred1P = np.zeros((len(trainingDays),sum(isOutput)))
        pred1M = np.zeros((len(trainingDays),sum(isOutput)))
        pred2P = np.zeros((len(trainingDays),sum(isOutput)))
        pred2M = np.zeros((len(trainingDays),sum(isOutput)))
        #pred3P = np.zeros((len(trainingDays),sum(isOutput)))
        #pred3M = np.zeros((len(trainingDays),sum(isOutput)))
        pred   = np.zeros((len(trainingDays),sum(isOutput)))
        for day in range(0,len(trainingDays)):
            for stock in range(0,sum(isOutput)):
                pred0P[day,stock] = (coeff[2])*(trainOutput[day,-1,stock])**2 + (coeff[1])*trainOutput[day,-1,stock] + (coeff[0] + h)
                pred0M[day,stock] = (coeff[2])*(trainOutput[day,-1,stock])**2 + (coeff[1])*trainOutput[day,-1,stock] + (coeff[0] - h)
                pred1P[day,stock] = (coeff[2])*(trainOutput[day,-1,stock])**2 + (coeff[1]+h)*trainOutput[day,-1,stock] + (coeff[0])
                pred1M[day,stock] = (coeff[2])*(trainOutput[day,-1,stock])**2 + (coeff[1]-h)*trainOutput[day,-1,stock] + (coeff[0])
                pred2P[day,stock] = (coeff[2]+h)*(trainOutput[day,-1,stock])**2 + (coeff[1])*trainOutput[day,-1,stock] + (coeff[0])
                pred2M[day,stock] = (coeff[2]-h)*(trainOutput[day,-1,stock])**2 + (coeff[1])*trainOutput[day,-1,stock] + (coeff[0])
                #pred3P[day,stock] = (coeff[3]+h)*(trainOutput[day,-1,stock])**3 + (coeff[2])*(trainOutput[day,-1,stock])**2 + (coeff[1])*trainOutput[day,-1,stock] + (coeff[0])
                #pred3M[day,stock] = (coeff[3]-h)*(trainOutput[day,-1,stock])**3 + (coeff[2])*(trainOutput[day,-1,stock])**2 + (coeff[1])*trainOutput[day,-1,stock] + (coeff[0])
                pred[day,stock]   = (coeff[2])*(trainOutput[day,-1,stock])**2 + (coeff[1])*trainOutput[day,-1,stock] + (coeff[0])

        gradient[0] = (err.maeFun(target,pred0P[0:200,:]) - err.maeFun(target,pred0M[0:200,:]))/(2*h)
        gradient[1] = (err.maeFun(target,pred1P[0:200,:]) - err.maeFun(target,pred1M[0:200,:]))/(2*h)
        gradient[2] = (err.maeFun(target,pred2P[0:200,:]) - err.maeFun(target,pred2M[0:200,:]))/(2*h)
        #gradient[3] = (err.maeFun(target,pred3P[0:200,:]) - err.maeFun(target,pred3M[0:200,:]))/(2*h)
       
        coeff = [coeff[x] - learningRate*gradient[x] for x in range(0,len(gradient))]

        error.append(err.maeFun(target,pred[0:200,:])) #maeFun(actual,pred)
        print "Results: " + str(error[-1]) + " Coefficients: " + str(coeff)
