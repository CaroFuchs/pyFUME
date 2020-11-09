import numpy as np

class DataSplitter(object):
    def __init__(self, dataX, dataY):
#        self.data=np.loadtxt(datapath,delimiter=',')
#        self.dataX=self.data[:,0:-1]
#        self.dataY=self.data[:,-1]
        self.dataX=dataX
        self.dataY=dataY
        
    def holdout(self, percentage_training=0.75,seed=None):
        universe=set(range(0,np.shape(self.dataX)[0]))
        trn=np.random.choice(self.dataX.shape[0], int(round(percentage_training*self.dataX.shape[0])), replace=False)
        tst=list(universe-set(trn))
    
        x_train=self.dataX[trn]
        x_test=self.dataX[tst]
        y_train=self.dataY[trn]
        y_test=self.dataY[tst]
        
        return x_train, y_train, x_test, y_test


    