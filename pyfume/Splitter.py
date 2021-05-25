import numpy as np
import random

class DataSplitter(object):
    """
        Creates an object that can (provide the indices to) split the data in a 
            training and test for model validation.
    """
    
#    def __init__(self):#, dataX, dataY):
#        self.data=np.loadtxt(datapath,delimiter=',')
#        self.dataX=self.data[:,0:-1]
#        self.dataY=self.data[:,-1]
#        self.dataX=dataX
#        self.dataY=dataY
        
    def holdout(self, dataX, dataY, percentage_training=0.75):
        """
            Splits the data in a training and test set using the hold-out method.
            
                Args:
                    dataX: The input data.
                    dataY: The output data (true label/golden standard).
                    percentage_training: Number between 0 and 1 that indicates the 
                        percentage of data that should be in the training data set 
                        (default = 0.75).
                    
                Returns:
                Tuple containing (x_train, y_train, x_test, y_test)
                        - x_train: Input variables of the training data.
                        - y_train: Output variables (true label/golden standard) of the training data.
                        - x_test: Input variables of the test data.
                        - y_test: Output variables (true label/golden standard) of the test data.
        """
        
        universe=set(range(0,np.shape(dataX)[0]))
        trn=np.random.choice(dataX.shape[0], int(round(percentage_training*dataX.shape[0])), replace=False)
        tst=list(universe-set(trn))

        x_train=dataX[trn]
        x_test=dataX[tst]
        y_train=dataY[trn]
        y_test=dataY[tst]
    
        return x_train, y_train, x_test, y_test

    def kfold(self, data_length, number_of_folds=10):
        """
            Provides the user with indices for 'k' number of  folds for the training 
                and testing of the model.
            
            Args:
                data_length: The total number of instances in the data sets 
                    (number of rows).
                number_of_folds: The number of folds the data should be split in 
                    (default = 10)
    
            Returns:
                A list with k (non-overlapping) sublists each containing the indices for one fold.
        """
        
        idx=np.arange(0, data_length)
        random.shuffle(idx)
         
        fold_indices= np.array_split(idx,number_of_folds)
        
        return fold_indices
    
        
