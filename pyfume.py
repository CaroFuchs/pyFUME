from BuildTakagiSugeno import BuildTSFIS
from Tester import SugenoFISTester
import numpy as np

class pyFUME(object):
    def __init__(self, datapath, nr_clus, method='Takagi-Sugeno', variable_names=None, **kwargs):
        self.datapath=datapath
        self.nr_clus=nr_clus
        self.method=method
        #self.variable_names=variable_names

        if method=='Takagi-Sugeno' or method=='Sugeno':
            self.FIS = BuildTSFIS(self.datapath, self.nr_clus, variable_names, **kwargs)
        else:
            raise Exception ("This modeling technique has not yet been implemented.")

    def get_model(self):
        if self.FIS.model is None:
            print ("ERROR: model was not created correctly, aborting.")
            exit(-1)
        else:
            return self.FIS.model

    def calculate_error(self, method="RMSE"):

        if method=="RMSE":
            return self._get_RMSE()
        else:
            # return self._get_MSE()
            raise Exception("Method '%s' not implemented yet" % (method))
        
    def _get_RMSE(self):

        # Calculate the mean squared error of the model using the test data set
        test = SugenoFISTester(self.FIS.model, self.FIS.x_test, self.FIS.y_test)
        RMSE = test.calculate_RMSE(variable_names=self.FIS.variable_names)
        #RMSE = list(RMSE.values())
        #print('The RMSE of the fuzzy system is', RMSE)
        return dict(RMSE)
        

    """
    def _get_RMSE(self):
        if self.FIS.error is None:
            print ("ERROR: RMSE was not calculated correctly, aborting.")
            exit(-1)
        else:
            return self.FIS.RMSE
    """ 

if __name__=='__main__':
    from numpy.random import seed
    seed(4)
   
    FIS = pyFUME(datapath='Concrete_data.csv', nr_clus=3, method='Takagi-Sugeno', operators=None)
    print ("The calculated error is:", FIS.calculate_error())