from .BuildTakagiSugeno import *
from .Clustering import *
from .EstimateAntecendentSet import *
from .EstimateConsequentParameters import *
from .LoadData import *
from .simpfulfier import *
from .SimpfulModelBuilder import *
from .Splitter import *
from .Tester import *

import numpy as np

class pyFUME(object):
    def __init__(self, datapath, nr_clus, method='Takagi-Sugeno', variable_names=None, merge_threshold=1., **kwargs):
        self.datapath=datapath
        self.nr_clus=nr_clus
        self.method=method
        self.dropped_fuzzy_sets = 0
        #self.variable_names=variable_names

        if method=='Takagi-Sugeno' or method=='Sugeno':
            self.FIS = BuildTSFIS(self.datapath, self.nr_clus, variable_names, merge_threshold=merge_threshold, **kwargs)
            self.dropped_fuzzy_sets = self.FIS._antecedent_estimator.get_number_of_dropped_fuzzy_sets()
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
        elif method=="MAE":
            return self._get_MAE()
        elif method=="MAPE":
            return self._get_MAPE()
        elif method=="RMSE":
            return self._get_RMSE()
        else:
            # return self._get_MSE()
            raise Exception("Method '%s' not implemented yet" % (method))
            
    def predict_test_data(self):
        #get the prediction for the test data set
        test = SugenoFISTester(self.FIS.model, self.FIS.x_test, self.FIS.y_test)
        pred = test.predict(variable_names=self.FIS.variable_names)
        return pred
        
    def _get_RMSE(self):

        # Calculate the mean squared error of the model using the test data set
        test = SugenoFISTester(self.FIS.model, self.FIS.x_test, self.FIS.y_test)
        RMSE = test.calculate_RMSE(variable_names=self.FIS.variable_names)
        #RMSE = list(RMSE.values())
        #print('The RMSE of the fuzzy system is', RMSE)
        return RMSE
    
    def _get_MSE(self):

        # Calculate the mean squared error of the model using the test data set
        test = SugenoFISTester(self.FIS.model, self.FIS.x_test, self.FIS.y_test)
        MSE = test.calculate_MSE(variable_names=self.FIS.variable_names)
        #RMSE = list(RMSE.values())
        #print('The RMSE of the fuzzy system is', RMSE)
        return MSE
        
    
    def _get_MAE(self):

        # Calculate the mean absolute error of the model using the test data set
        test = SugenoFISTester(self.FIS.model, self.FIS.x_test, self.FIS.y_test)
        MAE = test.calculate_MAE(variable_names=self.FIS.variable_names)
        return MAE
    
    def _get_MAPE(self):

        # Calculate the mean absolute percentage error of the model using the test data set
        test = SugenoFISTester(self.FIS.model, self.FIS.x_test, self.FIS.y_test)
        MAPE = test.calculate_MAPE(variable_names=self.FIS.variable_names)
        return MAPE

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
   
    FIS = pyFUME(datapath='Concrete_data.csv', nr_clus=3, method='Takagi-Sugeno',
     merge_threshold=.8, operators=None)
    print ("The calculated error is:", FIS.calculate_error())

    FIS.get_model().produce_figure("bla.pdf")