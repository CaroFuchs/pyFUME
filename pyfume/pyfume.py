from .BuildTakagiSugeno import BuildTSFIS
from .LoadData import DataLoader
from .Splitter import DataSplitter
from .SimpfulModelBuilder import SugenoFISBuilder
from .Clustering import Clusterer
from .EstimateAntecendentSet import AntecedentEstimator
from .FireStrengthCalculator import FireStrengthCalculator
from .EstimateConsequentParameters import ConsequentEstimator
from .Tester import SugenoFISTester
from .FeatureSelection import FeatureSelector
from .Sampler import Sampler
from .simpfulfier import SimpfulConverter

import numpy as np

class pyFUME(object):
    """
        Creates a new fuzzy model.
        
        Args:
            datapath: The path to the csv file containing the input data (argument 'datapath' or 'dataframe' should be specified by the user).
            dataframe: Pandas dataframe containing the input data (argument 'datapath' or 'dataframe' should be specified by the user).
            nr_clus: Number of clusters that should be identified in the data (default = 2).
            process_categorical: Boolean to indicate whether categorical variables should be processed (default = False).
            method: At this moment, only Takagi Sugeno models are supported (default = 'Takagi-Sugeno')
            variable_names: Names of the variables, if not specified the names will be read from the first row of the csv file (default = None).
            merge_threshold: Threshold for GRABS to drop fuzzy sets from the model. If the jaccard similarity between two sets is higher than this threshold, the fuzzy set will be dropped from the model.
            **kwargs: Additional arguments to change settings of the fuzzy model.

        Returns:
            An object containing the fuzzy model, information about its setting (such as its antecedent and consequent parameters) and the different splits of the data.
    """
    def __init__(self, datapath=None, dataframe=None, nr_clus=2, process_categorical=False, method='Takagi-Sugeno', variable_names=None, merge_threshold=1., **kwargs):

        if datapath is None and dataframe is None:
            raise Exception("Please specify a valid dataset.")

         #if nr_clus<2 and nr_clus!=None:
         #    raise Exception("Number of clusters should be greater than 1.")

        self.datapath=datapath
        self.nr_clus=nr_clus
        self.method=method
        self.dropped_fuzzy_sets = 0
        #self.variable_names=variable_names

        if method=='Takagi-Sugeno' or method=='Sugeno':
            if datapath is not None:
                self.FIS = BuildTSFIS(datapath=self.datapath, nr_clus=self.nr_clus, variable_names=variable_names, process_categorical=process_categorical, merge_threshold=merge_threshold, **kwargs)
            else:
                self.FIS = BuildTSFIS(dataframe=dataframe, nr_clus=self.nr_clus, variable_names=variable_names, process_categorical=process_categorical, merge_threshold=merge_threshold, **kwargs)
            if merge_threshold < 1.0:
                self.dropped_fuzzy_sets = self.FIS._antecedent_estimator.get_number_of_dropped_fuzzy_sets()
        else:
            raise Exception ("This modeling technique has not yet been implemented.")

    def get_model(self):
        """
        Returns the fuzzy model created by pyFUME.

        Returns:
            The fuzzy model (as an executable object).
        """          
        if self.FIS.model is None:
            print ("ERROR: model was not created correctly, aborting.")
            exit(-1)
        else:
            return self.FIS.model

    def calculate_error(self, method="MAE"):
        """
        Calculates the performance of the model given the test data.

            Args:
                method: The performance metric to be used to evaluate the model (default = 'MAE'). Choose from: Mean Absolute Error 
                ('MAE'), Mean Squared Error ('MSE'),  Root Mean Squared Error ('RMSE'), Mean Absolute Percentage 
                Error ('MAPE').
        
        Returns:
            The performance as expressed by the chosen performance metric.
        """   
        if method=="MSE":
            return self._get_MSE()
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
        """
        Calculates the predictions labels of the test data using the fuzzy model.

        Returns:
            Prediction labels.
        """
        #get the prediction for the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, variable_names=self.FIS.variable_names, golden_standard=self.FIS.y_test)
        pred, _ = test.predict()
        return pred
    
    def predict_label(self, xdata):
        """
        Calculates the predictions labels of a data set using the fuzzy model.

        Args:
            xdata: The input data (as numpy array with each row a different data instance and variables in the same order as in the original training data set) for which the labels should be calculated. 

        Returns:
            Prediction labels.
        """
        #get the prediction for a new data set
        model = self.get_model()
        test = SugenoFISTester(model=model, test_data=xdata, golden_standard=None, variable_names=self.FIS.variable_names)
        pred, _ = test.predict()
        return pred

    def test_model(self, xdata, ydata, error_metric='MAE'):
        """
        Calculates the performance of the model using the given data.

        Args:
            xdata: The input data (as numpy array with each row a different data instance and variables in the same order as in the original training data set) for which the labels should be calculated. 
            ydata: The target data (as single-column numpy array).
            error_metric: The error metric in which the performance should be expressed (default = 'MAE'). Choose from: Mean Absolute Error ('MAE'), Mean Squared Error ('MSE'),  Root Mean Squared Error ('RMSE'), Mean Absolute Percentage Error ('MAPE').

        Returns:
            The performance as expressed in the chosen metric.
        """        
        #get the prediction for a new data set
        model = self.get_model()
        test = SugenoFISTester(model=model, test_data=xdata, golden_standard=ydata, variable_names=self.FIS.variable_names)
        metric= test.calculate_performance(metric=error_metric)
        return metric

    def get_firing_strengths(self, data, normalize=True):
        """
        Calculates the (normalized) firing strength/ activition level of each rule for each data instance of the given data.

        Args:
            xdata: The input data (as numpy array with each row a different data instance and variables in the same order as in the original training data set) for which the labels should be calculated. 
            normalize: Boolean that indicates whether the retuned fiing strengths should be normalized (normalize = True) or not (normalize = False), When the firing strenghts are nomalized the summed fiing strengths for each data instance equals one.
        Returns:
            Firing strength/activition level of each rule (columns) for each data instance (rows).
        """          

        # Calculate the firing strengths
        fsc=FireStrengthCalculator(self.FIS.antecedent_parameters, self.FIS.nr_clus, self.FIS.variable_names)
        firing_strengths = fsc.calculate_fire_strength(data)
        if normalize == True:
            firing_strengths=firing_strengths/firing_strengths.sum(axis=1)[:,None]
        return firing_strengths
    
    def get_performance_per_fold(self):
        """
        Returns a list with the performances of each model that is created if crossvalidation is used when training..

        Returns:
            Perfomance of each cross validation model..
        """
        return FIS.FIS.MAE_per_fold
        
    def _get_RMSE(self):
        # Calculate the mean squared error of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, golden_standard=self.FIS.y_test, variable_names=self.FIS.variable_names)
        RMSE = test.calculate_RMSE()
        #RMSE = list(RMSE.values())
        #print('The RMSE of the fuzzy system is', RMSE)
        return RMSE
    
    def _get_MSE(self):
        # Calculate the mean squared error of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, golden_standard=self.FIS.y_test, variable_names=self.FIS.variable_names)
        MSE = test.calculate_MSE()
        #RMSE = list(RMSE.values())
        #print('The RMSE of the fuzzy system is', RMSE)
        return MSE
        
    
    def _get_MAE(self):
        # Calculate the mean absolute error of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, variable_names=self.FIS.variable_names, golden_standard=self.FIS.y_test)
        MAE = test.calculate_MAE()
        return MAE
    
    def _get_MAPE(self):
        # Calculate the mean absolute percentage error of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, golden_standard=self.FIS.y_test, variable_names=self.FIS.variable_names)
        MAPE = test.calculate_MAPE()
        return MAPE

if __name__=='__main__':
    from numpy.random import seed
    seed(4)
   
    FIS = pyFUME(datapath='Concrete_data.csv', nr_clus=3, method='Takagi-Sugeno',
     merge_threshold=.8, operators=None)
    print ("The calculated error is:", FIS.calculate_error())

    FIS.get_model().produce_figure("bla.pdf")