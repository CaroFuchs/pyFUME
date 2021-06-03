from collections import defaultdict
from math import sqrt
import numpy as np


class SugenoFISTester(object):
    """
    Creates a new Tester object to be able to calculate performance metrics of the fuzzy model.
    
    Args:
        model: The model for which the performance metrics should be calculated
        test_data: The data to be used to compute the performance metrics
        variable_names: A list of the variables names of the test data (which 
            should correspond with the variable names used in the model).
        golden_standard: The 'True' labels of the test data. If not provided, the 
            only predictions labels can be generated, but the error will not be 
            calculated (default = None).
        list_of_outputs: List of the output names (which should correspond with 
            the output names used in the model) (default: OUTPUT).
    """
    
    def __init__(self, model, test_data, variable_names, golden_standard=None, list_of_outputs=['OUTPUT']):
        super().__init__()
        self._model_to_test = model
        self._data_to_test = test_data
        self._golden_standard = golden_standard
        self._variable_names = variable_names
        self._list_of_outputs=list_of_outputs
        
    def predict(self):
        """
        Calculates the predictions labels of the test data using the fuzzy model.

        Returns:
            Tuple containing (result, error)
                - result: Prediction labels.
                - error: The difference between the prediction label and the 'true' label.
        """
        result = []
        for sample in self._data_to_test:
            for i, variable in enumerate(self._variable_names):
                self._model_to_test.set_variable(variable, sample[i])
            result.append(self._model_to_test.Sugeno_inference().get('OUTPUT'))
        result = np.array(result)
        if self._golden_standard is not  None:
            error = self._golden_standard - result
        else:
            error = np.nan
            # print('The true labels (golden standard) were not provided, so the error could not be calculated.')
        return result, error
    
    def calculate_performance(self, metric='MAE'):  
        """
        Calculates the performance of the model given the test data.

            Args:
                metric: The performance metric to be used to evaluate the model. Choose from: Mean Absolute Error 
                ('MAE'), Mean Squared Error ('MSE'),  Root Mean Squared Error ('RMSE'), Mean Absolute Percentage 
                Error ('MAPE').
        
        Returns:
            The performance as expressed by the chosen performance metric.
        """      
        if metric == 'MAE':
            err=self.calculate_MAE()
        elif metric == 'MSE':
            err=self.calculate_MSE()
        elif metric == 'RMSE':
            err=self.calculate_RMSE()        
        elif metric == 'MAPE':
            err=self.calculate_MAPE()
        else:
            print('The requested performance metric has not been implemented (yet).')
            
        return err
    
    def calculate_RMSE(self):
        """
        Calculates the Root Mean Squared Error of the model given the test data.
        
        Returns:
            The Root Mean Squared Error of the fuzzy model.
        """
        _, error=self.predict()
        return sqrt(np.mean(np.square(error)))
    
    
    def calculate_MSE(self):
        """
        Calculates the Mean Squared Error of the model given the test data.
        
        Returns:
            The Mean Squared Error of the fuzzy model.
        """
        _, error=self.predict()
        return np.mean(np.square(error))   
    
    def calculate_MAE(self):
        """
        Calculates the Mean Absolute Error of the model given the test data.
        
        Returns:
            The Mean Absolute Error of the fuzzy model.
        """
        _, error=self.predict()
        return np.mean(np.abs(error))
    
    def calculate_MAPE(self):
        """
        Calculates the Mean Absolute Percentage Error of the model given the test data.
        
        Returns:
            The Mean Absolute Percentage Error of the fuzzy model.
        """
        
        if self._golden_standard is None:
             raise Exception('To compute the MAPE the true label (golden standard) of the test data should be provided')
        
        _, error=self.predict()
        return np.mean(np.abs((error) / self._golden_standard)) * 100