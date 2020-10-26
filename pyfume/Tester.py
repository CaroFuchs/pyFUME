from collections import defaultdict
from math import sqrt
import numpy as np


class SugenoFISTester(object):
    """docstring for SugenoFISTester"""
    def __init__(self, model, test_data, golden_standard):
        super(SugenoFISTester, self).__init__()
        self._model_to_test = model
        self._data_to_test = test_data
        self._golden_standard = golden_standard

    def predict(self, variable_names, list_of_outputs=['OUTPUT']):
        result = []
        for sample in self._data_to_test:
            for i, variable in enumerate(variable_names):
                self._model_to_test.set_variable(variable, sample[i])
            result.append(self._model_to_test.Sugeno_inference().get('OUTPUT'))
        result = np.array(result) 
        error = self._golden_standard - result
        return result, error

    def calculate_RMSE(self, variable_names, list_of_outputs=['OUTPUT']):
        _, error=self.predict(variable_names, list_of_outputs)
        return sqrt(np.mean(np.square(error)))
    
    
    def calculate_MSE(self, variable_names, list_of_outputs=['OUTPUT']):
        _, error=self.predict(variable_names, list_of_outputs)
        return np.mean(np.square(error))   
    
    def calculate_MAE(self, variable_names, list_of_outputs=['OUTPUT']):
        _, error=self.predict(variable_names, list_of_outputs)
        return np.mean(np.abs(error))
    
    def calculate_MAPE(self, variable_names, list_of_outputs=['OUTPUT']):
        _, error=self.predict(variable_names, list_of_outputs)
        return np.mean(np.abs((error) / self._golden_standard)) * 100