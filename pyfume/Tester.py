from collections import defaultdict
from math import sqrt

class SugenoFISTester(object):
    """docstring for SugenoFISTester"""
    def __init__(self, model, test_data, golden_standard):
        super(SugenoFISTester, self).__init__()
        self._model_to_test = model
        self._data_to_test = test_data
        self._golden_standard = golden_standard

    def calculate_RMSE(self, variable_names, list_of_outputs=['OUTPUT']):
        # read names
        RMSE = defaultdict(float)
       
        for sample in self._data_to_test:
            for i, variable in enumerate(variable_names):
                self._model_to_test.set_variable(variable, sample[i])
            result = self._model_to_test.Sugeno_inference()
            
            for j, output in enumerate(list_of_outputs):
                RMSE[output] += (result[output] - self._golden_standard[j])**2

        for k,v in RMSE.items():
            RMSE[k] = sqrt(v/len(self._data_to_test))

        return RMSE
    
    
    def calculate_MSE(self, variable_names, list_of_outputs=['OUTPUT']):
        # read names
        MSE = defaultdict(float)
        
        for sample in self._data_to_test:
            for i, variable in enumerate(variable_names):
                self._model_to_test.set_variable(variable, sample[i])
            result = self._model_to_test.Sugeno_inference()
            
            for j, output in enumerate(list_of_outputs):
                MSE[output] += (result[output] - self._golden_standard[j])**2

        for k,v in MSE.items():
            MSE[k] = v/len(self._data_to_test)

        return MSE   
    
    def calculate_MAE(self, variable_names, list_of_outputs=['OUTPUT']):
        # read names
        MAE = defaultdict(float)
        
        for sample in self._data_to_test:
            for i, variable in enumerate(variable_names):
                self._model_to_test.set_variable(variable, sample[i])
            result = self._model_to_test.Sugeno_inference()
            
            for j, output in enumerate(list_of_outputs):
                MAE[output] += abs((result[output] - self._golden_standard[j]))

        for k,v in MAE.items():
            MAE[k] = sqrt(v/len(self._data_to_test))

        return MAE
    
    def calculate_MAPE(self, variable_names, list_of_outputs=['OUTPUT']):
        # read names
        MAPE = defaultdict(float)
        
        for sample in self._data_to_test:
            for i, variable in enumerate(variable_names):
                self._model_to_test.set_variable(variable, sample[i])
            result = self._model_to_test.Sugeno_inference()
            
            for j, output in enumerate(list_of_outputs):
                MAPE[output] += abs((result[output] - self._golden_standard[j])/self._golden_standard[j])

        for k,v in MAE.items():
            MAPE[k] = sqrt(v/len(self._data_to_test))

        return MAPE