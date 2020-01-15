from ModelEstimation import ModelCreator
from simpfulfier import SimpfulConverter
from collections import defaultdict
from math import sqrt

class SugenoFISBuilder(object):
    """docstring for SugenoFISBuilder"""
    def __init__(self, path=None, clusters=0, variable_names=None):
        super(SugenoFISBuilder, self).__init__()

        self._RC = ModelCreator(datapath=path, 
            nrclus=clusters, 
            varnames=variable_names
            )

        self._SC = SimpfulConverter(
            input_variables_names = variable_names,
            consequents_matrix = self._RC.cons,
            fuzzy_sets = self._RC.mfs
            )
        
        self._SC.generate_object()

        self.simpfulmodel = self._SC._fuzzyreasoner


class SugenoFISTester(object):
    """docstring for SugenoFISTester"""
    def __init__(self, model, test_data, golden_standard):
        super(SugenoFISTester, self).__init__()
        self._model_to_test = model
        self._data_to_test = test_data
        self._golden_standard = golden_standard

    def calculate_error(self, list_of_variables, 
        list_of_outputs=['OUTPUT']):
        # read names
        total_error = defaultdict(float)
        
        for sample in self._data_to_test:
            for i, variable in enumerate(list_of_variables):
                self._model_to_test.set_variable(variable, sample[i])
            result = self._model_to_test.Sugeno_inference()

            for j, output in enumerate(list_of_outputs):
                total_error[output] += (result[output] - self._golden_standard[j])**2

        for k,v in total_error.items():
            total_error[k] = sqrt(v/len(self._data_to_test))

        return total_error


class SugenoFISExperimenter(object):
    """docstring for SugenoFISExperimenter"""
    def __init__(self, arg):
        super(SugenoFISExperimenter, self).__init__()
        self.arg = arg
                        

        

if __name__ == '__main__':
    
    pass