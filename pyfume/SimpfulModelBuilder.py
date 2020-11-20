from .simpfulfier import *

class SugenoFISBuilder(object):
    """docstring for SugenoFISBuilder"""
    def __init__(self, antecedent_sets, consequent_parameters, 
        variable_names, extreme_values=None, operators=None, save_simpful_code=True, fuzzy_sets_to_drop=None):
        
        #super(SugenoFISBuilder, self).__init__()
        super().__init__()

        self._SC = SimpfulConverter(
            input_variables_names = variable_names,
            consequents_matrix = consequent_parameters,
            fuzzy_sets = antecedent_sets,
            operators = operators,
            extreme_values = extreme_values,
            fuzzy_sets_to_drop=fuzzy_sets_to_drop
            )
        
        if save_simpful_code==True:
            self._SC.save_code("Simpful_code.py")
        
        self._SC.generate_object()

        self.simpfulmodel = self._SC._fuzzyreasoner