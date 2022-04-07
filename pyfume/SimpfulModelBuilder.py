from .simpfulfier import *
import sys

class SugenoFISBuilder(object):
    """
        Builds the executable Simpful model.
        
        Args:
            antecedent_sets: The parameters for the antecedent sets.
            consequent_parameters: The parameters for the consequent function.
            variable_names: The names of the variables.
            extreme_values: Extreme values to determine the universe of discourse. 
                If these are not set, the model will function but it will not 
                be possible to plot the membership functions (default = None).
            operators=None
            save_simpful_code: True/False, determines if the Simpful code will 
                be saved to the same folder as the script (default = True).
            fuzzy_sets_to_drop: Fuzzy sets that should be droppped from the 
                model (default = None).
    
    """
    def __init__(self, antecedent_sets, consequent_parameters, 
        variable_names, normalization_values = None, model_order = 'first', extreme_values=None, operators=None, 
        save_simpful_code=True, fuzzy_sets_to_drop=None, verbose=True):
        #super(SugenoFISBuilder, self).__init__()
        super().__init__()
        
        # if normalization_values != None:
        #     print('variable names:', variable_names)
        #     print('Antecedent sets:', antecedent_sets)
        #     print('Consequent parameters:', consequent_parameters)
        #     print('Normalization values:', normalization_values)
        #     sys.exit()            

       
        self._SC = SimpfulConverter(
            input_variables_names = variable_names,
            consequents_matrix = consequent_parameters,
            fuzzy_sets = antecedent_sets,
            operators = operators,
            extreme_values = extreme_values,
            model_order = model_order,
            fuzzy_sets_to_drop=fuzzy_sets_to_drop,
            verbose = verbose)
        
        # save_simpful_code can be either True (save on default file), 
        # False (not not save anything), or a path to a file.
        if save_simpful_code==True:
            self._SC.save_code("Simpful_code.py")
        elif save_simpful_code != False:
            self._SC.save_code(save_simpful_code)
       
        self._SC.generate_object()

        self.simpfulmodel = self._SC._fuzzyreasoner
    
    def get_model(self):
        return self.simpfulmodel