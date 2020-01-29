from simpfulfier import SimpfulConverter

class SugenoFISBuilder(object):
    """docstring for SugenoFISBuilder"""
    def __init__(self, antecedent_sets, consequent_parameters, variable_names, save_simpful_code=1):
        super(SugenoFISBuilder, self).__init__()

        self._SC = SimpfulConverter(
            input_variables_names = variable_names,
            consequents_matrix = consequent_parameters,
            fuzzy_sets = antecedent_sets
            )
        
        if save_simpful_code==1:
            self._SC.save_code("Simpful_code.py")
        
        self._SC.generate_object()

        self.simpfulmodel = self._SC._fuzzyreasoner