from .SimpfulModelBuilder import SugenoFISBuilder

import numpy as np

class FireStrengthCalculator(object):
    def __init__(self, antecedent_parameters, nr_clus, variable_names, fuzzy_sets_to_drop=None, **kwargs):
        self.antecedent_parameters = antecedent_parameters
        self.nr_clus = nr_clus
        self.variable_names = variable_names
        self.what_to_drop = fuzzy_sets_to_drop
        if 'operators' not in kwargs.keys(): kwargs['operators'] = None
        
        
        # Build a first-order Takagi-Sugeno model using Simpful using dummy consequent parameters
        simpbuilder = SugenoFISBuilder(
            self.antecedent_parameters, 
            np.tile(1, (self.nr_clus, len(self.variable_names)+1)), 
            self.variable_names, 
            extreme_values = None,
            operators=kwargs["operators"], 
            save_simpful_code=False, 
            fuzzy_sets_to_drop=self.what_to_drop)

        self.dummymodel = simpbuilder.simpfulmodel
        
    def calculate_fire_strength(self, data):
        self.data=data
        # Calculate the firing strengths for each rule for each data point 
        firing_strengths=[]
        
        for i in range(0,len(self.data)):
            for j in range (0,len(self.variable_names)):
                self.dummymodel.set_variable(self.variable_names[j], self.data[i,j])
            firing_strengths.append(self.dummymodel.get_firing_strengths())
        self.firing_strengths=np.array(firing_strengths)
        return self.firing_strengths