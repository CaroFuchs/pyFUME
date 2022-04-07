from .SimpfulModelBuilder import SugenoFISBuilder

import numpy as np

class FireStrengthCalculator(object):
    """
        Creates a new fire strength calculator object.
        
        Args:
            antecedent_parameters: The parameters of the antecedent sets of the fuzzy model as given as output 
            by pyFUME's AntecedentEstimator clas (format: shape of the mf, parameters).
            nr_clus: Number of clusters in the data.
            variable_names: Names of the variables.
            fuzzy_sets_to_drop = Fuzzy sets identified by GRsABS to be dropped from the model (default = None).
            **kwargs: Additional arguments to change settings of the fuzzy model.
    """
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
            fuzzy_sets_to_drop=self.what_to_drop,
            verbose = False)

        self.dummymodel = simpbuilder.simpfulmodel
        
    def calculate_fire_strength(self, data):
        """
            Calculates the firing strength per rule of the fuzzy model given a data set.
            
            Args:
                data: The data of which the firing strengths per rule should be calculated.
                
            Returns:
                The firing strengths per rule per data point (rows: data point index, column: rule/cluster number).
        """

        self.data=data
        
        # Create a dictionary to pass to Simpful
        input_dict = {}
        for i in range(len(self.variable_names)):
            input_dict[self.variable_names[i]] = self.data[:,i]
        
        # Calculate the firing strengths for each rule for each data point         
        self.firing_strengths = np.array(self.dummymodel.get_firing_strengths(input_values = input_dict))     
        
        # firing_strengths=[]
        # for i in range(0,len(self.data)):
        #     for j in range (0,len(self.variable_names)):
        #         self.dummymodel.set_variable(self.variable_names[j], self.data[i,j])
        #     firing_strengths.append(self.dummymodel.get_firing_strengths())
        # self.firing_strengths=np.array(firing_strengths)
        return self.firing_strengths