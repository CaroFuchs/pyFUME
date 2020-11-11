from .LoadData import DataLoader
from .Splitter import DataSplitter
from .SimpfulModelBuilder import SugenoFISBuilder
from .Clustering import Clusterer
from .EstimateAntecendentSet import AntecedentEstimator
from .EstimateConsequentParameters import ConsequentEstimator
from .Tester import SugenoFISTester
from .FeatureSelection import FeatureSelector
import numpy as np


class BuildTSFIS(object):
    def __init__(self, datapath, nr_clus, variable_names=None, merge_threshold=1.0, **kwargs):
        self.datapath = datapath
        self.nr_clus = nr_clus
        self.variable_names = variable_names
        self._antecedent_estimator = None
        
        # Load the data
        if 'normalize' not in kwargs.keys(): kwargs['normalize'] = False       
        dl=DataLoader(self.datapath,normalize=kwargs['normalize'])
        self.variable_names=dl.variable_names        

        # Split the data using the hold-out method in a training (default: 75%) 
        # and test set (default: 25%).
        ds = DataSplitter(dl.dataX,dl.dataY)
        self.x_train, self.y_train, self.x_test, self.y_test = ds.holdout(dl.dataX, dl.dataY)

        # Perform feature selection if requested
        if 'feature_selection' not in kwargs.keys(): kwargs['feature_selection'] = False
        if kwargs['feature_selection'] == True:
            fs=FeatureSelector(self.x_train, self.y_train, self.nr_clus, self.variable_names)
            
            self.selected_feature_indices, self.variable_names=fs.wrapper(feature_selection_stop=0.05)
            self.x_train = self.x_train[:, self.selected_feature_indices]
            self.x_test = self.x_test[:, self.selected_feature_indices]
            
        
        # Cluster the training data (in input-output space) using FCM
        cl = Clusterer(self.x_train, self.y_train, self.nr_clus)
        
        if 'cluster_method' not in kwargs.keys(): kwargs['cluster_method'] = 'fcm'
        
        if kwargs['cluster_method'] == 'fcm':
            if 'fcm_m' not in kwargs.keys(): kwargs['fcm_m'] = 2
            if 'fcm_max_iter' not in kwargs.keys(): kwargs['fcm_maxiter'] = 1000
            if 'fcm_error' not in kwargs.keys(): kwargs['fcm_error'] = 0.005
            self.cluster_centers, self.partition_matrix, _ = cl.cluster(cluster_method='fcm', fcm_m=kwargs['fcm_m'], 
                fcm_maxiter=kwargs['fcm_maxiter'], fcm_error=kwargs['fcm_error'])
        elif kwargs['cluster_method'] == 'fstpso':
            if 'fstpso_n_particles' not in kwargs.keys(): kwargs['fstpso_n_particles'] = None
            if 'fstpso_maxiter' not in kwargs.keys(): kwargs['fstpso_maxiter'] = 100
            if 'fstpso_path_fit_dump' not in kwargs.keys(): kwargs['fstpso_path_fit_dump'] = None
            if 'fstpso_path_sol_dump' not in kwargs.keys(): kwargs['fstpso_path_sol_dump'] = None
            self.cluster_centers, self.partition_matrix, _ = cl.cluster(cluster_method='fstpso', 
                fstpso_n_particles=kwargs['fstpso_n_particles'], fstpso_max_iter=kwargs['fstpso_max_iter'],
                fstpso_path_fit_dump=kwargs['fstpso_path_fit_dump'], fstpso_path_sol_dump=kwargs['fstpso_path_sol_dump'])
            
        
        # Estimate the membership funtions of the system (default shape: gauss)
        if 'mf_shape' not in kwargs.keys(): kwargs['mf_shape'] = 'gauss'       
        self._antecedent_estimator = AntecedentEstimator(self.x_train, self.partition_matrix)

        self.antecedent_parameters = self._antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=merge_threshold)
        what_to_drop = self._antecedent_estimator._info_for_simplification
        

        # Build a first-order Takagi-Sugeno model using Simpful using dummy consequent parameters
        if 'operators' not in kwargs.keys(): kwargs['operators'] = None
        simpbuilder = SugenoFISBuilder(
            self.antecedent_parameters, 
            np.tile(1, (self.nr_clus, len(self.variable_names)+1)), 
            self.variable_names, 
            extreme_values = self._antecedent_estimator._extreme_values,
            operators=kwargs["operators"], 
            save_simpful_code=False, 
            fuzzy_sets_to_drop=what_to_drop)

        self.dummymodel = simpbuilder.simpfulmodel
        
        # Calculate the firing strengths for each rule for each data point 
        firing_strengths=[]
        print(self.variable_names)
        for i in range(0,len(self.x_train)):
            for j in range (0,len(self.variable_names)):
                self.dummymodel.set_variable(self.variable_names[j], self.x_train[i,j])
            firing_strengths.append(self.dummymodel.get_firing_strengths())
        self.firing_strengths=np.array(firing_strengths)
        
        
        # Estimate the parameters of the consequent (default: global fitting)
        if 'global_fit' not in kwargs.keys(): kwargs['global_fit'] = False  
        ce = ConsequentEstimator(self.x_train, self.y_train, self.firing_strengths)
        self.consequent_parameters = ce.suglms(self.x_train, self.y_train, self.firing_strengths, 
                                               global_fit=kwargs['global_fit'])

        # Build a first-order Takagi-Sugeno model using Simpful
        if 'save_simpful_code' not in kwargs.keys(): kwargs['save_simpful_code'] = True
        if 'operators' not in kwargs.keys(): kwargs['operators'] = None
                   
        simpbuilder = SugenoFISBuilder(
            self.antecedent_parameters, 
            self.consequent_parameters, 
            self.variable_names, 
            extreme_values = self._antecedent_estimator._extreme_values,
            operators=kwargs["operators"], 
            save_simpful_code=kwargs['save_simpful_code'], 
            fuzzy_sets_to_drop=what_to_drop)

        self.model = simpbuilder.simpfulmodel
