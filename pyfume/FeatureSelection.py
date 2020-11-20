from .Splitter import DataSplitter
from .SimpfulModelBuilder import SugenoFISBuilder
from .Clustering import Clusterer
from .EstimateAntecendentSet import AntecedentEstimator
from .EstimateConsequentParameters import ConsequentEstimator
from .Tester import SugenoFISTester
import numpy as np

class FeatureSelector(object):
    def __init__(self, dataX, dataY, nr_clus, variable_names, **kwargs):
        self.dataX=dataX
        self.dataY=dataY
        self.nr_clus = nr_clus
        self.variable_names = variable_names
       

                            
    def wrapper(self,**kwargs):
        
        # Check settings and complete with defaukt settings when needed
        if 'merge_threshold' not in kwargs.keys(): kwargs['merge_threshold'] = 1.0
        if 'cluster_method' not in kwargs.keys(): kwargs['cluster_method'] = 'fcm'        
        if kwargs['cluster_method'] == 'fcm':
            if 'fcm_m' not in kwargs.keys(): kwargs['fcm_m'] = 2
            if 'fcm_max_iter' not in kwargs.keys(): kwargs['fcm_maxiter'] = 1000
            if 'fcm_error' not in kwargs.keys(): kwargs['fcm_error'] = 0.005
        elif kwargs['cluster_method'] == 'fstpso':
            if 'fstpso_n_particles' not in kwargs.keys(): kwargs['fstpso_n_particles'] = None
            if 'fstpso_max_iter' not in kwargs.keys(): kwargs['fstpso_max_iter'] = 100
            if 'fstpso_path_fit_dump' not in kwargs.keys(): kwargs['fstpso_path_fit_dump'] = None
            if 'fstpso_path_sol_dump' not in kwargs.keys(): kwargs['fstpso_path_sol_dump'] = None
        if 'mf_shape' not in kwargs.keys(): kwargs['mf_shape'] = 'gauss'       
        if 'operators' not in kwargs.keys(): kwargs['operators'] = None
        if 'global_fit' not in kwargs.keys(): kwargs['global_fit'] = False  
        if 'operators' not in kwargs.keys(): kwargs['operators'] = None
        
        
        # Create a training and valiadation set for the feature selection phase
        ds = DataSplitter(self.dataX, self.dataY)
        x_feat, y_feat, x_val, y_val = ds.holdout(self.dataX, self.dataY)
        
        # Set initial values for the MAEs
        old_MAE=np.inf
        new_MAE=np.inf
        MAEs=[]
        
        # Create a set with the unselected (currently all) and selected (none yet) variables
        selected_features=[]
        unselected_features=list(range(0,np.size(x_feat,axis=1)))
        
        stop=False

        while stop == False: 
            MAEs= [np.inf]*np.size(x_feat,axis=1)
            
            for f in [x for x in unselected_features if x != -1]:
                considered_features = selected_features + [f]
                var_names=self.variable_names[considered_features] 
                feat=x_feat[:,considered_features]

                
                # Cluster the training data (in input-output space)
                cl = Clusterer(feat, y_feat, self.nr_clus)               
                
                if kwargs['cluster_method'] == 'fcm':
                    cluster_centers, partition_matrix, _ = cl.cluster(cluster_method='fcm', fcm_m=kwargs['fcm_m'], 
                        fcm_maxiter=kwargs['fcm_maxiter'], fcm_error=kwargs['fcm_error'])
                elif kwargs['cluster_method'] == 'fstpso':
                    cluster_centers, partition_matrix, _ = cl.cluster(cluster_method='fstpso', 
                        fstpso_n_particles=kwargs['fstpso_n_particles'], fstpso_max_iter=kwargs['fstpso_max_iter'],
                        fstpso_path_fit_dump=kwargs['fstpso_path_fit_dump'], fstpso_path_sol_dump=kwargs['fstpso_path_sol_dump'])
                else:
                    print('The requested clustering method is not (yet) implemented')
                     
                # Estimate the membership funtions of the system (default shape: gauss)
                antecedent_estimator = AntecedentEstimator(feat, partition_matrix)
        
                antecedent_parameters = antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=kwargs['merge_threshold'])
                what_to_drop = antecedent_estimator._info_for_simplification
        
                # Build a first-order Takagi-Sugeno model using Simpful using dummy consequent parameters
                simpbuilder = SugenoFISBuilder(
                    antecedent_parameters, 
                    np.tile(1, (self.nr_clus, len(var_names)+1)), 
                    var_names, 
                    extreme_values = antecedent_estimator._extreme_values,
                    operators=kwargs["operators"], 
                    save_simpful_code=False, 
                    fuzzy_sets_to_drop=what_to_drop)
        
                dummymodel = simpbuilder.simpfulmodel
                
                # Calculate the firing strengths for each rule for each data point 
                firing_strengths=[]
                for i in range(0,len(feat)):
                    for j in range (0,len(var_names)):
                        dummymodel.set_variable(var_names[j], feat[i,j])
                    firing_strengths.append(dummymodel.get_firing_strengths())
                firing_strengths=np.array(firing_strengths)
                
                
                # Estimate the parameters of the consequent
                ce = ConsequentEstimator(feat, y_feat, firing_strengths)
                consequent_parameters = ce.suglms(feat, y_feat, firing_strengths, 
                                                       global_fit=kwargs['global_fit'])
        
                # Build a first-order Takagi-Sugeno model using Simpful
                simpbuilder = SugenoFISBuilder(
                    antecedent_parameters, 
                    consequent_parameters, 
                    var_names, 
                    extreme_values = antecedent_estimator._extreme_values,
                    operators=kwargs["operators"], 
                    save_simpful_code=False, 
                    fuzzy_sets_to_drop=what_to_drop)
        
                model = simpbuilder.simpfulmodel
                
                # Test the model
                test = SugenoFISTester(model, x_val[:,considered_features],y_val)
                mae = test.calculate_MAE(variable_names=var_names)
                
                MAEs[f] = mae
            
            new_MAE=min(MAEs)
            #print('Unselected feaures:', unselected_features)
            #print('index new MAE:', MAEs.index(new_MAE))
            new_feature=unselected_features[MAEs.index(new_MAE)]
            
            #print('new feature', new_feature)
            del MAEs
            
            if new_MAE<old_MAE: #*feature_selection_stop:
                selected_features.append(new_feature)
                unselected_features[new_feature]=-1
                old_MAE=new_MAE
            else:
                
                stop = True 
       
        selected_feature_names = self.variable_names[selected_features]
        print('The following features were selected:',  selected_feature_names)
        
        return selected_features, selected_feature_names       
    
   
