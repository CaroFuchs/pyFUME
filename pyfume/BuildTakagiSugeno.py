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
        
        # Check keyword-arguments and complete with default settings if necessary
        if 'normalize' not in kwargs.keys(): kwargs['normalize'] = False 
        #if 'data_split_method' not in kwargs.keys(): kwargs['data_split_method'] = 'hold-out'
        if 'feature_selection' not in kwargs.keys(): kwargs['feature_selection'] = None
        if 'cluster_method' not in kwargs.keys(): kwargs['cluster_method'] = 'fcm'
        if kwargs['cluster_method'] == 'fcm':
            if 'fcm_m' not in kwargs.keys(): kwargs['fcm_m'] = 2
            if 'fcm_max_iter' not in kwargs.keys(): kwargs['fcm_maxiter'] = 1000
            if 'fcm_error' not in kwargs.keys(): kwargs['fcm_error'] = 0.005
        elif kwargs['cluster_method'] == 'fst-pso':
            if 'fstpso_n_particles' not in kwargs.keys(): kwargs['fstpso_n_particles'] = None
            if 'fstpso_max_iter' not in kwargs.keys(): kwargs['fstpso_max_iter'] = 100
            if 'fstpso_path_fit_dump' not in kwargs.keys(): kwargs['fstpso_path_fit_dump'] = None
            if 'fstpso_path_sol_dump' not in kwargs.keys(): kwargs['fstpso_path_sol_dump'] = None
        if 'mf_shape' not in kwargs.keys(): kwargs['mf_shape'] = 'gauss'       
        if 'operators' not in kwargs.keys(): kwargs['operators'] = None
        if 'global_fit' not in kwargs.keys(): kwargs['global_fit'] = False  
        if 'save_simpful_code' not in kwargs.keys(): kwargs['save_simpful_code'] = True
      
        
        # Load the data
        dl=DataLoader(self.datapath,normalize=kwargs['normalize'])
        self.variable_names=dl.variable_names        

        # Split the data using the hold-out method in a training (default: 75%) 
        # and test set (default: 25%).
        ds = DataSplitter()
       
        if kwargs['data_split_method'] == 'hold-out' or kwargs['data_split_method'] == 'holdout':
            print('Hold-out method selected.')
            self.x_train, self.y_train, self.x_test, self.y_test = ds.holdout(dl.dataX, dl.dataY)

            # Perform feature selection if requested
            if kwargs['feature_selection'] != None:
                fs=FeatureSelector(self.x_train, self.y_train, self.nr_clus, self.variable_names)
                
                if kwargs['feature_selection'] == 'wrapper':
                    self.selected_feature_indices, self.variable_names=fs.wrapper(feature_selection_stop=0.05)
                elif kwargs['feature_selection'] == 'fst-pso':
                    self.selected_feature_indices, self.variable_names= fs.fst_pso_feature_selection(max_iter=10) 
                
                self.x_train = self.x_train[:, self.selected_feature_indices]
                self.x_test = self.x_test[:, self.selected_feature_indices]
                
            
            # Cluster the training data (in input-output space) using FCM
            cl = ClusterFer(self.x_train, self.y_train, self.nr_clus)
            
            if kwargs['cluster_method'] == 'fcm':
                self.cluster_centers, self.partition_matrix, _ = cl.cluster(cluster_method='fcm', fcm_m=kwargs['fcm_m'], 
                    fcm_maxiter=kwargs['fcm_maxiter'], fcm_error=kwargs['fcm_error'])
            elif kwargs['cluster_method'] == 'fst-pso':
                self.cluster_centers, self.partition_matrix, _ = cl.cluster(cluster_method='fstpso', 
                    fstpso_n_particles=kwargs['fstpso_n_particles'], fstpso_max_iter=kwargs['fstpso_max_iter'],
                    fstpso_path_fit_dump=kwargs['fstpso_path_fit_dump'], fstpso_path_sol_dump=kwargs['fstpso_path_sol_dump'])
                
            
            # Estimate the membership funtions of the system (default shape: gauss)
            self._antecedent_estimator = AntecedentEstimator(self.x_train, self.partition_matrix)
    
            self.antecedent_parameters = self._antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=merge_threshold)
            what_to_drop = self._antecedent_estimator._info_for_simplification
            
    
            # Build a first-order Takagi-Sugeno model using Simpful using dummy consequent parameters
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
            for i in range(0,len(self.x_train)):
                for j in range (0,len(self.variable_names)):
                    self.dummymodel.set_variable(self.variable_names[j], self.x_train[i,j])
                firing_strengths.append(self.dummymodel.get_firing_strengths())
            self.firing_strengths=np.array(firing_strengths)
            
            
            # Estimate the parameters of the consequent (default: global fitting)
            ce = ConsequentEstimator(self.x_train, self.y_train, self.firing_strengths)
            self.consequent_parameters = ce.suglms(self.x_train, self.y_train, self.firing_strengths, 
                                                   global_fit=kwargs['global_fit'])
    
            # Build a first-order Takagi-Sugeno model using Simpful
                 
            simpbuilder = SugenoFISBuilder(
                self.antecedent_parameters, 
                self.consequent_parameters, 
                self.variable_names, 
                extreme_values = self._antecedent_estimator._extreme_values,
                operators=kwargs["operators"], 
                save_simpful_code=kwargs['save_simpful_code'], 
                fuzzy_sets_to_drop=what_to_drop)
    
            self.model = simpbuilder.simpfulmodel
            
        elif kwargs['data_split_method']=='cross_validation' or kwargs['data_split_method']=='k-fold_cross_validation' or kwargs['data_split_method']=='crossvalidation' or kwargs['data_split_method']=='cv':
            if 'number_of_folds' not in kwargs.keys(): kwargs['number_of_folds'] = 10
            print('K-fold cross validation was selected. The number of folds (k) equals', kwargs['number_of_folds'])
            
            #Create lists with test indices for each fold.
            fold_indices = ds.kfold(data_length=len(dl.dataX), number_of_folds=kwargs['number_of_folds'])
            
            # Create folder to store developed models
            import os, datetime, pickle
            self.folder_name= 'pyFUME run ' + datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
            
            os.mkdir(self.folder_name)
            
            
            MAE=[np.inf]*kwargs['number_of_folds']
            
            
            for fold_number in range(0, kwargs['number_of_folds']):
                print('Training the model for fold', fold_number)
                tst_idx=fold_indices[fold_number]
                np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
                trn_idx=np.concatenate(np.delete(fold_indices, fold_number, axis=0))
                
                self.x_train = np.array([dl.dataX[i,:] for i in trn_idx])
                self.x_test = np.array([dl.dataX[i,:] for i in tst_idx])                      
                self.y_train = np.array([dl.dataY[i] for i in trn_idx])
                self.y_test = np.array([dl.dataY[i] for i in tst_idx]) 
            
                        # Perform feature selection if requested
                if kwargs['feature_selection'] != None:
                    fs=FeatureSelector(self.x_train, self.y_train, self.nr_clus, self.variable_names)
                    
                    if kwargs['feature_selection'] == 'wrapper':
                        self.selected_feature_indices, self.variable_names=fs.wrapper(feature_selection_stop=0.05)
                    elif kwargs['feature_selection'] == 'fst-pso':
                        self.selected_feature_indices, self.variable_names= fs.fst_pso_feature_selection(max_iter=10) 
                    
                    self.x_train = self.x_train[:, self.selected_feature_indices]
                    self.x_test = self.x_test[:, self.selected_feature_indices]
                    
                
                # Cluster the training data (in input-output space) using FCM
                cl = Clusterer(self.x_train, self.y_train, self.nr_clus)
                
                if kwargs['cluster_method'] == 'fcm':
                    self.cluster_centers, self.partition_matrix, _ = cl.cluster(cluster_method='fcm', fcm_m=kwargs['fcm_m'], 
                        fcm_maxiter=kwargs['fcm_maxiter'], fcm_error=kwargs['fcm_error'])
                elif kwargs['cluster_method'] == 'fst-pso':
                    self.cluster_centers, self.partition_matrix, _ = cl.cluster(cluster_method='fstpso', 
                        fstpso_n_particles=kwargs['fstpso_n_particles'], fstpso_max_iter=kwargs['fstpso_max_iter'],
                        fstpso_path_fit_dump=kwargs['fstpso_path_fit_dump'], fstpso_path_sol_dump=kwargs['fstpso_path_sol_dump'])
                    
                
                # Estimate the membership funtions of the system (default shape: gauss)
                self._antecedent_estimator = AntecedentEstimator(self.x_train, self.partition_matrix)
        
                self.antecedent_parameters = self._antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=merge_threshold)
                what_to_drop = self._antecedent_estimator._info_for_simplification
                
        
                # Build a first-order Takagi-Sugeno model using Simpful using dummy consequent parameters
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
                for i in range(0,len(self.x_train)):
                    for j in range (0,len(self.variable_names)):
                        self.dummymodel.set_variable(self.variable_names[j], self.x_train[i,j])
                    firing_strengths.append(self.dummymodel.get_firing_strengths())
                self.firing_strengths=np.array(firing_strengths)
                
                
                # Estimate the parameters of the consequent (default: global fitting)
                ce = ConsequentEstimator(self.x_train, self.y_train, self.firing_strengths)
                self.consequent_parameters = ce.suglms(self.x_train, self.y_train, self.firing_strengths, 
                                                       global_fit=kwargs['global_fit'])
        
                # Build a first-order Takagi-Sugeno model using Simpful
                     
                simpbuilder = SugenoFISBuilder(
                    self.antecedent_parameters, 
                    self.consequent_parameters, 
                    self.variable_names, 
                    extreme_values = self._antecedent_estimator._extreme_values,
                    operators=kwargs["operators"], 
                    save_simpful_code=kwargs['save_simpful_code'], 
                    fuzzy_sets_to_drop=what_to_drop)
        
                self.model = simpbuilder.simpfulmodel
                
                # Save the created model in the dedicated folder
                filepath = './' + self.folder_name + '/Fold_' + str(fold_number) + '.pickle'
                
                pickle.dump(self.model, open(filepath, 'wb'))
                
                tester=SugenoFISTester(self.model, self.x_test, self.y_test)
                MAE[fold_number]=tester.calculate_MAE(variable_names=self.variable_names)
            print('The average MAE over ' + str(kwargs['number_of_folds']) +' folds is ', str(np.mean(MAE)) +' (with st. dev. ' + str(np.std(MAE)) + '). \nThe best model was created in fold ' +  str(np.argmin(MAE)) + ' with MAE = ' + str(np.min(MAE)) + '.')
                
                
