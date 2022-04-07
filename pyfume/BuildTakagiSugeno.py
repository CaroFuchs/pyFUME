from .LoadData import DataLoader
from .Splitter import DataSplitter
from .SimpfulModelBuilder import SugenoFISBuilder
from .Clustering import Clusterer
from .EstimateAntecendentSet import AntecedentEstimator
from .FireStrengthCalculator import FireStrengthCalculator
from .EstimateConsequentParameters import ConsequentEstimator
from .Tester import SugenoFISTester
from .FeatureSelection import FeatureSelector
from .Sampler import Sampler
import numpy as np


class BuildTSFIS(object):
    """
        Learns a a new  Takagi-Sugeno fuzzy model.
        
        Args:
            datapath: The path to the csv file containing the input data (argument 'datapath' or 'dataframe' should be specified by the user).
            dataframe: Pandas dataframe containing the input data (argument 'datapath' or 'dataframe' should be specified by the user).
            nr_clus: Number of clusters that should be identified in the data (default = 2).
            process_categorical: Boolean to indicate whether categorical variables should be processed (default = False).
            method: At this moment, only Takagi Sugeno models are supported (default = 'Takagi-Sugeno')
            variable_names: Names of the variables, if not specified the names will be read from the first row of the csv file (default = None).
            merge_threshold: Threshold for GRABS to drop fuzzy sets from the model. If the jaccard similarity between two sets is higher than this threshold, the fuzzy set will be dropped from the model.
            **kwargs: Additional arguments to change settings of the fuzzy model.

        Returns:
            An object containing the fuzzy model, information about its setting (such as its antecedent and consequent parameters) and the different splits of the data.
    """
    def __init__(self, datapath=None, dataframe=None, nr_clus=None, variable_names=None, 
            process_categorical=False, merge_threshold=1.0, **kwargs):

        self.datapath = datapath
        self.nr_clus = nr_clus
        self.variable_names = variable_names
        self._antecedent_estimator = None
        verbose = False

        if 'verbose' in kwargs.keys(): verbose = kwargs['verbose']
        # Check keyword-arguments and complete with default settings if necessary
        if 'model_order' not in kwargs.keys(): kwargs['model_order'] = 'first' 
        if 'normalize' not in kwargs.keys(): kwargs['normalize'] = False 
        if 'imputation' not in kwargs.keys(): kwargs['imputation'] = 'knn' # new
        if 'percentage_training' not in kwargs.keys(): kwargs['percentage_training'] = 0.75
        if 'oversampling' not in kwargs.keys(): kwargs['oversampling'] = False
        if kwargs['oversampling'] == True:
            if 'sampling_number_of_bins' not in kwargs.keys(): kwargs['sampling_number_of_bins'] = 2
            if 'sampling_histogram' not in kwargs.keys(): kwargs['sampling_histogram'] = False            
        if 'data_delimiter' not in kwargs.keys(): kwargs['data_delimiter'] = ','
        if 'data_split_method' not in kwargs.keys(): kwargs['data_split_method'] = 'hold-out'
        if 'feature_selection' not in kwargs.keys(): kwargs['feature_selection'] = None
        if 'fs_max_iter' not in kwargs.keys(): kwargs['fs_max_iter'] = 100
        if 'cluster_method' not in kwargs.keys(): kwargs['cluster_method'] = 'fcm'
        if 'm' not in kwargs.keys(): kwargs['m'] = '2'
        if kwargs['cluster_method'] == 'fcm':
            if 'fcm_max_iter' not in kwargs.keys(): kwargs['fcm_maxiter'] = 1000
            if 'fcm_error' not in kwargs.keys(): kwargs['fcm_error'] = 0.005
        elif kwargs['cluster_method'] == 'fst-pso':
            if 'fstpso_n_particles' not in kwargs.keys(): kwargs['fstpso_n_particles'] = None
            if 'fstpso_max_iter' not in kwargs.keys(): kwargs['fstpso_max_iter'] = 100
            if 'fstpso_path_fit_dump' not in kwargs.keys(): kwargs['fstpso_path_fit_dump'] = None
            if 'fstpso_path_sol_dump' not in kwargs.keys(): kwargs['fstpso_path_sol_dump'] = None
        elif kwargs['cluster_method'] == 'gk':
            if 'gk_max_iter' not in kwargs.keys(): kwargs['gk_maxiter'] = 1000
        if 'mf_shape' not in kwargs.keys(): kwargs['mf_shape'] = 'gauss'       
        if 'operators' not in kwargs.keys(): kwargs['operators'] = None
        if 'global_fit' not in kwargs.keys(): kwargs['global_fit'] = False  
        if 'save_simpful_code' not in kwargs.keys(): kwargs['save_simpful_code'] = True
        if 'verbose' not in kwargs.keys(): kwargs['verbose'] = True
        if 'cv_randomID' not in kwargs.keys(): kwargs['cv_randomID'] = False

        if self.nr_clus==None and kwargs['feature_selection'] == None:
            print('Error: please set pyFUME`s argument "nr_clus".')
            import sys
            sys.exit()
          

        # Load the data
        if self.datapath is None:
            dl=DataLoader(dataframe=dataframe, normalize=kwargs['normalize'], process_categorical=process_categorical, delimiter=kwargs['data_delimiter'], verbose=verbose)
        else:
            dl=DataLoader(self.datapath, normalize=kwargs['normalize'],  process_categorical=process_categorical, delimiter=kwargs['data_delimiter'], verbose=verbose)
        self.variable_names=dl.get_variable_names()
        
        if kwargs['normalize'] != False and kwargs['normalize'] != 'zscore':
            self.normalization_values=list(dl.get_normalization_values())
            self.minmax_norm_flag=True
        else:
            self.minmax_norm_flag = False
            self.normalization_values =  None
            
        self.dataX=dl.get_input_data()
        self.dataY=dl.get_target_data()
        
            
        # Create a DataSplitter object
        ds = DataSplitter()

        if kwargs['data_split_method'] == 'hold-out' or kwargs['data_split_method'] == 'holdout':
           
            if verbose: print(' * Hold-out method selected.')
            
            # Split the data using the hold-out method in a training (default: 75%) 
            # and test set (default: 25%).
            self.x_train, self.y_train, self.x_test, self.y_test = ds.holdout(dataX=self.dataX, dataY=self.dataY, percentage_training=kwargs['percentage_training'])
            # Check if there are any missing variables and impute them
            if np.isnan(self.dataX).any().any()== True:
                try:
                    from sklearn.impute import KNNImputer
                except ImportError:
                    raise Exception('pyFUME tried to impute missing values, but couldn`t find \'sklearn\'. Please pip install sklearn to proceed.')

                if verbose: print('Warning: Your data contains missing values that will be imputed using KNN.')   
                imputer = KNNImputer(n_neighbors=3, weights="uniform")
                self.x_train=imputer.fit_transform(self.x_train)
                self.x_test=imputer.fit_transform(self.x_test)

            
            if kwargs['oversampling'] == True:
                sample = Sampler(train_x = self.x_train, train_y=self.y_train, number_of_bins =  kwargs['sampling_number_of_bins'], histogram =  kwargs['sampling_histogram'])
                self.x_train, self.y_train = sample.oversample()
                    
            # Perform feature selection if requested
            if kwargs['feature_selection'] != None and kwargs['feature_selection'] != False:
                if 'performance_metric' not in kwargs.keys(): kwargs['performance_metric'] = 'MAE'
                fs = FeatureSelector(self.x_train, self.y_train, self.nr_clus, self.variable_names, model_order= kwargs['model_order'], performance_metric = kwargs['performance_metric'], verbose=kwargs['verbose'])
                
                # Keep copies of the training and test set before dropping unselected features
                self.x_train_before_fs=self.x_train.copy()
                self.x_test_before_fs=self.x_test.copy()
                            
                if kwargs['feature_selection'] == 'wrapper' or kwargs['feature_selection'] == 'sfs' or kwargs['feature_selection'] == 'SFS':
                    self.selected_feature_indices, self.variable_names=fs.wrapper()
                elif kwargs['feature_selection'] == 'logwrapper':
                    self.selected_feature_indices, self.selected_variable_names, self.log_indices, self.log_variable_names = fs.log_wrapper()
                elif kwargs['feature_selection'] == 'fst-pso' or kwargs['feature_selection'] == 'fstpso' or kwargs['feature_selection'] == 'pso' or kwargs['feature_selection'] == True:
                    self.selected_feature_indices, self.variable_names, self.nr_clus= fs.fst_pso_feature_selection(max_iter=kwargs['fs_max_iter']) 
                self.x_train = self.x_train[:, self.selected_feature_indices]
                self.x_test = self.x_test[:, self.selected_feature_indices]
                
            elif kwargs['feature_selection'] == None:
                self.selected_variable_names= self.variable_names
            
            # Cluster the data, log-transform when needed.
            if kwargs['feature_selection'] == 'logwrapper':
                # Use log transformed variables if needed
                self.log_x_train = self.x_train.copy()
                for i in self.log_indices:
                    self.log_x_train[i]= np.log(self.x_train[i])
                cl = Clusterer(x_train=self.log_x_train, y_train=self.y_train, nr_clus=self.nr_clus, verbose=verbose)
            else:                
                # Cluster the training data (in input-output space)
                cl = Clusterer(x_train=self.x_train, y_train=self.y_train, nr_clus=self.nr_clus, verbose=verbose)
            
            if kwargs['cluster_method'] == 'fcm':
                self.cluster_centers, self.partition_matrix, _ = cl.cluster(method='fcm', fcm_m=kwargs['m'], 
                    fcm_maxiter=kwargs['fcm_maxiter'], fcm_error=kwargs['fcm_error'])
            elif kwargs['cluster_method'] == 'gk':
                self.cluster_centers, self.partition_matrix, _ = cl.cluster(method='gk')
            elif kwargs['cluster_method'] == 'fst-pso':
                self.cluster_centers, self.partition_matrix, _ = cl.cluster(method='fstpso', 
                    fstpso_n_particles=kwargs['fstpso_n_particles'], fstpso_max_iter=kwargs['fstpso_max_iter'],
                    fstpso_path_fit_dump=kwargs['fstpso_path_fit_dump'], fstpso_path_sol_dump=kwargs['fstpso_path_sol_dump'])
            else: 
                print('ERROR: Choose a valid clustering method.')
                import sys
                sys.exit()
                    
                
            # Estimate the membership funtions of the system (default shape: gauss)
            self._antecedent_estimator = AntecedentEstimator(x_train=self.x_train, partition_matrix=self.partition_matrix)
    
            self.antecedent_parameters = self._antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=merge_threshold)
            what_to_drop = self._antecedent_estimator._info_for_simplification
            
            # Calculate the firing strengths
            fsc=FireStrengthCalculator(antecedent_parameters=self.antecedent_parameters, nr_clus=self.nr_clus, variable_names=self.variable_names,  **kwargs)
            self.firing_strengths = fsc.calculate_fire_strength(data=self.x_train)
  
            # Estimate the parameters of the consequent
            ce = ConsequentEstimator(x_train=self.x_train, y_train=self.y_train, firing_strengths=self.firing_strengths)
            if kwargs['model_order'] == 'first':
                self.consequent_parameters = ce.suglms()
            elif kwargs['model_order'] == 'zero':
                self.consequent_parameters = ce.zero_order()
            else:
                raise Exception("pyFUME currently only supports zero-order (model_order = 'zero') and first-order (model_order = 'first') fuzzy models.")
                    
            # Build a first-order Takagi-Sugeno model using Simpful
            simpbuilder = SugenoFISBuilder(
                self.antecedent_parameters, 
                self.consequent_parameters, 
                self.selected_variable_names, 
                normalization_values = self.normalization_values,
                extreme_values = self._antecedent_estimator._extreme_values,
                model_order=kwargs["model_order"],
                operators=kwargs["operators"], 
                save_simpful_code=kwargs['save_simpful_code'], 
                fuzzy_sets_to_drop=what_to_drop, verbose=verbose)
    
            self.model = simpbuilder.simpfulmodel
            
        elif kwargs['data_split_method']=='cross_validation' or kwargs['data_split_method']=='k-fold_cross_validation' or kwargs['data_split_method']=='crossvalidation' or kwargs['data_split_method']=='cv' or kwargs['data_split_method']=='kfold':
            if 'number_of_folds' not in kwargs.keys(): kwargs['number_of_folds'] = 10
            if verbose: print('K-fold cross validation was selected. The number of folds (k) equals', kwargs['number_of_folds'])
            if 'performance_metric' not in kwargs.keys(): kwargs['performance_metric'] = 'MAE'

            #Create lists with test indices for each fold.
            self.fold_indices = ds.kfold(data_length=len(self.dataX), number_of_folds=kwargs['number_of_folds'])
            self.performance_metric = kwargs['performance_metric']
            # import indices from external file
            # import pickle
            # import pandas as pd

            #fold_indices=pd.read_csv('./fold_indices.csv', header=None)  
            #self.fold_indices=fold_indices.to_numpy()
                        
            # Create folder to store developed models
            import os, datetime, pickle

            if kwargs['cv_randomID'] == True:
                try:
                    import uuid
                except ImportError:
                    raise Exception('pyFUME tried to generate random IDs, but couldn`t find \'uuid\'. Please pip install uuid to proceed.')
                
                self.folder_name= 'pyFUME runID ' + str(uuid.uuid4())
            elif kwargs['cv_randomID'] == False:
                self.folder_name= 'pyFUME run ' + datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
            
            owd = os.getcwd()
            os.mkdir(self.folder_name)
            os.chdir('./' + self.folder_name)
            
            self.performance_metric_per_fold=[np.inf]*kwargs['number_of_folds']
            
            
            for fold_number in range(0, kwargs['number_of_folds']):
                if verbose: print('Training the model for fold', fold_number)
                tst_idx=self.fold_indices[fold_number]
                tst_idx = tst_idx[~np.isnan(tst_idx)]
                tst_idx = [int(x) for x in tst_idx]
                np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
                trn_idx=np.concatenate(np.delete(self.fold_indices, fold_number, axis=0))
                trn_idx = trn_idx[~np.isnan(trn_idx)]
                trn_idx = [int(x) for x in trn_idx]

                self.x_train = np.array([self.dataX[i,:] for i in trn_idx])
                self.x_test = np.array([self.dataX[i,:] for i in tst_idx])                      
                self.y_train = np.array([self.dataY[i] for i in trn_idx])
                self.y_test = np.array([self.dataY[i] for i in tst_idx]) 
                
                if np.isnan(self.dataX).any().any()== True:
                    try:
                        from sklearn.impute import KNNImputer
                    except ImportError:
                        raise Exception('pyFUME tried to impute missing values, but couldn`t find \'sklearn\'. Please pip install sklearn to proceed.')

                    if verbose: print('Warning: Your data contains missing values that will be imputed using KNN.')   
                    imputer = KNNImputer(n_neighbors=3, weights="uniform")
                    self.x_train=imputer.fit_transform(self.x_train)
                    self.x_test=imputer.fit_transform(self.x_test)
                
                if kwargs['oversampling'] == True:
                    sample = Sampler(train_x = self.x_train, train_y=self.y_train, number_of_bins =  kwargs['sampling_number_of_bins'], histogram =  kwargs['sampling_histogram'])
                    self.x_train, self.y_train = sample.oversample()
            
                # Perform feature selection if requested
                if kwargs['feature_selection'] != None and kwargs['feature_selection'] != False:
                    fs = FeatureSelector(self.x_train, self.y_train, self.nr_clus, self.variable_names, model_order= kwargs['model_order'], performance_metric = kwargs['performance_metric'], verbose=kwargs['verbose'])
                    self.x_train_before_fs=self.x_train.copy()
                    self.x_test_before_fs=self.x_test.copy()
                    
                    if kwargs['feature_selection'] == 'wrapper' or kwargs['feature_selection'] == 'sfs' or kwargs['feature_selection'] == 'SFS':
                        self.selected_feature_indices, self.selected_variable_names=fs.wrapper(feature_selection_stop=0.05)
                    elif kwargs['feature_selection'] == 'logwrapper':
                        self.selected_feature_indices, self.selected_variable_names, self.log_indices, self.log_variable_names = fs.log_wrapper()
                    elif kwargs['feature_selection'] == 'fst-pso' or kwargs['feature_selection'] == 'fstpso' or kwargs['feature_selection'] == 'pso' or kwargs['feature_selection'] == True:
                        self.selected_feature_indices, self.selected_variable_names, self.nr_clus= fs.fst_pso_feature_selection(max_iter=kwargs['fs_max_iter']) 
                    
                    self.x_train = self.x_train[:, self.selected_feature_indices]
                    self.x_test = self.x_test[:, self.selected_feature_indices]
                elif kwargs['feature_selection'] == None:
                    self.selected_variable_names= self.variable_names                    
                
                # Cluster the data, log-transform when needed.
                if kwargs['feature_selection'] == 'logwrapper':
                    # Use log transformed variables if needed
                    self.log_x_train = self.x_train.copy()
                    for i in self.log_indices:
                        self.log_x_train[i]= np.log(self.x_train[i])
                    cl = Clusterer(x_train=self.log_x_train, y_train=self.y_train, nr_clus=self.nr_clus, verbose=verbose)
                else:                
                    # Cluster the training data (in input-output space)
                    cl = Clusterer(x_train=self.x_train, y_train=self.y_train, nr_clus=self.nr_clus, verbose=verbose)
                
                
                if kwargs['cluster_method'] == 'fcm':
                    self.cluster_centers, self.partition_matrix, _ = cl.cluster(method='fcm', fcm_m=kwargs['m'], 
                        fcm_maxiter=kwargs['fcm_maxiter'], fcm_error=kwargs['fcm_error'])
                elif kwargs['cluster_method'] == 'fst-pso':
                    self.cluster_centers, self.partition_matrix, _ = cl.cluster(method='fstpso', 
                        fstpso_n_particles=kwargs['fstpso_n_particles'], fstpso_max_iter=kwargs['fstpso_max_iter'],
                        fstpso_path_fit_dump=kwargs['fstpso_path_fit_dump'], fstpso_path_sol_dump=kwargs['fstpso_path_sol_dump'])
                elif kwargs['cluster_method'] == 'gk':
                    self.cluster_centers, self.partition_matrix, _ = cl.cluster(method='gk')
                else: 
                    print('ERROR: Choose a valid clustering method.')
                    import sys
                    sys.exit()    
                
                # Estimate the membership funtions of the system (default shape: gauss)
                self._antecedent_estimator = AntecedentEstimator(self.x_train, self.partition_matrix)
        
                self.antecedent_parameters = self._antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=merge_threshold)
                what_to_drop = self._antecedent_estimator._info_for_simplification
        
                # Calculate the firing strengths
                fsc=FireStrengthCalculator(antecedent_parameters=self.antecedent_parameters, nr_clus=self.nr_clus, variable_names=self.selected_variable_names, **kwargs)
                self.firing_strengths = fsc.calculate_fire_strength(data=self.x_train)
                
                # Estimate the parameters of the consequent
                ce = ConsequentEstimator(self.x_train, self.y_train, self.firing_strengths)
                self.consequent_parameters = ce.suglms()
        
                # Build a first-order Takagi-Sugeno model using Simpful
                     
                simpbuilder = SugenoFISBuilder(
                    self.antecedent_parameters, 
                    self.consequent_parameters, 
                    self.selected_variable_names, 
                    normalization_values = self.normalization_values, 
                    extreme_values = self._antecedent_estimator._extreme_values,
                    operators=kwargs["operators"], 
                    save_simpful_code='Fold_' + str(fold_number) +'_Simpful_code.py', 
                    fuzzy_sets_to_drop=what_to_drop,
                    verbose=False)
        
                self.model = simpbuilder.simpfulmodel
                
                # Save the created model in the dedicated folder
                filename = 'Fold_' + str(fold_number) + '.pickle'
                
                pickle.dump(self, open(filename, 'wb'))
                
                tester=SugenoFISTester(model=self.model, test_data=self.x_test, variable_names=self.selected_variable_names, golden_standard=self.y_test)
                self.performance_metric_per_fold[fold_number]=tester.calculate_performance(metric=self.performance_metric)
                
            # set working directory back to where script is stored
            os.chdir(owd)
            print('The average ' + self.performance_metric + ' over ' + str(kwargs['number_of_folds']) +' folds is ' + str(np.mean(self.performance_metric_per_fold)) +' (with st. dev. ' + str(np.std(self.performance_metric_per_fold)) + '). \nThe best model was created in fold ' +  str(np.argmin(self.performance_metric_per_fold)) + ' with ' + self.performance_metric +  ' = ' + str(np.min(self.performance_metric_per_fold)) + '.')
                
        elif kwargs['data_split_method'] == 'no_split':
            if verbose: print('No test data will be split off, all data will be used for training.')
            
            self.x_train = self.dataX.copy()
            self.y_train = self.dataY.copy()
            
            if np.isnan(self.dataX).any().any()== True:
                try:
                    from sklearn.impute import KNNImputer
                except ImportError:
                    raise Exception('pyFUME tried to impute missing values, but couldn`t find \'sklearn\'. Please pip install sklearn to proceed.')

                if verbose: print('Warning: Your data contains missing values that will be imputed using KNN.')
    
                imputer = KNNImputer(n_neighbors=3, weights="uniform")
                self.x_train=imputer.fit_transform(self.x_train)
            
            if kwargs['oversampling'] == True:
                sample = Sampler(train_x = self.x_train, train_y=self.y_train, number_of_bins =  kwargs['sampling_number_of_bins'], histogram =  kwargs['sampling_histogram'])
                self.x_train, self.y_train = sample.oversample()
                    
            # Perform feature selection if requested
            if kwargs['feature_selection'] != None and kwargs['feature_selection'] != False:
                if 'performance_metric' not in kwargs.keys(): kwargs['performance_metric'] = 'MAE'
                fs=FeatureSelector(self.x_train, self.y_train, self.nr_clus, self.variable_names, model_order= kwargs['model_order'], performance_metric = kwargs['performance_metric'], verbose=kwargs['verbose'])
                
                # Keep copies of the training and test set before dropping unselected features
                self.x_train_before_fs=self.x_train.copy()
                            
                if kwargs['feature_selection'] == 'wrapper' or kwargs['feature_selection'] == 'sfs' or kwargs['feature_selection'] == 'SFS':
                    self.selected_feature_indices, self.variable_names=fs.wrapper()
                elif kwargs['feature_selection'] == 'logwrapper':
                    self.selected_feature_indices, self.selected_variable_names, self.log_indices, self.log_variable_names = fs.log_wrapper()
                elif kwargs['feature_selection'] == 'fst-pso' or kwargs['feature_selection'] == 'fstpso' or kwargs['feature_selection'] == 'pso' or kwargs['feature_selection'] == True:
                    self.selected_feature_indices, self.variable_names, self.nr_clus= fs.fst_pso_feature_selection(max_iter=kwargs['fs_max_iter']) 
                self.x_train = self.x_train[:, self.selected_feature_indices]
                
            elif kwargs['feature_selection'] == None:
                self.selected_variable_names= self.variable_names
                
            # Cluster the data, log-transform when needed.
            if kwargs['feature_selection'] == 'logwrapper':
                # Use log transformed variables if needed
                self.log_x_train = self.x_train.copy()
                for i in self.log_indices:
                    self.log_x_train[i]= np.log(self.x_train[i])
                cl = Clusterer(x_train=self.log_x_train, y_train=self.y_train, nr_clus=self.nr_clus, verbose=verbose)
            else:                
                # Cluster the training data (in input-output space)
                cl = Clusterer(x_train=self.x_train, y_train=self.y_train, nr_clus=self.nr_clus, verbose=verbose)
            
            if kwargs['cluster_method'] == 'fcm':
                self.cluster_centers, self.partition_matrix, _ = cl.cluster(method='fcm', fcm_m=kwargs['m'], 
                    fcm_maxiter=kwargs['fcm_maxiter'], fcm_error=kwargs['fcm_error'])
            elif kwargs['cluster_method'] == 'gk':
                self.cluster_centers, self.partition_matrix, _ = cl.cluster(method='gk')
            elif kwargs['cluster_method'] == 'fst-pso':
                self.cluster_centers, self.partition_matrix, _ = cl.cluster(method='fstpso', 
                    fstpso_n_particles=kwargs['fstpso_n_particles'], fstpso_max_iter=kwargs['fstpso_max_iter'],
                    fstpso_path_fit_dump=kwargs['fstpso_path_fit_dump'], fstpso_path_sol_dump=kwargs['fstpso_path_sol_dump'])
            else: 
                print('ERROR: Choose a valid clustering method.')
                import sys
                sys.exit()   
            
            # Estimate the membership funtions of the system (default shape: gauss)
            self._antecedent_estimator = AntecedentEstimator(self.x_train, self.partition_matrix)
    
            self.antecedent_parameters = self._antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=merge_threshold)
            what_to_drop = self._antecedent_estimator._info_for_simplification
            
            # Calculate the firing strengths
            fsc=FireStrengthCalculator(self.antecedent_parameters, self.nr_clus, self.variable_names, **kwargs)
            self.firing_strengths = fsc.calculate_fire_strength(self.x_train)
  
            # Estimate the parameters of the consequent
            ce = ConsequentEstimator(self.x_train, self.y_train, self.firing_strengths)
            if kwargs['model_order'] == 'first':
                self.consequent_parameters = ce.suglms()
            elif kwargs['model_order'] == 'zero':
                self.consequent_parameters = ce.zero_order()
            else:
                raise Exception("pyFUME currently only supports zero-order (model_order = 'zero') and first-order (model_order = 'first') fuzzy models.")
                    
            # Build a first-order Takagi-Sugeno model using Simpful
            simpbuilder = SugenoFISBuilder(
                self.antecedent_parameters, 
                self.consequent_parameters, 
                self.selected_variable_names,  
                normalization_values = self.normalization_values,
                extreme_values = self._antecedent_estimator._extreme_values,
                model_order=kwargs["model_order"],
                operators=kwargs["operators"], 
                save_simpful_code=kwargs['save_simpful_code'], 
                fuzzy_sets_to_drop=what_to_drop,
                verbose=verbose)
    
            self.model = simpbuilder.simpfulmodel
            
        else:
            print('ERROR: invalid data splitting method chosen. Training will be aborted.')
            import sys
            sys.exit()
