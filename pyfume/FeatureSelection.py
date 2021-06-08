from .Splitter import DataSplitter
from .SimpfulModelBuilder import SugenoFISBuilder
from .Clustering import Clusterer
from .EstimateAntecendentSet import AntecedentEstimator
from .FireStrengthCalculator import FireStrengthCalculator
from .EstimateConsequentParameters import ConsequentEstimator
from .Tester import SugenoFISTester
import numpy as np

class FeatureSelector(object):
    """
        Creates a new feature selection object.
        
        Args:
            dataX: The input data.
            dataY: The output data (true label/golden standard).
            nr_clus: Number of clusters that should be identified in the data.
            variable_names: Names of the variables
            **kwargs: Additional arguments to change settings of the fuzzy model.
    """
    def __init__(self, dataX, dataY, nr_clus, variable_names, model_order='first', performance_metric='MAE', verbose=True, **kwargs):
        self.dataX=dataX
        self.dataY=dataY
        self.nr_clus = nr_clus
        self.variable_names = variable_names
        self.performance_metric = performance_metric
        self.model_order=model_order
        self.verbose=verbose
        
    def wrapper(self,**kwargs):
        """
            Performs feature selection using the wrapper method.
            
            Args:
                **kwargs: Additional arguments to change settings of the fuzzy model.
                
            Returns:
                Tuple containing (selected_features, selected_feature_names)
                    - selected_features: The indices of the selected features.
                    - selected_feature_names: The names of the selected features. 

        """
            
        # Create a training and valiadation set for the feature selection phase
        ds = DataSplitter()
        x_feat, y_feat, x_val, y_val = ds.holdout(self.dataX, self.dataY)
        
        # Set initial values for the performance
        old_performance=np.inf
        new_performance=np.inf
        perfs=[]
        
        # Create a set with the unselected (currently all) and selected (none yet) variables
        selected_features=[]
        unselected_features=list(range(0,np.size(x_feat,axis=1)))
        
        stop=False

        while stop == False: 
            perfs= [np.inf]*np.size(x_feat,axis=1)
            
            for f in [x for x in unselected_features if x != -1]:
                from itertools import compress 
                considered_features = selected_features + [f]
                var_names = [self.variable_names[i] for i in considered_features]
                feat=x_feat[:,considered_features]
                x_validation=x_val[:,considered_features] 
                if self.verbose==True: 
                    print('Evaluating feature sub set including:', var_names)
                
                try:
                    perfs[f] = self._evaluate_feature_set(x_data=feat, y_data=y_feat, x_val=x_validation, y_val=y_val, nr_clus=self.nr_clus, var_names=var_names, model_order=self.model_order, performance_metric = self.performance_metric, **kwargs)
                except RuntimeError:
                    raise Exception('ERROR: main module was not safely imported. Feature selection exploits multiprocessing, so please add a `if _name_ == `_main_`: `-line to your main script. See https://docs.python.org/2/library/multiprocessing.html#windows for further info.')
                    import sys
                    sys.exit(1)
                    
            new_performance=min(perfs)
            new_feature=unselected_features[perfs.index(new_performance)]
            
            #print('new feature', new_feature)
            del perfs
            
            if new_performance<old_performance: #*feature_selection_stop:
                selected_features.append(new_feature)
                unselected_features[new_feature]=-1
                old_performance=new_performance
            else:
                print('The selected features have a', self.performance_metric, 'of:', old_performance)
                stop = True 
       
        # selected_feature_names = self.variable_names[selected_features]
        selected_feature_names = [self.variable_names[i] for i in selected_features]
        print('The following features were selected:',  selected_feature_names)
        
        return selected_features, selected_feature_names

    def fst_pso_feature_selection(self,max_iter=100, min_clusters=2, max_clusters=10, performance_metric='MAE', **kwargs):
        """
            Perform feature selection using the FST-PSO [1] variant of the Integer and Categorical 
            PSO (ICPSO) proposed by Strasser and colleagues [2]. ICPSO hybridizes PSO and Estimation of Distribution 
            Algorithm (EDA), which makes it possible to convert a discrete problem to the (real-valued) 
            problem of estimating the distribution vector of a probabilistic model. Each fitness 
            evaluation a random solution is generated according to the probability distribution 
            encoded by the particle. Because the implementation is a variant on FST-PSO, the optimal 
            settings for the PSO are set automatically.

            If the number of clusters is set to None, this method simultaneously choses the optimal 
            number of clusters.

            [1] Nobile, M. S., Cazzaniga, P., Besozzi, D., Colombo, R., Mauri, G., & Pasi, G. (2018). 
            Fuzzy Self-Tuning PSO: A settings-free algorithm for global optimization. Swarm and 
            evolutionary computation, 39, 70-85.
            
            [2] Strasser, S., Goodman, R., Sheppard, J., & Butcher, S. (2016). A new discrete 
            particle swarm optimization algorithm. In Proceedings of the Genetic and Evolutionary 
            Computation Conference 2016 (pp. 53-60). 
            
            Args:
                max_iter: The maximum number of iterations used in the PSO (default = 10).
                min_clusters: The minimum number of clusters to be identified in the data set (only 
                when nr_clusters = None)
                max_clusters: The maximum number of clusters to be identified in the data set (only 
                when nr_clusters = None)
                performance_metric: The performance metric on which each solution is evaluated (default
                Mean Absolute Error (MAE))
                **kwargs: Additional arguments to change settings of the fuzzy model.
                
            Returns:
                Tuple containing (selected_features, selected_feature_names, optimal_number_clusters)
                    - selected_features: The indices of the selected features.
                    - selected_feature_names: The names of the selected features.
                    - optimal_number_clusters: If initially nr_clusters = None, this argument encodes the optimal number of clusters in the data set. If nr_clusters is not None, the optimal_number_clusters is set to nr_clusters.

        """
        
        from fstpso import FuzzyPSO


        FP = FuzzyPSO()
        
        # Create the search space for feature selection with number of dimensions D
        D = np.size(self.dataX,1)
        
        s=list([[True, False]]*D)
        
        # Add dimension for cluster number selection
        if self.nr_clus == None:
            s.append(list(range(min_clusters,max_clusters+1)))
        
        # Set search space
        FP.set_search_space_discrete(s)
        
        # Set the fitness function
        args={'x_train': self.dataX, 'y_train': self.dataY, 'verbose':self.verbose}
        FP.set_fitness(self._function, arguments=args)
        
        # solve problem with FST-PSO
        _, best_performance, best_solution = FP.solve_with_fstpso(max_iter=max_iter)
        
        if self.nr_clus == None:
            selected_features=best_solution[:-1]
        else:
           selected_features=best_solution 
    
        
        # Show best solution with fitness value
        varnams=[i for indx,i in enumerate(self.variable_names) if selected_features[indx]]
        print('The following features have been selected:', varnams, 'with a', self.performance_metric, 'of', round(best_performance,2))
        
        if self.nr_clus == None:
            optimal_number_clusters=best_solution[-1]
        else:
            optimal_number_clusters = self.nr_clus
            
            
        return selected_features, varnams, optimal_number_clusters

#    def fun(self, particle):
#        return sum(particle)
    
    def _function(self, particle, arguments, verbose=True, **kwargs):
        from itertools import compress 
        if self.nr_clus == None:
            A = arguments['x_train'][:,particle[:-1]]
            varnams=list(compress(self.variable_names, particle[:-1]))
            nr_clus=particle[-1]
        else:
            A = arguments['x_train'][:,particle[:]]
            varnams=list(compress(self.variable_names, particle[:]))
            nr_clus=self.nr_clus
        
        if A.shape[1]==0: ## If no features are selected, return a infinite high error
            error=np.inf
        else:
            error=self._evaluate_feature_set(x_data=A, y_data=arguments['y_train'], nr_clus=nr_clus, var_names=varnams, model_order=self.model_order, performance_metric=self.performance_metric, **kwargs)
            
        if verbose: print(" * Fitness: %.3f" % error)
        return error
    
    def _evaluate_feature_set(self, x_data, y_data, nr_clus, var_names, model_order='first', performance_metric='MAE', fs_number_of_folds=3, **kwargs):
        # Check settings and complete with default settings when needed
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
        if 'verbose' not in kwargs.keys(): kwargs['verbose'] = False
                
        # Split the data using the hold-out method in a training (default: 75%) 
        # and test set (default: 25%).
        ds = DataSplitter()

        if fs_number_of_folds==1: ##### feauture selection with hold-out
            x_train, y_train, x_val, y_val = ds.holdout(x_data, y_data) 

            cl = Clusterer(x_train=x_train, y_train=y_train, nr_clus=nr_clus)               
                
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
            antecedent_estimator = AntecedentEstimator(x_train, partition_matrix)

            antecedent_parameters = antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=kwargs['merge_threshold'])
            what_to_drop = antecedent_estimator._info_for_simplification

            # Build a first-order Takagi-Sugeno model using Simpful using dummy consequent parameters
            fsc=FireStrengthCalculator(antecedent_parameters, nr_clus, var_names, **kwargs)
            firing_strengths = fsc.calculate_fire_strength(x_train)
            
            # Estimate the parameters of the consequent
            ce = ConsequentEstimator(x_train, y_train, firing_strengths)
            
            if self.model_order=='first':
                consequent_parameters = ce.suglms()
            elif self.model_order== 'zero':
                consequent_parameters = ce.zero_order()

            # Build a first-order Takagi-Sugeno model using Simpful
            simpbuilder = SugenoFISBuilder(
                antecedent_parameters, 
                consequent_parameters, 
                var_names, 
                extreme_values = antecedent_estimator._extreme_values,
                operators=kwargs['operators'], 
                model_order=self.model_order,
                save_simpful_code=False, 
                fuzzy_sets_to_drop=what_to_drop,
                verbose=kwargs['verbose'])

            model = simpbuilder.simpfulmodel
            
            # Test the model
            test = SugenoFISTester(model=model, test_data=x_val, golden_standard=y_val,variable_names=var_names)
            performance= test.calculate_performance(metric=self.performance_metric)
        
        elif fs_number_of_folds>1:  ##### feauture selection with cross validation
            fold_indices = ds.kfold(data_length=np.shape(x_data)[0], number_of_folds=fs_number_of_folds)
            # perf=[np.inf]*fs_number_of_folds

            arg=[]
            for fold_number in range(0, fs_number_of_folds):
                
                # Choose the indices for training and testing for this fold
                tst_idx=fold_indices[fold_number]
                np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
                trn_idx=np.concatenate(np.delete(fold_indices, fold_number, axis=0))       # Use all indices, except the ones that are used for testing
                
                x_train = np.array([x_data[i,:] for i in trn_idx])
                x_val = np.array([x_data[i,:] for i in tst_idx])                      
                y_train = np.array([y_data[i] for i in trn_idx])
                y_val = np.array([y_data[i] for i in tst_idx]) 
                
                arg.append([x_train, y_train, x_val, y_val, nr_clus, var_names])
            
            try:
                from multiprocessing import Pool
            except ImportError:
                    raise Exception('pyFUME uses multiprocessing to parallelize computations, but couldn`t find \'multiprocessing\'. Please pip install multiprocessing to proceed.')
            
            try:
                with Pool(fs_number_of_folds) as p:
                    perf=p.starmap(func=self._create_model, iterable=arg)
            except RuntimeError:
                raise Exception('ERROR: main module was not safely imported. Feature selection exploits multiprocessing, so please add a `if _name_ == `_main_`: `-line to your main script. See https://docs.python.org/2/library/multiprocessing.html#windows for further info')
                
                import os
                import signal
                os.kill(os.getppid(), signal.SIGTERM)
                
            performance = np.mean(perf)
            
            return performance


            #     # Cluster the training data (in input-output space)
            #     cl = Clusterer(x_train=x_train, y_train=y_train, nr_clus=nr_clus)                               
                
            #     if kwargs['cluster_method'] == 'fcm':
            #         cluster_centers, partition_matrix, _ = cl.cluster(cluster_method='fcm', fcm_m=kwargs['fcm_m'], 
            #             fcm_maxiter=kwargs['fcm_maxiter'], fcm_error=kwargs['fcm_error'])
            #     elif kwargs['cluster_method'] == 'gk':
            #         cluster_centers, partition_matrix, _ = cl.cluster(cluster_method='gk')
            #     elif kwargs['cluster_method'] == 'fstpso':
            #         cluster_centers, partition_matrix, _ = cl.cluster(cluster_method='fstpso', 
            #             fstpso_n_particles=kwargs['fstpso_n_particles'], fstpso_max_iter=kwargs['fstpso_max_iter'],
            #             fstpso_path_fit_dump=kwargs['fstpso_path_fit_dump'], fstpso_path_sol_dump=kwargs['fstpso_path_sol_dump'])
            #     else:
            #         print('The requested clustering method is not (yet) implemented')
                     
            #     # Estimate the membership funtions of the system (default shape: gauss)
            #     antecedent_estimator = AntecedentEstimator(x_train, partition_matrix)

            #     antecedent_parameters = antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=kwargs['merge_threshold'])
            #     what_to_drop = antecedent_estimator._info_for_simplification

            #     # Build a first-order Takagi-Sugeno model using Simpful using dummy consequent parameters
            #     fsc=FireStrengthCalculator(antecedent_parameters, nr_clus, var_names, **kwargs)
            #     firing_strengths = fsc.calculate_fire_strength(x_train)

            #     # Estimate the parameters of the consequent
            #     ce = ConsequentEstimator(x_train, y_train, firing_strengths)
            
            #     if self.model_order=='first':
            #         consequent_parameters = ce.suglms()
            #     elif self.model_order== 'zero':
            #         consequent_parameters = ce.zero_order()
                    
            #     # Build a first-order Takagi-Sugeno model using Simpful
            #     simpbuilder = SugenoFISBuilder(
            #         antecedent_parameters, 
            #         consequent_parameters, 
            #         var_names, 
            #         extreme_values = antecedent_estimator._extreme_values,
            #         operators=kwargs["operators"], 
            #         model_order=self.model_order,
            #         save_simpful_code=False, 
            #         fuzzy_sets_to_drop=what_to_drop,
            #         verbose=kwargs['verbose'])

            #     model = simpbuilder.simpfulmodel
                
            #     # Test the model
            #     test = SugenoFISTester(model=model, test_data=x_val, golden_standard=y_val,variable_names=var_names)
            #     perf[fold_number]= test.calculate_performance(metric=self.performance_metric) 

            # performance = np.mean(perf)

        # return performance
   
    def _create_model(self, x_train, y_train, x_test, y_test, nr_clus, var_names, **kwargs):
        # Cluster the training data (in input-output space)
        cl = Clusterer(x_train=x_train, y_train=y_train, nr_clus=nr_clus)
        cluster_centers, partition_matrix, _ = cl.cluster(cluster_method='fcm')                               
                
        # if kwargs['cluster_method'] == 'fcm':
        #     cluster_centers, partition_matrix, _ = cl.cluster(cluster_method='fcm', fcm_m=kwargs['fcm_m'], 
        #         fcm_maxiter=kwargs['fcm_maxiter'], fcm_error=kwargs['fcm_error'])
        # elif kwargs['cluster_method'] == 'gk':
        #     cluster_centers, partition_matrix, _ = cl.cluster(cluster_method='gk')
        # elif kwargs['cluster_method'] == 'fstpso':
        #     cluster_centers, partition_matrix, _ = cl.cluster(cluster_method='fstpso', 
        #         fstpso_n_particles=kwargs['fstpso_n_particles'], fstpso_max_iter=kwargs['fstpso_max_iter'],
        #         fstpso_path_fit_dump=kwargs['fstpso_path_fit_dump'], fstpso_path_sol_dump=kwargs['fstpso_path_sol_dump'])
        # else:
        #     print('The requested clustering method is not (yet) implemented')
             
        # # Estimate the membership funtions of the system (default shape: gauss)
        antecedent_estimator = AntecedentEstimator(x_train, partition_matrix)
        
        antecedent_parameters = antecedent_estimator.determineMF()
        #antecedent_parameters = antecedent_estimator.determineMF(mf_shape=kwargs['mf_shape'], merge_threshold=kwargs['merge_threshold'])
        what_to_drop = antecedent_estimator._info_for_simplification

        # Build a first-order Takagi-Sugeno model using Simpful using dummy consequent parameters
        fsc=FireStrengthCalculator(antecedent_parameters, nr_clus, var_names)
        firing_strengths = fsc.calculate_fire_strength(x_train)

        # Estimate the parameters of the consequent
        ce = ConsequentEstimator(x_train, y_train, firing_strengths)
    
        if self.model_order=='first':
            consequent_parameters = ce.suglms()
        elif self.model_order== 'zero':
            consequent_parameters = ce.zero_order()
            
        # Build a first-order Takagi-Sugeno model using Simpful
        simpbuilder = SugenoFISBuilder(
            antecedent_parameters, 
            consequent_parameters, 
            var_names, 
            extreme_values = antecedent_estimator._extreme_values, 
            model_order=self.model_order,
            save_simpful_code=False, 
            fuzzy_sets_to_drop=what_to_drop)
        
        # simpbuilder = SugenoFISBuilder(
        #     antecedent_parameters, 
        #     consequent_parameters, 
        #     var_names, 
        #     extreme_values = antecedent_estimator._extreme_values,
        #     operators=kwargs['operators'], 
        #     model_order=self.model_order,
        #     save_simpful_code=False, 
        #     fuzzy_sets_to_drop=what_to_drop,
        #     verbose=kwargs['verbose'])

        model = simpbuilder.simpfulmodel
        
        # Test the model
        test = SugenoFISTester(model=model, test_data=x_test, golden_standard=y_test, variable_names=var_names)
        perf = test.calculate_performance(metric=self.performance_metric) 
        return perf
