from LoadData import DataLoader
from Splitter import DataSplitter
from SimpfulModelBuilder import SugenoFISBuilder
from Clustering import Clusterer
from EstimateAntecendentSet import AntecedentEstimator
from EstimateConsequentParameters import ConsequentEstimator
from Tester import SugenoFISTester


class BuildTSFIS(object):
    def __init__(self, datapath, nr_clus, variable_names=None, **kwargs):
        self.datapath = datapath
        self.nr_clus = nr_clus
        self.variable_names = variable_names
        
        # Load the data
        if 'normalize' not in kwargs.keys(): kwargs['normalize'] = False       
        dl=DataLoader(self.datapath,normalize=kwargs['normalize'])
        self.variable_names=dl.variable_names        

        # Split the data using the hold-out method in a training (default: 75%) 
        # and test set (default: 25%).
        ds = DataSplitter(dl.dataX,dl.dataY)
        self.x_train, self.y_train, self.x_test, self.y_test = ds.holdout(dl.dataX, dl.dataY)
        
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

        ae = AntecedentEstimator(self.x_train, self.partition_matrix)
        self.antecedent_parameters = ae.determineMF(self.x_train, self.partition_matrix, mf_shape=kwargs['mf_shape'])
        
        # Estimate the parameters of the consequent (default: global fitting)
        if 'global_fit' not in kwargs.keys(): kwargs['global_fit'] = True  
        ce = ConsequentEstimator(self.x_train, self.y_train, self.partition_matrix)
        self.consequent_parameters = ce.suglms(self.x_train, self.y_train, self.partition_matrix, 
                                               global_fit=kwargs['global_fit'])
        
        # Build a first-order Takagi-Sugeno model using Simpful
        if 'save_simpful_code' not in kwargs.keys(): kwargs['save_simpful_code'] = True           
        simpbuilder = SugenoFISBuilder(self.antecedent_parameters, self.consequent_parameters, self.variable_names, operators=kwargs["operators"], save_simpful_code=kwargs['save_simpful_code'])
        self.model = simpbuilder.simpfulmodel

        """        
        # Calculate the mean squared error of the model using the test data set
        test = SugenoFISTester(self.model, self.x_test,self.y_test)
        RMSE = test.calculate_RMSE(variable_names=self.variable_names)
        self.RMSE = list(RMSE.values())
        print('The RMSE of the fuzzy system is', self.RMSE)
        """