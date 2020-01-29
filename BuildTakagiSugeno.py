import numpy as np
from LoadData import DataLoader
from Splitter import DataSplitter
from SimpfulModelBuilder import SugenoFISBuilder
from Clustering import Clusterer
from EstimateAntecendentSet import AntecedentEstimator
from EstimateConsequentParameters import ConsequentEstimator
from Tester import SugenoFISTester


class BuildTSFIS(object):
    def __init__(self, datapath, nr_clus, variable_names=None):
        self.datapath = datapath
        self.nr_clus = nr_clus
        self.variable_names = variable_names
        
        # Load the data
        dl=DataLoader(self.datapath,normalize=0)
        self.variable_names=dl.variable_names        

        # Split the data using the hold-out method in a training (default: 75%) 
        # and test set (default: 25%).
        ds = DataSplitter(dl.dataX,dl.dataY)
        self.x_train, self.y_train, self.x_test, self.y_test = ds.holdout(dl.dataX, dl.dataY)
        
        # Cluster the training data (in input-output space) using FCM
        cl = Clusterer(self.x_train, self.y_train, self.nr_clus)
        self.cluster_centers, self.partition_matrix, _ = cl.cluster(method="fcm")
        
        # Estimate the membership funtions of the system (default shape: 2gauss)
        ae = AntecedentEstimator(self.x_train, self.partition_matrix)
        self.antecedent_parameters = ae.determineMF(self.x_train, self.partition_matrix)
        
        # Estimate the parameters of the consequent (default: global fitting)
        ce = ConsequentEstimator(self.x_train, self.y_train, self.partition_matrix)
        self.consequent_parameters = ce.suglms(self.x_train, self.y_train, self.partition_matrix)
        
        # Build a first-order Takagi-Sugeno model using Simpful
        simpbuilder = SugenoFISBuilder(self.antecedent_parameters, self.consequent_parameters, self.variable_names)
        self.model = simpbuilder.simpfulmodel
        
        # Calculate the mean squared error of the model using the test data set
        test = SugenoFISTester(self.model, self.x_test,self.y_test)
        error = test.calculate_RMSE(variable_names=self.variable_names)
        self.err = list(error.values())
        print('The RMSE of the fuzzy system is', self.err)

