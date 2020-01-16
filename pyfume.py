from Splitter import DataSplitter
from ModelBuilder import SugenoFISBuilder
from Clustering import Clusterer
from EstimateAntecendentSet import AntecedentEstimator
from EstimateConsequentParameters import ConsequentEstimator
from Tester import SugenoFISTester


class BuildTSFIS(object):
    def __init__(self, datapath, nr_clus, variable_names):
        self.datapath=datapath
        self.nr_clus=nr_clus
        self.variable_names=variable_names
        
        # Split the data using the hold-out method in a training (default: 75%) 
        # and test set (default: 25%).
        ds=DataSplitter(self.datapath)
        self.x_train, self.y_train, self.x_test, self.y_test = ds.holdout(ds.dataX,ds.dataY,percentage_training=0.75)
        
        # Cluster the training data (in input-output space) using FCM
        cl=Clusterer(self.x_train, self.y_train, self.nr_clus)
        _,self.partition_matrix,_=cl.fcm(cl.data, self.nr_clus)
        
        # Estimate the membership funtions of the system (default shape: 2gauss)
        ae=AntecedentEstimator(self.x_train, self.partition_matrix, mf_shape='2gauss')
        self.antecedent_sets = ae.determineMF(self.x_train, self.partition_matrix)
        
        # Estimate the parameters of the consequent (default: global fitting)
        ce=ConsequentEstimator(self.x_train, self.y_train, self.partition_matrix)
        self.consequent_parameters = ce.suglms(self.x_train, self.y_train, self.partition_matrix)
        
        # Build a first-order Takagi-Sugeno model using Simpful
        simpbuilder=SugenoFISBuilder(self.antecedent_sets, self.consequent_parameters, self.variable_names)
        self.model=simpbuilder.simpfulmodel
        
        # Calculate the mean squared error of the model using the test data set
        test=SugenoFISTester(self.model, self.x_test,self.y_test)
        error=test.calculate_MSE(variable_names=variable_names)
        print('The RMSE of the system is', error)
        

if __name__=='__main__':
 FIS = BuildTSFIS(datapath='../data/dataset6_withlabels.txt', nr_clus=3, variable_names=['v1','v2'])
 print (dir(FIS.model))