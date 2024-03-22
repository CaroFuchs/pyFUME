from pyfume import *

# Set the path to the data and choose the number of clusters
path='./Concrete_data.csv'
nr_clus=3

# Load and normalize the data using min-max normalization
dl=DataLoader(path,normalize='minmax')
variable_names=dl.variable_names 
dataX=dl.dataX
dataY=dl.dataY

# Split the data using the hold-out method in a training (default: 75%) 
# and test set (default: 25%).
ds = DataSplitter()
x_train, y_train, x_test, y_test = ds.holdout(dataX=dl.dataX, dataY=dl.dataY)

# Select features relevant to the problem
fs=FeatureSelector(dataX=x_train, dataY=y_train, nr_clus=nr_clus, variable_names=variable_names)
selected_feature_indices, variable_names=fs.wrapper()

# Adapt the training and test input data after feature selection
x_train = x_train[:, selected_feature_indices]
x_test = x_test[:, selected_feature_indices]
      
# Cluster the training data (in input-output space) using FCM with default settings
cl = Clusterer(x_train=x_train, y_train=y_train, nr_clus=nr_clus)
cluster_centers, partition_matrix, _ = cl.cluster(method="fcm")
     
# Estimate the membership funtions of the system (default: mf_shape = gaussian)
ae = AntecedentEstimator(x_train=x_train, partition_matrix=partition_matrix)
antecedent_parameters = ae.determineMF()

# Calculate the firing strength of each rule for each data instance        
fsc=FireStrengthCalculator(antecedent_parameters=antecedent_parameters, nr_clus=nr_clus, variable_names=variable_names)
firing_strengths = fsc.calculate_fire_strength(data=x_train)

# Estimate the parameters of the consequent functions
ce = ConsequentEstimator(x_train=x_train, y_train=y_train, firing_strengths=firing_strengths)
consequent_parameters = ce.suglms()
        
# Build a first-order Takagi-Sugeno model using Simpful. Specify the optional 
# 'extreme_values' argument to specify the universe of discourse of the input
# variables if you which to use Simpful's membership function plot functionalities.
simpbuilder = SugenoFISBuilder(antecedent_sets=antecedent_parameters, consequent_parameters=consequent_parameters, variable_names=variable_names)
model = simpbuilder.get_model()

# Calculate the mean squared error (MSE) of the model using the test data set
test=SugenoFISTester(model=model, test_data=x_test, variable_names=variable_names, golden_standard=y_test)
MSE = test.calculate_MSE()

print('The mean squared error of the created model is', MSE)