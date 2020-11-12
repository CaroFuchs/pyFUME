# pyFUME

pyFUME is a Python package for automatic Fuzzy Models Estimation from data [1].
pyFUME contains functions to estimate the antecedent sets and the consequent parameters of a Takagi-Sugeno fuzzy model directly from data. This information is then used to create an executable fuzzy model using the Simpful library.
pyFUME also provides facilities for the evaluation of performance from a statistical standpoint.

## Usage
For the following example, we use the Concrete Compressive Strength data set [2] as can be found in the UCI repository.
The  code  in  Example 1  is  simple  and  easy  to  use,  making it  ideal  to  use  for  practitioners  who  wish  to  use  the  default settings or only wish to use few non-default settings using additional input arguments (Example 2). 
Users that wish to deviate from  the  default  settings  can  use  the code  as shown  in  Example 3.

### Example 1
```
from pyfume import pyFUME

# Set the path to the data and choose the number of clusters
path='./Concrete_data.csv'
nc=3

# Generate the Takagi-Sugeno FIS
FIS = pyFUME(datapath=path, nr_clus=nc)

# Calculate and print the accuracy of the generated model
MAE=FIS.calculate_error(method="MAE")
print ("The estimated error of the developed model is:", MAE)

## Use the FIS to predict the compressive strength of a new concrete sample
# Extract the model from the FIS object
model=FIS.get_model()

# Set the values for each variable
model.set_variable('Cement', 300.0)
model.set_variable('BlastFurnaceSlag', 50.0)
model.set_variable('FlyAsh', 0.0)
model.set_variable('Water', 175.0)
model.set_variable('Superplasticizer',0.7)
model.set_variable('CoarseAggregate', 900.0)
model.set_variable('FineAggregate', 600.0)
model.set_variable('Age', 45.0)

# Perform inference and print predicted value
print(model.Sugeno_inference(['OUTPUT']))
```

### Example 2
```
from pyfume import pyFUME

# Set the path to the data and choose the number of clusters
path='./Concrete_data.csv'
nc=3

# Generate the Takagi-Sugeno FIS
FIS = pyFUME(datapath=path, nr_clus=nc, normaliztion='minmax', feature_selection=True)

# Calculate and print the accuracy of the generated model
MAE=FIS.calculate_error(method="MAE")
print ("The estimated error of the developed model is:", MAE)

## Use the FIS to predict the compressive strength of a new concrete sample
# Extract the model from the FIS object
model=FIS.get_model()

# Set the values for each variable
model.set_variable('Cement', 300.0)
model.set_variable('BlastFurnaceSlag', 50.0)
model.set_variable('FlyAsh', 0.0)
model.set_variable('Water', 175.0)
model.set_variable('Superplasticizer',0.7)
model.set_variable('CoarseAggregate', 900.0)
model.set_variable('FineAggregate', 600.0)
model.set_variable('Age', 45.0)

# Perform inference and print predicted value
print(model.Sugeno_inference(['OUTPUT']))
```

### Example 3

```
from LoadData import DataLoader
from Splitter import DataSplitter
from ModelBuilder import SugenoFISBuilder
from Clustering import Clusterer
from EstimateAntecendentSet import AntecedentEstimator
from EstimateConsequentParameters import ConsequentEstimator
from Tester import SugenoFISTester

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
ds = DataSplitter(dl.dataX,dl.dataY)
x_train, y_train, x_test, y_test = ds.holdout(dataX, dataY)

# Select features relevant to the problem
fs=FeatureSelector(self.x_train, self.y_train, self.nr_clus, self.variable_names)
selected_feature_indices, variable_names=fs.wrapper()

# Adapt the training and test input data after feature selection
x_train = x_train[:, selected_feature_indices]
x_test = x_test[:, selected_feature_indices]
      
# Cluster the training data (in input-output space) using FCM with default settings
cl = Clusterer(x_train, y_train, nr_clus)
cluster_centers, partition_matrix, _ = cl.cluster(method="fcm")
     
# Estimate the membership funtions of the system (default: mf_shape = gaussian)
ae = AntecedentEstimator(x_train, partition_matrix)
antecedent_parameters = ae.determineMF()
        
# Estimate the parameters of the consequent functions
ce = ConsequentEstimator(x_train, y_train, partition_matrix)
consequent_parameters = ce.suglms(x_train, y_train, partition_matrix)
        
# Build a first-order Takagi-Sugeno model using Simpful. Specify the optional 
# 'extreme_values' argument to specify the universe of discourse of the input
# variables if you which to use Simpful's membership function plot functionalities.
simpbuilder = SugenoFISBuilder(antecedent_parameters, consequent_parameters, variable_names)
model = simpbuilder.get_model()
        
# Calculate the mean squared error (MSE) of the model using the test data set
MAE = test.calculate_MAE(variable_names=variable_names)

print('The mean absolute error of the created model is', MAE)
```

## Installation

`pip install pyfume`


## Further information
If you need further information, please write an e-mail to Caro Fuchs: c.e.m.fuchs(at)tue.nl.


## References
[1] Fuchs, C., Spolaor, S., Nobile, M. S., & Kaymak, U. (2020) "pyFUME: a Python package for fuzzy model estimation". In 2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE) (pp. 1-8). IEEE.

[2] I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998). http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength


