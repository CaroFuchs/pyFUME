# pyFUME

pyFUME is a Python package for automatic Fuzzy Models Estimation from data [1].
pyFUME contains functions to estimate the antecedent sets and the consequent parameters of a Takagi-Sugeno fuzzy model directly from data. This information is then used to create an executable fuzzy model using the Simpful library.
pyFUME also provides facilities for the evaluation of performance.
For more information about pyFUME's functionalities, please check the [online documentation](https://pyfume.readthedocs.io/en/latest/).

## Usage
For the following example, we use the Concrete Compressive Strength data set [2] as can be found in the UCI repository.
The  code  in  Example 1  is  simple  and  easy  to  use,  making it  ideal  to  use  for  practitioners  who  wish  to  use  the  default settings or only wish to use few non-default settings using additional input arguments (Example 2). 
Users that wish to deviate from  the  default  settings  can  use  the code  as shown  in  Example 3.
The code of the Simpful model that is generated is automatically saved (in the same location as the pyFUME script is ran from) under the name 'Simpful_code.py'

## Note
Please be aware that pyFUME's feature selection functionality makes use of multiprocessing. 
When feature selection is used, the main script should always be guarded by including "if \_\_name\_\_ == '\_\_main\_\_':" in the header the script.
When the Spyder IDE is used, one should include "if \_\_name\_\_ == '\_\_main\_\_' and '\_\_file\_\_' in globals():".

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
FIS = pyFUME(datapath=path, nr_clus=nc, feature_selection='fst-pso')

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
```

### Example 4

```
from pyfume import pyFUME
import pandas as pd
import numpy as np

# Read a Pandas dataframe (using the Pandas library)
df = pd.read_csv('.\Concrete_data.csv')

# Generate the Takagi-Sugeno FIS
FIS = pyFUME(dataframe=df, nr_clus=2)

# Calculate and print the accuracy of the generated model
MAE=FIS.calculate_error(method="MAE")
print ("The estimated error of the developed model is:", MAE)

### Use the FIS to predict the compressive strength of a new concrete samples

## Using Simpful's syntax (NOTE: This approach ONLY works for models built using non-normalized data!)   
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
print('The output using Simpfuls "set_variable" functionality is:', model.Sugeno_inference(['OUTPUT']))

## Using pyFUME's syntax (NOTE: This approach DOES work for models built using normalized data!)
# Create numpy array (matrix) in which each row is a data instance to be processed
new_data_one_instance=np.array([[300, 50,0,175,0.7,900,600,45]]) 
prediction_labels_one_instance=FIS.predict_label(new_data_one_instance)
print('The output using pyFUMEs "predict_label" functionality is:', prediction_labels_one_instance)

# Example in which output for multiple data instances is computed
new_data_multiple_instances=np.array([[300, 50,0,175,0.7,900,600,45],[500, 75,30,200,0.9,600,760,39],[250, 40,10,175,0.3,840,360,51]]) 
prediction_labels_multiple_instance=FIS.predict_label(new_data_multiple_instances)
print('The output using pyFUMEs "predict_label" functionality is:', prediction_labels_multiple_instance)

### Plot the actual values vs the predicted values of the test data using the matplotlib library

# Predict the labels of the test data
pred = FIS.predict_test_data()

# Get the actual labels of the test data
_, actual = FIS.get_data(data_set='test')

# Create scatterplot
import matplotlib.pyplot as plt 
plt.scatter(actual, pred)
plt.xlabel('Actual value') 
plt.ylabel('Predicted value')
plt.plot([0,85],[0,85],'r')     # Add a reference line
plt.show()


```

## Installation

`pip install pyfume`


## Further information
If you need further information, please write an e-mail to Caro Fuchs: c.e.m.fuchs(at)tue.nl.


## References
[1] Fuchs, C., Spolaor, S., Nobile, M. S., & Kaymak, U. (2020) "pyFUME: a Python package for fuzzy model estimation". In 2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE) (pp. 1-8). IEEE.

[2] I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998). http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength


