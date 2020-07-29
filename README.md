# pyFUME

pyFUME is a Python package for automatic Fuzzy Models Estimation from data.
pyFUME contains functions to estimate the antecedent sets and the consequent parameters of a Takagi-Sugeno fuzzy model directly from data. This information is then used to create an executable fuzzy model using the Simpful library.
pyFUME also provides facilities for the evaluation of performance from a statistical standpoint.

## Usage
For the following example, we use the Concrete Compressive Strength data set [1] as can be found in the UCI repository
```
from pyfume import *

# Set the path to the data and choose the number of clusters
path='./Concrete_data.csv'
nc=3

# Generate the Takagi-Sugeno FIS
FIS = pyFUME(datapath=path, nr_clus=nc)

# Calculate and print the accuracy of the generated model
print ("The calculated error is:", FIS.calculate_error())

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

## Installation

`pip install pyfume`


## Further information
If you need further information, please write an e-mail to Caro Fuchs: c.e.m.fuchs(at)tue.nl.


[1] I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998). http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength


