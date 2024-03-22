from pyfume import pyFUME
import pandas as pd
import numpy as np

# Read a Pandas dataframe (using the Pandas library)
df = pd.read_csv('./Concrete_data.csv')

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