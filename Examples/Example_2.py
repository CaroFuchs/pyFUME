from pyfume import pyFUME

# Set the path to the data and choose the number of clusters
path='./Concrete_data.csv'
nc=3

# Generate the Takagi-Sugeno FIS
FIS = pyFUME(datapath=path, nr_clus=nc, feature_selection='wrapper')

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