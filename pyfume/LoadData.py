import numpy as np
from copy import deepcopy

class DataLoader(object):
    """
        Creates an object that loads data from csv files and normalizes this data if requested.
        
        Args:
            datapath: The path to where the CSV file can be found. The data 
                should be delimited using commas.
            dataframe: Data can be loaded by the user and specified as a dataframe. 
                If a dataframe is specified, the datapath is automatically ignored.
            variable_names: If the CSV file does not contain the variable names,
                the user can specify them here. If this argument is not specified,
                the variable names are read from the first line of the CSV file
                (default = None). 
            normalize: If switch on, the data will be normalized. The user can 
                choose between 'minmax' or 'zscore' normalization (default = False). 
            delimiter: Specify the symbol used to separate data in the dataset (default = ',').
            verbose: Enable verbose mode (default = True).
    """
    
    def __init__(self, datapath=None, dataframe=None, process_categorical=False, 
        variable_names=None, normalize=False, delimiter=',', verbose=True, log_variables = None):
        # user specified a dataframe?
        if dataframe is not None:
            
            import pandas as pd

            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Please specify a valid dataframe.")

            # valid dataframe specified: proceed with creating pyFUME's data structures

            new_data_frame = deepcopy(dataframe)
            output_name=dataframe.columns[-1]
            
            if process_categorical:
                if verbose: print(" * Processing categorical data.")
                new_data_frame = pd.get_dummies(new_data_frame) 
                new_data_frame = new_data_frame[ [ col for col in new_data_frame.columns if col != output_name ] + [output_name] ]

            else:
                list_categorical = self._check_categorical(dataframe)    
                if verbose: 
                    print(" * Dropping categorical data.")
                    print("   Detected categorical variables:", list_categorical)
                
                for cat in list_categorical:
                    new_data_frame = new_data_frame.drop(cat, axis=1)
            
            # DEBUG
            new_data_frame.to_csv("__new_dataset.csv", index=False)
            
            # Convert to matrix for pyFUME processing and 
            # store the names of the columns as variable names.
            self.data = new_data_frame.to_numpy()           
            self.variable_names = np.array(list(new_data_frame.columns)[:-1])
            
            if verbose:
                print(" * Variable names:", self.variable_names)
                print(" * Dataframe %dx%d imported in pyFUME." % self.data.shape)
            

        # pyFUME's conventional behavior
        else:

            if variable_names is None:
                variable_names = np.genfromtxt(datapath, dtype='str', delimiter=delimiter, max_rows=1)
                if verbose ==True:
                    print('The following variable names were detected from the input file: ', variable_names)
                self.output_name=variable_names[-1]
                self.variable_names=variable_names[:-1]
                self.data=np.genfromtxt(datapath, delimiter=delimiter, skip_header=1, filling_values=np.nan)
            else:
                self.variable_names=variable_names
                self.data=np.genfromtxt(datapath, delimiter=delimiter, filling_values=np.nan)
                
        self.dataX=self.data[:,0:-1]
        self.dataY=self.data[:,-1]

        if log_variables is not None:
            if all(isinstance(e, (str)) for e in log_variables):
                idx = []
                for var in log_variables:
                    i = np.argwhere(self.variable_names == var)[0][0]
                    idx.append(i)   
                log_variables = np.array(idx)

            if all(isinstance(e, (int, np.integer)) for e in log_variables):
            # log transform variables
                for var in log_variables:
                    self.dataX[var,:] = np.log(self.dataX[var,:])
                if verbose: print('The following variables were log-transformed:', self.variable_names[log_variables])
            else: raise TypeError("Please specify valid variable indices (as int) or variable names (as strings) for variables to log transform.")


        if normalize=='minmax' or normalize=='linear' or normalize==True:
            if verbose: print('The data is normalized using min-max normalization.')
            self.raw_dataX=self.dataX.copy()
            
            mini=self.dataX.min(axis=0)
            maxi=self.dataX.max(axis=0)
            self.normalization_values=zip(self.variable_names, mini, maxi)

            self.dataX = (self.dataX - self.dataX.min(axis=0)) / (self.dataX.max(axis=0)-self.dataX.min(axis=0))

        elif normalize == 'zscore':
            if verbose: print('The data is normalized using z-score normalization.')
            self.dataX = (self.dataX - self.dataX.mean(axis=0)) / self.dataX.std(axis=0)
        else:
            if verbose: print('The data will not be normalized.')
            

            

    def _check_categorical(self, DF):
        cols = DF.columns[:-1] # last one is ALWAYS the output variable
        num_cols = DF._get_numeric_data().columns
        categ_cols = list( set(cols)-set(num_cols))
        return categ_cols
    
    def get_input_data(self):
        return self.dataX
    
    def get_target_data(self):
        return self.dataY
    
    def get_all_data(self):
        return self.data
    
    def get_variable_names(self):
        return self.variable_names
    
    def get_normalization_values(self):
        return self.normalization_values
    
    def get_non_normalized_x_data(self):
        return self.raw_dataX


if __name__ == "__main__":

    pass