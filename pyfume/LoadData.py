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
        variable_names=None, normalize=False, delimiter=',', verbose=True):

        # user specified a dataframe?
        if dataframe is not None:
            
            import pandas as pd

            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Please specify a valid dataframe.")

            # valid dataframe specified: proceed with creating pyFUME's data structures

            new_data_frame = deepcopy(dataframe)
            
            if process_categorical:
                if verbose: print(" * Processing categorical data.")
                new_data_frame = pd.get_dummies(new_data_frame) 

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
            self.variable_names = list(new_data_frame.columns)
            
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

        if normalize=='minmax' or normalize=='linear' or normalize==True:
            self.dataX = (self.dataX - np.abs(self.dataX).min(axis=0)) / (np.abs(self.dataX).max(axis=0)-np.abs(self.dataX).min(axis=0))

        elif normalize == 'zscore':
            self.dataX = (self.dataX - self.dataX.mean(axis=0)) / self.dataX.std(axis=0)
            

    def _check_categorical(self, DF):
        cols = DF.columns[:-1] # last one is ALWAYS the output variable
        num_cols = DF._get_numeric_data().columns
        categ_cols = list( set(cols)-set(num_cols))
        return categ_cols



if __name__ == "__main__":

    import pandas as pd
    DF = pd.read_csv("examples\\serious-injury-outcome-indicators-2000-19.csv")
    DL = DataLoader(dataframe=DF, process_categorical=False)