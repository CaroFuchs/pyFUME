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
    """
    
    def __init__(self, datapath=None, dataframe=None, process_categorical=False, variable_names=None, normalize=False, delimiter=','):

        # user specified a dataframe?
        if dataframe is not None:
            
            import pandas as pd

            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Please specify a valid dataframe.")

            # valid dataframe specified: proceed with creating pyFUME's data structures

            if process_categorical:
                print(" * Processing categorical data: ENABLED.")
                list_categorical = self._check_non_categorical(dataframe)    
                new_data_frame = deepcopy(dataframe)
                new_data_frame = pd.get_dummies(new_data_frame) 
                new_data_frame.to_csv("bla.csv")
                self.data = new_data_frame.to_numpy()
                self.variable_names = new_data_frame.columns

            print(" * Dataframe imported in pyFUME.")
            

        # pyFUME's conventional behavior
        else:

            if variable_names is None:
                variable_names = np.genfromtxt(datapath, dtype='str', delimiter=delimiter, max_rows=1)
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
            

    def _check_non_categorical(self, DF):
        cols = DF.columns[:-1] # last one is ALWAYS the output variable
        num_cols = DF._get_numeric_data().columns
        categ_cols = list( set(cols)-set(num_cols))
        print(" * Detected categorical variables:", categ_cols)
        return categ_cols


    def _replace_column(self, DF, column, dummy):
        print (dummy)
        exit()
        DF = DF.drop(column, axis=1)
        DF = DF.join(dummy)
        print(DF)
        exit()

if __name__ == "__main__":

    import pandas as pd
    DF = pd.read_csv("examples\\serious-injury-outcome-indicators-2000-19.csv")
    #print (DF)
    DL = DataLoader(dataframe=DF, process_categorical=True)