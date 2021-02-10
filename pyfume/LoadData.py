import numpy as np

class DataLoader(object):
    """
        Creates an object that loads data from csv files and normalizes this data if requested.
        
        Args:
            datapath: The path to where the CSV file can be found. The data 
                should be delimited using commas.
            variable_names: If the CSV file does not contain the variable names,
                the user can specify them here. If this argument is not specified,
                the variable names are read from the first line of the CSV file
                (default = None). 
            normalize: If switch on, the data will be normalized. The user can 
                choose between 'minmax' or 'zscore' normalization (default = False). 
    """
    
    def __init__(self, datapath, variable_names=None, normalize=False,delimiter=','):
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
        
