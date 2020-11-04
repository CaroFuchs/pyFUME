import numpy as np

class DataLoader(object):
        def __init__(self, datapath, variable_names=None, normalize=False):
            if variable_names is None:
                variable_names = np.genfromtxt(datapath, dtype='str', delimiter=',', max_rows=1)
                print('The following variable names were detected from the input file: ', variable_names)
                self.output_name=variable_names[-1]
                self.variable_names=variable_names[:-1]
                self.data=np.loadtxt(datapath, delimiter=',', skiprows=1)
            else:
                self.variable_names=variable_names
                self.data=np.loadtxt(datapath, delimiter=',')
                
            self.dataX=self.data[:,0:-1]
            self.dataY=self.data[:,-1]
            
            if normalize=='minmax' or normalize=='linear' or normalize==True:
                self.dataX = (self.dataX - np.abs(self.dataX).min(axis=0)) / (np.abs(self.dataX).max(axis=0)-np.abs(self.dataX).min(axis=0))

            elif normalize == 'zscore':
                self.dataX = (self.dataX - self.dataX.mean(axis=0)) / self.dataX.std(axis=0)
            
