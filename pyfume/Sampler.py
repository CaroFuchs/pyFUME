import numpy as np

class Sampler(object):  
    """
        Creates a new Sampler object that makes it possible to oversample unbalanced data sets to make them more balanced.
        
        Args:
            train_x: The input data.
            train_y: The output data (true label/golden standard) on basis which will be sampled.
            number_of_bins: Number of clusters that should be identified in the data.
            histogram: True/False flag that determines whether a histogram of the frequencies of the output data will be plotted of both the old and new (= sampled) situation (default = False). The package 'matplotlib.pyplot' is required for this functionality.
    """ 

    def __init__(self,train_x, train_y, number_of_bins = 5, histogram = False):
        self.train_x = train_x
        self.train_y = train_y
        self.number_of_bins = number_of_bins
        self.histogram = histogram
        
    def oversample(self):
        """
        Created a more balanced data set by oversampling underrepresented data instances (based on values of the output variable) in the data set.

        Returns:
            Tuple containing (new_train_x, new_train_y)
                - new_train_x: The  oversampled input data metrix.
                - new_train_y: The oversampled output data matrix.
        """           
        print('Your training data will be oversampled.')
        data = np.append(self.train_x, self.train_y.reshape(len(self.train_y),1), axis=1)
        hist, bins = np.histogram(self.train_y, bins=self.number_of_bins)
        max_len=max(hist)
      
        for i in range(1,self.number_of_bins):
            if i==1:
                c = data[self.train_y <= bins[i]]
                indices = np.random.choice(len(c), max_len-len(c), replace=True)
                data_new= np.append(data, c[indices], axis=0)
            elif i>1 & i != self.number_of_bins-1:
                c = data[(self.train_y > bins[i-1]) & (self.train_y <= bins[i])]
                indices = np.random.choice(len(c), max_len-len(c), replace=True)
                data_new= np.append(data_new, c[indices], axis=0)
                        
        new_train_x = data_new[:,:-1]
        new_train_y = data_new[:,-1]
        
        if self.histogram == True:
            import matplotlib.pyplot as plt
            plt.hist(new_train_y, bins=self.number_of_bins, alpha=0.5, label='Oversampled data', color= 'blue')
            plt.hist(self.train_y, bins=self.number_of_bins, alpha=0.5, label='Raw data', color='red')
            plt.legend(loc='upper right')
            plt.show()
    
        return new_train_x, new_train_y