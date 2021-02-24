import numpy as np

class Sampler(object):  
    def __init__(self,train_x, train_y, number_of_bins = 5, histogram = False):
        self.train_x = train_x
        self.train_y = train_y
        self.number_of_bins = number_of_bins
        self.histogram = histogram
        
    def oversample(self):
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