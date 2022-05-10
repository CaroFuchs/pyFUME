from collections import defaultdict
from math import sqrt
import numpy as np


class SugenoFISTester(object):
    """
    Creates a new Tester object to be able to calculate performance metrics of the fuzzy model.
    
    Args:
        model: The model for which the performance metrics should be calculated
        test_data: The data to be used to compute the performance metrics
        variable_names: A list of the variables names of the test data (which 
            should correspond with the variable names used in the model).
        golden_standard: The 'True' labels of the test data. If not provided, the 
            only predictions labels can be generated, but the error will not be 
            calculated (default = None).
        list_of_outputs: List of the output names (which should correspond with 
            the output names used in the model) (default: OUTPUT).
    """
    
    def __init__(self, model, test_data, variable_names, golden_standard=None, list_of_outputs=['OUTPUT']):
        super().__init__()
        self._model_to_test = model
        self._data_to_test = test_data
        self._golden_standard = golden_standard
        self._variable_names = variable_names
        self._list_of_outputs=list_of_outputs
        
    def predict(self):
        """
        Calculates the predictions labels of the test data using the fuzzy model.

        Returns:
            Tuple containing (result, error)
                - result: Prediction labels.
                - error: The difference between the prediction label and the 'true' label.
        """
        result = []
        for sample in self._data_to_test:
            for i, variable in enumerate(self._variable_names):
                self._model_to_test.set_variable(variable, sample[i])
            result.append(self._model_to_test.Sugeno_inference().get('OUTPUT'))
        result = np.array(result)
        if self._golden_standard is not  None:
            error = self._golden_standard - result
        else:
            error = np.nan
            # print('The true labels (golden standard) were not provided, so the error could not be calculated.')
        return result, error
    
    def calculate_performance(self, metric='MAE'):  
        """
        Calculates the performance of the model given the test data.

            Args:
                metric: The performance metric to be used to evaluate the model. Choose from: Mean Absolute Error 
                ('MAE'), Mean Squared Error ('MSE'),  Root Mean Squared Error ('RMSE'), Mean Absolute Percentage 
                Error ('MAPE').
        
        Returns:
            The performance as expressed by the chosen performance metric.
        """      
        if metric == 'MAE':
            err=self.calculate_MAE()
        elif metric == 'MSE':
            err=self.calculate_MSE()
        elif metric == 'RMSE':
            err=self.calculate_RMSE()        
        elif metric == 'MAPE':
            err=self.calculate_MAPE()
        elif metric == 'accuracy':
            err=self.calculate_accuracy()
        elif metric == 'AUC':
            err=self.calculate_AUC()
        else:
            print('The requested performance metric has not been implemented (yet).')
            
        return err
    
    def calculate_RMSE(self):
        """
        Calculates the Root Mean Squared Error of the model given the test data.
        
        Returns:
            The Root Mean Squared Error of the fuzzy model.
        """
        _, error=self.predict()
        return sqrt(np.mean(np.square(error)))
    
    
    def calculate_MSE(self):
        """
        Calculates the Mean Squared Error of the model given the test data.
        
        Returns:
            The Mean Squared Error of the fuzzy model.
        """
        _, error=self.predict()
        return np.mean(np.square(error))   
    
    def calculate_MAE(self):
        """
        Calculates the Mean Absolute Error of the model given the test data.
        
        Returns:
            The Mean Absolute Error of the fuzzy model.
        """
        _, error=self.predict()
        return np.mean(np.abs(error))
    
    def calculate_MAPE(self):
        """
        Calculates the Mean Absolute Percentage Error of the model given the test data.
        
        Returns:
            The Mean Absolute Percentage Error of the fuzzy model.
        """
        
        if self._golden_standard is None:
             raise Exception('To compute the MAPE the true label (golden standard) of the test data should be provided.')
        
        _, error=self.predict()
        return np.mean(np.abs((error) / self._golden_standard)) * 100
    
    def calculate_accuracy(self, threshold = 0.5):
        """
        Calculates the accuracy of the model for binary problems, given the test data and a discretization threshold .
        
        Args:
            treshold: The treshold to discretize the predicted output in binary categories.
        Returns:
            The accuracy of the fuzzy model.
        """
        
        confusion_matrix= self.generate_confusion_matrix(threshold=threshold)
        acc = round((confusion_matrix['TP']+confusion_matrix['TN'])/(confusion_matrix['TP']+confusion_matrix['TN']+confusion_matrix['FP']+confusion_matrix['FN']),3)
        return acc
    
    def generate_confusion_matrix(self, threshold = 0.5):
        """
        Calculates the confusion matrix for binary output data.
        
        Args:
            treshold: The treshold to discretize the predicted output in binary categories.
        
        Returns:
            Dictionary containing the confusion martrix.
        """
        # Check if golden standard is present
        if self._golden_standard is None:
            raise Exception('To calculate the confusion matrix, the true label (golden standard) of the test data should be provided.') 
        
        # Get the predicted values for the test data
        ypred, _ =self.predict()
        
        # discretize ypred using a user-specified thresehold
        discrete_ypred = np.digitize(ypred,bins=[threshold])
        
        # Create the confusion matrix as a dictionary
        confusion_matrix = dict()
        
        confusion_matrix['TP'] = np.sum(np.logical_and(discrete_ypred == 1, self._golden_standard == 1))
        confusion_matrix['TN'] = np.sum(np.logical_and(discrete_ypred == 0, self._golden_standard == 0))
        confusion_matrix['FP'] = np.sum(np.logical_and(discrete_ypred == 1, self._golden_standard == 0))
        confusion_matrix['FN'] = np.sum(np.logical_and(discrete_ypred == 0, self._golden_standard == 1))
        return confusion_matrix
    
    def calculate_AUC(self, number_of_slices=25, show_plot = False):
        """
        Calculates the area under the ROC curve (AUC) for models with binary output data.
        
        Args:
            number_of_slices: More slices give a higher precision of the AUC, against the cost of higher computational costs.
        
        Returns:
            AUC.
        """
        
        ROC = np.array([])
        for T in np.linspace(0,1,number_of_slices):
            con_mat = self.generate_confusion_matrix(threshold = T) 
            TPR = self.calculate_TPR(con_mat)
            FPR = self.calculate_FPR(con_mat)
            ROC=np.append(ROC, [FPR,TPR])
        ROC = ROC.reshape(-1, 2)
        
        fpr, tpr = ROC[:, 0], ROC[:, 1]
        AUC = 0
        for k in range(0,number_of_slices-1):
            AUC = AUC + ((fpr[k]-fpr[k+1]) * tpr[k+1]) + ((1/2) * (fpr[k]- fpr[k+1]) * (tpr[k]- tpr[k+1]))
        
        if show_plot:
            import matplotlib.pyplot as plt 
            plt.figure(figsize=(16,8))
            plt.scatter(ROC[:,0],ROC[:,1],s=100)
            plt.plot([0, 1], [0, 1], ls = '--', c = 'darkgrey')
            plt.title('ROC Curve with AUC = %.2f' %AUC)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')        
        
        return AUC
                
        
        
    def calculate_TPR(self, confusion_matrix):
        """
        Calculates the true positive rate, given the confusion matrix for binary output data.
        
        Args:
            confusion matrix: confusion matrix (dict) containing TP, FP, TN and FN.
        
        Returns:
            True positive rate.
        """
        return confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
        
    def calculate_FPR(self, confusion_matrix):
        """
        Calculates the false positive rate, given the confusion matrix for binary output data.
        
        Args:
            confusion matrix: confusion matrix (dict) containing TP, FP, TN and FN.
        
        Returns:
            False positive rate.
        """
        return confusion_matrix['FP'] / (confusion_matrix['FP'] + confusion_matrix['TN'])
    
    
    