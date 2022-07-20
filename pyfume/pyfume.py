from .BuildTakagiSugeno import BuildTSFIS
from .LoadData import DataLoader
from .Splitter import DataSplitter
from .SimpfulModelBuilder import SugenoFISBuilder
from .Clustering import Clusterer
from .EstimateAntecendentSet import AntecedentEstimator
from .FireStrengthCalculator import FireStrengthCalculator
from .EstimateConsequentParameters import ConsequentEstimator
from .Tester import SugenoFISTester
from .FeatureSelection import FeatureSelector
from .Sampler import Sampler
from .simpfulfier import SimpfulConverter

import numpy as np

class pyFUME(object):
    """
        Creates a new fuzzy model.
        
        Args:
            datapath: The path to the csv file containing the input data (argument 'datapath' or 'dataframe' should be specified by the user).
            dataframe: Pandas dataframe containing the input data (argument 'datapath' or 'dataframe' should be specified by the user).
            nr_clus: Number of clusters that should be identified in the data (default = 2).
            process_categorical: Boolean to indicate whether categorical variables should be processed (default = False).
            method: At this moment, only Takagi Sugeno models are supported (default = 'Takagi-Sugeno')
            variable_names: Names of the variables, if not specified the names will be read from the first row of the csv file (default = None).
            merge_threshold: Threshold for GRABS to drop fuzzy sets from the model. If the jaccard similarity between two sets is higher than this threshold, the fuzzy set will be dropped from the model.
            **kwargs: Additional arguments to change settings of the fuzzy model.

        Returns:
            An object containing the fuzzy model, information about its setting (such as its antecedent and consequent parameters) and the different splits of the data.
    """
    def __init__(self, datapath=None, dataframe=None, nr_clus=2, process_categorical=False, method='Takagi-Sugeno', variable_names=None, merge_threshold=1., **kwargs):
      
        if datapath is None and dataframe is None:
            raise Exception("ERROR: a dataset was not specified. Please either use the datapath or dataframe arguments.")

         #if nr_clus<2 and nr_clus!=None:
         #    raise Exception("Number of clusters should be greater than 1.")

        self.datapath=datapath
        self.nr_clus=nr_clus
        self.method=method
        self.dropped_fuzzy_sets = 0
        #self.variable_names=variable_names

        if method=='Takagi-Sugeno' or method=='Sugeno':
            if datapath is not None:
                self.FIS = BuildTSFIS(datapath=self.datapath, nr_clus=self.nr_clus, variable_names=variable_names, process_categorical=process_categorical, merge_threshold=merge_threshold, **kwargs)
            else:
                self.FIS = BuildTSFIS(dataframe=dataframe, nr_clus=self.nr_clus, variable_names=variable_names, process_categorical=process_categorical, merge_threshold=merge_threshold, **kwargs)
            if merge_threshold < 1.0:
                self.dropped_fuzzy_sets = self.FIS._antecedent_estimator.get_number_of_dropped_fuzzy_sets()
        else:
            raise Exception ("This modeling technique has not yet been implemented.")

    def calculate_error(self, method="MAE"):
        """
        Calculates the performance of the model given the test data.

            Args:
                method: The performance metric to be used to evaluate the model (default = 'MAE'). Choose from: Mean Absolute Error 
                ('MAE'), Mean Squared Error ('MSE'),  Root Mean Squared Error ('RMSE'), Mean Absolute Percentage 
                Error ('MAPE').
        
        Returns:
            The performance as expressed by the chosen performance metric.
        """   
        if method=="MSE":
            return self._get_MSE()
        elif method=="MAE":
            return self._get_MAE()
        elif method=="MAPE":
            return self._get_MAPE()
        elif method=="RMSE":
            return self._get_RMSE()
        else:
            # return self._get_MSE()
            raise Exception("Method '%s' not implemented yet" % (method))
            
    def predict_test_data(self):
        """
        Calculates the predictions labels of the test data using the fuzzy model.

        Returns:
            Prediction labels.
        """
        #get the prediction for the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, variable_names=self.FIS.variable_names, golden_standard=self.FIS.y_test)
        pred, _ = test.predict()
        return pred
    
    def predict_label(self, xdata):
        """
        Calculates the predictions labels of a data set using the fuzzy model.

        Args:
            xdata: The input data (as numpy array with each row a different data instance and variables in the same order as in the original training data set) for which the labels should be calculated. 

        Returns:
            Prediction labels.
        """
        # Normalize the input data if needed
        if self.FIS.minmax_norm_flag==True:
            norm_val=self.FIS.normalization_values
            variable_names, min_values, max_values = zip(*norm_val)
            xdata = (xdata - np.array(min_values)) / (np.array(max_values) - np.array(min_values))
        
        #get the prediction for a new data set
        model = self.get_model()
        test = SugenoFISTester(model=model, test_data=xdata, golden_standard=None, variable_names=self.FIS.variable_names)
        pred, _ = test.predict()
        return pred
    
    def normalize_values(self, data):
        """
        Calculates the normalized values of a data point, using the same scaling 
        that was used to training data of the model. This method only works when 
        the data was normalized using the min-max method.

        Args:
            xdata: The input data (as numpy array with each row a different data instance and variables in the same order as in the original training data set) for which the normalized values should be calculated. 

        Returns:
            Normalized values.
        """        
        if self.FIS.minmax_norm_flag==True:
            norm_val=self.FIS.normalization_values
            variable_names, min_values, max_values = zip(*norm_val)
            normalized_data = (data - np.array(min_values)) / (np.array(max_values) - np.array(min_values)) 
           
        elif self.FIS.minmax_norm_flag==False:
            raise Exception('The model was not trained on normalized data, normalization is aborted.')    
        
        return normalized_data
    
    
    def denormalize_values(self, data):
        """
        Takes normalized data points, and returns the denormalized (raw) values
        of that data point. This method only works when during modeling the 
        data was normalized using the min-max method.

        Args:
            xdata: The input data (as numpy array with each row a different data instance and variables in the same order as in the original training data set) for which the normalized values should be calculated. 

        Returns:
            Normalized values.
        """        
        if self.FIS.minmax_norm_flag==True:
            if np.amin(data)<0 or np.amax(data) >1:
                print('WARNING: The given value(s) are not between 0 and 1, the denormalization is performed by extrapolating.')
            
            norm_val=self.FIS.normalization_values
            variable_names, min_values, max_values = zip(*norm_val)
            # print(min_values, max_values)
            denormalized_data = (data * (np.array(max_values) - np.array(min_values))) + np.array(min_values)
        elif self.FIS.minmax_norm_flag==False:
            raise Exception('The model was not trained on normalized data, normalization is aborted.')    
        
        return denormalized_data

    def test_model(self, xdata, ydata, error_metric='MAE'):
        """
        Calculates the performance of the model using the given data.

        Args:
            xdata: The input data (as numpy array with each row a different data instance and variables in the same order as in the original training data set) for which the labels should be calculated. 
            ydata: The target data (as single-column numpy array).
            error_metric: The error metric in which the performance should be expressed (default = 'MAE'). Choose from: Mean Absolute Error ('MAE'), Mean Squared Error ('MSE'),  Root Mean Squared Error ('RMSE'), Mean Absolute Percentage Error ('MAPE').

        Returns:
            The performance as expressed in the chosen metric.
        """        
        #get the prediction for a new data set
        model = self.get_model()
        test = SugenoFISTester(model=model, test_data=xdata, golden_standard=ydata, variable_names=self.FIS.variable_names)
        metric= test.calculate_performance(metric=error_metric)
        return metric

    #######################
    ###     GETTERS     ###
    #######################

    def get_model(self):
        """
        Returns the fuzzy model created by pyFUME.

        Returns:
            The fuzzy model (as an executable object).
        """          
        if self.FIS.model is None:
            raise Exception("ERROR: model was not created correctly, aborting.")
        else:
            return self.FIS.model
        
    def get_firing_strengths(self, data, normalize=True):
        """
        Calculates the (normalized) firing strength/ activition level of each rule for each data instance of the given data.

        Args:
            xdata: The input data (as numpy array with each row a different data instance and variables in the same order as in the original training data set) for which the labels should be calculated. 
            normalize: Boolean that indicates whether the retuned fiing strengths should be normalized (normalize = True) or not (normalize = False), When the firing strenghts are nomalized the summed fiing strengths for each data instance equals one.
        Returns:
            Firing strength/activition level of each rule (columns) for each data instance (rows).
        """          

        # Calculate the firing strengths
        fsc=FireStrengthCalculator(self.FIS.antecedent_parameters, self.FIS.nr_clus, self.FIS.variable_names)
        firing_strengths = fsc.calculate_fire_strength(data)
        if normalize == True:
            firing_strengths=firing_strengths/firing_strengths.sum(axis=1)[:,None]
        return firing_strengths
    
    def get_performance_per_fold(self):
        """
        Returns a list with the performances of each model that is created if crossvalidation is used when training..

        Returns:
            Perfomance of each cross-validation model.
        """
        return self.FIS.performance_metric_per_fold
    
    def get_fold_indices(self):
        """
        Returns a list with the fold indices of each model that is created if crossvalidation is used when training.

        Returns:
            Fold indices.
        """
        return self.FIS.fold_indices
        
    def _get_RMSE(self):
        # Calculate the mean squared error of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, golden_standard=self.FIS.y_test, variable_names=self.FIS.variable_names)
        RMSE = test.calculate_RMSE()
        #RMSE = list(RMSE.values())
        #print('The RMSE of the fuzzy system is', RMSE)
        return RMSE
    
    def _get_MSE(self):
        # Calculate the mean squared error of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, golden_standard=self.FIS.y_test, variable_names=self.FIS.variable_names)
        MSE = test.calculate_MSE()
        #RMSE = list(RMSE.values())
        #print('The RMSE of the fuzzy system is', RMSE)
        return MSE
        
    
    def _get_MAE(self):
        # Calculate the mean absolute error of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, variable_names=self.FIS.variable_names, golden_standard=self.FIS.y_test)
        MAE = test.calculate_MAE()
        return MAE
    
    def _get_MAPE(self):
        # Calculate the mean absolute percentage error of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, golden_standard=self.FIS.y_test, variable_names=self.FIS.variable_names)
        MAPE = test.calculate_MAPE()
        return MAPE
    
    def _get_accuracy(self):
        # Calculate the accuraccy of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, golden_standard=self.FIS.y_test, variable_names=self.FIS.variable_names)
        accuracy = test.calculate_accuracy()
        return accuracy
    
    def calculate_AUC(self, number_of_slices=100, show_plot = False):
        # Calculate the area under the ROC curve of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, golden_standard=self.FIS.y_test, variable_names=self.FIS.variable_names)
        AUC = test.calculate_AUC(number_of_slices, show_plot)
        return AUC

    
    def get_confusion_matrix(self):
        # Calculate the confusion matrix of the model using the test data set
        test = SugenoFISTester(model=self.FIS.model, test_data=self.FIS.x_test, golden_standard=self.FIS.y_test, variable_names=self.FIS.variable_names)
        con_mat = test.generate_confusion_matrix()
        return con_mat
    
    def get_data(self, data_set='test'):
        """
        Returns the test or training data set.
        
        Args:
            data_set: Used to specify whether the function should return the training (data_set = "train"), test set (data_set = "test") or both training and test data (data_set = "all"). By default, the function returns the test set. 

        Returns:
            Tuple (x_data, y_data) containing the test or training data set.
        """  
      
        if data_set == 'train':
            return self.FIS.x_train, self.FIS.y_train
        elif data_set == 'test':
            return self.FIS.x_test, self.FIS.y_test
        elif data_set == 'all':
            xdata = np.concatenate((self.FIS.x_train, self.FIS.x_test), axis=0)
            ydata = np.concatenate((self.FIS.y_train, self.FIS.y_test), axis=0)
            return xdata, ydata
        else:
            print('Please specify whether you would like to receive the training (data_set = "train"), test set (data_set = "test") or all data (data_set = "all").')
            
    def get_cluster_centers(self):
        """
        Returns the cluster centers as identified by pyFUME.
        
        Returns:
            cluster centers.
        """  
        return self.FIS.cluster_centers 
    
    
    #######################
    ### PLOT FACILITIES ###
    #######################

    def plot_mf(self, variable_name, output_file='', highlight_element=None, highlight_mf=None, ax = None, xscale = 'linear'):
        """
        Uses Simpful's plotting facilities to plot the membership functions of
        the pyFUME model.

        Args:
            variable_name: The variable name whose membership functions should be plotted.
            output_file: Path and filename where the plot must be saved. By default, the file is not saved.
            highlight_element: Show the memberships of a specific element of the universe of discourse in the figure.
            highlight_mf: String indicating the linguistic term/fuzzy set to highlight in the plot.
            ax: The motplotlib ax where the variable will be plotted.

        """ 
        self.get_model().plot_variable(var_name = variable_name, outputfile = output_file, TGT = highlight_element, highlight = highlight_mf, ax = ax, xscale = xscale)
                
    def plot_all_mfs(self, output_file='', figures_per_row=4):
        """
        Plots the membership functions of all the variables  in the pyFUME model,
        each in their own sub figure.

        Args:
            output_file: path and filename where the plot must be saved.
            figures_per_row: The number of sub figures per row in the figure.
        """
        self.get_model().produce_figure(outputfile=output_file, max_figures_per_row=figures_per_row)
    
    def plot_consequent_parameters(self, rule_number, output_file='', set_title = True, set_legend = True, ax = None):
        """
        Plots the consequent coeffecients of a given rule in a bar chart. If 
        the training data was normalized, the coeffiecients are plotted as-is. 
        If the data was not normalized, the coefficients are normalized to 
        enhance comparability.

        Args:
            output_file: path and filename where the plot must be saved.
            figures_per_row: The number of figures per row.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
                
        # Start counting from 1 instead of 0
        rule_number=rule_number-1
        
        # Get the required data from the pyFUME model
        labels= self.FIS.selected_variable_names
        consequent_parameters=self.FIS.consequent_parameters
        nr_rules=len(consequent_parameters)
        nr_variables = len(labels)
        
        # Standardize the parameters if data was not normalized
        if self.FIS.minmax_norm_flag == False:
            standard_deviations=np.std(self.FIS.x_train, axis=0)
            std_y=np.std(self.FIS.y_train)
            parameters=np.zeros((nr_variables,nr_rules))
            for rule in range(0,nr_rules):
                consequent=consequent_parameters[rule]
                n=np.zeros(nr_variables)
                for var in range(0,nr_variables):
                    std=standard_deviations[var]
                    parameter=consequent[var]
                    norm= (std/std_y)*parameter
                    n[var]=norm
                parameters[:,rule]= n
                
        # Used the raw values if data was normalized        
        elif self.FIS.minmax_norm_flag == True:
            parameters=np.zeros((nr_variables,nr_rules))
            for rule in range(0,nr_rules):
                consequent=consequent_parameters[rule]
                n=np.zeros(nr_variables)
                for var in range(0,nr_variables):
                    n[var]=consequent[var]
                parameters[:,rule]= n
        
        # Color the bars in the plot based on the relationship to the target variable
        cc=['colors']*len(parameters[:,rule_number])
        for n,val in enumerate(parameters[:,rule_number]):
            if val<0:
                cc[n]='firebrick'
            elif val>=0:
                cc[n]='navy'
                
        # Create the information for the legend
        legend_elements = [Patch(facecolor='firebrick', label="Negatively related to target variable"),
                           Patch(facecolor='navy', label="Positively related to target variable")]
        
        # Create the plot
        ax.barh(labels, np.abs(parameters[:,rule_number]), align='center', color = cc)
        ax.grid(color='grey', linestyle='dotted', linewidth=1.5)
        ax.invert_yaxis()
        if self.FIS.minmax_norm_flag == False:
            fig_title = 'Standardized consequent parameters for rule ' + str(rule_number+1)
        elif self.FIS.minmax_norm_flag == True:
            fig_title = 'Consequent parameters for rule ' + str(rule_number+1)    
        if set_title== True: ax.set_title(fig_title);
        if set_legend == True: ax.legend(handles=legend_elements)
        # fig.tight_layout()
        
        # Save the plot if requested, otherwise just show the plot to the user
        if ax != None:
            return ax
        elif output_file != "":
            fig = ax.get_figure() 
            fig.savefig(output_file)
        else:
            plt.show()
        

    def _denormalize_antecedent_set(self, data, normalization_values):
        """
        Takes a normalized antecedent set, and returns the denormalized parameters
        defining that set. This method only works when during modeling the 
        data was normalized using the min-max method.
    
        Args:
            xdata: The input data (as numpy array with each row a different data instance and variables in the same order as in the original training data set) for which the normalized values should be calculated. 
    
        Returns:
            Normalized values.
        """        
    
        (_, min_value, max_value) = normalization_values

        x=data[-1]

        denormalized_mu = (x[0] * (np.array(max_value) - np.array(min_value))) + np.array(min_value)
        denormalized_sigma = (x[1] * (np.array(max_value) - np.array(min_value))) 

        denormalized_set = tuple([data[0], [denormalized_mu, denormalized_sigma]])

        return denormalized_set
    
    def plot_denormalized_mf(self, variable_name, output_file='', highlight_element=None, highlight_mf=None, ax = None, xscale = 'linear'):
        normalization_values = self.FIS.normalization_values 
        antecedent_sets= self.FIS.antecedent_parameters
        
        if normalization_values == None:
            raise Exception('ERROR: The input data for the pyFUME model was not normalized during training. Denormaliztaion is therefore not possible.')
        
        # Check if variables were removed (during feature selection)
        to_keep=[]
        for i in range(0,len(normalization_values)):
            if normalization_values[i][0] in self.FIS.selected_variable_names:
                to_keep.append(i)
        
        # Keep only the values for variables that were selected
        normalization_values = [normalization_values[i] for i in to_keep]
        
        # Denormalize the antecedent set parameters
        denormed_antecedent_sets = []
        
        x=0
        cnt=0
        for i in range(0, len(antecedent_sets)):
             norm_vals = normalization_values[x]
             denormed_set=self._denormalize_antecedent_set(antecedent_sets[i], norm_vals)
             denormed_antecedent_sets.append(denormed_set)
             cnt+=1
            
             if cnt == self.nr_clus:
                x+=1
                cnt = 0
        
        UoD = []
        _, mi, ma = zip(*self.FIS.normalization_values)
        for i in range(0,len(mi)):
            # UoD.append(tuple((mi[i]-0.05*ma[i],ma[i]+0.05*ma[i])))
            UoD.append(tuple((mi[i],ma[i])))

        
        simpbuilder = SugenoFISBuilder(
            denormed_antecedent_sets, 
            np.tile(1, (self.nr_clus, len(self.FIS.selected_variable_names)+1)), 
            self.FIS.selected_variable_names, 
            extreme_values = UoD,
            save_simpful_code='trial.py', 
            fuzzy_sets_to_drop=self.FIS._antecedent_estimator._info_for_simplification,
            verbose = False)
        dummymodel = simpbuilder.simpfulmodel
        
        # Plot the requested variable using Simpful
        dummymodel.plot_variable(var_name = variable_name, outputfile = output_file, TGT = highlight_element, highlight = highlight_mf, ax = ax, xscale = xscale)
      
if __name__=='__main__':
    from numpy.random import seed
    seed(4)
   
    FIS = pyFUME(datapath='Concrete_data.csv', nr_clus=3, method='Takagi-Sugeno',
     merge_threshold=.8, operators=None)
    print ("The calculated error is:", FIS.calculate_error())

    FIS.get_model().produce_figure("bla.pdf")