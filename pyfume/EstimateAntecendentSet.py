import numpy as np
from scipy.optimize import curve_fit

def is_complete(G):
    nodelist = G.nodes
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == n*(n-1)/2

class AntecedentEstimator(object):
    """
        Creates a new antecedent estimator object.
        
        Args:
            x_train: The input data.
            partition_matrix: The partition matrix of the input data (generated 
                by a clustering the data).
    """ 
    
    def __init__(self, x_train, partition_matrix):
        self.xtrain = x_train 
        self.partition_matrix = partition_matrix
        self._info_for_simplification = None
        self._calculate_all_extreme_values()
                        
        
    def determineMF(self, mf_shape='gauss', merge_threshold=1.0):
        """
            Estimates the parameters of the membership functions that are used 
            as antecedents of the fuzzy rules.
            
            Args:
                mf_shape: The desired shape of the fuzzy sets. The user can choose
                    from 'gauss' (gaussian), 'gauss2' (double gaussion) or 'sigmf' 
                    (sigmoidal) (default = gauss).
                merge_threshold: Threshold for the merging of fuzzy sets for 
                    the GRABS approach. By default no merging takes place 
                    (default = 1.0).
            Returns:
                A list with the estimated parameters of the membership functions (format if gauss: mu, sigma; if gauss2: mu1, sigma1, mu2, sigma2)
        """         
        mf_list=[]

        # mf_list is structured as follows:
        # - an outer list of V variables
        # - each item of the outer list contains C fuzzy set, one for each cluster
        number_of_variables = self.xtrain.shape[1]
        for i in range(0, number_of_variables):
            xin = self.xtrain[:,i]
            if all(y in (0, 1) for y in xin):           # Add noise to binary variables
                noise = np.random.normal(0,0.001,xin.shape[0]) 
                xin= xin + noise
                
            for j in range(0, self.partition_matrix.shape[1]):
                mfin = self.partition_matrix[:,j]
                mf, xx = self._convexMF(xin=xin, mfin=mfin)
                prm = self._fitMF(x=xx, mf=mf, mf_shape=mf_shape)
                mf_list.append(prm) 
        
        if merge_threshold < 1.0:
            self._check_similarities(mf_list, number_of_variables, threshold=merge_threshold)

        #print(self._info_for_simplification)

        return mf_list

    def is_subclique(self,G,nodelist):
        H = G.subgraph(nodelist)
        n = len(nodelist)
        return H.size() == n*(n-1)/2

    def _extreme_values_for_variable(self, v):
        return min(self.xtrain.T[v]), max(self.xtrain.T[v])

    def _calculate_all_extreme_values(self):
        num_variables = len(self.xtrain.T)
        self._extreme_values = [self._extreme_values_for_variable(v) for v in range(num_variables)]

    def _check_similarities(self, mf_list, number_of_variables,
            threshold=1., approx_points=100):

        number_of_clusters = len(mf_list)//number_of_variables
        
        from collections import defaultdict

        things_to_be_removed = defaultdict(list)


        """ 
            This function assesses the pair-wise similarities between 
            the clusters mapped on each variable.
            It returns a dictionary of this kind:
            variable -> list of similar couples for that variable + jaccard sim

        """

        for v in range(number_of_variables):

            mi, ma = self._extreme_values_for_variable(v)
            points = np.linspace(mi, ma, approx_points)

            for c1 in range(number_of_clusters):
                for c2 in range(c1+1, number_of_clusters):

                    index1 = v*number_of_clusters + c1
                    index2 = v*number_of_clusters + c2
                    funname1, params1 = mf_list[index1]
                    funname2, params2 = mf_list[index2]

                    if funname1== "gauss":
                        
                        from numpy import linspace, array

                        first_cluster = array([self._gaussmf(x, params1[0], params1[1]) for x in points])
                        second_cluster = array([self._gaussmf(x, params2[0], params2[1]) for x in points])

                        intersection = sum([min(x,y) for x,y in zip(first_cluster, second_cluster)])
                        union        = sum([max(x,y) for x,y in zip(first_cluster, second_cluster)])

                        jaccardsim = (intersection/union)

                        if jaccardsim>threshold:
                            things_to_be_removed[v].append([c1,c2,jaccardsim])

                            #print("%.2f is fine" % jaccardsim)

                    else:
                        raise Exception("Not implemented yet")

        #for k,v in things_to_be_removed.items():            print (k, v)
        #exit()
        """
            This function must return a dictionary of items like:
            (variable, cluster) -> mapped_cluster
        """

        self._info_for_simplification = {}
        for var_num, value in things_to_be_removed.items():

            subcomponents = self._create_graph(value)

            for subcomp in subcomponents:
                #print (is_complete(subcomp))

                if is_complete(subcomp):
                    retained = list(subcomp.nodes())[0]
                    #print ("retain: %d" % retained)
                    for el in list(subcomp.nodes()):
                        if el!=retained:
                            self._info_for_simplification[(var_num, el)]  = retained                    
                    
        dropped_stuff = self.get_number_of_dropped_fuzzy_sets()
        print (" * %d antecedent clauses will be simplified using a threshold %.2f" % (dropped_stuff, threshold))
        self._info_for_simplification

    def get_number_of_dropped_fuzzy_sets(self):
        return len(self._info_for_simplification)

    def _create_graph(self, list_of_arcs):
        from networkx import Graph, connected_components
        G = Graph()
        nodelist = []
        for arc in list_of_arcs:
            G.add_edge(arc[0], arc[1])
            nodelist.append(arc[0])
            nodelist.append(arc[1])
        S = [G.subgraph(c).copy() for c in connected_components(G)]
        return S
            
       
    def _convexMF(self, xin, mfin, norm=1, nc=1000):
        
        # Calculates the convex membership function that envelopes a given set of
        # data points and their corresponding membership values. 
        
        # Input:
        # Xin: N x 1 input domain (column vector)
        # MFin: N x 1 corresponding membership values 
        # nc: number of alpha cut values to consider (default=101)
        # norm: optional normalization flag (0: do not normalize, 1 : normalize, 
        # default=1)
        #
        # Output:
        # mf: membership values of convex function
        # x: output domain values    
        
        # Normalize the membership values (if requested)
        if norm == 1:
            mfin = np.divide(mfin, np.max(mfin))
        
        # Initialize auxilary variables
        acut = np.linspace(0,np.max(mfin),nc)
        mf= np.full(2*nc, np.nan)
        x=np.full(2*nc, np.nan)
        
        if np.any(mfin>0):
            x[0] = np.min(xin[mfin>0])
            x[nc]=np.max(xin[mfin>0])
            mf[0]=0
            mf[nc] = 0 
        
        # Determine the elements in the alpha cuts    
        for i in range(0,nc):
            if np.any(mfin>acut[i]):
                x[i]=np.min(xin[mfin>acut[i]])
                x[i+nc]=np.max(xin[mfin>acut[i]])
                mf[i]=acut[i]
                mf[i+nc]=acut[i]
                
        # # Determine the elements in the alpha cuts    
        # for i in range(0,nc):
        #     tmp1 = mfin>acut[i]
        #     if len(tmp1)==0:
        #         tmp=xin[tmp1]
        #         np.sort(tmp)
        #         x[i]=tmp[0]
        #         x[i+nc]=tmp[-1]
        #         mf[i]=acut[i]
        #         mf[i+nc]=acut[i]
                  
        #Delete NaNs
        idx=np.isnan(x)
        x=x[idx==False]
        mf=mf[idx==False]  
        
        # Sort vectors based on membership value (descending order)
        indmf=mf.argsort(axis=0)
        indmf=np.flipud(indmf)
        mf=mf[indmf]
        x=x[indmf]
        
        # Find duplicate values for x and onlykeep the ones with the highest membership value
        _, ind = np.unique(x, return_index=True, return_inverse=False, return_counts=False, axis=None)
        mf=mf[ind]
        x=x[ind]
        
        # Sort vectors based on x value (ascending order)
        indx=x.argsort(axis=0)
        mf=mf[indx]
        x=x[indx]
        
        xval=np.linspace(np.min(x),np.max(x),nc)
        mf=np.interp(xval, x, mf, left=None, right=None, period=None)
        x=xval;
        return mf, x
    
    def _fitMF(self,x,mf,mf_shape='gauss'):
        # Fits parametrized membership functions to a set of pointwise defined 
        # membership values.
        #
        # Input:
        # x:  N x 1 domain of input variable
        # mf: N x 1 membership values for input data x 
        # shape: Type of membership function to fit (possible values: 'gauss', 
        # 'gauss2' and 'sigmf')
        #
        # Output:
        # param: matrix of membership function parameters
    
        if mf_shape == 'gauss':
            # Determine initial parameters
            mu = sum(x * mf) / sum(mf)
            mf[mf==0] = np.finfo(np.float64).eps
            sig = np.mean(np.sqrt(-((x-mu)**2)/(2*np.log(mf))))
            
            # Fit parameters to the data using least squares
#            print('mu=', mu, 'sig=', sig)
            param, _ = curve_fit(self._gaussmf, x, mf, p0 = [mu, sig], bounds=((-np.inf, 0), (np.inf, np.inf)), maxfev = 10000)
       
        elif mf_shape == 'gauss2':
            # Determine initial parameters
            mu1 = x[mf>=0.95][0]
            mu2 = x[mf>=0.95][-1]
            xmf =x[mf>=0.5]
            sig1 = (mu1 - xmf[0])/(np.sqrt(2*np.log(2)))
            sig2 = (xmf[-1]-mu2)/(np.sqrt(2*np.log(2)))
            if sig1==0.0:
                sig1=0.1
            if sig2==0.0:
                sig2=0.1
            
            # Fit parameters to the data using least squares
#            print('mu1',mu1,'sig1',sig1,'mu2',mu2,'sig2',sig2)
            param, _ = curve_fit(self._gauss2mf, x, mf, p0 = [mu1, sig1, mu2, sig2], maxfev=1000, bounds=((-np.inf, 0,-np.inf, 0), (np.inf, np.inf,np.inf, np.inf)))
            
        elif mf_shape == 'sigmf':
            # Determine initial parameters
            if np.argmax(mf)-np.argmin(mf) > 0:         # if sloping to the right
                if len(x[mf>=0.5])>0:
                    c = x[mf>=0.5][0]
                    s = 1
                elif len(x[mf>=0.5])==0:                   # if there are no datapoints with membership larger than 0
                    c = x[0]
                    s = 1                    
            elif np.argmax(mf)-np.argmin(mf) < 0:       # if sloping to the left
                if len(x[mf<=0.5])>0:
                    c = x[mf<=0.5][0]
                    s = 1
                elif len(x[mf<=0.5])==0:                   # if there are no datapoints with membership smaller than 0
                    c = x[-1]
                    s = 1             
            # Fit parameters of the function to the data using non-linear least squares  
            try:
                param, _ = curve_fit(self._sigmf, x, mf, p0 = [c, s], maxfev=1000)
            except RuntimeError:
                print('pyFUME attempted to fit sigmoidal shaped membership functions, but was unable to find fitting parameters. pyFUME will now terminate. Please consider using a different shape for the membership functions.')
                import sys
                sys.exit()
        
        return mf_shape, param
        
    def _gaussmf(self,x, mu, sigma, a=1):
        # x:  (1D array)
        # mu: Center of the bell curve (float)
        # sigma: Width of the bell curve (float)
        # a: normalizes the bell curve, for normal fuzzy set a=1 (float) 
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    
    def _gauss2mf(self,x, mu1, sigma1, mu2, sigma2):
        # x: Data
        # mu1: Center of the leftside bell curve
        # sigma1: Standard deviation that determines the width of the leftside bell curve
        # mu2: Center of the rightside bell curve
        # sigma2: Standard deviation that determines the width of the rightside bell curve
        y = np.ones(len(x))
        idx1 = x <= mu1
        idx2 = x > mu2
        y[idx1] = self._gaussmf(x[idx1], mu1, sigma1)
        y[idx2] = self._gaussmf(x[idx2], mu2, sigma2)
        return y
    
    def _sigmf(self,x, c, s):
        # x: data
        # b: x where mf is 0.5
        # c: Controls 'width' of the sigmoidal region about `b` (magnitude); also
        #    which side of the function is open (sign). A positive value of `a`
        #    means the left side approaches 0.0 while the right side approaches 1,
        #    a negative value of `c` means the opposite.
        return 1. / (1. + np.exp(- s * (x - c)))