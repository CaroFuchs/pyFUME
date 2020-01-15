import numpy as np
from scipy.optimize import curve_fit
import numpy.matlib
from scipy.spatial.distance import cdist
from random import randint

class RuleCreator(object):
    def __init__(self, datapath, nrclus,varnames=None,m=2,max_it=1000,error=0.005,seed=None,shape='gauss2',global_fit=1):
        self.data=np.loadtxt(datapath,delimiter=',')
        self.nrclus=nrclus
        self.varnames=varnames
        self.shape=shape

        # The first row of the data contains the variable names
        self.varnames=self.data[0,:]
        
        # The last collumn of the data contains the prediction labels
        self.dataX=self.data[:,0:-1]
        self.dataY=self.data[:,-1]
        
        # Split the data in a training and test set
        self.xtrn, self.ytrn, self.xtst, self.ytst = splitTrainTest(dataX,dataY, train_perc=0.75)
        
        # Cluster the data with FCM
        _,self.partitionmatrix,_= self.fcm(self.xtrn,nrclus,m,max_it,error,seed)
        
        # Estimate the antecedent sets of the fuzzy rules
        self.MFs = self.determineMF(self.xtrn,self.partitionmatrix,shape)
        
        # Estimate the consequent parematers of the fuzzy rules
        self.Cons = self.suglms(self.xdata,self.ydata,self.partitionmatrix,global_fit)

    def determineMF(self,x,f,shape='gauss2'):
        mflist=[]
        for i in range(0,x.shape[1]):
            xin=x[:,i]
            for j in range(0,f.shape[1]):
                mfin= f[:,j]
                mf, xx = self.convexMF(xin, mfin)
                prm = self.fitMF(xx,mf, shape)
                mflist.append(prm) 
        return mflist
       
    def convexMF(self,xin, mfin, norm=1, nc=1000):
        
        # Calculates the convex membership function that envelopes a given set of
        # data points and their corresponding membership values. 
        
        # Input:
        # Xin: N x 1 input domain (column vector)
        # MFin: N x 1correspoding membership values 
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
    
    def fitMF(self,x,mf,shape='gauss2'):
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
        
    
        if shape == 'gauss':
            # Determine initial parameters
            mu = sum(x * mf) / sum(mf)
            sig = np.sqrt(sum(mf*(x - mu)**2))
            # Fit parameters to the data using least squares
            param, _ = curve_fit(self.gaussmf, x, mf, p0 = [1, mu, sig])
       
        elif shape == 'gauss2':
            # Determine initial parameters
            mu1 = x[mf>=0.95][0]
            mu2 = np.flipud(x[mf>=0.95])[0]
            xmf =x[mf>=0.5]
            sig1 = np.divide(mu1 - xmf[0],np.sqrt(2*np.log(2)));
            sig2 = np.divide(np.flipud(xmf)[0]-mu2,np.sqrt(2*np.log(2)));
            
            # Fit parameters to the data using least squares
            param, _ = curve_fit(self.gauss2mf, x, mf, p0 = [mu1, sig1, mu2, sig2])
            
        elif shape == 'sigmf':
            # Determine initial parameters
            if np.argmax(mf)-np.argmin(mf) > 0:         # if sloping to the right
                c = x[mf>=0.5][0]
                s = 1
            elif np.argmax(mf)-np.argmin(mf) < 0:       # if sloping to the left
                c = x[mf<=0.5][0]
                s = -1
            # Fit parameters of the function to the data using non-linear least squares           
            param, _ = curve_fit(self.sigmf, x, mf, p0 = [c, s])
        
        return shape, param
        
    def gaussmf(self,x, mu, sigma, a=1):
        # x:  (1D array)
        # mu: Center of the bell curve (float)
        # sigma: Width of the bell curve (float)
        # a: normalizes the bell curve, for normal fuzzy set a=1 (float) 
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    
    def gauss2mf(self,x, mu1, sigma1, mu2, sigma2):
        # x: Data
        # mu1: Center of the leftside bell curve
        # sigma1: Standard deviation that determines the width of the leftside bell curve
        # mu2: Center of the rightside bell curve
        # sigma2: Standard deviation that determines the width of the rightside bell curve
        y = np.ones(len(x))
        idx1 = x <= mu1
        idx2 = x > mu2
        y[idx1] = self.gaussmf(x[idx1], mu1, sigma1)
        y[idx2] = self.gaussmf(x[idx2], mu2, sigma2)
        return y
    
    def sigmf(self,x, c, s):
        # x: data
        # b: x where mf is 0.5
        # c: Controls 'width' of the sigmoidal region about `b` (magnitude); also
        #    which side of the function is open (sign). A positive value of `a`
        #    means the left side approaches 0.0 while the right side approaches 1,
        #    a negative value of `c` means the opposite.
        return 1. / (1. + np.exp(- s * (x - c)))
    
    
    def suglms(self,x,y,f,df=0,global_fit=1):
    # SUGLMS estimates the consequent parameters in the Sugeno-Takagi model
    #	 using least squares.
    #
    #    Input:
    #       X .....	input data matrix
    #	    Y .....	output data vector
    #       F ..... fuzzy partition matrix (membership degrees),
    #		        optional, defaults to ones(size(y)) for
    #		        which SUGLMS is a standard linear regression
    #       DF ... default value returned when the sum of grades
    #               equals to one (optional, defaults to 0)
    #	    FLAG .. set to 1 to get local weighted LMS estimates
    #    
    #    Output:
    #       P .....	consequents parameters for every cluster
    #	    Ym ....	global model output for the given input data
    #	    Yl ....	output of local submodels (corresponding to clusters)
    #	    Ylm ...	output of local submodels with data corresponding to
    #               degrees < 0.2 masked with NaN's (for plots)
    #
    #    Example:
    #	x = (0:0.02:1)'; y = sin(7*x);
    #       f = mgrade(x',mfequ(x,2,3))';
    #       [p,ym,yl,ylm] = suglms([x ones(size(x))],y,f);
    #	subplot(211); plot(x,ylm,'.',x,[y ym]); title('Fitting y = sin(7*x)')
    #	subplot(212); plot(x,f); title('Membership functions')
    # (c) Robert Babuska, 1994-95
    
    #################
    
    
    # Check if input X contains one column of ones (for the constant). If not, add it.
        u=np.unique(x[:,-1])
        if u.shape[0]!=1 or u[0]!=1:
            x = np.hstack((x,np.ones((x.shape[0],1))))
    
        # Find the number of data points (mx & mx) , the number of variables (nx) and the
        # number of clusters (nf) 
        mx,nx=x.shape
        mf,nf=f.shape
        
        # Calculate the sum of the degree of fulfillement (DOF) for each data point
        sumDOF=np.sum(f, 1)
        
        
        # When degree of fulfillment is zero (which means no rule is applicable), set to one
        NoRule = sumDOF == 0
        sumDOF[NoRule] = 1
        sumDOF = np.matlib.repmat(sumDOF,nf,1).T
        
        
        # Auxillary variables
        f1 = x.flatten()
        s = np.matlib.repmat(f1,nf,1).T
        xx = np.reshape(s, (nx,nf*mx), order='F')
        s = xx.T  
        x1=np.reshape(s,(mx,nf*nx),order='F') 
        x=x.T                                # reshape data matrix
        
        if nf == 1:
           global_fit = 1
        
        if global_fit == 0:                                           # local weighted least mean squares estimates   
            # (reshaped) vector of f devided by the sum of each row of f
            # (normalised membership degree)
            xx = (f.T.flatten()/sumDOF.T.flatten())
            
            # reshape partition matrix
            s= np.matlib.repmat(xx,nx,1).T
            f1 = np.reshape(s, (mx,nf*nx), order='F')                # reshape partition matrix
            x1 = f1*x1
        
            # Find least squares solution
            xp = np.linalg.lstsq(x1,y,rcond=None)
            p=np.reshape(xp[0],(nf,nx), order='F')
        
            # Local models
            yl = np.transpose(x).dot(np.transpose(p))						                    #
            
            # Global model
            ym = x1.dot(p.flatten()) + df*NoRule
            ylm = yl.copy()
            
            # Mask all memberships < 0.2 with NaN's for plots
            ylm[f<0.2] = np.NaN
        
        elif global_fit == 1:                                             # Global least mean squares estimates
            # preallocate variables
            p=np.zeros((nf,nx))
            yl=np.zeros((mx,nf))
            for i in range (0,nf):
                f1 = np.tile(f[:,i],(nx,1)).T
                x1 = np.sqrt(f1)*np.transpose(x) 
                zz=np.sqrt(f[:,i])
                yx=zz*y[0]
        
                # Find least squares solution
                xp = np.linalg.lstsq(x1,yx,rcond=None)
                p[i,:]=xp[0].T
               
                # Global mode
                yl[:,i] = x.T.dot(p[i,:].T)
                ym = yl.copy()
                ylm = yl.copy()
                
                # Mask all memberships < 0.2 with NaN's for plots
                ylm[f<0.2] = np.NaN    		   
        
        return p #,ym,yl,ylm
    
    def fcm(self,data, n_clusters, m=2, max_it=1000, error=0.005, seed=None):
    	#data: 2d array, size (N, S). N is the number of instances; S is the number of variables.
    	#n_clusters: number of clusters
    	#m: fuzzy clustering coefficient
    	#max_it: maximum number of iterations, default=1000
    	#error: stopping criterion, default=0.005
    	#seed: seed for random initialization of u matrix
    
    	n_instances = data.shape[0]
    	n_variables = data.shape[1]
    
    	#randomly initaliaze u
    	if seed:
    		np.random.seed(seed=seed)
    	u = np.random.rand(n_instances, n_clusters)
    	u = np.fmax(u, np.finfo(np.float64).eps)
    	ut = u.T
    
    	for it in range(0,max_it):
    		#copy old u matrix
    		u_old = ut.copy()
    		u_old /= np.ones((n_clusters, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    		u_old = np.fmax(u_old, np.finfo(np.float64).eps)
    
    		#elevate to m
    		um = u_old ** m
    
    		#calculate cluster centers
    		centers = um.dot(data) / (np.ones((data.shape[1], 1)).dot(np.atleast_2d(um.sum(axis=1))).T)
    
    		#calculate distances
    		dist = cdist(centers, data, metric='euclidean')
    		dist = np.fmax(dist, np.finfo(np.float64).eps)
    
    		#calculate objective
    		jm = (um * dist ** 2).sum()
    
    		#calculate new u matrix
    		ut = dist ** (- 2. / (m - 1))
    		ut /= np.ones((n_clusters, 1)).dot(np.atleast_2d(ut.sum(axis=0)))
    
    		#stopping criterion
    		if np.linalg.norm(ut - u_old) < error:
    			break
    
    	u = ut.T
    	return centers, u, jm
    
    def splitTrainTest(dataX,dataY, train_perc=0.75):
        universe=set(range(0,dataX.shape[0]))
        trn=np.random.choice(dataX.shape[0], int(round(train_perc*dataX.shape[0])), replace=False)
        tst=list(universe-set(trn))
        
        xtrn=dataX[trn]
        xtst=dataX[tst]
        ytrn=dataY[trn]
        ytst=dataY[tst]

        return xtrn, ytrn, xtst, ytst
    
if __name__=='__main__':
    
    pass