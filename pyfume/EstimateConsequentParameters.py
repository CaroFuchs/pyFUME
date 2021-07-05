import numpy as np
from scipy.optimize import curve_fit
import numpy.matlib

class ConsequentEstimator(object):
    """
        Creates a new consequent estimator object.
        
        Args:
            x_train: The input data.
            y_train: The output data (true label/golden standard).
            firing_strengths: Matrix containing the degree to which each rule 
                fires for each data instance.
    """
    
    def __init__(self, x_train, y_train, firing_strengths):
        self.x_train=x_train
        self.y_train=y_train
        self.firing_strengths=firing_strengths
        
    def zero_order(self):
        """
            Estimates the consequent parameters of the zero-order Sugeno-Takagi model using normalized means.
        
            Args:
                df: default value returned when the sum of grades equals to one (default = 0).
        
            Returns:
                The parameters for the consequent function.
        """
        p=np.zeros((self.firing_strengths.shape[1]))
        for clus in range(0,self.firing_strengths.shape[1]): 
            fs=self.firing_strengths[:,clus]
            fs = np.fmax(fs, np.finfo(np.float64).eps)        # avoid 0's in the matrix
            normalized_weights=fs/fs.sum(0)
            s=np.multiply(normalized_weights, self.y_train) 
            p[clus]=sum(s)
        return p
            
    def suglms(self, global_fit=False, df=0):
        """
            Estimates the consequent parameters in the first-order Sugeno-Takagi model using least squares.
        
            Args:
                global_fit: Use the local (global_fit=False) or global (global_fit=True) least mean squares estimates. Global estimates functionality is still in beta mode, so use with caution.
                df: default value returned when the sum of grades equals to one (default = 0).
        
            Returns:
                The parameters for the consequent function.
        """
        
        x=self.x_train.copy()
        y=self.y_train.copy()
        f=self.firing_strengths.copy()
        
        # Check if input X contains one column of ones (for the constant). If not, add it.
        u=np.unique(x[:,-1])
        if u.shape[0]!=1 or u[0]!=1:
            x = np.hstack((x,np.ones((x.shape[0],1))))
    
        # Find the number of data points (mx & mf) , the number of variables (nx) and the
        # number of clusters (nf) 
        mx,nx=x.shape
        mf,nf=f.shape
        
        # Calculate the sum of the degree of fulfillement (DOF) for each data point
        sumDOF=np.sum(f, 1)
        
        
        # When degree of fulfillment is zero (which means no rule is applicable), set to one
        NoRule = sumDOF == 0
        sumDOF[NoRule] = 1
        sumDOF = np.matlib.repmat(sumDOF,nf,1).T
        
        
        if nf == 1:
           global_fit = False
        
        if global_fit == True:                # Global least mean squares estimates 

            # Still under construction!
            
            # Auxillary variables
            f1 = x.flatten()
            s = np.matlib.repmat(f1,nf,1).T
            xx = np.reshape(s, (nx,nf*mx), order='F')
            s = xx.T  
            x1=np.reshape(s,(mx,nf*nx),order='F') 
            x=x.T                                # reshape data matrix
            
            # (reshaped) vector of f devided by the sum of each row of f
            # (normalised membership degree)
            xx = (f.T.flatten()/sumDOF.T.flatten())
            
            # reshape partition matrix
            s= np.matlib.repmat(xx,nx,1).T
            f1 = np.reshape(s, (mx,nf*nx), order='F')                
            x1 = f1*x1
        
            # Find least squares solution
#            xp = np.linalg.lstsq(x1,y,rcond=None)
            
            # Perform QR decomposition
            Q,R = np.linalg.qr(x1) # qr decomposition of A
            Qy = np.dot(Q.T,y) # computing Q^T*b (project b onto the range of A)
            xx = np.linalg.solve(R,Qy)
            
            p=np.reshape(xx,(nf,nx), order='F')
        
            # Local models
            yl = np.transpose(x).dot(np.transpose(p))                                           #
            
            # Global model
            ym = x1.dot(p.flatten()) + df*NoRule
            ylm = yl.copy()
            
            # Mask all memberships < 0.2 with NaN's for plots
            ylm[f<0.2] = np.NaN
        
        elif global_fit == False:                                         # local weighted least mean squares estimates
            # Pre-allocate variable
            p=np.zeros([nf,nx])
            
            for i in range (0,nf):
                # Select firing strength of the selected rule
                w= f[:,i]
                
                # Weight input with firing strength
                xw = x * np.sqrt(w[:,np.newaxis])
                
                # Weight output with firing strength
                yw = y * np.sqrt(w)
                
                # Perform least squares with weighted input and output
                prm,_,_,_=np.linalg.lstsq(xw, yw, rcond=None)
                p[i]=prm
                       
        return p #,ym,yl,ylm
