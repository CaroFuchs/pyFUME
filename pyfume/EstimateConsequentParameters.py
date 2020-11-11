import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import numpy.matlib
import statsmodels.api as sm

class ConsequentEstimator(object):
    def __init__(self, x_train, y_train, firing_strengths):
        self.x_train=x_train
        self.y_train=y_train
        self.firing_strengths=firing_strengths
        
    def suglms(self, x_train, y_train, firing_strengths, global_fit=False, df=0):
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
        
        x=x_train.copy()
        y=y_train.copy()
        f=firing_strengths.copy()
        
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
            yl = np.transpose(x).dot(np.transpose(p))						                    #
            
            # Global model
            ym = x1.dot(p.flatten()) + df*NoRule
            ylm = yl.copy()
            
            # Mask all memberships < 0.2 with NaN's for plots
            ylm[f<0.2] = np.NaN
        
        elif global_fit == False:                                         # local weighted least mean squares estimates
            # Pre-allocate variable
            p=np.zeros([nf,nx])
            
            for i in range (0,nf):
                '''
                # Weight the points with the firing strength of the rule
                #W=np.sqrt(np.diag(f[:,i]))
                W=np.array([f[:,i],]*nx).transpose()
                x1 = np.multiply(W,x)
                y1 = np.multiply(f[:,i],y)
               
                # Perform QR decomposition
                Q,R = np.linalg.qr(x1)      # qr decomposition of x1
                Qy = np.dot(Q.T,y1)         # computing Q^T*b (project by onto the range of x1)
                
                #perform least-squares
                p[i] = np.linalg.solve(R,Qy)
                
                wls_model=sm.WLS(y, x, weights=f[:,i]) 
                results = wls_model.fit()
                p[i]=np.array(results.params)               
        '''
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
    
    def sugfunc(self, x1, x2, a, b, c):
        return a*x1 + b*x2 + c   
    