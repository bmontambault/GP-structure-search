import numpy as np
from scipy.stats import norm,multivariate_normal
from scipy.optimize import fmin_cg

class GP:

    def __init__(self,X,Y,kernel):
        self.X=X
        self.Y=Y
        self.kernel=kernel

        self.mean_vector=np.array([self.kernel.mean(xm) for xm in self.X])
        self.gram=np.matrix([[kernel.cov(xm,xn) for xm in X] for xn in X])+10**-9*np.identity(len(X))
        self.inverted_gram=np.linalg.inv(self.gram)
        
  def neg_log_ml(self,cov_args):
        in_bounds=True
        for i in xrange(len(cov_args)):
            lower=self.kernel.cov_bounds[i][0]
            upper=self.kernel.cov_bounds[i][1]
            if cov_args[i]<lower or cov_args[i]>upper:
                in_bounds=False
        if in_bounds==False:
            return 1e309
        else:
            gram=np.matrix([[self.kernel.cov(xm,xn,cov_args) for xm in self.X] for xn in self.X])+10**-9*np.identity(len(self.X))
            return -1*np.log(multivariate_normal(self.mean_vector,gram).pdf(self.Y))

    def optimize_cov_args(self):
        opt=fmin_cg(self.neg_log_ml,self.kernel.cov_args,disp=False,retall=False)
        self.kernel.update_args(opt)
        return GP(self.X,self.Y,self.kernel)

    def bic(self):
        return (2*self.neg_log_ml(self.kernel.cov_args))+(len(self.kernel.cov_args)*np.log(len(self.X)))
        
    def prediction(self,x):
        mean=self.kernel.mean(x)
        vector=np.array([self.kernel.cov(x,xm) for xm in self.X])
        return np.sum(mean+np.dot(np.dot(self.inverted_gram,vector),(self.Y-self.mean_vector)))
