import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import basinhopping

class GP:
    """
    Gaussian Process for regression.
    Attributes:
        X: numpy array of observed X values.
        Y: numpy array of observed Y values.
        kernel: chosen Kernel subclass.
        mean_vector: numpy array of all values in X put through kernel's mean function.
        cov_matrix: len(X) by len(X) matrix of all pairs of all values in X put through kernel's cov function.
        inverted_cov_matrix: cov_matrix put through numpy's inversion function.
    """
    def __init__(self,X,Y,kernel,optimize=False,niter=100):
        """
        Init GP with observed data X,Y data and chosen kernel.
        Args:
            optimize: if True optimize selected kernel's cov_args to best fit data.
        """
        self.X=X
        self.Y=Y
        self.kernel=kernel
        self.mean_vector=np.array([self.kernel.mean(xm) for xm in self.X])
        if optimize==True:
            self.optimize()
        self.cov_matrix=np.matrix([[kernel.cov(xm,xn) for xm in X] for xn in X])+10**-9*np.identity(len(X))
        self.inverted_cov_matrix=np.linalg.inv(self.cov_matrix)        

    def optimize(self,niter=100):
        """
        Optimize kernel's hyperparameters using basin hopping.
        Args:
            niter: iterations in basin hopping function.
        Returns: optimized covariance parameters for kernel.
        """
        def optimized_marginal_likelihood(cov_args):
            """
            Negative log marginal likelihood to be minimized.
            if cov_args is out of bounds specified by kernel return inf
            """
            in_bounds=True
            for i in xrange(len(cov_args)):
                lower=self.kernel.cov_bounds[i][0]
                upper=self.kernel.cov_bounds[i][1]
                if cov_args[i]<lower or cov_args[i]>upper:
                    in_bounds=False
            if in_bounds==False:
                return 1e309
            else:
                cov_matrix=np.matrix([[self.kernel.cov(xm,xn,cov_args) for xm in self.X] for xn in self.X])+10**-9*np.identity(len(self.X))
            return -1*np.log(multivariate_normal(self.mean_vector,cov_matrix).pdf(self.Y))
        opt=basinhopping(optimized_marginal_likelihood,self.kernel.cov_args,niter=niter).x
        self.kernel.update_args(list(opt))
        
    def marginal_likelihood(self):
        """
        Determine marginal likelihood of selected kernel given it's cov function's parameters.
        Returns: log marginal likelihood of selected kernel given cov_args.
        """
        return np.log(multivariate_normal(self.mean_vector,self.cov_matrix).pdf(self.Y))

    def bic(self):
        """
        Determines Bayesian Information Criterion of selected kernel given it's cov function's parameters.
        Returns: BIC of selected kernel given cov_args.
        """
        return (2*self.marginal_likelihood())-(len(self.kernel.cov_args)*np.log(len(self.X)))

    def prediction(self,x):
        """
        Predicts y value for new x value.
        Returns: predicted x value.
        """
        mean=self.kernel.mean(x)
        cov_vector=np.array([self.kernel.cov(x,xm) for xm in self.X])
        return np.sum(mean+np.dot(np.dot(self.inverted_cov_matrix,cov_vector),(self.Y-self.mean_vector)))
