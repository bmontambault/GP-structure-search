import numpy as np
from operator import add,mul
from random import uniform

class Kernel(object):
    """
    Kernel defining covariance and mean functions for gaussian process regression
    Attributes:
        cov_bounds: list of tuples containing lower and upper bounds for each of the parameters in the kernel's covariance function.
        cov_args: list of parameters for the kernel's covariance function.
        mean_args: list of parameters for the kernel's mean function.
    """

    def __init__(self,cov_bounds,cov_args,mean_args):
        """
        Inits Kernel object with bounds and arguments for covariance and mean functions
        Raises:
            ValueError: if cov_args are outside of cov_bounds
        """
        in_bounds=True
        for i in xrange(len(cov_args)):
            if cov_args[i]<cov_bounds[i][0] or cov_args[i]>cov_bounds[i][1]:
                in_bounds=False
        if in_bounds==False:
            raise ValueError('covariance parameters out of bounds')
        self.cov_bounds=cov_bounds
        self.cov_args=cov_args
        self.mean_args=mean_args
    
    def mean(self,x):
        """
        Default mean function for kernels.
        Args:
            x: value to be put through mean function
        Returns: default mean value for gaussian process with this kernel as defined by first element in mean_args (usually 0).
        """
        return self.mean_args[0]

    def update_args(self,new_cov_args):
        """
        Update parameters for kernel's covariance Function
        Args:
            new_cov_args: list of values to replace old parameters
        Returns: None
        """
        self.cov_args=new_cov_args


class RandomNoise(Kernel):
    """
    Kernel instance where observations are expected to be random noise.
    """
    def __init__(self,cov_args=[.1],mean_args=[0]):
        Kernel.__init__(self,[(0,1e309)],cov_args,mean_args)

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        if x1==x2:
            return cov_args[0]
        else:
            return 0


class SquaredExponential(Kernel):
    """
    Kernel instance where observations are expected to be grouped by distance.
    """
    def __init__(self,cov_args=[1,1],mean_args=[0]):
        Kernel.__init__(self,[(1,1e309),(.00001,1e309)],cov_args,mean_args)

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        return cov_args[0]*np.exp(-(abs(x1-x2)**2)/(2*(cov_args[1]**2)))


class Linear(Kernel):
    """
    Kernel instance where observations are expected to be linear.
    """
    def __init__(self,cov_args=[1,0],mean_args=[0,0]):
        Kernel.__init__(self,[(1,1e309),(0,1e309)],cov_args,mean_args)

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        return cov_args[0]*(x1-cov_args[1])*(x2-cov_args[1])

    def mean(self,x):
        return self.mean_args[0]+x*self.mean_args[1]

    
class Operation(Kernel):
    """
    Operation defining kernel that is the result of combining two kernels
    """
    def __init__(self,k1,k2,operation):
        Kernel.__init__(self,k1.cov_bounds+k2.cov_bounds,k1.cov_args+k2.cov_args,k1.mean_args+k2.mean_args)
        self.k1=k1
        self.k2=k2
        self.partitioned_cov_args=[k1.cov_args,k2.cov_args]
        self.operation=operation

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        k_args=[[cov_args[i] for i in xrange(len(self.partitioned_cov_args[0]))],[cov_args[i] for i in xrange(len(self.partitioned_cov_args[0]),len(self.partitioned_cov_args[1])+len(self.partitioned_cov_args[0]))]]
        return self.operation(self.k1.cov(x1,x2,k_args[0]),self.k2.cov(x1,x2,k_args[1]))

    def mean(self,x):
        return self.operation(self.k1.mean(x),self.k2.mean(x))


class Add(Operation):
    """
    Addition between two kernels
    """
    def __init__(self,k1,k2):
        Operation.__init__(self,k1,k2,add)


class Mult(Operation):
    """
    Multiplication between two kernels
    """
    def __init__(self,k1,k2):
        Operation.__init__(self,k1,k2,mul)


class ChangePoint(Kernel):
    
    def __init__(self,k1,k2,cov_args):
        Kernel.__init__(self,k1.cov_bounds+k2.cov_bounds+[(-1e309,1e309),(-1e309,1e309)],k1.cov_args+k2.cov_args+cov_args,k1.mean_args+k2.mean_args)
        self.k1=k1
        self.k2=k2
        
    def sig(self,x,cov_args):
        return .5*(1+np.tanh((cov_args[0]-x)/cov_args[1]))

    def set_args(self,cov_args):
        self.k1_args=[cov_args[i] for i in xrange(len(self.k1.cov_args))]
        self.k2_args=[cov_args[j] for j in xrange(len(self.k1.cov_args),len(self.k1.cov_args)+len(self.k2.cov_args))]
        self.cp_args=[cov_args[-2],cov_args[-1]]


class Det(ChangePoint):

    def __init__(self,k1,k2,cov_args=[0,1]):
        ChangePoint.__init__(self,k1,k2,cov_args)

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        self.set_args(cov_args)
        return self.sig(x1,self.cp_args)*self.k1.cov(x1,x2,self.k1_args)*self.sig(x2,self.cp_args)+(1-self.sig(x1,self.cp_args))*self.k2.cov(x1,x2,self.k2_args)*(1-self.sig(x2,self.cp_args))
        
    def mean(self,x,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        self.set_args(cov_args)
        return self.sig(x,self.cp_args)*self.k1.mean(x)+(1-self.sig(x,self.cp_args))*self.k2.mean(x)
    

class Prob(ChangePoint):

    def __init__(self,k1,k2,cov_args=[0,1]):
        ChangePoint.__init__(self,k1,k2,cov_args)
        self.clusters=[[],[]]

    def assign(self,x,cp_args):
        if x not in self.clusters[0] and x not in self.clusters[1]:
            probs=[self.sig(x,cp_args),(1-self.sig(x,cp_args))]
            factor=sum(probs)
            probs=[p/factor for p in probs]
            if uniform(0,1)<max(probs):
                index=probs.index(max(probs))
            else:
                index=probs.index(min(probs))
            self.clusters[index].append(x)

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        self.set_args(cov_args)
        self.assign(x1,self.cp_args)
        self.assign(x2,self.cp_args)
        if x1 in self.clusters[0] and x2 in self.clusters[0]:
            return self.k1.cov(x1,x2,self.k1_args)
        elif x1 in self.clusters[1] and x2 in self.clusters[1]:
            return self.k2.cov(x1,x2,self.k2_args)
        else:
            return 0

    def mean(self,x,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        self.set_args(cov_args)
        self.assign(x,self.cp_args)
        if x in self.clusters[0]:
            return self.k1.mean(x)
        else:
            return self.k2.mean(x)      
    


