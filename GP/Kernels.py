import numpy as np
from random import uniform
from scipy.stats import norm

class Kernel(object):

    def __init__(self,cov_bounds,cov_args,mean_args):
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
        return self.mean_args[0]

    def update_args(self,new_cov_args):
        self.cov_args=new_cov_args


class RandomNoise(Kernel):

    def __init__(self,cov_args=[0],mean_args=[0]):
        Kernel.__init__(self,[(0,1e309)],cov_args,mean_args)

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        if x1==x2:
            return cov_args[0]
        else:
            return 0


class SquaredExponential(Kernel):

    def __init__(self,cov_args=[1,1],mean_args=[0]):
        Kernel.__init__(self,[(1,1e309),(.00001,1e309)],cov_args,mean_args)

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        return cov_args[0]*np.exp(-(abs(x1-x2)**2)/(2*(cov_args[1]**2)))


class Linear(Kernel):

    def __init__(self,cov_args=[1,0],mean_args=[0,0]):
        Kernel.__init__(self,[(1,1e309),(0,1e309)],cov_args,mean_args)

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        return cov_args[0]*(x1-cov_args[1])*(x2-cov_args[1])

    def mean(self,x):
        return self.mean_args[0]+x*self.mean_args[1]


class Add(Kernel):

    def __init__(self,k1,k2):
        Kernel.__init__(self,k1.cov_bounds+k2.cov_bounds,k1.cov_args+k2.cov_args,k1.mean_args+k2.mean_args)
        self.k1=k1
        self.k2=k2
        
    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        k1_args=[cov_args[i] for i in xrange(len(self.k1.cov_args))]
        k2_args=[cov_args[j] for j in xrange(len(self.k1.cov_args),len(cov_args))]
        return self.k1.cov(x1,x2,k1_args)+self.k2.cov(x1,x2,k2_args)

    def mean(self,x):
        return self.k1.mean(x)+self.k2.mean(x)


class Mult(Kernel):

    def __init__(self,k1,k2):
        Kernel.__init__(self,k1.cov_bounds+k2.cov_bounds,k1.cov_args+k2.cov_args,k1.mean_args+k2.mean_args)
        self.k1=k1
        self.k2=k2
        
    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        k1_args=[cov_args[i] for i in xrange(len(self.k1.cov_args))]
        k2_args=[cov_args[j] for j in xrange(len(self.k1.cov_args),len(cov_args))]
        return self.k1.cov(x1,x2,k1_args)*self.k2.cov(x1,x2,k2_args)

    def mean(self,x):
        return self.k1.mean(x)*self.k2.mean(x)


class ChangePoint(Kernel):

    def __init__(self,k1,k2,cov_args=[0,1]):
        Kernel.__init__(self,k1.cov_bounds+k2.cov_bounds,k1.cov_args+k2.cov_args,k1.mean_args+k2.mean_args)
        self.k1=k1
        self.k2=k2

    def mean(self,x,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        midpoint=cov_args[0]
        speed=cov_args[1]
        print midpoint,speed,x
        gauss=norm(midpoint,speed).pdf(x)
        if x<midpoint:
            term1=1-gauss
        else:
            term1=gauss
        term2=1-term1
        terms=[term1,term2]
        factor=sum(terms)
        term1=term1/factor
        term2=term2/factor
        return (self.k1.mean*term1)+(self.k2.mean*term2)
        

class DetChangePoint(Kernel):

    def __init__(self,k1,k2,cov_args=[0,1]):
        Kernel.__init__(self,k1.cov_bounds+k2.cov_bounds+[(-1e309,1e309),(-1e309,1e309)],k1.cov_args+k2.cov_args+cov_args,k1.mean_args+k2.mean_args)
        self.k1=k1
        self.k2=k2

    def sig(self,x,cov_args):
        return .5*(1+np.tanh((cov_args[0]-x)/cov_args[1]))

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        k1_args=[cov_args[i] for i in xrange(len(self.k1.cov_args))]
        k2_args=[cov_args[j] for j in xrange(len(self.k1.cov_args),len(self.k1.cov_args)+len(self.k2.cov_args))]
        cp_args=[cov_args[-2],cov_args[-1]]
        return self.sig(x1,cp_args)*self.k1.cov(x1,x2,k1_args)*self.sig(x2,cp_args)+(1-self.sig(x1,cp_args))*self.k2.cov(x1,x2,k2_args)*(1-self.sig(x2,cp_args))

    def mean(self,x,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        cp_args=[cov_args[-2],cov_args[-1]]
        return self.sig(x,cp_args)*self.k1.mean(x)+(1-self.sig(x,cp_args))*self.k2.mean(x)


class ProbChangePoint(Kernel):

    def __init__(self,k1,k2,cov_args=[0,1]):
        Kernel.__init__(self,k1.cov_bounds+k2.cov_bounds+[(-1e309,1e309),(-1e309,1e309)],k1.cov_args+k2.cov_args+cov_args,k1.mean_args+k2.mean_args)
        self.k1=k1
        self.k2=k2

    def sig(self,x,cov_args):
        return .5*(1+np.tanh((cov_args[0]-x)/cov_args[1]))

    def cov(self,x1,x2,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        k1_args=[cov_args[i] for i in xrange(len(self.k1.cov_args))]
        k2_args=[cov_args[j] for j in xrange(len(self.k1.cov_args),len(self.k1.cov_args)+len(self.k2.cov_args))]
        cp_args=[cov_args[-2],cov_args[-1]]

        term1=self.sig(x1,cp_args)+self.sig(x2,cp_args)
        term2=(1-self.sig(x1,cp_args))+(1-self.sig(x2,cp_args))
        terms=[term1,term2]
        kerns=[self.k1,self.k2]
        args=[k1_args,k2_args]
        factor=term1+term2
        terms=[term1/factor,term2/factor]
        if uniform(0,1)<max(terms):
            index=terms.index(max(terms))
        else:
            index=terms.index(min(terms))
        return kerns[index].cov(x1,x2,args[index])   
        
    def mean(self,x,cov_args=None):
        if cov_args==None:
            cov_args=self.cov_args
        k1_args=[cov_args[i] for i in xrange(len(self.k1.cov_args))]
        k2_args=[cov_args[j] for j in xrange(len(self.k1.cov_args),len(self.k1.cov_args)+len(self.k2.cov_args))]
        cp_args=[cov_args[-2],cov_args[-1]]
        kerns=[self.k1,self.k2]
        terms=[self.sig(x,cp_args),(1-self.sig(x,cp_args))]
        args=[k1_args,k2_args]
        factor=sum(terms)
        terms=[i/factor for i in terms]
        if uniform(0,1)<max(terms):
            index=terms.index(max(terms))
        else:
            index=terms.index(min(terms))
        return kerns[index].mean(x)
