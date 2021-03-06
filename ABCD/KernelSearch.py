class KernelSearch:
    """
    Search for best possible kernel for a gaussian process given observations and a space of kernels.
    Attributes:
        grammar: Grammar instance defining how strings representing combinations of kernels are generated.
        language: Language instance defining how strings are translated into Kernel objects.
    """
    def __init__(self,grammar,language):
        """
        Init Kernel search with specified Grammar and Language instances.
        """
        self.grammar=grammar
        self.language=language

    def search(self,X,Y,scoring='bic'):
        """
        Search space of kernels defined by grammar and language based on fit to observed X and Y values.
        Args:
            X: numpy array of observed X values.
            Y: numpy array of observed Y values.
            scoring: how kernels are scored during search. If \'bic\' kernels are scored using Bayesian information criterion.
                     if \'pcfg\' kernels are scored by multiplying the marginal likelihood of the kernel and the probability of the string representing that kernel being generated by the grammar.
        Returns: highest scoring string, Kernel instance, GP instance, and score.
        Raises:
            ValueError: if scoring is something other than \'bic\' or \'pcfg\'.
        """
        string=(self.grammar.start,1)
        while True:
            print string
            parent=string
            string=self.__score(X,Y,parent,scoring)
            if string==parent:
                print string
                return GP(X,Y,self.language.translate(string[0]))

    def __score(self,X,Y,parent,scoring,iterations=1000):
        if scoring=='bic':
            return self.__score_bic(X,Y,parent)
        else:
            return self.__score_pcfg(X,Y,parent,iterations)

    def __score_bic(self,X,Y,parent):
        string,prob=parent
        kernel=self.language.translate(string)
        gp=GP(X,Y,kernel)
        score=gp.bic()
        for child in self.grammar.expand(string,1):
            child_string,child_prob=child
            child_kernel=self.language.translate(child_string)
            child_gp=GP(X,Y,child_kernel)
            child_score=child_gp.bic()
            if child_score>score:
                score=child_score
                string=child_string
                prob=child_prob
                kernel=child_kernel
                gp=child_gp
        return (string,child_prob)

    def __score_pcfg(self,X,Y,parent,iterations):
        string,prob=parent
        kernel=self.language.translate(string)
        likelihood=GP(X,Y,kernel).marginal_likelihood()
        string_atts={string:[prob,likelihood]}
        distribution={}
        state=string
        for i in xrange(iterations):
            if state!=string:
                if uniform(0,1)<string_atts[string][0]:
                    state=self.__accept(string_atts,state,string)
                    if state in distribution:
                        distribution[state]+=1
                    else:
                        distribution[state]=1
            for new_string,new_prob in self.grammar.expand(*parent):
                if new_string not in string_atts:
                    new_kernel=self.language.translate(new_string)
                    new_likelihood=GP(X,Y,new_kernel).marginal_likelihood()
                    string_atts[new_string]=[new_prob,new_likelihood]
                if uniform(0,1)<string_atts[new_string][0]:
                    state=self.__accept(string_atts,state,new_string)
                    if state in distribution:
                        distribution[state]+=1
                    else:
                        distribution[state]=1
        print distribution
        string=max(distribution, key=distribution.get)
        return (string,string_atts[string][0])
                        
    def __accept(self,atts,state,new_state):
        weights=[atts[state][1],atts[new_state][1]]
        non_neg_factor=min(weights)-1
        non_neg_weights=[x-non_neg_factor for x in weights]
        factor=sum(non_neg_weights)
        normalized=[x/factor for x in non_neg_weights]
        if uniform(0,1)<max(normalized):
            index=normalized.index(max(normalized))
        else:
            index=normalized.index(min(normalized))
        if index==0:
            return state
        else:
            return new_state
