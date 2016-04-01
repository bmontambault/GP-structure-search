class Grammar:
    """
    Define a grammar for producing strings.
    """
    def __init__(self,start,rules):
        """
        Init Grammar with start symbol and production rules.
        Args:
            start: string of len=1 that begins all expansions within grammar.
            rules: dictionary with strings of len=1 as keys and lists of tuples possible child strings of len=1 and there probabilities as values.
        """
        self.start=start
        self.rules=rules
        
    def expand(self,state=None,prob=1):
        """
        Expand string by each symbol also in rules.
        Args:
            state: current state to be expanded.
            prob: probability of current state.
        Returns: generator with all possible expansions of state in tuples (string,prob).
        """
        if state==None:
            state=self.start
        for n,sym in enumerate(state):
            if sym in self.rules:
                for new_sym,new_prob in self.rules[sym]:
                    new=state[:n]+'('+new_sym+')'+state[n+1:]
                    yield new,prob*new_prob

            
