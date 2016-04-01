class Language:
    """
    Define a language for translating strings to kernels.
    Attributes:
        kernel_mappings: Dictionary mapping kernel_strings to kernels.
        operation_mappings: Dictionary mapping operation_strings to operations.
    """
    def __init__(self,kernel_strings,operation_strings,kernels,operations):
        """
        Init language specifying mappings between strings and kernels for kernels and operations.
        Args:
            kernel_strings: list of strings to be mapped onto kernels.
            operation_strings: list of strings to be mapped onto operations.
            kernels: list of Kernel objects, with one corresponding to each string in kernel_strings.
            operations: list of Operation objects, with one corresponding to each string in operation_strings.
        """
        self.kernel_mappings={kernel_strings[i]:kernels[i] for i in xrange(len(kernel_strings))}
        self.operation_mappings={operation_strings[i]:operations[i] for i in xrange(len(operation_strings))}

    def translate(self,string):
        """
        Translate string to Kernel object.
        Args:
            string: input string to be translated.
        Returns: Kernel instance.
        """
        if '(' not in string:
            return self.kernel_mappings[string]
        queue=[]
        while len(string)>0:
            up_to_end=string[string.find("(")+1:string.find(")")]
            if '(' in up_to_end:
                inner=up_to_end.split('(')[-1]
            else:
                inner=up_to_end
            remaining=''.join(string.split('('+inner+')'))
            string=remaining
            if len(inner)==3:
                queue.append(self.operation_mappings[inner[1]](self.kernel_mappings[inner[0]],self.kernel_mappings[inner[2]]))
            elif len(inner)==2:
                if inner[0] in self.operation_mappings:
                    queue.append(self.operation_mappings[inner[0]](queue[-1],self.kernel_mappings[inner[1]]))
                else:
                    queue.append(self.operation_mappings[inner[1]](queue[-1],self.kernel_mappings[inner[0]]))
            elif len(inner)==1:
                queue.append(self.operation_mappings[inner](queue[-1],queue[-2]))
        kernel=queue[-1]
        return kernel
        

