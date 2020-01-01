import numpy as np

""" This file contains the definitions of nodes in a computational graph - 
    namely Operations, variables (used for weights/bias), and placeholders
    (used for training data inputs). Also defines a graph structure (not actually
    used), and a session class, used to actually run the maths. """
    
#------------------------------------------------------------------------

""" Math Operations """

class Operation:
    
    def __init__(self, input_nodes=[]):
        
        self.input_nodes = input_nodes
        self.consumers = []
        
        # Append operation to list of consumers to previous inputs
        for i in input_nodes:
            i.consumers.append(self)
        
        _default_graph.operations.append(self)
        
        
    def compute(self):
        pass #defined by each operation
        
###
class add(Operation):
    
    def __init__(self, x, y):
        
        # super refers to parent class, i.e. this does the initializing of add as an Operation class
        super().__init__([x,y]) 
    
    def compute(self, x_val, y_val):
        return x_val + y_val
   
###   
class matmul(Operation):
    
    def __init__(self, a, b):
        
        super().__init__([a,b])
        
    def compute(self, a_val, b_val):
        return np.dot(a_val, b_val)
        
class sigmoid(Operation):

    def __init__(self, a):
        super().__init__([a])

    def compute(self, a_val):
        return 1 / (1 + np.exp(-a_val))
        
class softmax(Operation):

    def __init__(self, a):
        super().__init__([a])

    def compute(self, a_val):
        # columns within a single row sum to 1
        # axis = 1 collapses columns i.e. merge horizontally
        # Forms a row vector even though paper-math intuitively gives column vector
        # [:, None] transposes the sum vector to a column vector: [a,b] --> [[a], [b]]
        # Then just element wise division of matrix a by column vector
        return np.exp(a_val) / np.sum(np.exp(a_val), axis=1)[:, None] 
    
class log(Operation):
    
    def __init__(self, x):
        super().__init__([x])
        
    def compute(self, x_val):
        return np.log(x_val)
    
class elmul(Operation): # element wise multiplication
    
    def __init__(self, x, y):
        super().__init__([x, y])
        
    def compute(self, x_val, y_val):
        return x_val * y_val
    
class axis_sum(Operation): # sum over a given axis in a tensor
    
    def __init__(self, A, axis = None):
        super().__init__([A])
        self.axis = axis
        
    def compute(self, A_val):
        return np.sum(A_val, self.axis)

class negative(Operation): #compute negative of input element wise

    def __init__(self, x):
        super().__init__([x])
        
    def compute(self, x_val):
        return -x_val

""" ----------------- Other node types -------------------------------"""

class placeholder:
    
    def __init__(self):
        
        self.consumers = []
        
        _default_graph.placeholders.append(self) # append to graph's list of placeholders


class variable:
    
    def __init__(self, initial_value = None):
        
        self.value = initial_value
        self.consumers = []
        
        _default_graph.variables.append(self)
        

""" Graph Class """

class graph:
    
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
        
    def as_default(self):
        global _default_graph
        _default_graph = self




""" ---------------- Define a session instance -------------------------- """
        
class session:
    """Represents a particular execution of a computational graph"""

    def run(self, op, feed_dict={}):
        """Computes the output of an operation

        Args:
          op: The operation whose output we'd like to compute.
          feed_dict: A dictionary that maps placeholders to values for this session
        """

        # Perform a post-order traversal of the graph to bring the nodes into the right order
        nodes_postorder = traverse_postorder(op)

        # Iterate all nodes to determine their value / output
        for node in nodes_postorder:

            if type(node) == placeholder:
                # Set the node value to the placeholder value from feed_dict
                node.output = feed_dict[node] # Ergo set node x as {x: [1,2]}
                
            elif type(node) == variable:
                # Set the node value to the variable's value attribute
                node.output = node.value
                
            else:  # Operation
                # Get the input values for this operation from node_values
                node.inputs = [input_node.output for input_node in node.input_nodes]

                # Compute the output of this operation
                node.output = node.compute(*node.inputs)

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        # Return the requested node value
        return op.output


def traverse_postorder(op):
    """Performs a post-order traversal, returning a list of nodes
    in the order in which they have to be computed - i.e. first element in the list
    is the first node to be computed (towards inputs), and last in list is final output
    of calculation (output side).

    Args:
       operation: The operation to start traversal at
    """

    nodes_postorder = []

    def recurse(node):
        # operations are only nodes which can have multiple input nodes
        if isinstance(node, Operation): 
            for input_node in node.input_nodes:
                recurse(input_node)
                
        nodes_postorder.append(node) #doing this last puts node last - temporally

    recurse(op)
    return nodes_postorder

