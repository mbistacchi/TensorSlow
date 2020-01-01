from queue import Queue
import numpy as np
import operations

""" N.B  - grad_table is a dict of actual grad values for a given loss
         - grad_reg maps defined analytical functions to their analytical
         gradient functions wrt x where x is input to that node """


""" The end product - change variables (W, b) by specified chained gradients """

class grad_desecent_optimizer:
    
    def __init__(self, mu): # mu = learning rate
        self.mu = mu
        
    def minimize(self, loss): # loss = J = cross entropy
        mu = self.mu
        
        class minimize_op(Operation): # defining minimize operation node
            def compute(self):
                # compute gradients
                grad_table = compute_gradients(loss) # define later - is dictionary
                
                for node in grad_table: # iterate across all variables
                    if type(node) == variable: # i.e. Weights, bias
                    
                        grad = grad_table[node]
                        # Take a step along the direction of the negative gradient
                        node.value -= mu * grad
    
        return minimize_op()
    



""" Decorator for registering the grad function of an operation type
    e.g. put the value for sigmoid key as the gradient of sigmoid """
    
_grad_reg = {} # A dictionary that maps operations to their grad function
class registergrad: 

    def __init__(self, op_type): # create new decorator
        self._op_type = eval(op_type) # ?????
        
    """ __call__ allows the instance of the class to be called as function """    
    def __call__(self, f): # registers f as the gradient function for the operation
        
        _grad_reg[self._op_type] = f # insert value for the key
        return f # why need to return?
    
    
    

""" Gradients for the Operations
    chain rule the gradients i.e. d/dx = d/dy(x) * dy(x)/dx
    i.e. takes in grad for output, returns grad for input
    The functions define the __call__ of register_grad class """

@registergrad("negative")
def _negative_grad(op, grad): # op is the negative operation
    return -grad # Given grad wrt -x, grad wrt x = -grad

@registergrad("log")
def log_grad(op, grad):
    x = op.inputs[0]
    return grad/x # Given grad wrt log(x), grad wrt x = grad /x

@registergrad("sigmoid")
def sigmoid_gradient(op, grad):
    sigmoid = op.output
    return grad * sigmoid * (1 - sigmoid)

@registergrad("elmul")
def elmul_grad(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad * B, grad * A] # can seperate out grads of products

@registergrad("matmul")
def matmul_grad(op, grad):
    A = op.inputs[0]
    B = op.inputs[1]
    return [grad.dot(B.T), A.T.dot(grad)] # can seperate out grads of products
    
@registergrad("add")
def add_grad(op, grad): # would just be return [grad, grad] but
                        # have to wizardry for shape(a) != shape(b)

    a = op.inputs[0]
    b = op.inputs[1]

    grad_wrt_a = grad
    while np.ndim(grad_wrt_a) > len(a.shape):
        grad_wrt_a = np.sum(grad_wrt_a, axis=0)
    for axis, size in enumerate(a.shape):
        if size == 1:
            grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

    grad_wrt_b = grad
    while np.ndim(grad_wrt_b) > len(b.shape):
        grad_wrt_b = np.sum(grad_wrt_b, axis=0)
    for axis, size in enumerate(b.shape):
        if size == 1:
            grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

    return [grad_wrt_a, grad_wrt_b]

@registergrad("axis_sum")
def axis_sum_gradient(op, grad):

    A = op.inputs[0]

    output_shape = np.array(A.shape)
    output_shape[op.axis] = 1
    tile_scaling = A.shape // output_shape
    grad = np.reshape(grad, output_shape)
    return np.tile(grad, tile_scaling)

@registergrad("softmax")
def softmax_grad(op, grad):
    softmax = op.output
    return (grad - np.reshape(np.sum(grad * softmax, 1), [-1, 1])) * softmax
  

""" ------------------------------------------------------------------- """
    
""" Computes the gradient values for grad_table """

def compute_gradients(loss):
    # grad_table[node] is key for gradient of the loss wrt nodes output
    grad_table = {} 
    grad_table[loss] = 1 # d loss/ d loss = 1
    
    """ Now perform breadth-first search, backward from loss """
    
    #Visited/Queue to structure / record back-propogation
    visited = set() # unordered collection, no duplicates
    queue = Queue() # FIFO data structure
    visited.add(loss) 
    queue.put(loss) # start back-prop at loss (final node)
    
    while not queue.empty():
        
        node = queue.get() # get first node put in at last step of back-prop
        
        if node != loss: # for all nodes but final loss node
            
            grad_table[node] = 0 # initialize
            
            for consumer in node.consumers: # for the nodes consumers
                
                # loss grad for the consumer is known
                loss_grad_wrt_consumer_output = grad_table[consumer]
                
                # call the analytical grad function for that operation type
                consumer_op_type = consumer.__class__
                back_prop = _grad_reg[consumer_op_type] 
                
                # Apply chain rule - enter the arguments for the __call__ of the grad function (op, grad)
                loss_grads_wrt_consumer_inputs = back_prop(consumer, loss_grad_wrt_consumer_output)
                
                # If node under consideration is the only input to the consumer under consideration...
                if len(consumer.input_nodes) == 1:
                    
                    grad_table[node] += loss_grads_wrt_consumer_inputs # simple update
                    
                else:
                    # if more than one input to node under consideration's consumer under consideration
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node) # index it
                    
                    # Just add the node's contribution to the loss gradient - other contributions from other rounds of back-prop
                    loss_grad_wrt_node = loss_grads_wrt_consumer_inputs[node_index_in_consumer_inputs]
                    grad_table[node] += loss_grad_wrt_node
                    
        # Update queue for further backpropogation
        if hasattr(node, "input_nodes"): # if node has input nodes
            for input_node in node.input_nodes: # for those input nodes
                if not input_node in visited: # if haven't flagged as visited node
                    visited.add(input_node)   # now flag as visited
                    queue.put(input_node)     # add node to queue for next step of back-propogation
    
    return grad_table