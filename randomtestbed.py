import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

class Operation:
    """Represents a graph node that performs a computation.

    An `Operation` is a node in a `Graph` that takes zero or
    more objects as input, and produces zero or more objects
    as output.
    """

    def __init__(self, input_nodes=[]):
        """Construct Operation
        """
        self.input_nodes = input_nodes

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)

        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)

    def compute(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass


class add(Operation):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        return x_value + y_value



class matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        """Construct matmul

        Args:
          a: First matrix
          b: Second matrix
        """
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        """Compute the output of the matmul operation

        Args:
          a_value: First matrix value
          b_value: Second matrix value
        """
        return a_value.dot(b_value)




class placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self):
        """Construct placeholder
        """
        self.consumers = []

        # Append this placeholder to the list of placeholders in the currently active default graph
        _default_graph.placeholders.append(self)



class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
    """

    def __init__(self, initial_value=None):
        """Construct Variable

        Args:
          initial_value: The initial value of this variable
        """
        self.value = initial_value
        self.consumers = []

        # Append this variable to the list of variables in the currently active default graph
        _default_graph.variables.append(self)


class Graph:
    """Represents a computational graph
    """

    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self



class Session:
    """Represents a particular execution of a computational graph.
    """

    def run(self, operation, feed_dict={}):
        """Computes the output of an operation

        Args:
          operation: The operation whose output we'd like to compute.
          feed_dict: A dictionary that maps placeholders to values for this session
        """

        # Perform a post-order traversal of the graph to bring the nodes into the right order
        nodes_postorder = traverse_postorder(operation)

        # Iterate all nodes to determine their value
        for node in nodes_postorder:

            if type(node) == placeholder:
                # Set the node value to the placeholder value from feed_dict
                node.output = feed_dict[node]
            elif type(node) == Variable:
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
        return operation.output


def traverse_postorder(operation):
    """Performs a post-order traversal, returning a list of nodes
    in the order in which they have to be computed

    Args:
       operation: The operation to start traversal at
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


# Create red points centered at (-2, -2)
red_points = np.random.randn(50, 2) - 2*np.ones((50, 2))

# Create blue points centered at (2, 2)
blue_points = np.random.randn(50, 2) + 2*np.ones((50, 2))

class sigmoid(Operation):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self, a):
        """Construct sigmoid

        Args:
          a: Input node
        """
        super().__init__([a])

    def compute(self, a_value):
        """Compute the output of the sigmoid operation

        Args:
          a_value: Input value
        """
        return 1 / (1 + np.exp(-a_value))


class softmax(Operation):
    """Returns the softmax of a.
    """

    def __init__(self, a):
        """Construct softmax

        Args:
          a: Input node
        """
        super().__init__([a])

    def compute(self, a_value):
        """Compute the output of the softmax operation

        Args:
          a_value: Input value
        """
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]


class log(Operation):
    """Computes the natural logarithm of x element-wise.
    """

    def __init__(self, x):
        """Construct log

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the log operation

        Args:
          x_value: Input value
        """
        return np.log(x_value)
    
    
class multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        """Construct multiply

        Args:
          x: First multiplicand node
          y: Second multiplicand node
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the multiply operation

        Args:
          x_value: First multiplicand value
          y_value: Second multiplicand value
        """
        return x_value * y_value  
    

class negative(Operation):
    """Computes the negative of x element-wise.
    """

    def __init__(self, x):
        """Construct negative

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the negative operation

        Args:
          x_value: Input value
        """
        return -x_value


class reduce_sum(Operation):
    """Computes the sum of elements across dimensions of a tensor.
    """

    def __init__(self, A, axis=None):
        """Construct reduce_sum

        Args:
          A: The tensor to reduce.
          axis: The dimensions to reduce. If `None` (the default), reduces all dimensions.
        """
        super().__init__([A])
        self.axis = axis

    def compute(self, A_value):
        """Compute the output of the reduce_sum operation

        Args:
          A_value: Input tensor value
        """
        return np.sum(A_value, self.axis)





class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute(self):
                # Compute gradients
                grad_table = compute_gradients(loss)

                # Iterate all variables
                for node in grad_table:
                    if type(node) == Variable:
                        # Retrieve gradient for this variable
                        grad = grad_table[node]

                        # Take a step along the direction of the negative gradient
                        node.value -= learning_rate * grad

        return MinimizationOperation()




# A dictionary that will map operations to gradient functions
_gradient_registry = {}


class RegisterGradient:
    """A decorator for registering the gradient function for an op type.
    """

    def __init__(self, op_type):
        """Creates a new decorator with `op_type` as the Operation type.
        Args:
          op_type: The name of an operation
        """
        self._op_type = eval(op_type)

    def __call__(self, f):
        """Registers the function `f` as gradient function for `op_type`."""
        _gradient_registry[self._op_type] = f
        return f



def compute_gradients(loss):
    # grad_table[node] will contain the gradient of the loss w.r.t. the node's output
    grad_table = {}

    # The gradient of the loss with respect to the loss is just 1
    grad_table[loss] = 1

    # Perform a breadth-first search, backwards from the loss
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        # If this node is not the loss
        if node != loss:
            #
            # Compute the gradient of the loss with respect to this node's output
            #
            grad_table[node] = 0

            # Iterate all consumers
            for consumer in node.consumers:

                # Retrieve the gradient of the loss w.r.t. consumer's output
                lossgrad_wrt_consumer_output = grad_table[consumer]

                # Retrieve the function which computes gradients with respect to
                # consumer's inputs given gradients with respect to consumer's output.
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]

                # Get the gradient of the loss with respect to all of consumer's inputs
                lossgrads_wrt_consumer_inputs = bprop(consumer, lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    # If there is a single input node to the consumer, lossgrads_wrt_consumer_inputs is a scalar
                    grad_table[node] += lossgrads_wrt_consumer_inputs

                else:
                    # Otherwise, lossgrads_wrt_consumer_inputs is an array of gradients for each input node

                    # Retrieve the index of node in consumer's inputs
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)

                    # Get the gradient of the loss with respect to node
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]

                    # Add to total gradient
                    grad_table[node] += lossgrad_wrt_node

        #
        # Append each input node to the queue
        #
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    # Return gradients for each visited node
    return grad_table

@RegisterGradient("negative")
def _negative_gradient(op, grad):
    """Computes the gradients for `negative`.

    Args:
      op: The `negative` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `negative` op.

    Returns:
      Gradients with respect to the input of `negative`.
    """
    return -grad








@RegisterGradient("log")
def _log_gradient(op, grad):
    """Computes the gradients for `log`.

    Args:
      op: The `log` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `log` op.

    Returns:
      Gradients with respect to the input of `log`.
    """
    x = op.inputs[0]
    return grad/x

@RegisterGradient("sigmoid")
def _sigmoid_gradient(op, grad):
    """Computes the gradients for `sigmoid`.

    Args:
      op: The `sigmoid` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `sigmoid` op.

    Returns:
      Gradients with respect to the input of `sigmoid`.
    """

    sigmoid = op.output

    return grad * sigmoid * (1 - sigmoid)



@RegisterGradient("multiply")
def _multiply_gradient(op, grad):
    """Computes the gradients for `multiply`.

    Args:
      op: The `multiply` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `multiply` op.

    Returns:
      Gradients with respect to the input of `multiply`.
    """

    A = op.inputs[0]
    B = op.inputs[1]

    return [grad * B, grad * A]

@RegisterGradient("matmul")
def _matmul_gradient(op, grad):
    """Computes the gradients for `matmul`.

    Args:
      op: The `matmul` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `matmul` op.

    Returns:
      Gradients with respect to the input of `matmul`.
    """

    A = op.inputs[0]
    B = op.inputs[1]

    return [grad.dot(B.T), A.T.dot(grad)]


@RegisterGradient("add")
def _add_gradient(op, grad):
    """Computes the gradients for `add`.

    Args:
      op: The `add` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `add` op.

    Returns:
      Gradients with respect to the input of `add`.
    """
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

@RegisterGradient("reduce_sum")
def _reduce_sum_gradient(op, grad):
    """Computes the gradients for `reduce_sum`.

    Args:
      op: The `reduce_sum` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `reduce_sum` op.

    Returns:
      Gradients with respect to the input of `reduce_sum`.
    """
    A = op.inputs[0]

    output_shape = np.array(A.shape)
    output_shape[op.axis] = 1
    tile_scaling = A.shape // output_shape
    grad = np.reshape(grad, output_shape)
    return np.tile(grad, tile_scaling)









@RegisterGradient("softmax")
def _softmax_gradient(op, grad):
    """Computes the gradients for `softmax`.

    Args:
      op: The `softmax` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `softmax` op.

    Returns:
      Gradients with respect to the input of `softmax`.
    """

    softmax = op.output
    return (grad - np.reshape(
        np.sum(grad * softmax, 1),
        [-1, 1]
    )) * softmax



# Create a new graph
Graph().as_default()

X = placeholder()
c = placeholder()

# Initialize weights randomly
W = Variable(np.random.randn(2, 2))
b = Variable(np.random.randn(2))

# Build perceptron
p = softmax(add(matmul(X, W), b))

# Build cross-entropy loss
J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

# Build minimization op
minimization_op = GradientDescentOptimizer(learning_rate=0.01).minimize(J)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)

}

# Create session
session = Session()

# Perform 100 gradient descent steps
for step in range(1000):
    J_value = session.run(J, feed_dict)
    if step % 100 == 0:
        print("Step:", step, " Loss:", J_value)
    session.run(minimization_op, feed_dict)

# Print final result
W_value = session.run(W)
print("Weight matrix:\n", W_value)
b_value = session.run(b)
print("Bias:\n", b_value)



W_value =  np.array([[ 1.27496197 -1.77251219], [ 1.11820232 -2.01586474]])
b_value = np.array([-0.45274057 -0.39071841])

# Plot a line y = -x
x_axis = np.linspace(-4, 4, 100)
y_axis = -W_value[0][0]/W_value[1][0] * x_axis - b_value[0]/W_value[1][0]
plt.plot(x_axis, y_axis)

# Add the red and blue points
plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')
plt.show()







