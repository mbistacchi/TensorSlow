import numpy as np
import operations

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
    in the order in which they have to be computed

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

