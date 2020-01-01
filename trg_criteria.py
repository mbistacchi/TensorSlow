import numpy as np
import graph_run
import operations

red = np.random.randn(50,2) - 2*np.ones((50,2)) # centered at -2, -2
blu = np.random.randn(50,2) + 2*np.ones((50,2)) # centered at 2, 2

# Create a new graph
graph().as_default()

X = placeholder()
c = placeholder()

W = variable([
    [1, -1],
    [1, -1]
])

b = variable([0, 0])
p = softmax(add(matmul(X, W), b))

# Cross-entropy loss
J = negative(axis_sum(axis_sum(elmul(c, log(p)), axis=1)))

session1 = session()
print(session1.run(J, {
    X: np.concatenate((blu, red)),
    c: [[1, 0]] * len(blu) + [[0, 1]] * len(red) })) 
    # e.g. [[1, 0]]*2 = [[1, 0], [1, 0]]
    # as X is just blu appended with red, c matrix is the same


""" Have -J = argmax over W, b
                                        product over i->N
                                        product over j->C:
                                        p(c^_i = c_i)
                                        = ... p_i,j ^{I(c_i = j)}
                                        = ... p_i,j ^ c_i,j

"""
