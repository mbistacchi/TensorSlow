import numpy as np
import matplotlib.pyplot as plt
import operations



red = np.random.randn(50,2) - 2*np.ones((50,2)) # centered at -2, -2
blu = np.random.randn(50,2) + 2*np.ones((50,2)) # centered at 2, 2

# Create a new graph
graph().as_default()


""" Define Graph structure - inputs / computations """

X = placeholder()
c = placeholder()

# Initialize weights randomly
W = variable(np.random.randn(2, 2))
b = variable(np.random.randn(2))

# Build perceptron
p = softmax(add(matmul(X, W), b))

# Build cross-entropy loss
J = negative(axis_sum(axis_sum(elmul(c, log(p)), axis=1)))

# Build minimization op
minimization_op = grad_desecent_optimizer(mu=0.01).minimize(J)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blu, red)),
    c:
        [[1, 0]] * len(blu)
        + [[0, 1]] * len(red)

}

""" Run! """

# Create session
session1 = session()

# Perform n gradient descent steps
for step in range(1000):
    J_value = session1.run(J, feed_dict)
    if step % 100 == 0:
        print("Step:", step, " Loss:", J_value)
    session1.run(minimization_op, feed_dict)

# Print final result
W_value = session1.run(W)
print("Weight matrix:\n", W_value)
b_value = session1.run(b)
print("Bias:\n", b_value)



#W_value =  np.array([[ 1.27496197 -1.77251219], [ 1.11820232 -2.01586474]])
#b_value = np.array([-0.45274057 -0.39071841])

# Plot a line y = -x
x_axis = np.linspace(-4, 4, 100)
y_axis = - W_value[0][0]/W_value[1][0] * x_axis - b_value[0]/W_value[1][0]
plt.plot(x_axis, y_axis)

# Add the red and blue points
plt.scatter(red[:,0], red[:,1], color='red')
plt.scatter(blu[:,0], blu[:,1], color='blue')
plt.show()