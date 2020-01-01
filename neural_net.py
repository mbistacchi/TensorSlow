import numpy as np
import matplotlib.pyplot as plt
import operations
#import gradient

n = 50000 # number of gradient iterations

"""Create data """
# Create two clusters of red points centered at (0, 0) and (1, 1), respectively. 
red = np.concatenate((
    0.2*np.random.randn(50, 2) + np.array([[0, 0]]*50),
    0.2*np.random.randn(50, 2) + np.array([[1, 1]]*50)
))

# Create two clusters of blue points centered at (0, 1) and (1, 0), respectively.
blu = np.concatenate((
    0.2*np.random.randn(50, 2) + np.array([[0, 1]]*50),
    0.2*np.random.randn(50, 2) + np.array([[1, 0]]*50)
))


x_min = min([ min(red[:,0]), min(blu[:, 0]) ])
x_max = max([ max(red[:,0]), max(blu[:, 0]) ])
y_min = min([ min(red[:, 1]), min(blu[:, 1]) ])
y_max = max([ max(red[:,0]), max(blu[:, 0]) ]) 


# Plot them
plt.scatter(red[:,0], red[:,1], color='red')
plt.scatter(blu[:,0], blu[:,1], color='blue')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()







""" Create and run neural net """

# Create a new graph
graph().as_default()

# Create training input placeholder
X = placeholder()

# Create placeholder for the training classes: remember, c is bit matrix for training
c = placeholder()

# Build a hidden layer(s)
W_hidden1 = variable(np.array([[ 7.05926791, -5.36370548], [-8.65650006 , 9.70797293]])) # variable(np.random.randn(2, 2))
b_hidden1 = variable([-1.09156871,  0.42934381]) #variable(np.random.randn(2))
p_hidden1 = sigmoid( add(matmul(X, W_hidden1), b_hidden1) )

W_hidden2= variable(np.random.randn(2, 2))
b_hidden2= variable(np.random.randn(2))
p_hidden2= sigmoid( add(matmul(p_hidden1, W_hidden2), b_hidden2) )

W_hidden3= variable(np.random.randn(2, 2))
b_hidden3= variable(np.random.randn(2))
p_hidden3= sigmoid( add(matmul(p_hidden2, W_hidden3), b_hidden3) )

W_hidden4= variable(np.random.randn(2, 2))
b_hidden4= variable(np.random.randn(2))
p_hidden4= sigmoid( add(matmul(p_hidden3, W_hidden4), b_hidden4) )

# Build the output layer
W_output = variable(np.random.randn(2, 2))
b_output = variable(np.random.randn(2))
p_output = softmax( add(matmul(p_hidden4, W_output), b_output) )

# Build cross-entropy loss
J = negative(axis_sum(axis_sum(elmul(c, log(p_output)), axis=1)))

# Build minimization op
minimization_op = grad_desecent_optimizer(mu = 0.02).minimize(J)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blu, red)),
    c:
        [[1, 0]] * len(blu)
        + [[0, 1]] * len(red)
    
}

# Create session
session1 = session()

# Perform n gradient descent steps
for step in range(n):
    J_value = session1.run(J, feed_dict)
    
    if step % (n / 10)== 0:
        print("Step:", step, " Loss:", J_value)
    if J_value < 0.1:
        break
    
    session1.run(minimization_op, feed_dict)

# Print output layer weights
print("Output layer weight matrix:\n", W_output_value)
b_output_value = session1.run(b_output)
print("Output layer bias:\n", b_output_value)


""" Visualize classification boundary """

xs = np.linspace(x_min, x_max)
ys = np.linspace(y_min, y_max)
pred_classes = []
for x in xs:
    for y in ys:
        pred_class = session1.run(p_output,
                              feed_dict={X: [[x, y]]})[0] # 0 for p(class = 0)
        pred_classes.append((x, y, pred_class.argmax()))
        
xs_p, ys_p = [], []
xs_n, ys_n = [], []
for x, y, c in pred_classes:
    if c == 0:
        xs_n.append(x)
        ys_n.append(y)
    else:
        xs_p.append(x)
        ys_p.append(y)
        
plt.plot(xs_p, ys_p, 'ro', xs_n, ys_n, 'bo')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()