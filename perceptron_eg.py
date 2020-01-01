import numpy as np
import matplotlib.pyplot as plt
import operations
import graph_run


"""Create and Plot some data"""
red = np.random.randn(50,2) - 2*np.ones((50,2)) # centered at -2, -2
blu = np.random.randn(50,2) + 2*np.ones((50,2)) # centered at 2, 2

plt.scatter(red[:, 0], red[:, 1], color = 'red')
plt.scatter(blu[:, 0], blu[:, 1], color = 'blue')


graph().as_default()

# class = 1 if wx + b > 0
# class = 2 if wx + b < 0

# prob(is blue) = prob(class = 1)

""" First eg - classify single x point """
x = placeholder()
w = variable([1, 1]) # weight vector for blue
b = variable(0)
p = sigmoid( add(matmul(w, x), b) )

session1 = session()
print(session1.run(p, {
    x: [3, 2]
}))

"""Second eg - classify a batch"""

graph().as_default()

X = placeholder()
W = variable([ # vector weight 1,1 for blue, -1,-1 for red
        [1, -1],
        [1, -1]
        ])

b = variable([0,0])

p = softmax(add(matmul(X, W), b))

session2 = session()

# training data is the union of blue and red points
output_probs = session2.run(p, {X: np.concatenate((blu, red))})
print(output_probs[0:2])

