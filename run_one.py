import operations
import graph_run

# Create a new graph
graph().as_default()
session1 = session()
# Create variables
A = variable([[1, 0], [0, -1]])
b = variable([1, 1])

# Create placeholder
x = placeholder()

# Create hidden node y
y = matmul(A, x)

# Create output node z
z = add(y, b)


output = session1.run(z, {x: [1, 2] }) # out = z = Ax + b
print(output)
