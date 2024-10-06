# -----------------------------------------------------------------------------------------
# Purpose: 
#
# Run:
#     python3 multi_perceptron_xor.py
# -----------------------------------------------------------------------------------------
import random 
import matplotlib.pyplot as plt

# Define variables needed for plotting.
storage = {}
N_LAYER_PERCEPTRON = 2

# Define perceptron weights for P0, P1, and P2
w = [
    (0.9,-0.6,-0.5), #P0 
    (0.2,0.6,0.6),   #P1
    (-0.9,0.6,0.6)   #P2
] # Set of weight vector will yield to XOR gate behavior

# Code Snippet 1-1 Python Implementation of Perceptron Function 
# First element in vector x must be 1.
# Length of w and x must be n+1 for neuron with n inputs.
def compute_output(w,x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i]     # Compute sum of weighted inputs
    # Apply sign function
    if z < 0:
        return -1
    else:
        return 1

def hidden_layer(x):
    result = []
    for p_node in range(N_LAYER_PERCEPTRON):
        result.append(compute_output(w[p_node], x))
    return result

def multi_perceptron(x):
    last_output_input = [1]
    last_output_input = last_output_input + hidden_layer(x)
    return compute_output(w[N_LAYER_PERCEPTRON], last_output_input)

# Define training examples.
x_train = [
    (1.0,-1.0,-1.0),
    (1.0,1.0,-1.0),
    (1.0,-1.0,1.0),
    (1.0,1.0,1.0)
] # Inputs


result = multi_perceptron(x_train[0])
print(x_train[0], end="|")
print(result)

result = multi_perceptron(x_train[1])
print(x_train[1], end="|")
print(result)


result = multi_perceptron(x_train[2])
print(x_train[2], end="|")
print(result)


result = multi_perceptron(x_train[3])
print(x_train[3], end="|")
print(result)