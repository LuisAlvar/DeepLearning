# -----------------------------------------------------------------------------------------
# Purpose: 
#
# Run:
#     python3 perceptron_learning_in_detail.py
# -----------------------------------------------------------------------------------------
import random 
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define variables needed for plotting.
weight_iteration = 0
storage = {}
ONE_TO_NEG_N_POWER = 1e-5
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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

# Code Snippet 1-2 Initialization Code for Our Perceptron Learning Example
# Code Snippet 1-5 Extended Version Of Initialization Code with Function to Plot the Output
def save_learning(w):
    global weight_iteration
    global storage
    print('w0 =','%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])
    storage[weight_iteration] = copy.deepcopy(w)
    weight_iteration += 1

def update(frame):
    global ax
    global storage
    ax.clear()
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    x1, x2 = np.meshgrid(x1, x2)
    cur_weight = storage.get(frame)
    z = cur_weight[0] + cur_weight[1] * x1 + cur_weight[2] * x2
    ax.plot_surface(x1, x2, z, cmap='coolwarm', alpha=0.7)
    final_weight = storage.get(len(storage)-1)
    z = final_weight[0] + final_weight[1] * x1 + final_weight[2] * x2
    ax.plot_surface(x1, x2, z, cmap='coolwarm', alpha=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$z$')
    ax.set_zlim(-1.5,1.5)
    ax.set_title('Change in Final Weight Vector For NAND Gate')

def show_learning():
    global fig
    ani = FuncAnimation(fig, update, frames=np.arange(0, len(storage)), interval=1000)
    plt.show()


# Define variables needed to control training process.
random.seed(7) # To make repeatable
LEARNING_RATE = 0.1
index_list = [0,1,2,3] # Used to randomize order

# Define training examples.
x_train = [
    (1.0,-1.0,-1.0),
    (1.0,-1.0,1.0),
    (1.0,1.0,-1.0),
    (1.0,1.0,1.0)
] # Inputs
y_train = [1.0,1.0,1.0,-1.0] # Output (ground truth)

# Define perceptron weights. 
w = [0.2, -0.6, 0.25] # Initialize to some "random" numbers

# Print initial weights
save_learning(w)

# Perceptron training loop.
all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list)
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w,x) # Perceptron function
        if p_out != y: # Update weights when wrong
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            save_learning(w) # Show updated weights

# show plotted graph
show_learning()