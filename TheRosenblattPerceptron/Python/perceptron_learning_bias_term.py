# -----------------------------------------------------------------------------------------
# Purpose: 
#   We remove the bias terms w_0 and x_0 within the weight vector and input vector.
#   By removing it we understand that the bias terms acts as an adjustment threshold because we compute it within the summation
#   Whereas, having a explicit fixed threshold. We substract it from the summation at the end.  
# Run:
#     python3 perceptron_learning_bias_term.py
# -----------------------------------------------------------------------------------------
import random 
import matplotlib.pyplot as plt

# Define variables needed for plotting.
color_list = ['r-','m-','y-','c-','b-','g-']
color_index = 0
weight_iteration = 0
storage = {}
theta = -0.4              # an explicit threshold instead of a bias terms (i.e., an adjustment threshold) 
ONE_TO_NEG_N_POWER = 1e-5

# Code Snippet 1-1 Python Implementation of Perceptron Function 
def compute_output(w,x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i]     # Compute sum of weighted inputs
    # Apply sign function
    # Revised version: we now have a fix threshold value  
    if z - theta < 0:
        return -1
    else:
        return 1

# Code Snippet 1-2 Initialization Code for Our Perceptron Learning Example
# Code Snippet 1-5 Extended Version Of Initialization Code with Function to Plot the Output
def save_learning(w):
    global weight_iteration
    print('theta =','%5.2f' % theta, ', w1 =', '%5.2f' % w[0], ', w2 =', '%5.2f' % w[1])
    x = [-2.0, 2.0]
    if abs(w[1]) < ONE_TO_NEG_N_POWER:
        y = [
          -w[0]/(ONE_TO_NEG_N_POWER)*(-2.0)+(theta/(ONE_TO_NEG_N_POWER)),
          -w[0]/(ONE_TO_NEG_N_POWER)*(2.0)+(theta/(ONE_TO_NEG_N_POWER))
        ]
    else:
        y = [
          -w[0]/w[1]*(-2.0)+(theta/w[1]),
          -w[0]/w[1]*(2.0)+(theta/w[1])
        ]
    storage[weight_iteration] = y
    weight_iteration += 1

def show_learning():
    global color_index
    curr_index = 1
    plt.plot([1.0],[1.0],'b_',markersize=12)
    plt.plot([-1.0,1.0,-1.0],[1.0,-1.0,-1.0], 'r+', markersize=12)
    plt.axis([-2,2,-2,2])
    plt.xlabel('x1')
    plt.ylabel('x2')
    x = [-2.0,2.0]
    for obj in storage:
        str_line_label = "w_line " + str(curr_index)
        plt.plot(x,storage.get(obj), color_list[color_index], label=str_line_label)
        color_index += 1
        curr_index += 1
    plt.legend()
    plt.show()

# Define variables needed to control training process.
random.seed(7) # To make repeatable
LEARNING_RATE = 0.1
index_list = [0,1,2,3] # Used to randomize order

# Define training examples.
x_train = [
    (-1.0,-1.0),
    (-1.0,1.0),
    (1.0,-1.0),
    (1.0,1.0)
] # Inputs
y_train = [1.0,1.0,1.0,-1.0] # Output (ground truth)

# Define perceptron weights. 
w = [-0.6, 0.25] # Initialize to some "random" numbers
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
            for j in range(len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            save_learning(w)


show_learning()
