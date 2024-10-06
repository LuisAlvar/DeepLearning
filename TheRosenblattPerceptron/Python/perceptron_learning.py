import random 

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
def show_learning(w):
    print('w0 =','%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])

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
show_learning(w)

# Perceptron training loop.
all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list)
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w,x) # Perceptron function
        
        if y != p_out: # Update weights when wrong
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            show_learning(w) # Show updated weights