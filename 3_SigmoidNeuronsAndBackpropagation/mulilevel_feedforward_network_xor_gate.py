import numpy as np

def neuron_w(input_count):
  weights = np.zeros(input_count + 1)
  for i in range(1, (input_count + 1)):
    weights[i] = np.random.uniform(-1.0,1.0)
  return weights

def show_learning():
  print('Current weigts:')
  for i, w in enumerate(n_w):
    print('neuron ', i, ': w0 = ', '%5.2f' % w[0], ', w1 = ', '%5.2f' % w[1], ', w2 = ', '%5.2f' % w[2])
  print('----------------------------------')

def forward_pass(x):
  global n_y
  n_y[0] = np.tanh(np.dot(n_w[0], x)) # Neuron N_0
  n_y[1] = np.tanh(np.dot(n_w[1], x)) # Neuron N_1
  n2_input = np.array([1.0, n_y[0], n_y[1]]) # 1.0 is bias
  z2 = np.dot(n_w[2], n2_input)
  n_y[2] = 1.0 / (1.0 + np.exp(-z2))

def backward_pass(y_truth):
  global n_error
  error_prime = -(y_truth - n_y[2])                 # Derivative of loss-func MSE'
  derivative = n_y[2] * (1.0 - n_y[2])              # Logistic derivative S'(x) = S(x)(1-S(x)) of Neuron N_2
  n_error[2] = error_prime * derivative             # Error for Neuron N_2 = MSE' * S(1-S)
  derivative = 1.0 - n_y[1]**2                      # tanh derivative tan'(x) = 1 - tanh^2(x) of Neuron N_1
  n_error[1] = n_error[2] * n_w[2][2] * derivative  # Error for Neuron N_1 = Error of N_2 * weight of N_22 * derivative
  derivative = 1.0 - n_y[0]**2                      # tanh derivative tanh'(x) = 1 - tanh^2(x) of Neuron N_0
  n_error[0] = n_error[2] * n_w[2][1] * derivative  # Error for Neuron N_0 = Error of N_2 * weight of N_21 * derivative

def adjust_weights(x):
  global n_w
  n_w[0] -= (LEARNING_RATE * x * n_error[0])
  n_w[1] -= (LEARNING_RATE * x * n_error[1])
  n2_inputs = np.array([1.0, n_y[0], n_y[1]])         # 1.0 is bias
  n_w[2] -= (LEARNING_RATE * n2_inputs * n_error[2])  # 


np.random.seed(3) # to make repeatable 
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3] # Used to randomize order

# Define training examples
x_train = [
  np.array([1.0,-1.0,-1.0]),
  np.array([1.0,-1.0,1.0]),
  np.array([1.0,1.0,-1.0]),
  np.array([1.0,1.0,1.0])
]

y_train = [0.0,1.0,1.0,0.0] # Output (ground truth)

n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]
n_y = [0, 0, 0]
n_error = [0, 0, 0]

# Network training loop
all_correct = False
while not all_correct: # Train until converged
  all_correct = True
  np.random.shuffle(index_list) # Randomize order
  for i in index_list: # Train on all examples
    forward_pass(x_train[i])
    backward_pass(y_train[i])
    adjust_weights(x_train[i])
    show_learning() # show updated weights
  for i in range(len(x_train)): # Check if converged
    forward_pass(x_train[i])
    print('x1 =', '%4.1f' % x_train[i][1], ', x2 =', '%4.1f' % x_train[i][2], ', y =', '%.4f' % n_y[2])
    if (((y_train[i] < 0.5) and (n_y[2] >= 0.5)) or ((y_train[i] >= 0.5) and (n_y[2] < 0.5))):
      all_correct = False

