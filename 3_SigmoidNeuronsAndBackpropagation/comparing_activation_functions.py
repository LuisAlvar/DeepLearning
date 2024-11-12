import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def step_function(x, threshold=0.5):
    return np.where(x >= threshold, 1, 0)

def tanh_function(x):
    return np.tanh(x)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Generate input values
x = np.linspace(-3, 3, 400)

# Calculate outputs for each function
step_values = step_function(x)
tanh_values = tanh_function(x)
sigmoid_values = sigmoid_function(x)

# Plotting the activation functions
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(x, step_values, label='Step Function')
plt.axhline(0.5, color='gray', linestyle='--')
plt.axvline(0.5, color='gray', linestyle='--')
plt.title('Step Function (Threshold = 0.5)')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, tanh_values, label='Tanh Function', color='orange')
plt.title('Tanh Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, sigmoid_values, label='Sigmoid Function', color='green')
plt.title('Sigmoid Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualizing neurons in sequence
def neuron_chain_output(activation_func, inputs):
    outputs = [inputs[0]]
    for i in range(1, len(inputs)):
        outputs.append(activation_func(outputs[-1] + inputs[i]))
    return np.array(outputs)

# Sequential inputs (close to threshold for illustration)
inputs = np.linspace(0.4, 0.6, 10)

# Outputs of neurons in sequence
step_outputs = neuron_chain_output(step_function, inputs)
tanh_outputs = neuron_chain_output(tanh_function, inputs)
sigmoid_outputs = neuron_chain_output(sigmoid_function, inputs)

# Plotting the sequential neuron outputs
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(inputs, step_outputs, label='Step Function Outputs')
plt.title('Sequential Neurons (Step Function)')
plt.xlabel('Input Index')
plt.ylabel('Output')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(inputs, tanh_outputs, label='Tanh Function Outputs', color='orange')
plt.title('Sequential Neurons (Tanh Function)')
plt.xlabel('Input Index')
plt.ylabel('Output')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(inputs, sigmoid_outputs, label='Sigmoid Function Outputs', color='green')
plt.title('Sequential Neurons (Sigmoid Function)')
plt.xlabel('Input Index')
plt.ylabel('Output')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
