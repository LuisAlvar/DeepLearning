# -----------------------------------------------------------------------------------------
# Purpose: 
# We picked the three weightsm and we ended up with a perceptron that 
# behaves like a NAND gate if we view the inputs as Boolean values. 
#
# You can see that the z-value is far enough form zero ina ll cases, so you should be 
# able to adjust one of the weights by 0.1 in either direction and still end up with 
# the same behavior. 
#
# Logic:
# We find the upper and lower bound of vector that results in NAND gate behavior.
# Then we plot these result into a line graph, where we can visual all possible values that 
# will result in NAND gate behavior. 
#
# Run:
#     python3 all_possible_weights.py
# -----------------------------------------------------------------------------------------

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

w = [0.9,-0.6,-0.5]
scale_by = 0.1
data_set = [
    [1.0,-1.0,-1.0],
    [1.0,-1.0,1.0],
    [1.0,1.0,-1.0],
    [1.0,1.0,1.0]
]
original_results = [
 compute_output(w, data_set[0]),
 compute_output(w, data_set[1]),
 compute_output(w, data_set[2]),
 compute_output(w, data_set[3])
]

outer_index = 0 
inner_index = 0

new_upper_w = w
# finding the upper bound new_w
upper_result = []
while(1):
    new_upper_w = [i+scale_by for i in new_upper_w]
    upper_result = [
     compute_output(new_upper_w, data_set[0]),
     compute_output(new_upper_w, data_set[1]),
     compute_output(new_upper_w, data_set[2]),
     compute_output(new_upper_w, data_set[3])
    ]
    if(original_results != upper_result): 
        new_upper_w = [i-scale_by for i in new_upper_w]
        break
print(new_upper_w)

# finding the lower bound new_w
new_lower_w = w
lower_result = []
while(1):
    new_lower_w = [i-scale_by for i in new_lower_w]
    lower_result = [
     compute_output(new_lower_w, data_set[0]),
     compute_output(new_lower_w, data_set[1]),
     compute_output(new_lower_w, data_set[2]),
     compute_output(new_lower_w, data_set[3])
    ]
    if(original_results != lower_result): 
        new_lower_w = [i+scale_by for i in new_lower_w]
        break
print(new_lower_w)


import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,0.5,1])
y = np.array(new_upper_w)

plt.plot(x,y)

y1 = np.array(new_lower_w)

plt.plot(x, y1, '--')
plt.xlabel('x-axis')
plt.ylabel('weight-axis')

for i, (xi,yi) in enumerate(zip(x,y)):
    plt.annotate(f'({xi},{yi})', (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')

for i, (xi,yi) in enumerate(zip(x,y1)):
    plt.annotate(f'({xi},{yi})', (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')


plt.fill_between(x, y, y1, color='grey', alpha=0.5)
plt.grid(True)
plt.show()
