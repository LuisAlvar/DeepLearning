import numpy as np
import matplotlib.pyplot as plt

# Define the function, where 
def f(w0,w1):
    return 0.3*w1 + 1.0*w0

# Create a grid of x and y values
w0 = np.linspace(-2, 2, 100)
w1 = np.linspace(-2, 2, 100)
w0, w1 = np.meshgrid(w0, w1)

# Calculate the corresponding z values
z = f(w0, w1)

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(w0, w1, z, cmap='coolwarm')

# Add labels and title
ax.set_xlabel('$w_0$')
ax.set_ylabel('$w_1$')
ax.set_zlabel('x_0 = 1.0, x_1 = 0.3')

# Add a color bar
fig.colorbar(surf)

# Show the plot
plt.show()
