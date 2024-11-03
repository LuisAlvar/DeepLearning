import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y):
    return (x - 3)**2 - (y - 3)**2

# Create a grid of x and y values
x = np.linspace(-30,30, 400)
y = np.linspace(-30,30, 400)
x, y = np.meshgrid(x, y)

# Calculate the corresponding z values
z = f(x, y)

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x, y, z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot with a Valley and a Hill')

# Add a color bar
fig.colorbar(surf)

# Show the plot
plt.show()
