#---------------------------------------------------------------------------------
# Purpose: 
# When having x1 and x2 as input variables to a f(x1,x2).
# We can observe how manually changing the weight vector changings the 3d plane.
# This reminds me of linear algebar, be in 3D, where you may have a:
# reflection, slide, growth, or a shrink against a set of 4 points connected together. 
# However, in this case, the w_0 causes the 3d plane to move up and down - independently. 
# 
# A change in w_1, will cause the center of the surface to tilt parallel 
# to x2. Image the x2 axis in the center of the surface, as it acts a seesaw, 
# where the surface can tilt back and front. 
# 
# Whereas, a change in w_2, will cause the center of the surface to tilt parallel 
# to x1. Image the x1 axis in the center of the sure, as it acts a seesaw, 
# where the surface can tilt back and front.
#
#---------------------------------------------------------------------------------
# pip install ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Scale

def update_plot(val):
    w0 = bias_weight.get()
    w1 = weight_1.get()
    w2 = weight_2.get()

    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    x1, x2 = np.meshgrid(x1, x2)
    z = w0 + w1 * x1 + w2 * x2
    
    ax.clear()
    ax.plot_surface(x1, x2, z, cmap='coolwarm', alpha=0.7)

    ax.set_title('Interactive Plot with Sliders')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$z$')
    ax.set_zlim(-1.5,1.5)
    ax.grid(True)    
    canvas.draw()

# Create the main window
root = tk.Tk()
root.title("Interactive Surface with Weight Vector Sliders")

# Create a matplotlib figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()


# Create the w_0 slider
bias_weight = Scale(root, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.1, label='w0', command=update_plot)
bias_weight.pack()

# Create the w_1 slider
weight_1 = Scale(root, from_=0.0, to=-1.0, orient=tk.HORIZONTAL, resolution=0.1, label='w1', command=update_plot)
weight_1.pack()

# Create the w_2 slider
weight_2 = Scale(root, from_=0.0, to=-1.0, orient=tk.HORIZONTAL, resolution=0.1, label='w2', command=update_plot)
weight_2.pack()

# Initialize the plot
update_plot(None)

# Start the Tkinter event loop
root.mainloop()
