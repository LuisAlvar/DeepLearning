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
