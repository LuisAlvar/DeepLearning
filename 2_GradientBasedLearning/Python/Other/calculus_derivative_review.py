# pip install mplcursors
# pip install sympy
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import mplcursors

# Step 1: Define the function and calculate the derivative
x = sp.symbols('x')
f = x**2
f_prime = sp.diff(f, x)

# Convert the sympy expressions to numpy functions
f_lambdified = sp.lambdify(x, f, 'numpy')
f_prime_lambdified = sp.lambdify(x, f_prime, 'numpy')

# Step 2: Create x values
x_values = np.linspace(-10, 10, 400)

# Calculate y values for both the function and its derivative
y_values = f_lambdified(x_values)
y_prime_values = f_prime_lambdified(x_values)

# Plot the function and its derivative
fig, ax = plt.subplots(figsize=(10, 6))
line_f, = ax.plot(x_values, y_values, label='$f(x) = x^2$')
line_f_prime, = ax.plot(x_values, y_prime_values, label="$f'(x) = 2x$", linestyle='--')

# Add labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Function and Its Derivative')
ax.legend()
ax.grid(True)

# Add tooltips for the function
cursor_f = mplcursors.cursor(line_f, hover=True)
@cursor_f.connect("add")
def on_add_f(sel):
    x, y = sel.target
    sel.annotation.set_text(f'Function:\nx: {x:.2f}\ny: {y:.2f}\n Derivative: {f_prime_lambdified(x):.2f}')

# Add tooltips for the derivative
cursor_f_prime = mplcursors.cursor(line_f_prime, hover=True)
@cursor_f_prime.connect("add")
def on_add_f_prime(sel):
    x, y = sel.target
    sel.annotation.set_text(f'Derivative:\nx: {x:.2f}\ny: {y:.2f}')

# Show the plot
plt.show()
