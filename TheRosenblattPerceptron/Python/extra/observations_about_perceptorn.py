import matplotlib.pyplot as plt
import numpy as np

# Data
training_examples = 100
inputs = np.random.randn(training_examples)  # Example input values
desired_outputs = np.random.choice([0, 1], size=training_examples)  # Binary desired outputs

# Bias weight adjustments
incorrect_predictions = np.random.choice([0, 1], size=training_examples)  # Binary correctness
bias_adjustments = incorrect_predictions * (desired_outputs - 0.5)  # Simplified adjustment calculation

# Feature weight adjustments
weight_adjustments = inputs * incorrect_predictions  # Proportional to input values

# Plotting bias weight adjustments
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(range(training_examples), bias_adjustments, color='blue')
plt.axhline(0, color='black',linewidth=0.5)
plt.title('Bias Weight Adjustments')
plt.xlabel('Training Example')
plt.ylabel('Adjustment Magnitude')

# Plotting feature weight adjustments
plt.subplot(1, 2, 2)
plt.bar(range(training_examples), weight_adjustments, color='orange')
plt.axhline(0, color='black',linewidth=0.5)
plt.title('Weight Adjustments Proportional to Input')
plt.xlabel('Training Example')
plt.ylabel('Adjustment Magnitude')

plt.tight_layout()
plt.show()
