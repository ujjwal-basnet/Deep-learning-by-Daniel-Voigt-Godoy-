import torch
import numpy as np 

torch.manual_seed(42)
import matplotlib.pyplot as plt
# Generate synthetic data
num_samples = 100
X = np.random.rand(num_samples, 1) * 10
y = 2 * X + 1 + np.random.randn(num_samples, 1) * 2

plt.plot(X, y,'.')