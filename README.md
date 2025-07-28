# Designed Quadrature

Python implementation of Designed Quadrature for efficient numerical integration, based on Bansal & Keshavarzzadeh (2021).

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from designed_quadrature import designed_quadrature

# Generate quadrature rule
d = 3      # Dimension
p = 5      # Polynomial order
n_s = 50   # Number of nodes

XW, residual = designed_quadrature(d, p, n_s)

# Extract nodes and weights
nodes = XW[:, :d]
weights = XW[:, -1]

# Integrate a function
def f(x):
    return np.sum(x**2)  # Sum of squares

integral = sum(f(node) * weight for node, weight in zip(nodes, weights))
print(f"∫f(x)dx = {integral:.10f}")  # Should be 3.0 for d=3
```

## Running Tests

```bash
pytest test_designed_quadrature.py -v
```

## Example

```bash
python example.py
```

## Features

- Exactly integrates polynomials up to specified degree
- Outperforms Monte Carlo by orders of magnitude for smooth functions
- Supports Gaussian and uniform measures
- Works in arbitrary dimensions

## Algorithm

Uses regularized Newton's method to find quadrature nodes and weights that satisfy polynomial exactness conditions. Initialization via Latin Hypercube Sampling ensures good starting points.