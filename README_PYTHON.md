# Designed Quadrature - Python Implementation

This is a Python implementation of the Designed Quadrature method described in:

Bansal, P. and Keshavarzzadeh, V. (2021) "Designed Quadrature to Approximate Integrals in Maximum Simulated Likelihood Estimation"

## Overview

Designed Quadrature (DQ) is a method for numerical integration that exactly integrates polynomials up to a specified degree. For polynomial integrands, it significantly outperforms Monte Carlo methods with the same number of function evaluations.

## Features

- Complete Python implementation matching the MATLAB algorithm
- Support for both Gaussian and uniform measures
- Total degree and hyperbolic cross index sets
- Comprehensive test suite comparing to Monte Carlo and Sobol sequences
- Warnings for small negative weights (numerical precision at constraint boundaries)
- Automatic retry mechanism in tests for robustness

## Installation

```bash
pip install numpy scipy
```

## Usage

```python
from designed_quadrature import designed_quadrature

# Generate quadrature rule
d = 3      # dimension
p = 5      # polynomial order
n_s = 50   # number of nodes

XW, residual = designed_quadrature(d, p, n_s)

# Extract nodes and weights
nodes = XW[:, :-1]  # shape: (n_s, d)
weights = XW[:, -1]  # shape: (n_s,)

# Integrate a function
def f(x):
    return x[0]**2 + x[1]**2  # example polynomial

integral = sum(f(nodes[i]) * weights[i] for i in range(n_s))
```

## Running Tests

```bash
python -m pytest test_designed_quadrature.py -v
```

## Implementation Notes

1. **Negative Weights**: The implementation allows up to 5% of weights to be slightly negative (magnitude < 1/n_s) due to numerical precision at constraint boundaries. This matches the behavior observed in the MATLAB implementation.

2. **Initialization**: Uses Latin Hypercube Sampling from scipy.stats.qmc for better initial point distribution.

3. **Constraints**: Implements soft constraints using Heaviside-like functions to match MATLAB's behavior.

4. **Regularization**: Uses adaptive Tikhonov regularization with parameters that adjust based on residual magnitude.

5. **Retry Logic**: The `designed_quadrature` function automatically retries up to 3 times with different random initializations if the negative weight constraints are violated. This makes the function more robust without requiring user intervention.

## Test Results

The test suite demonstrates that Designed Quadrature:
- Exactly integrates polynomials up to the specified degree (errors < 1e-10)
- Outperforms Monte Carlo by orders of magnitude for polynomial functions
- Provides more accurate results than Quasi-Monte Carlo (Sobol) for polynomial integrands

## Files

- `designed_quadrature.py`: Main implementation (consolidated single file)
- `test_designed_quadrature.py`: Comprehensive test suite
- `example_usage.py`: Examples demonstrating usage and comparison with Monte Carlo

## Key Functions

- `designed_quadrature()`: Main function to generate quadrature rules
- `total_degree_indices()`: Generate polynomial indices for total degree
- `hyperbolic_cross_indices()`: Generate polynomial indices for hyperbolic cross
- `pol_mul_g()`: Hermite polynomials for Gaussian quadrature
- `pol_mul_jacobi()`: Jacobi polynomials for uniform quadrature

## References

1. Bansal, P. and Keshavarzzadeh, V. (2021) "Designed Quadrature to Approximate Integrals in Maximum Simulated Likelihood Estimation"
2. Original MATLAB implementation available in the `matlab/` directory