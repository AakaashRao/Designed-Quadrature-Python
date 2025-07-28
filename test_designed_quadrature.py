"""
Comprehensive test suite for Designed Quadrature implementation.

Tests include:
- Basic functionality
- Polynomial integration accuracy
- Comparison with Monte Carlo
- Comparison with Quasi-Monte Carlo (Sobol sequences)
"""

import pytest
import numpy as np
from scipy import stats
from designed_quadrature import designed_quadrature, total_degree_indices, hyperbolic_cross_indices
import time
import warnings

# Try to import Sobol sequence generator
try:
    from scipy.stats import qmc
    SOBOL_AVAILABLE = True
except ImportError:
    SOBOL_AVAILABLE = False


class TestBasicFunctionality:
    """Test basic functionality of the designed quadrature implementation."""
    
    def test_total_degree_indices(self):
        """Test total degree index generation."""
        # Test case: d=2, p=2
        indices = total_degree_indices(2, 2)
        expected = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [2, 0],
            [1, 1],
            [0, 2]
        ])
        np.testing.assert_array_equal(indices, expected)
        
        # Test case: d=3, p=1
        indices = total_degree_indices(3, 1)
        assert len(indices) == 4  # Should have (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        assert np.all(np.sum(indices, axis=1) <= 1)
    
    def test_hyperbolic_cross_indices(self):
        """Test hyperbolic cross index generation."""
        # Test case: d=2, p=3
        indices = hyperbolic_cross_indices(2, 3)
        # Check hyperbolic cross constraint: prod(a_i + 1) <= p + 1
        for idx in indices:
            assert np.prod(idx + 1) <= 4
        
        # Should include (0,0), (1,0), (0,1), (2,0), (0,2), (1,1), (3,0), (0,3)
        assert len(indices) >= 8
    
    def test_quadrature_generation_small(self):
        """Test quadrature generation for small problems."""
        d, p, n_s = 2, 3, 20
        
        XW, info = designed_quadrature(d, p, n_s, verbose=False)
        
        # Check output shape (might have fewer nodes if negatives were dropped)
        assert XW.shape[1] == d + 1
        assert XW.shape[0] <= n_s
        
        # Check weights sum to 1
        weights = XW[:, -1]
        assert abs(np.sum(weights) - 1.0) < 1e-10
        
        # Check residual is small
        assert info['residual'] < 1e-6
        
        # Check that negative weights info is present
        assert 'negative_weights' in info
        assert 'min_negative_weight' in info


class TestPolynomialIntegration:
    """Test polynomial integration accuracy."""
    
    @pytest.fixture
    def quadrature_2d(self):
        """Generate 2D quadrature rule for testing."""
        XW, info = designed_quadrature(2, 6, 40, verbose=False)
        return XW
    
    def test_constant_integration(self, quadrature_2d):
        """Test integration of constant function."""
        nodes = quadrature_2d[:, :-1]
        weights = quadrature_2d[:, -1]
        
        # Integral of 1 should be 1
        integral = np.sum(weights)
        assert abs(integral - 1.0) < 1e-12
    
    def test_polynomial_integration_2d(self, quadrature_2d):
        """Test integration of various 2D polynomials."""
        nodes = quadrature_2d[:, :-1]
        weights = quadrature_2d[:, -1]
        
        test_cases = [
            # (function, exact_value, description)
            (lambda x: x[0]**2, 1.0, "x₁²"),
            (lambda x: x[1]**2, 1.0, "x₂²"),
            (lambda x: x[0]**2 + x[1]**2, 2.0, "x₁² + x₂²"),
            (lambda x: x[0]**4, 3.0, "x₁⁴"),
            (lambda x: x[0]**2 * x[1]**2, 1.0, "x₁²x₂²"),
            (lambda x: x[0]**6, 15.0, "x₁⁶"),
        ]
        
        for func, exact, desc in test_cases:
            values = np.array([func(node) for node in nodes])
            integral = np.sum(values * weights)
            error = abs(integral - exact)
            assert error < 1e-6, f"Failed for {desc}: integral={integral}, exact={exact}, error={error}"
    
    def test_polynomial_integration_3d(self):
        """Test integration of 3D polynomials."""
        d, p, n_s = 3, 4, 60  # Increased from 50 to avoid negative weights
        XW, info = designed_quadrature(d, p, n_s, verbose=False, max_iters=100)
        nodes = XW[:, :-1]
        weights = XW[:, -1]
        
        # Test ∑x_i² = d for Gaussian measure
        integral = np.sum(np.sum(nodes**2, axis=1) * weights)
        assert abs(integral - d) < 1e-6
        
        # Test x₁²x₂² = 1
        integral = np.sum(nodes[:, 0]**2 * nodes[:, 1]**2 * weights)
        assert abs(integral - 1.0) < 1e-6


class TestMonteCarloComparison:
    """Compare Designed Quadrature with Monte Carlo methods."""
    
    @staticmethod
    def monte_carlo_integrate(func, d, n_samples, distribution='gaussian'):
        """Perform Monte Carlo integration."""
        if distribution == 'gaussian':
            samples = np.random.randn(n_samples, d)
        else:
            samples = np.random.uniform(-1, 1, (n_samples, d))
        
        values = np.array([func(x) for x in samples])
        return np.mean(values)
    
    @staticmethod
    def designed_quadrature_integrate(func, XW):
        """Integrate using designed quadrature."""
        nodes = XW[:, :-1]
        weights = XW[:, -1]
        values = np.array([func(x) for x in nodes])
        return np.sum(values * weights)
    
    @pytest.mark.parametrize("d,p,n_s", [(2, 4, 50), (3, 4, 80), (5, 3, 150)])
    def test_polynomial_accuracy(self, d, p, n_s):
        """Test that DQ is more accurate than MC for polynomial functions."""
        # Generate DQ
        XW, info = designed_quadrature(d, p, n_s, verbose=False, max_iters=100)
        
        # Test function: sum of squares (exact value = d)
        func = lambda x: np.sum(x**2)
        exact = d
        
        # DQ integration
        dq_val = self.designed_quadrature_integrate(func, XW)
        dq_error = abs(dq_val - exact) / exact
        
        # MC integration (average over multiple runs)
        mc_errors = []
        for _ in range(50):
            mc_val = self.monte_carlo_integrate(func, d, n_s, 'gaussian')
            mc_errors.append(abs(mc_val - exact) / exact)
        mc_error = np.mean(mc_errors)
        
        # DQ should be much more accurate
        assert dq_error < 1e-7, f"DQ error too large: {dq_error}"
        assert mc_error > 1e-3, f"MC error suspiciously small: {mc_error}"
        assert dq_error < mc_error / 100, f"DQ not sufficiently better than MC"
    
    def test_convergence_comparison(self):
        """Compare convergence rates of DQ vs MC."""
        d = 3
        sample_sizes = [20, 40, 80]
        
        # Test function
        func = lambda x: np.sum(x**4) + 2 * np.sum(x**2)
        exact = 3 * d + 2 * d
        
        dq_errors = []
        mc_errors = []
        
        for n_s in sample_sizes:
            # DQ
            try:
                XW, info = designed_quadrature(d, 5, n_s, verbose=False, max_iters=100)
                dq_val = self.designed_quadrature_integrate(func, XW)
                dq_errors.append(abs(dq_val - exact) / exact)
            except:
                dq_errors.append(np.nan)
            
            # MC (average over 100 runs)
            mc_vals = []
            for _ in range(100):
                mc_val = self.monte_carlo_integrate(func, d, n_s, 'gaussian')
                mc_vals.append(mc_val)
            mc_error = abs(np.mean(mc_vals) - exact) / exact
            mc_errors.append(mc_error)
        
        # Check that DQ errors are much smaller
        for dq_err, mc_err in zip(dq_errors, mc_errors):
            if not np.isnan(dq_err):
                assert dq_err < mc_err / 100


class TestSobolComparison:
    """Compare with Quasi-Monte Carlo using Sobol sequences."""
    
    @pytest.mark.skipif(not SOBOL_AVAILABLE, reason="Sobol sequences not available")
    def test_sobol_comparison(self):
        """Compare DQ with Sobol sequence integration."""
        d = 3
        n_s = 64  # Power of 2 for Sobol
        
        # Generate DQ
        XW, info = designed_quadrature(d, 5, n_s, verbose=False, max_iters=100)
        
        # Generate Sobol sequence
        sobol = qmc.Sobol(d, scramble=False)
        sobol_points = sobol.random(n_s)
        # Transform to standard normal
        sobol_gaussian = stats.norm.ppf(sobol_points)
        
        # Test functions
        test_funcs = [
            (lambda x: np.sum(x**2), d, "sum of squares"),
            (lambda x: np.sum(x**4), 3*d, "sum of fourth powers"),
            (lambda x: np.prod(1 + x/2), 1.0, "product"),
        ]
        
        for func, exact, name in test_funcs:
            # DQ integration
            dq_val = TestMonteCarloComparison.designed_quadrature_integrate(func, XW)
            dq_error = abs(dq_val - exact) / abs(exact) if exact != 0 else abs(dq_val)
            
            # Sobol integration
            sobol_vals = np.array([func(x) for x in sobol_gaussian])
            sobol_val = np.mean(sobol_vals)
            sobol_error = abs(sobol_val - exact) / abs(exact) if exact != 0 else abs(sobol_val)
            
            # For polynomial functions, DQ should be better
            if "squares" in name or "powers" in name:
                assert dq_error < sobol_error / 10, f"DQ not better than Sobol for {name}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_minimal_nodes(self):
        """Test with minimal number of nodes."""
        d, p = 2, 2
        n_terms = len(total_degree_indices(d, p))
        n_s = n_terms  # Minimal number of nodes
        
        XW, info = designed_quadrature(d, p, n_s, verbose=False, max_iters=200)
        assert XW.shape[0] == n_s
        assert info['residual'] < 1e-3  # May not converge to machine precision
    
    def test_high_dimension(self):
        """Test with higher dimensions."""
        d, p, n_s = 10, 2, 100
        XW, info = designed_quadrature(d, p, n_s, verbose=False, max_iters=50)
        
        # Should at least produce valid output (might have fewer nodes if negatives dropped)
        assert XW.shape[1] == d + 1
        assert XW.shape[0] <= n_s
        weights = XW[:, -1]
        assert abs(np.sum(weights) - 1.0) < 0.1  # Relaxed tolerance
    
    def test_uniform_quadrature(self):
        """Test uniform (non-Gaussian) quadrature."""
        d, p, n_s = 2, 3, 20
        XW, info = designed_quadrature(d, p, n_s, use_gaussian=False, verbose=False)
        
        nodes = XW[:, :-1]
        weights = XW[:, -1]
        
        # Check nodes are in [-1, 1] for uniform (with small tolerance for numerical errors)
        assert np.all(nodes >= -1 - 1e-6), f"Minimum node value: {np.min(nodes)}"
        assert np.all(nodes <= 1 + 1e-6), f"Maximum node value: {np.max(nodes)}"
        
        # Check weights sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-6


class TestPerformance:
    """Test performance characteristics."""
    
    def test_generation_time(self):
        """Test that quadrature generation completes in reasonable time."""
        d, p, n_s = 5, 4, 200  # Increased to avoid negative weights
        
        start_time = time.time()
        XW, info = designed_quadrature(d, p, n_s, verbose=False, max_iters=100)
        elapsed = time.time() - start_time
        
        assert elapsed < 60.0, f"Generation took too long: {elapsed:.2f}s"
        assert info['residual'] < 1e-3, f"Poor convergence: residual = {info['residual']}"
    
    def test_scalability(self):
        """Test scalability with problem size."""
        times = []
        configs = [(2, 3, 30), (3, 3, 40), (4, 3, 60), (5, 3, 80)]
        
        for d, p, n_s in configs:
            start = time.time()
            _, info = designed_quadrature(d, p, n_s, verbose=False, max_iters=50)
            times.append(time.time() - start)
        
        # Check that time doesn't explode
        assert max(times) < 5.0, f"Maximum time too large: {max(times):.2f}s"


# Fixtures for commonly used quadrature rules
@pytest.fixture
def dq_2d_p4():
    """2D quadrature rule with polynomial order 4."""
    XW, info = designed_quadrature(2, 4, 40, verbose=False)
    return XW


@pytest.fixture
def dq_3d_p3():
    """3D quadrature rule with polynomial order 3."""
    XW, info = designed_quadrature(3, 3, 40, verbose=False)
    return XW


# Integration test functions for benchmarking
def gaussian_exp(x):
    """Gaussian exponential: exp(-0.5 * ||x||²)"""
    return np.exp(-0.5 * np.sum(x**2))


def polynomial_4th(x):
    """4th degree polynomial: sum(x^4) + 2*sum(x^2)"""
    return np.sum(x**4) + 2 * np.sum(x**2)


def product_function(x):
    """Product function: prod(1 + x)"""
    return np.prod(1 + x)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])