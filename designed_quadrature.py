"""
Designed Quadrature for Maximum Simulated Likelihood Estimation

Python implementation of the Designed Quadrature method described in:
Bansal, P. and Keshavarzzadeh, V. (2021) "Designed Quadrature to Approximate 
Integrals in Maximum Simulated Likelihood Estimation"

This module provides functions to generate quadrature rules that exactly integrate
polynomials up to a specified degree and outperform Monte Carlo methods.
"""

import numpy as np
from scipy.stats import norm, qmc
from scipy.special import eval_hermite, eval_jacobi
from scipy import linalg
from itertools import combinations_with_replacement, chain


def total_degree_indices(d, k):
    """
    Computes all multi-indices of degree k or less in d dimensions.
    
    Parameters:
    -----------
    d : int
        Number of dimensions
    k : int
        Maximum polynomial degree
        
    Returns:
    --------
    a : ndarray
        Array of shape (n, d) where each row is a multi-index
        
    Example:
    --------
    k = 2, d = 2 -> [[0, 0],
                     [1, 0], 
                     [0, 1],
                     [2, 0],
                     [1, 1],
                     [0, 2]]
    """
    # Generate all combinations with replacement for each degree from 0 to k
    raw = chain.from_iterable(
        combinations_with_replacement(range(d), s)
        for s in range(k+1)
    )
    
    # Convert each tuple of repeated dims into a count-vector
    indices = []
    for t in raw:
        count_vec = np.bincount(t, minlength=d)
        indices.append(count_vec)
    
    return np.array(indices, dtype=int)


def hyperbolic_cross_indices(d, p):
    """
    Generate hyperbolic cross indices for dimension d and order p.
    
    Parameters:
    -----------
    d : int
        Number of dimensions
    p : int
        Maximum polynomial degree
        
    Returns:
    --------
    indices : ndarray
        Array of multi-indices satisfying the hyperbolic cross constraint
    """
    # Start with empty list
    indices = []
    
    # Helper function to generate all indices recursively
    def generate_indices(current_index, dim, remaining_product):
        if dim == d:
            if np.prod(np.array(current_index) + 1) <= p + 1:
                indices.append(current_index.copy())
            return
        
        # Try all possible values for current dimension
        for val in range(p + 1):
            if val == 0 or remaining_product * (val + 1) <= p + 1:
                current_index[dim] = val
                generate_indices(current_index, dim + 1, 
                               remaining_product * (val + 1) if val > 0 else remaining_product)
    
    # Start recursion
    current = np.zeros(d, dtype=int)
    generate_indices(current, 0, 1)
    
    return np.array(indices, dtype=int)




def quad_int_mul_u_sens(a, b, use_gaussian=True):
    """
    Compute quadrature integral for multivariate polynomial.
    
    Parameters:
    -----------
    a : ndarray
        Multi-index specifying polynomial degrees
    b : ndarray
        Points at which to evaluate (n_s x d array)
    use_gaussian : bool
        If True, use monomials for Gaussian quadrature
        If False, use monomials for uniform quadrature
        
    Returns:
    --------
    c : ndarray
        Product of monomials evaluated at points
    cdm : ndarray
        Derivatives with respect to each dimension
    """
    n_s, d = b.shape
    cmatrix = []
    cdmatrix = []
    
    # Evaluate monomials for each dimension
    for i in range(d):
        xi = b[:, i]
        # Use monomials x^a[i]
        c_i = xi**a[i]
        if a[i] > 0:
            cd_i = a[i] * xi**(a[i]-1)
        else:
            cd_i = np.zeros_like(xi)
        cmatrix.append(c_i)
        cdmatrix.append(cd_i)
    
    cmatrix = np.column_stack(cmatrix)
    cdmatrix = np.column_stack(cdmatrix)
    
    # Compute product over all dimensions
    c = np.ones(n_s)
    for i in range(d):
        c *= cmatrix[:, i]
    
    # Compute derivatives
    cdm = np.zeros((n_s, d))
    for r in range(d):
        ind = list(range(d))
        ind.remove(r)
        cd = np.ones(n_s)
        for i in range(d - 1):
            cd *= cmatrix[:, ind[i]]
        cd *= cdmatrix[:, r]
        cdm[:, r] = cd
    
    return c, cdm


def gaussian_moment(a):
    """
    Compute the exact integral of x^a with standard normal distribution.
    
    For multivariate Gaussian N(0,I), the integral of x1^a1 * x2^a2 * ... * xd^ad
    equals the product of univariate integrals.
    
    For univariate standard normal:
    - If a is odd: integral = 0
    - If a is even (a = 2k): integral = (2k-1)!! = 1*3*5*...*(2k-1)
    
    Parameters:
    -----------
    a : ndarray
        Multi-index of polynomial degrees
        
    Returns:
    --------
    moment : float
        The exact integral value
    """
    moment = 1.0
    for ai in a:
        if ai % 2 == 1:  # Odd power
            return 0.0
        else:  # Even power
            k = ai // 2
            # Compute (2k-1)!! = 1*3*5*...*(2k-1)
            double_factorial = 1.0
            for j in range(1, k + 1):
                double_factorial *= (2*j - 1)
            moment *= double_factorial
    return moment


def uniform_moment(a):
    """
    Compute the exact integral of x^a over [-1,1]^d with uniform measure.
    
    For uniform distribution on [-1,1]:
    - If a is odd: integral = 0
    - If a is even: integral = 2/(a+1)
    
    Parameters:
    -----------
    a : ndarray
        Multi-index of polynomial degrees
        
    Returns:
    --------
    moment : float
        The exact integral value
    """
    # For uniform on [-1,1]^d with density 1/2^d
    moment = 1.0
    for ai in a:
        if ai % 2 == 1:  # Odd power
            return 0.0
        else:  # Even power
            moment *= 2.0 / (ai + 1)
    # Account for the density 1/2^d
    moment /= 2**len(a)
    return moment


def cons_computation(b, parm, use_gaussian=True):
    """
    Compute constraint residual and derivative for position variables.
    
    Parameters:
    -----------
    b : ndarray
        Position variables
    parm : float
        Regularization parameter
    use_gaussian : bool
        If True, use Gaussian limits; if False, use uniform limits
        
    Returns:
    --------
    res : ndarray
        Residual values
    resd : ndarray
        Derivative values
    """
    b = np.asarray(b).reshape(-1)
    
    # Set limit based on distribution type
    lim = 8.0 if use_gaussian else 1.0
    
    # Heaviside function: 0 if |b| <= lim, 1 if |b| > lim
    heaviside = (np.abs(b) > lim).astype(float)
    
    # Constraint: (|b| - lim)^2 when |b| > lim, 0 otherwise
    res = heaviside * (np.abs(b) - lim)**2 * parm
    
    # Derivative
    resd = heaviside * 2 * (np.abs(b) - lim) * np.sign(b) * parm
    
    return res, resd


def cons_w_computation(w, parm):
    """
    Compute constraint residual and derivative for weight variables.
    Enforces w > delta using a smooth penalty.
    
    Parameters:
    -----------
    w : ndarray
        Weight variables
    parm : float
        Regularization parameter
        
    Returns:
    --------
    resw : ndarray
        Residual values
    resdw : ndarray
        Derivative values
    """
    w = np.asarray(w).reshape(-1)
    
    # Following MATLAB: enforce w > delta
    delta = 1e-6
    x = w - delta
    
    # Smooth penalty that is 0 when w >= delta
    # For w < delta, use quadratic penalty
    mask = x < 0
    resw = np.zeros_like(w)
    resdw = np.zeros_like(w)
    
    if np.any(mask):
        resw[mask] = x[mask]**2 * parm
        resdw[mask] = 2 * x[mask] * parm
    
    return resw, resdw


def auto_tikhonov(delta, use_gaussian=True):
    """
    Compute Tikhonov regularization parameter automatically.
    
    Parameters:
    -----------
    delta : float
        Current residual norm
    use_gaussian : bool
        If True, use parameters for Gaussian case
        
    Returns:
    --------
    dtikh : float
        Tikhonov regularization parameter
    """
    if use_gaussian:
        if delta > 100:
            return 250
        elif delta > 10:
            return 150
        elif delta > 1:
            return delta * 10
        elif delta > 0.1:
            return delta * 5
        elif delta > 0.01:
            return delta * 2
        else:
            return delta * 1
    else:
        if delta > 100:
            return 100
        elif delta > 10:
            return 10
        elif delta > 1:
            return 5
        elif delta > 0.1:
            return 1
        elif delta > 0.01:
            return 0.1
        else:
            return 0.01


def lhs(n_samples, d):
    """Generate Latin Hypercube Sample using scipy."""
    sampler = qmc.LatinHypercube(d=d)
    return sampler.random(n=n_samples)


def designed_quadrature(d, p, n_s, use_total_degree=True, use_gaussian=True, 
                       max_iters=5000, tol=1e-8, verbose=False):
    """
    Generate designed quadrature rules.
    
    Parameters:
    -----------
    d : int
        Dimension
    p : int
        Polynomial order  
    n_s : int
        Number of sample points
    use_total_degree : bool
        If True, use total degree indices. If False, use hyperbolic cross
    use_gaussian : bool
        If True, generate Gaussian quadrature. If False, generate uniform
    max_iters : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
        
    Returns:
    --------
    XW : ndarray
        Array with nodes (first d columns) and weights (last column).
        Negative weights are dropped and weights are renormalized.
    info : dict
        Dictionary containing:
        - 'residual': Final residual norm
        - 'negative_weights': Fraction of weights that were negative and dropped
        - 'min_negative_weight': Minimum negative weight before dropping (or 0 if none)
    """
    # Try up to 3 times with different random initializations
    best_result = None
    best_neg_fraction = 1.0
    
    for attempt in range(3):
        XW, delta, n_neg, min_neg = _designed_quadrature_single_attempt(
            d, p, n_s, use_total_degree, use_gaussian, max_iters, tol, verbose
        )
        
        neg_fraction = n_neg / n_s
        
        # Keep the result with the fewest negative weights
        if neg_fraction < best_neg_fraction:
            best_result = (XW, delta, n_neg, min_neg)
            best_neg_fraction = neg_fraction
        
        # If we get a perfect result (no negative weights), return immediately
        if n_neg == 0:
            break
            
        if verbose and n_neg > 0:
            print(f"Attempt {attempt + 1}: {n_neg} negative weights ({neg_fraction:.1%})")
    
    XW, delta, n_neg, min_neg = best_result
    
    # Drop negative weights and renormalize
    if n_neg > 0:
        weights = XW[:, -1]
        positive_mask = weights > 0
        XW = XW[positive_mask]
        # Renormalize weights to sum to 1
        XW[:, -1] = XW[:, -1] / np.sum(XW[:, -1])
        
        if verbose:
            print(f"Dropped {n_neg} negative weights and renormalized remaining {np.sum(positive_mask)} weights")
    
    info = {
        'residual': delta,
        'negative_weights': n_neg / n_s,
        'min_negative_weight': min_neg if n_neg > 0 else 0.0
    }
    
    return XW, info


def _designed_quadrature_single_attempt(d, p, n_s, use_total_degree=True, use_gaussian=True, 
                                       max_iters=5000, tol=1e-8, verbose=False):
    """
    Single attempt at generating designed quadrature rules.
    
    Returns:
    --------
    XW : ndarray
        Array with nodes and weights (including any negative weights)
    delta : float
        Final residual norm
    n_neg : int
        Number of negative weights
    min_neg : float
        Minimum negative weight value (or 0 if none)
    """
    # Generate index set
    if use_total_degree:
        aind = total_degree_indices(d, p)
    else:
        aind = hyperbolic_cross_indices(d, p)
    
    n_terms = aind.shape[0]
    
    # Initialize nodes and weights
    if use_gaussian:
        # Latin hypercube design with normal transformation
        b = norm.ppf(lhs(n_s, d))
        # Compute norm of each row (each sample point)
        rad = np.linalg.norm(b, axis=1)
        cent = 1  # Centralize for high dimensions
        w = np.exp(-(rad - cent * np.mean(rad))**2 / 2)
        w = w / np.sum(w)  # Only normalize to sum=1
    else:
        # Latin hypercube design for uniform case
        b = lhs(n_s, d) * 2 - 1
        w = (1.0 / n_s) * np.ones(n_s)  # Normalize to sum=1
    
    # Initialize solution vector
    # Store b column-wise (each column of b is stored consecutively)
    xnew = []
    for i in range(d):
        xnew.extend(b[:, i])
    xnew.extend(w)
    xnew = np.array(xnew)
    
    # Right hand side - set target moments for each polynomial
    RHS = np.zeros(n_terms + n_s * (d + 1))
    for i in range(n_terms):
        if use_gaussian:
            RHS[i] = gaussian_moment(aind[i])
        else:
            RHS[i] = uniform_moment(aind[i])
    
    delta = 1
    iters = []
    
    for count in range(max_iters):
        if delta <= tol:
            break
        xold = xnew.copy()
        
        # Extract current nodes and weights
        b = np.zeros((n_s, d))
        for i in range(d):
            b[:, i] = xold[i*n_s:(i+1)*n_s]
        w = xold[d*n_s:(d+1)*n_s]
        
        # Compute residual and Jacobian
        R = np.zeros(n_terms)
        J = np.zeros((n_terms, (d+1)*n_s))
        
        for i in range(n_terms):
            a = aind[i, :]
            Ri, Rdij = quad_int_mul_u_sens(a, b, use_gaussian)
            R[i] = w @ Ri
            
            for j in range(d):
                Rsens = Rdij[:, j]
                J[i, j*n_s:(j+1)*n_s] = w * Rsens
            J[i, d*n_s:(d+1)*n_s] = Ri
        
        # Compute constraint residuals and Jacobians
        # Compute target moments for regularization scaling
        parm_target = np.zeros(n_terms)
        for i in range(n_terms):
            if use_gaussian:
                parm_target[i] = gaussian_moment(aind[i])
            else:
                parm_target[i] = uniform_moment(aind[i])
        parm = 1 / np.linalg.norm(R - parm_target)
        parm = max(parm, 1000)  # Increased to enforce constraints more strongly
        
        Rcons = []
        Jcons = []
        
        for i in range(d):
            res, resd = cons_computation(b[:, i], parm, use_gaussian)
            Rcons.extend(res)
            Jcons.extend(resd)
        
        resw, resdw = cons_w_computation(w, parm)
        Rcons.extend(resw)
        Jcons.extend(resdw)
        
        Rcons = np.array(Rcons)
        Jcons_diag = np.diag(Jcons)
        
        # Augmented system
        R_aug = np.concatenate([R, Rcons]) - RHS
        J_aug = np.vstack([J, Jcons_diag])
        
        delta = np.linalg.norm(R_aug)
        
        # Select regularization parameter
        dtikh = auto_tikhonov(delta, use_gaussian)
        
        # Compute Newton step (Tikhonov regularization)
        try:
            JTJ = J_aug.T @ J_aug
            JTR = J_aug.T @ R_aug
            step = linalg.solve(JTJ + dtikh * np.eye(JTJ.shape[0]), JTR)
        except linalg.LinAlgError:
            if verbose:
                print(f"Linear solve failed at iteration {count}")
            break
        
        # Update solution
        xnew = xold - step
        
        # Newton decrement
        ndecr = np.sqrt(step @ (J_aug.T @ R_aug))
        
        if verbose and ((count + 1) % 10 == 0 or count < 5):
            print(f"Iteration {count + 1}: Residual norm = {delta:.6e}")
        
        iters.append(delta)
        
        # Check convergence criteria
        if delta / ndecr > 10000:
            if verbose:
                print(f"Newton decrement too small: {ndecr/delta:.6e}. Increase number of points.")
                print(f"R[0] = {R[0]:.6e}, target = {parm_target[0]:.6e}")
                print(f"R norm without constraints = {np.linalg.norm(R - parm_target):.6e}")
            break
        
        if delta > 1e30:
            if verbose:
                print("Failed miserably! The regularization parameter is too small")
            break
    else:
        # This executes if we didn't break (i.e., reached max_iters)
        if verbose:
            print(f"Failed to converge within {max_iters} iterations")
    
    # Final extraction
    b = np.zeros((n_s, d))
    for i in range(d):
        b[:, i] = xnew[i*n_s:(i+1)*n_s]
    w = xnew[d*n_s:(d+1)*n_s]
    
    # No scaling needed - we're already working with the correct measure
    
    # Check for negative weights
    n_neg = 0
    min_neg = 0.0
    if np.any(w < 0):
        n_neg = np.sum(w < 0)
        min_neg = np.min(w)
        
        if verbose:
            print(f"Found {n_neg} negative weight(s) (min={min_neg:.2e})")
    
    XW = np.column_stack([b, w])
    
    if verbose and delta <= tol:
        print(f"\nConverged in {count + 1} iterations with residual norm = {delta:.6e}")
    
    return XW, delta, n_neg, min_neg