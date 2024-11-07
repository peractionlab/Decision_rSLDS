import numpy as np
import matplotlib.pyplot as plt
import os
import dill

def compute_snr(x, n):
    """
    Compute the Signal-to-Noise Ratio (SNR) in decibels (dB).
    
    Parameters:
    - x: Signal.
    - n: Noise.
    
    Returns:
    - SNR in dB.
    """
    sig_power = np.sum(x**2) / x.shape[0]
    noise_power = np.sum((x - n)**2) / x.shape[0]
    snr_10 = 10 * np.log10(sig_power / noise_power)
    return snr_10

def tanh(x):
    """
    Compute the hyperbolic tangent function.
    
    Parameters:
    - x: Input value.
    
    Returns:
    - Hyperbolic tangent of x.
    """
    if x < 10:
        return 1 - 2 / (np.exp(2 * x) + 1)
    else:
        return 1

def find_attractor(model):
    """
    Find the attractors for each discrete state in the model.
    
    Parameters:
    - model: The model object.
    
    Returns:
    - List of attractors.
    """
    atts = []
    for k in range(model.K):
        atts.append(np.linalg.pinv((np.eye(model.D) - model.dynamics.As[k])).dot(model.dynamics.bs[k]))
    return atts

def compute_original_x(model, x_infer, C_refer, d_refer):
    """
    Compute the original latent states from the inferred latent states.
    
    Parameters:
    - model: The model object.
    - x_infer: Inferred latent states.
    - C_refer: Reference observation matrix.
    - d_refer: Reference bias vector.
    
    Returns:
    - Original latent states.
    """
    C_est_inv = np.linalg.pinv(model.emissions.Cs[0]).squeeze()
    M = C_est_inv.dot(C_refer).squeeze()
    M_inv = np.linalg.pinv(M).squeeze()
    n = C_est_inv.dot((d_refer - model.emissions.ds[0])).squeeze()
    x_orig = M_inv.dot((x_infer - n).transpose()).transpose()
    return x_orig

def latent_space_transform(model, C_refer, d_refer):
    """
    Transform the model's latent space to the original latent space.
    
    Parameters:
    - model: The model object.
    - C_refer: Reference observation matrix.
    - d_refer: Reference bias vector.
    """
    # Project to the original latent space
    C_est_inv = np.linalg.pinv(model.emissions.Cs[0]).squeeze()
    M = C_est_inv.dot(C_refer).squeeze()
    M_inv = np.linalg.pinv(M).squeeze()
    n = C_est_inv.dot((d_refer - model.emissions.ds[0])).squeeze()
    
    A_prime = []
    b_prime = []
    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        A_prime.append(M_inv.dot(A).dot(M))
        b_prime.append(M_inv.dot(A).dot(n) + M_inv.dot(b - n))
    
    for k in range(model.K):
        model.dynamics.As[k] = A_prime[k]
        model.dynamics.bs[k] = b_prime[k]
    
    R_prime = model.transitions.Rs.dot(M)
    r_prime = model.transitions.Rs.dot(n) + model.transitions.r
    
    model.transitions.Rs = R_prime
    model.transitions.r = r_prime

    model.emissions.Cs[0] = C_refer
    model.emissions.ds[0] = d_refer

def save_model(path, model, elbos, q):
    """
    Save the model, ELBOs, and q to a file.
    
    Parameters:
    - path: File path to save the model.
    - model: The model object.
    - elbos: Evidence Lower Bound (ELBO) values.
    - q: Variational distribution.
    """
    with open(path, 'wb') as f:
        dill.dump(model, f)
        dill.dump(elbos, f)
        dill.dump(q, f)
    
def load_model(path):
    """
    Load the model, ELBOs, and q from a file.
    
    Parameters:
    - path: File path to load the model from.
    
    Returns:
    - model: The model object.
    - elbos: Evidence Lower Bound (ELBO) values.
    - q: Variational distribution.
    """
    model = None
    q = None
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = dill.load(f)
            elbos = dill.load(f)
            # q = dill.load(f)
        return model, elbos, q
    else:
        raise FileNotFoundError