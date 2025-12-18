import numpy as np

def apply_maxwellian_physics(v_parallel, mu, temperature=1.0, density=1.0):
    """Law: Particles follow a Maxwell-Boltzmann distribution in velocity space."""
    # f0 ~ exp(-(v_par^2 + mu*B)/T)
    v_sq = v_parallel**2 + mu # Simplified energy term
    f0 = density * np.exp(-v_sq / (2 * temperature))
    return f0

def calculate_heat_flux(dist_func, v_parallel):
    """Law: Heat flux is the 3rd moment of the distribution function."""
    # Q = Integral( v^3 * f dV )
    flux = np.sum((v_parallel**3) * dist_func)
    return flux
