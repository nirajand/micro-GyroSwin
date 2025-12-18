import zarr
import numpy as np
import os
from utils import apply_maxwellian_physics, calculate_heat_flux

def generate_physics_dataset(path="data/plasma.zarr", samples=50):
    if not os.path.exists("data"): os.makedirs("data")
    # 5D Grid: (T, X, Y, Vpar, Mu)
    shape = (samples, 4, 16, 16, 8, 8)
    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=True)
    
    dist_ds = root.create_dataset("dist_func", shape=shape, chunks=(1, 4, 16, 16, 8, 8), dtype='f4')
    flux_ds = root.create_dataset("target_flux", shape=(samples, 1), dtype='f4')

    v_par = np.linspace(-3, 3, 8)
    mu = np.linspace(0, 5, 8)

    print(f"Generating physics-consistent 5D data...")
    for i in range(samples):
        # Create a 5D distribution based on Maxwellian physics
        grid = np.zeros(shape[1:])
        for t in range(4):
            temp_fluctuation = 1.0 + 0.1 * np.random.randn()
            grid[t] = apply_maxwellian_physics(v_par[:, None], mu[None, :], temperature=temp_fluctuation)
        
        dist_ds[i] = grid
        flux_ds[i] = calculate_heat_flux(grid, v_par)

if __name__ == "__main__":
    generate_physics_dataset()
