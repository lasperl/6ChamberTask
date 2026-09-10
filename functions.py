import os, yaml, h5py
import numpy as np
import scipy as sc


# PARAMETERS
N_nodes = 4
sf = 1/30 #sampling frequency, s
# pix2cm = 1 #0.08 #approximation
score_th = 0.0 #threshold score from sleap to keep prediction
pos_gap = 0.5 #s, interpolation gap for positions
interp_gap = 2#s, interpolation gap for between nodes interpolations
jump_size=5 #in cm
jump_duration = 0.2 #in s
smooth_speed=0.1
speed_th = 5#cm/s

def load_comparison_data(folder, rat, keys):
    repo = os.path.join(folder, rat, 'Behavior')
    with h5py.File(os.path.join(repo, 'comparison.hdf5'), mode='r') as f:
        def read_dataset(path):
            d = f[path]
            if d.shape == ():
                return d[()]
            return d[:]
        if isinstance(keys, str):
            return read_dataset(f'comparison/{keys}')
        return {
            k: read_dataset(f'comparison/{k}')
            for k in keys
        }

def occ_map(xy, bins2d, smooth=False, sigma=1):
    valid = ~np.isnan(xy).any(axis=1)
    occ, _ = np.histogramdd(xy[valid], bins=bins2d)
    occ = occ * sf
    return nanGaussianSmooth(occ, sigma=sigma, truncate=2) if smooth else occ

def load_data(folder, rat, phase, keys):
    repo = os.path.join(folder, rat, 'Behavior', phase, 'sleap_data')
    with h5py.File(os.path.join(repo, 'position.hdf5'), mode='r') as f:
        def read_dataset(path):
            d = f[path]
            # scalar dataset
            if d.shape == ():
                return d[()]
            # array dataset
            return d[:]
        if isinstance(keys, str):
            return read_dataset(f'behavior/{keys}')
        return {
            k: read_dataset(f'behavior/{k}')
            for k in keys
        }

def nanGaussianSmooth(data, sigma=1.0, truncate=4.0):
    '''adapted from:
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python/'''

    U = np.array(data)
    mask_nans = np.isnan(U)

    V = U.copy()
    V[np.isnan(U)] = 0
    VV = sc.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = 0 * U.copy() + 1
    W[np.isnan(U)] = 0
    WW = sc.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

    Z = VV / WW

    Z[mask_nans] = np.nan

    return Z