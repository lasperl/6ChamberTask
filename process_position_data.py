# IMPORTS
import collections, collections.abc
for _n in ('Mapping', 'MutableMapping', 'Sequence', 'MutableSequence',
           'Set', 'MutableSet', 'Iterable', 'Iterator', 'Callable',
           'Hashable', 'Container'):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

import os
import h5py
import yaml
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from skimage.transform import ProjectiveTransform
from Toolbox.signals.smooth.kernelsmoothing import smooth1d
from Toolbox.behavior import preprocessing as prep
import Toolbox.segments as seg

# Fixing dependency issues
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if not hasattr(sc, 'interp'):
    sc.interp = np.interp

# FUNCTIONS
def get_sleap_data(sl_file="", nodes=2):
    '''
    Parameters
    ----------
    sl_file : str
        SLEAP output file, expect 'io' or 'slp'
    nodes: scalar
        number of nodes that were tracked

    Returns
    -------
    pos : array, (timestamps, nodes, 2d coordinates)
        list of coordinates for tracked nodes over video frames
    nodes_names: list of str,
        list of nodes labels
    point_score: array, (timestamps, nodes)
        confidence score for individual predictions
    h5file: hdf5 container

    '''

    # open file
    with h5py.File(sl_file, 'r') as h5file:

        # map prediction tuples onto arrays
        if sl_file.endswith('io') or sl_file.endswith('h5'):
            locations = h5file["tracks"][:].T
            locations = np.reshape(locations, (-1, nodes, 2))
            arr = list()
            for i in range(len(locations)):
                for j in range(len(locations[0])):
                    arr.append(locations[i, j])
            pos = np.array(arr)

            point_scores = h5file["point_scores"][:].T

            # get node names
            node_names = [n.decode() for n in h5file["node_names"][:]]

        elif sl_file.endswith('slp'):
            print('there are no nodes labels in file')

            arr = h5file['pred_points'][:]
            pos = np.array(list(map(list, arr)))

            point_scores = pos[:, -1]
            pos = pos[:, :2]  # .reshape((-1,nodes,2))

            node_names = []

        else:
            print('Unexpected fiel format')

        # get frames indices for which there are predictions
        # frameIDX = np.array( list( map( list, h5file['frames'][:] ) ) )[:,2]

        point_scores = np.reshape(point_scores, (-1, nodes))
        frames = np.where(~np.isnan(point_scores))
        frameIDX = np.array(list(map(list, frames))).T
        frameIDX = np.unique(frameIDX[:, 0])

        # sometimes there are more than one prediction instance per frame (WHY?? training?)
        # this is to select only single predictions (firsts) per frame
        """no need to use np.unique, there is only 1 instance"""
        # framesID = np.array( list( map( list, h5file['instances'][:] ) ) )[:,2]
        # _, select, _ = np.unique( framesID, return_counts=True, return_index=True)

        # organize data per node
        pos = np.dstack([pos[i::nodes][:] for i in range(nodes)])

        # TODO
        # handle squeletl
        # tracking of instance?
        # nodes mapping with labels?

    return pos, frameIDX, node_names, point_scores, h5file
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
def project2area(original_coords=[], projection_coords=[]):
    original_coords = np.array(original_coords)
    projection_coords = np.array(projection_coords)

    conversion_fcn = ProjectiveTransform()
    conversion_fcn.estimate(original_coords, projection_coords)

    return conversion_fcn
def _relative2line(xy, line2d, direction='below'):
    x = xy[:, 0]
    y = xy[:, 1]

    relative = (y - line2d[0][1]) * (line2d[1][0] - line2d[0][0]) - (x - line2d[0][0]) * (line2d[1][1] - line2d[0][1])

    if direction == 'below':
        return relative < 0
    if direction == 'above':
        return relative > 0
def two_gaussians(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return (
            A1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) +
            A2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
    )
def intersect_gaussians(mu1, mu2, sigma1, sigma2):
    """
    Solve intersection of two unit-height Gaussians.
    Returns all real intersection points.
    """
    a = 1 / (2 * sigma1 ** 2) - 1 / (2 * sigma2 ** 2)
    b = mu2 / (sigma2 ** 2) - mu1 / (sigma1 ** 2)
    c = mu1 ** 2 / (2 * sigma1 ** 2) - mu2 ** 2 / (2 * sigma2 ** 2) - np.log(sigma2 / sigma1)

    roots = np.roots([a, b, c])
    return np.real(roots[np.isreal(roots)])

def GetGaussianThresh_BM(sm_ghi, user_confirmation=True, PlotFig=True):
    sm_ghi = np.asarray(sm_ghi)
    neg_val = False

    if np.any(sm_ghi < 0):
        neg_val = True
        cor = abs(sm_ghi.min())
        sm_ghi = sm_ghi + cor

    # Histogram
    Y, X_edges = np.histogram(sm_ghi, bins=1000)
    X = (X_edges[:-1] + X_edges[1:]) / 2

    # Initial guess
    p0 = [
        Y.max(), X[np.argmax(Y)], np.std(sm_ghi) / 2,
                                  Y.max() / 2, np.mean(sm_ghi), np.std(sm_ghi)
    ]

    popt, _ = curve_fit(two_gaussians, X, Y, p0=p0, maxfev=10000)
    A1, mu1, std1, A2, mu2, std2 = popt

    # Gaussian intersections
    b = intersect_gaussians(mu1, mu2, std1, std2)

    gamma_thresh = b[(b < mu1) & (b > mu2)]
    if gamma_thresh.size == 0:
        gamma_thresh = b[(b > mu1) & (b < mu2)]

    gamma_thresh = gamma_thresh[0] if gamma_thresh.size else None

    AshD = abs(mu2 - mu1) / np.sqrt(2 * (std1 ** 2 + std2 ** 2))

    # Plotting
    if PlotFig or gamma_thresh is None:
        plt.figure()
        plt.plot(X, uniform_filter1d(Y, size=5), 'k', label='Histogram')
        plt.plot(X, two_gaussians(X, *popt), 'r', linewidth=2)

        if gamma_thresh is not None:
            plt.axvline(gamma_thresh, color='r', linewidth=3)

        plt.show()

        if user_confirmation:
            try:
                in_val = int(input("happy? 1/0 "))
            except Exception:
                in_val = 0
        else:
            in_val = 1

        if in_val == 0:
            print("Please show me where the two peaks are")
            peaks = plt.ginput(2)
            peaks = np.array(peaks)

            p0 = [
                peaks[0, 1], peaks[0, 0], abs(peaks[0, 0] - peaks[1, 0]) / 2,
                peaks[1, 1], peaks[1, 0], abs(peaks[0, 0] - peaks[1, 0]) / 2
            ]

            popt, _ = curve_fit(two_gaussians, X, Y, p0=p0, maxfev=10000)
            A1, mu1, std1, A2, mu2, std2 = popt

            b = intersect_gaussians(mu1, mu2, std1, std2)
            gamma_thresh = b[(b > mu1) & (b < mu2)]
            gamma_thresh = gamma_thresh[0] if gamma_thresh.size else None

            AshD = np.sqrt(2) * abs(mu2 - mu1) / abs(std1 + std2)

            plt.figure()
            plt.plot(X, uniform_filter1d(Y, size=5), 'k')
            plt.plot(X, two_gaussians(X, *popt), 'g', linewidth=2)

            if gamma_thresh is not None:
                plt.axvline(gamma_thresh, color='g', linewidth=3)
            else:
                print("Select desired cut off threshold")
                gamma_thresh = plt.ginput(1)[0][0]

            plt.show()

    if neg_val and gamma_thresh is not None:
        gamma_thresh -= cor

    return gamma_thresh, mu1, mu2, std1, std2, AshD
def cutoff_1d(x, fac=0.5):
    # compute threshold peak rate
    from scipy import stats
    from functools import partial

    def my_kde_bandwidth(obj, fac=1. / 5):
        """We use Scott's Rule, multiplied by a constant factor."""
        return np.power(obj.n, -1. / (obj.d + 4)) * fac

    kde = stats.gaussian_kde(x, bw_method=partial(my_kde_bandwidth, fac=fac))
    x_eval = np.linspace(x.min() - 1, x.max() + 1, 500)

    import Toolbox.signals.core as sig
    threshold = sig.localminima(kde(x_eval), x=x_eval, method='gradient')[0][0]

    return threshold
def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel
def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y
def _proj_line(key):
    if not divisions.get(key):
        return None
    return project_fcn(np.array(divisions[key], dtype=float))


# PARAMETERS
N_nodes = 4
sf = 1 / 30  # sampling frequency, s
# pix2cm = 1 #0.08 #approximation
score_th = 0.0  # threshold score from sleap to keep prediction
pos_gap = 0.5  # s, interpolation gap for positions
interp_gap = 2  # s, interpolation gap for between nodes interpolations
jump_size = 5  # in cm
jump_duration = 0.2  # in s
smooth_speed = 0.1
speed_th = 5  # cm/s

bins = [np.linspace(-25, 50, 31),
        np.linspace(-25, 100, 51)]

folder = "/data07/Lina/6Chamber_LbO/Data_analysis"

rats = [
    'F_B1C1R1',
    'F_B1C1R3',
    'M_B1C1R1',
    'M_B1C1R3',
    'F_B2C1R1',
    'F_B2C1R3',
    'M_B2C1R1',
    'M_B2C1R3',
    'F_B3C1R1',
    'F_B3C1R3',
    'M_B3C1R1',
    'M_B3C1R3'
]

phases = [
    'Baseline_Closed',
    'Baseline_Open',
    'Observation_Neutral',
    'Observation_Shock',
    'Recall_Closed',
    'Recall_Open'
]

# MAIN SCRIPT
# Load common parameters
with open(os.path.join(folder, 'overall_info.yaml')) as fh:
    overall_info = yaml.safe_load(fh)

all_xy_head = {rat: {} for rat in rats}
all_xy_back = {rat: {} for rat in rats}
all_hd = {rat: {} for rat in rats}

for rat in rats:
    rat_folder = os.path.join(folder, '{}/Behavior'.format(rat))
    rat_info = yaml.safe_load(open(os.path.join(folder, rat, 'meta_data.yaml')))
    shock_side = rat_info['shock_side']
    order_first = rat_info['first_obs']

    for phase in phases:

        print('processing: {}, {}'.format(rat, phase))
        folder_analysis = os.path.join(rat_folder, phase)

        data = {}
        repo = os.path.join(rat_folder, phase, 'sleap_data')
        npz = np.load(os.path.join(repo, 'missinginstanceSLEAP_predictions_eye.npz'), allow_pickle=True)
        nodes_loc = npz['xy'].swapaxes(1, 2)
        scores = npz['scores']
        time = np.arange(nodes_loc.shape[0]) * sf
        start_end = rat_info['phase'][phase]['start-end']

        # estimate calibration transform
        project_fcn = project2area(original_coords=rat_info['phase'][phase]['corners_coords'],
                                   projection_coords=overall_info['arena']['coords'])

        divisions = rat_info['phase'][phase].get('compartments', {})
        left = _proj_line('left')
        right = _proj_line('right')
        middle = _proj_line('middle_between')

        # exclude low-confidence points (if threshold is set)
        idx_out = np.dstack([(scores < score_th), (scores < score_th)]).swapaxes(1, 2)
        nodes_loc[idx_out] = np.nan

        nodes_xy = nodes_loc[start_end[0]:start_end[1]]  # same length

        if len(nodes_xy) != len(time):
            idx_stop = np.min([len(nodes_xy), len(time)])
            time = time[:idx_stop]
            nodes_xy = nodes_xy[:idx_stop]

        # clean up data
        for n in range(nodes_xy.shape[-1]):
            # nodes_xy[:,:,n][_relative2line( nodes_xy[:,:,n],line2d, direction=discard)] = np.nan
            nodes_xy[:, :, n] = project_fcn(nodes_xy[:, :, n])
            nodes_xy[:, :, n] = prep.remove_jumps(time, nodes_xy[:, :, n], jump_size, jumpduration=jump_duration)[0]
            nodes_xy[:, :, n] = prep.interpolate_gaps(time, nodes_xy[:, :, n], pos_gap)

        # interpolates node positions based on nodes distance and alternate node position
        nodes_xy[:, :, 1], nodes_xy[:, :, 0], _ = prep.check_diode_distance(nodes_xy[:, :, 1], nodes_xy[:, :, 0],
                                                                            threshold=10)
        try:
            nodes_xy[:, :, 0], nodes_xy[:, :, 1] = prep.interpolate_hd(time, nodes_xy[:, :, 0], nodes_xy[:, :, 1],
                                                                       interp_gap=interp_gap)
        except Exception as e:
            print(f"No HD interpolation ({e})")

        # final smoothing
        for n in range(nodes_xy.shape[-1]):
            nodes_xy[:, :, n] = smooth1d(nodes_xy[:, :, n], axis=0, delta=sf, bandwidth=0.05, unbiased=True,
                                         nansaszero=True)

        xy_snout = nodes_xy[:, :, 0]
        xy_head = nodes_xy[:, :, 1]
        xy_back = nodes_xy[:, :, 2]

        plt.figure(figsize=(7, 7))
        plt.plot(xy_head[:, 0], xy_head[:, 1], color='k')
        # plt.axhline(37.5, color='crimson', ls='--', lw=1.5, label='shock/safe divider')
        for ln, name, col in [(left, 'left', 'tab:blue'),
                              (right, 'right', 'tab:green'),
                              (middle, 'mid', 'tab:orange')]:
            if ln is not None:
                plt.plot(ln[:, 0], ln[:, 1], color=col, ls='--', lw=1.5, label=name)
        plt.legend()
        plt.legend()
        plt.title('Raw Traces (head)')
        plt.savefig(os.path.join(repo, 'Traces_head_xy.png'),
                    transparent=False)
        plt.show()
        plt.close()

        arena_coords = np.array(overall_info['arena']['coords'])
        center_arena = arena_coords.mean(axis=0)

        ######### TRACES ARE BEING FLIPPED AFTER THIS ####################

        # align maps so that shock side is the same for everyone (now all hve shock side on the right!)
        if shock_side == 'left':
            center_arena = arena_coords.mean(axis=0)

            for ln in (left, right, middle):
                if ln is not None:
                    ln[:, 1] = 2 * center_arena[1] - ln[:, 1]

            arena_coords = np.array(overall_info['arena']['coords'])
            center_arena = arena_coords.mean(axis=0)
            xy_head[:, 1] = 2 * center_arena[1] - xy_head[:, 1]
            xy_snout[:, 1] = 2 * center_arena[1] - xy_snout[:, 1]
            xy_back[:, 1] = 2 * center_arena[1] - xy_back[:, 1]

            plt.figure(figsize=(7, 7))
            plt.plot(xy_head[:, 0], xy_head[:, 1], color='k')
            # plt.axhline(37.5, color='crimson', ls='--', lw=1.5, label='shock/safe divider')
            for ln, name, col in [(left, 'left', 'tab:blue'),
                                  (right, 'right', 'tab:green'),
                                  (middle, 'mid', 'tab:orange')]:
                if ln is not None:
                    plt.plot(ln[:, 0], ln[:, 1], color=col, ls='--', lw=1.5, label=name)
            plt.title('Raw Traces (flipped; head)')
            plt.savefig(os.path.join(repo, 'Traces_head_xy_flip.png'),
                        transparent=False)
            plt.show()
            plt.close()

            safe = _relative2line(xy_head, left, direction='above')
            shock = _relative2line(xy_head, right, direction='below')
            between = _relative2line(xy_head, middle, direction='above')
            center = ~safe & ~shock & ~between

        # compute head direction
        head_direction = prep.compute_head_direction(xy_snout, xy_head, dx=sf, smooth=0.1)

        # compute veolcity
        raw_velocity = prep.compute_velocity(xy_head, dx=sf, smooth=None)
        velocity = prep.compute_velocity(xy_head, dx=sf, smooth=smooth_speed)
        max_speed = np.nanmax(np.vstack([np.abs(prep.compute_velocity(nodes_xy[:, :, n], dx=sf, smooth=smooth_speed))
                                         for n in range(nodes_xy.shape[-1])]), axis=0)
        speed = np.abs(velocity)

        # xy_head_half = len(xy_head) // 2
        # xy_head1 = xy_head[:xy_head_half]
        # xy_head2 = xy_head[xy_head_half:]

        # Divide arena into two halves
        line2d = np.array([[0, 37.5], [25, 37.5]])

        if shock_side == 'right':
            # safe = _relative2line(xy_head, top_chamber, direction='below')
            # shock = _relative2line(xy_head, bottom_chamber, direction='above')
            # between = _relative2line(xy_head, line_midbtw, direction='below')
            # center = ~safe & ~shock & ~between
            shock = _relative2line(xy_head, left, direction='below')
            safe = _relative2line(xy_head, right, direction='above')
            between = _relative2line(xy_head, middle, direction='below')
            center = ~safe & ~shock & ~between

        regions = [('safe', safe, 'tab:blue'),
                ('shock', shock, 'tab:green'),
                ('between', between, 'tab:orange'),
                ('center', center, 'tab:red')]

        fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharex=True, sharey=True)
        for ax, (name, mask, col) in zip(axes, regions):
            ax.plot(xy_head[:, 0], xy_head[:, 1], color='0.85', lw=0.5, zorder=1)
            ax.plot(xy_head[mask, 0], xy_head[mask, 1],
                    'o', ms=2, color=col, alpha=0.4, zorder=2)
            for ln in (left, right, middle):
                if ln is not None:
                    ax.plot(ln[:, 0], ln[:, 1], color='k', ls='--', lw=1)
            ax.set_title(f'{name}  ({np.sum(mask) / len(mask) * 100:.1f}%)')
            ax.set_aspect('equal')
            ax.set(xlabel='X (cm)', ylabel='Y (cm)')

        fig.suptitle(f'{rat} — {phase}: head positions by category')
        fig.savefig(os.path.join(repo, 'traces_by_category.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

        line_halfsplit = (np.asarray(left) + np.asarray(right)) / 2

        left_col = ~between
        safe_side = _relative2line(xy_head, line_halfsplit, direction='above')
        safe_half = left_col & safe_side
        shock_half = left_col & ~safe_side

        half_regions = [('safe_half', safe_half, 'tab:blue'),
                        ('shock_half', shock_half, 'tab:green'),
                        ('neutral_between', between, 'tab:orange')]

        fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True)
        for ax, (name, mask, col) in zip(axes, half_regions):
            ax.plot(xy_head[:, 0], xy_head[:, 1], color='0.85', lw=0.5, zorder=1)
            ax.plot(xy_head[mask, 0], xy_head[mask, 1],
                    'o', ms=2, color=col, alpha=0.4, zorder=2)
            for ln in (left, right, middle, line_halfsplit):
                if ln is not None:
                    ax.plot(ln[:, 0], ln[:, 1], color='k', ls='--', lw=1)
            ax.set_title(f'{name}  ({np.sum(mask) / len(mask) * 100:.1f}%)')
            ax.set_aspect('equal')
            ax.set(xlabel='X (cm)', ylabel='Y (cm)')

        fig.suptitle(f'{rat} — {phase}: head positions by category (half-split)')
        fig.savefig(os.path.join(repo, 'traces_by_category_half.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

        # count occupancy
        P_head_in_safe = np.sum(safe) / len(safe) * 100
        P_head_in_shock = np.sum(shock) / len(shock) * 100
        P_head_in_between = np.sum(between) / len(between) * 100
        P_head_center = np.sum(center) / len(center) * 100

        print(f'{rat} {phase}: % in Safe = {P_head_in_safe:.1f} '
              f'Occupancy: % Between Chambers = {P_head_in_between:.1f}; % in Shock = {P_head_in_shock:.1f}; % in Center = {P_head_center:.1f} '
              f'(Total % = {P_head_in_safe + P_head_in_between + P_head_in_shock + P_head_center:.1f})')

        # Split in middle (2 sides)
        head_shock_side = _relative2line(xy_head, line2d, direction='below')
        P_head_shock_side = np.sum(head_shock_side) / len(head_shock_side) * 100
        back_shock_side = _relative2line(xy_back, line2d, direction='below')
        P_back_shock_side = np.sum(back_shock_side) / len(back_shock_side) * 100

        # compute occupancy head
        occupancy, occup_bins = np.histogramdd(xy_head, bins=bins)
        occupancy = occupancy * sf
        occupancy = nanGaussianSmooth(occupancy, sigma=1, truncate=2)

        # compute occupancy back
        occupancy_back, occup_bins = np.histogramdd(xy_back, bins=bins)
        occupancy_back = occupancy_back * sf
        occupancy_back = nanGaussianSmooth(occupancy_back, sigma=1, truncate=2)

        # Compute average speed in each 2D bin
        speed_perbin, x_edges, y_edges, binnumber = binned_statistic_2d(
            xy_head[:, 0], xy_head[:, 1], np.nan_to_num(speed), statistic='mean', bins=bins)
        speed_perbin = nanGaussianSmooth(speed_perbin, sigma=1, truncate=2)

        # Compute time spent freezing
        #speed_th = cutoff_1d(np.log10(np.abs(velocity[~np.isnan(velocity)])), fac=0.5)
        speed_th = np.log10(0.05)

        low_speed = seg.Segment.fromlogical(np.log10(speed) < speed_th,
                                            x=time, interpolate=True)
        low_speed = low_speed.join(gap=0.5)
        low_speed = low_speed[low_speed.duration > 2]

        if len(low_speed) == 0:
            freezing = 0
        else:
            freezing = np.sum(low_speed.duration) / (time[-1] - time[0]) * 100

        frozen = np.zeros(len(time), dtype=bool)
        if len(low_speed) > 0:
            for a, b in zip(np.atleast_1d(low_speed.start), np.atleast_1d(low_speed.stop)):
                frozen[(time >= a) & (time <= b)] = True

        avg_speed = {}
        for name, mask, _ in regions:
            m = mask & ~np.isnan(speed)
            avg_speed[name] = speed[m].mean() if m.any() else np.nan

        names = [name for name, _, _ in regions]
        values = [avg_speed[name] for name in names]
        colors = [col for _, _, col in regions]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, values, color=colors)

        for b, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('mean speed (cm/s)')
        ax.set_title(f'{rat} — {phase}: average speed per region')
        plt.tight_layout()
        plt.savefig(os.path.join(repo, 'speed_by_region.png'), dpi=150)
        plt.show()
        plt.close()

        freezing_by_region = {}
        freezing_by_region_total = {}
        total_frozen = np.sum(frozen)

        for name, mask, _ in regions:
            in_region = np.sum(mask)
            frozen_here = np.sum(frozen & mask)
            # % of time in this region that was freezing (conditional)
            freezing_by_region[name] = frozen_here / in_region * 100 if in_region else np.nan
            freezing_by_region_total[name] = frozen_here / len(frozen) * 100 if total_frozen else np.nan

        for name, v in freezing_by_region.items():
            print(f'{name:15s}: {v:.1f} % freezing')

        all_xy_head[rat][phase] = xy_head.copy()
        all_xy_back[rat][phase] = xy_back.copy()
        all_hd[rat][phase] = head_direction.copy()

        # save to h5 file
        with h5py.File(os.path.join(repo, 'position.hdf5'), 'w') as f:
            f['behavior/xy_snout'] = xy_snout
            f['behavior/xy_head'] = xy_head
            f['behavior/xy_back'] = xy_back
            f['behavior/time'] = time
            f['behavior/head_direction'] = head_direction
            f['behavior/velocity'] = velocity
            f['behavior/speed'] = speed
            f['behavior/max_speed'] = max_speed
            f['behavior/mouv_dir'] = np.angle(velocity)
            f['behavior/occupancy'] = occupancy
            f['behavior/occupancy_back'] = occupancy_back
            f['behavior/bin_speed'] = speed_perbin
            f['behavior/nodes_xy'] = nodes_xy
            f['behavior/shock_side'] = shock_side
            f['behavior/order_obs'] = order_first
            f['behavior/gender'] = rat_info['subject']['sex']
            f['behavior/batch'] = rat_info['subject']['batch']
            f['behavior/freezing'] = freezing  # in %
            f['behavior/frozen_boolean'] = frozen  # boolean array when freezing
            f['behavior/safe'] = safe
            f['behavior/shock'] = shock
            f['behavior/between'] = between
            f['behavior/center'] = center
            f['behavior/P_head_in_safeCh'] = P_head_in_safe
            f['behavior/P_head_in_shockCh'] = P_head_in_shock
            f['behavior/P_head_inbetween'] = P_head_in_between
            f['behavior/P_head_center'] = P_head_center
            f['behavior/safe_half'] = safe_half
            f['behavior/shock_half'] = shock_half
            f['behavior/neutral_between'] = between
            f['behavior/P_safe_half'] = np.sum(safe_half) / len(safe_half) * 100
            f['behavior/P_shock_half'] = np.sum(shock_half) / len(shock_half) * 100
            f['behavior/P_neutral_between'] = np.sum(between) / len(between) * 100
            for name, _, _ in regions:
                f[f'behavior/freezing_{name}'] = freezing_by_region[name]
                f[f'behavior/freezing_oftotal_{name}'] = freezing_by_region_total[name]
                f[f'behavior/speed_{name}'] = avg_speed[name]

        fig, ax = plt.subplots(1, 4, figsize=(20, 8), sharex=False, sharey=False)
        sns.histplot(np.degrees(head_direction), binwidth=10, ax=ax[0])
        ax[0].set(xlabel='head direction (degree)')

        coords = [c for c in overall_info['arena']['coords']]
        coords.append(coords[0])
        coords = np.array(coords)
        y_centers = bins[1][:-1] + np.diff(bins[1]) / 2

        ax[1].plot(coords.T[0], coords.T[1], color='black')
        ax[1].plot(xy_head[:, 0], xy_head[:, 1],
                   'o', ms=2.0, color='blue', alpha=0.1, label='occupancy head')
        ax[1].plot(xy_snout[:, 0], xy_snout[:, 1],
                   'o', ms=2.0, color='green', alpha=0.1, label='occupancy snout')

        ax[1].plot(xy_head[speed > 5, 0], xy_head[speed > 5, 1],
                   ms=2.0, color='red', alpha=0.5, label='speed>5cm/s')
        ax[1].set(xlabel='X (cm)', ylabel='Y (cm)')
        ax[1].legend()

        ax[2].plot(y_centers, np.sum(occupancy, axis=0), label='head')
        ax[2].plot(y_centers, np.sum(occupancy_back, axis=0), label='back')
        ax[2].legend()
        ax[2].set(xlabel='(cm)', ylabel='time(s)',
                  title="%time in shock, back: {:.2F}, head: {:.2F}".format(P_head_shock_side, P_back_shock_side))

        sns.histplot(np.log10(speed[~np.isnan(speed)]),
                     stat='density', kde=True, ax=ax[3])
        ax[3].axvline(speed_th, lw=1.0, color='black')
        ax[3].set(xlabel='log(speed)',
                  title="% freezing = {:.2F}".format(freezing),
                  # xlim=(0,25), ylim=(0,0.5)
                  )

        fig.savefig(os.path.join(repo, 'position.png'),
                    transparent=False)
        plt.close()

        fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=False)
        ax[0].plot(time, speed, 'ro', markersize=1.0, color='grey')
        ax[0].plot(time, np.abs(raw_velocity), 'ro', markersize=1.0, color='grey')
        ax[0].axhline(0.05, color='red', alpha=0.5,
                      label='{:.2f}% below 0.05'.format(np.sum(speed < 0.05) / len(velocity) * 100))
        ax[0].set(ylabel='speed (cm/s)', xlabel='')
        ax[0].legend()

        ax[1].plot(time, np.degrees(head_direction), 'ro', markersize=1.0, color='grey')
        ax[1].set(ylabel='head direction', xlabel='time(s)', ylim=(0, 360), xlim=(time[0], time[-1]))


        fig.savefig(os.path.join(repo, 'speed_HeadDirection_dist.png'),transparent=False)
        plt.close()

        # Occupancy Simple
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.heatmap(occupancy, cmap='binary', vmin=0, vmax=2)
        fig.savefig(os.path.join(repo, 'occupancy.png'),
                    transparent=False)
        # fig.savefig(os.path.join(repo, 'occupancy.svg'),
        #             transparent=False)
        plt.close()

        # Speed
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plt.imshow(speed_perbin.T, origin='lower',
                   # extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                   aspect='auto', cmap='viridis')
        plt.colorbar(label='Average Speed')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Average Speed per 2D Position Bin')
        # plt.show()
        fig.savefig(os.path.join(repo, 'speed_bin.png'),
                    transparent=False)
        plt.close()

        # Occupancy with histogram
        x, y = xy_head[:, 0], xy_head[:, 1]

        sns.set_theme(style='ticks')
        g = sns.jointplot(x=x, y=y, kind='hist',
                          bins=(30, 30),
                          cmap='crest', cbar=True,
                          marginal_kws=dict(element='step'))
        g.set_axis_labels('x', 'y')
        g.ax_joint.set_xlim(0, 25)
        g.ax_joint.set_ylim(0, 75)
        g.figure.text(0.01, 0.01, f'n = {len(x)} frames',
                 fontsize=7, color='grey', ha='left', va='bottom')
        # g.ax_joint.plot(coords[:,0], coords[:, 1], 'k-', lw=1)
        # g.ax_joint.set_aspect('equal')
        # g.set_title(f'Occupancy for {phase} (Shock side: {shock_side})')
        g.savefig(os.path.join(repo, 'occupancy_joint.png'))
        # g.savefig(os.path.join(repo, 'occupancy_joint.svg'))
        plt.close(g.figure)

        # Occupancy with Histogram + % in Shock Side barplot
        # g1 = sns.jointplot(x=x, y=y, kind='hist',
        #                    bins=(30, 30),
        #                    cmap='crest', cbar=False,
        #                    marginal_kws=dict(element='step'), color='#3b6978')
        # g1.set_axis_labels('x', 'y')
        # g1.ax_joint.axhline(37.5, color='grey', linestyle=':', linewidth=0.8)
        # g1.ax_joint.set_xlim(0, 25)
        # g1.ax_joint.set_ylim(0, 75)
        #
        # fig = g1.figure
        # fig.subplots_adjust(right=0.80)
        #
        # joint_pos = g1.ax_joint.get_position()
        # marg_pos = g1.ax_marg_y.get_position()
        #
        # bar_ax = fig.add_axes([marg_pos.x1 + 0.03, joint_pos.y0,
        #                        0.08, joint_pos.height])
        # bar_ax.bar(0, P_head_in_shock, width=0.6, color='k')
        # bar_ax.set_ylim(0, 100)
        # bar_ax.set_xticks([])
        # bar_ax.set_ylabel('Time Spent on Shock Side (in %)')
        # bar_ax.yaxis.set_label_position('right')
        # bar_ax.yaxis.tick_right()
        # g1.figure.suptitle(f'Occupancy for {rat} in {phase} (Shock side: {shock_side})')
        # fig.text(0.01, 0.01, f'n = {len(x)} frames',
        #          fontsize=7, color='grey', ha='left', va='bottom')
        #
        # g1.savefig(os.path.join(repo, 'occupancy_joint_quant.png'))
        # #g1.savefig(os.path.join(repo, 'occupancy_joint_quant.svg'))
        # plt.close(fig)

        # # Occupancy with Histogram + % in Shock Side barplot: FIRST
        # x_first, y_first = xy_head1[:, 0], xy_head1[:, 1]
        #
        # g2 = sns.jointplot(x=x_first, y=y_first, kind='hist',
        #                    bins=(30, 30),
        #                    cmap='flare', cbar=False,
        #                    marginal_kws=dict(element='step'), color='#edaf80')
        # g2.set_axis_labels('x', 'y')
        # g2.ax_joint.axhline(37.5, color='grey', linestyle=':', linewidth=0.8)
        # g2.ax_joint.set_xlim(0, 25)
        # g2.ax_joint.set_ylim(0, 75)
        #
        # fig = g2.figure
        # fig.subplots_adjust(right=0.80)
        #
        # joint_pos = g2.ax_joint.get_position()
        # marg_pos = g2.ax_marg_y.get_position()
        #
        # bar_ax = fig.add_axes([marg_pos.x1 + 0.03, joint_pos.y0,
        #                        0.08, joint_pos.height])
        # bar_ax.bar(0, P_head_in_shock1, width=0.6, color='k')
        # bar_ax.set_ylim(0, 100)
        # bar_ax.set_xticks([])
        # bar_ax.set_ylabel('Time Spent on Shock Side (in %)')
        # bar_ax.yaxis.set_label_position('right')
        # bar_ax.yaxis.tick_right()
        # g2.figure.suptitle(f'Occupancy for {rat} in First Half of {phase} (Shock side: {shock_side})')
        # fig.text(0.01, 0.01, f'n = {len(x_first)} frames',
        #          fontsize=7, color='grey', ha='left', va='bottom')
        #
        # g2.savefig(os.path.join(repo, 'occupancy_joint_quant_first.png'))
        # #g2.savefig(os.path.join(repo, 'occupancy_joint_quant_first.svg'))
        # plt.close(fig)
        #
        # # Occupancy with Histogram + % in Shock Side barplot: SECOND
        # x_second, y_second = xy_head2[:, 0], xy_head2[:, 1]
        #
        # g3 = sns.jointplot(x=x_second, y=y_second, kind='hist',
        #                    bins=(30, 30),
        #                    cmap='flare', cbar=False,
        #                    marginal_kws=dict(element='step'), color='#edaf80')
        # g3.set_axis_labels('x', 'y')
        # g3.ax_joint.axhline(37.5, color='grey', linestyle=':', linewidth=0.8)
        # g3.ax_joint.set_xlim(0, 25)
        # g3.ax_joint.set_ylim(0, 75)
        #
        # fig = g3.figure
        # fig.subplots_adjust(right=0.80)
        #
        # joint_pos = g3.ax_joint.get_position()
        # marg_pos = g3.ax_marg_y.get_position()
        #
        # bar_ax = fig.add_axes([marg_pos.x1 + 0.03, joint_pos.y0,
        #                        0.08, joint_pos.height])
        # bar_ax.bar(0, P_head_in_shock2, width=0.6, color='k')
        # bar_ax.set_ylim(0, 100)
        # bar_ax.set_xticks([])
        # bar_ax.set_ylabel('Time Spent on Shock Side (in %)')
        # bar_ax.yaxis.set_label_position('right')
        # bar_ax.yaxis.tick_right()
        # g3.figure.suptitle(f'Occupancy for {rat} in Second Half of {phase} (Shock side: {shock_side})')
        # fig.text(0.01, 0.01, f'n = {len(x_second)} frames',
        #          fontsize=7, color='grey', ha='left', va='bottom')
        #
        # g3.savefig(os.path.join(repo, 'occupancy_joint_quant_second.png'))
        # #g3.savefig(os.path.join(repo, 'occupancy_joint_quant_second.svg'))
        # plt.close(fig)

        # # FIRST HALF
        # valid1 = ~np.isnan(xy_head1).any(axis=1)
        # x1, y1 = xy_head1[valid1, 0], xy_head1[valid1, 1]
        #
        # # Occupancy with Histogram + % in Shock Side barplot
        # g3 = sns.jointplot(x=x1, y=y1, kind='hist',
        #                    bins=(15, 15),
        #                    cmap='flare', cbar=False,
        #                    marginal_kws=dict(element='step'), color='#3b6978')
        # g3.set_axis_labels('x', 'y')
        # g3.ax_joint.axhline(37.5, color='grey', linestyle=':', linewidth=0.8)
        # g3.ax_joint.set_xlim(0, 25)
        # g3.ax_joint.set_ylim(0, 75)
        #
        # fig = g3.figure
        # fig.subplots_adjust(right=0.80)
        #
        # joint_pos = g3.ax_joint.get_position()
        # marg_pos = g3.ax_marg_y.get_position()
        #
        # bar_ax = fig.add_axes([marg_pos.x1 + 0.03, joint_pos.y0,
        #                        0.08, joint_pos.height])
        # bar_ax.bar(0, P_head_in_shock, width=0.6, color='#3b6978')
        # bar_ax.set_ylim(0, 100)
        # bar_ax.set_xticks([])
        # bar_ax.set_ylabel('Time Spent on Shock Side (in %)')
        # bar_ax.yaxis.set_label_position('right')
        # bar_ax.yaxis.tick_right()
        # g1.figure.suptitle(f'Occupancy for {rat} in First Half of {phase} (Shock side: {shock_side})')
        #
        # g1.savefig(os.path.join(repo, 'occupancy_joint_quant_firsthalf.png'))
        # plt.close(fig)

        # smoothed kde
        # g2 = sns.jointplot(x=x, y=y, kind='kde',
        #                   cmap='crest', fill=True,
        #                   thresh=0.02, levels=100, bw_adjust=0.6, cbar=True)
        # g2.set_axis_labels('x', 'y')
        # #g2.ax_joint.plot(coords[:, 0], coords[:, 1], 'k-', lw=1)
        # # g.ax_joint.set_aspect('equal')
        # g2.savefig(os.path.join(repo, 'occupancy_joint_kde.png'))
        # plt.close(g2.figure)

        # Velocity path
        nodes_loc_check = fill_missing(xy_head)
        thx_vel_fly0 = smooth_diff(nodes_loc_check[:])

        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(121)
        ax1.plot(xy_head[:, 0], xy_head[:, 1], 'k')
        # ax1.set_xlim(0, 1024)
        ax1.set_xticks([])
        # ax1.set_ylim(0, 1024)
        ax1.set_yticks([])
        ax1.set_title('head tracks')

        vmin = 0
        vmax = 1

        kp = thx_vel_fly0

        ax2 = fig.add_subplot(122)
        plotts = ax2.scatter(xy_head[:, 0], xy_head[:, 1], c=kp, s=3, vmin=vmin, vmax=vmax)
        # ax2.set_xlim(0, 1024)
        ax2.set_xticks([])
        # ax2.set_ylim(0, 1024)
        ax2.set_yticks([])
        ax2.set_title('head tracks colored by speed')
        cbar = fig.colorbar(plotts, ax=ax2)
        cbar.set_label('Velocity')
        plt.show()
        plt.savefig(os.path.join(repo, 'velocity.png'))
        plt.close()

        # Freezing and occupancy
        fine_bins = [np.arange(bins[0][0], bins[0][-1] + 0.5, 0.5),
                     np.arange(bins[1][0], bins[1][-1] + 0.5, 0.5)]

        occ_fine, _ = np.histogramdd(xy_head, bins=fine_bins)
        occ_fine = occ_fine * sf
        occ_fine = nanGaussianSmooth(occ_fine, sigma=0.8 / 0.5, truncate=2)

        vmax = np.nanpercentile(occ_fine[occ_fine > 0], 99)
        extent = [bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]]

        fig = plt.figure(figsize=(8, 7))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 6, 0.4], wspace=0.15)
        bax = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])

        # occupancy map
        im = ax.imshow(occ_fine.T, origin='lower', extent=extent,
                       cmap='crest', aspect='equal', vmin=0, vmax=vmax)

        # chamber dividers
        for ln in (left, right, middle):
            if ln is not None:
                ax.plot(ln[:, 0], ln[:, 1], color='k', ls='--', lw=1)

        # freezing % label per region
        for name, mask, _ in regions:
            if np.sum(mask) == 0:
                continue
            cx = np.nanmean(xy_head[mask, 0])
            cy = np.nanmean(xy_head[mask, 1])
            ax.text(cx, cy, f'{freezing_by_region[name]:.1f}% \n ({freezing_by_region_total[name]:.1f}%)',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white',
                    )

        ax_x1 = np.nanmax(xy_head[:, 0])
        ax_y0 = np.nanmin(xy_head[:, 1])
        sx = ax_x1 - 2
        sy = ax_y0 + 6
        ax.text(sx, sy, '⚡', ha='center', va='center', fontsize=22, color='gold')

        ax.set(xlabel='X (cm)', ylabel='Y (cm)',
               title=f'{rat}: {phase}\noccupancy + % freezing per region')
        pad = 5
        ax.set_xlim(np.nanmin(xy_head[:, 0]) - pad, np.nanmax(xy_head[:, 0]) + pad)
        ax.set_ylim(np.nanmin(xy_head[:, 1]) - pad, np.nanmax(xy_head[:, 1]) + pad)

        fig.colorbar(im, cax=cax, orientation='vertical', label='occupancy')

        # left: total session freezing
        bax.bar(0, freezing, width=0.6, color='0.3')
        bax.set_ylim(0, 100)
        bax.set_xticks([])
        bax.set_ylabel('total % freezing')
        bax.text(0, freezing + 1.5, f'{freezing:.1f}%', ha='center', va='bottom', fontsize=9)
        fig.savefig(os.path.join(repo, 'occupancy_freezing_by_region.png'),
                    bbox_inches='tight', dpi=150)
        plt.show()
        plt.close()


        # Occupancy Map
        plt.figure(figsize=(8, 7))
        im = plt.imshow(occ_fine.T, origin='lower', extent=extent,
                        cmap='binary', aspect='equal', vmin=0, vmax=vmax)
        pad = 5
        plt.xlabel('X (cm)');
        plt.ylabel('Y (cm)')
        plt.title(f'{rat}: {phase} occupancy')
        plt.xlim(np.nanmin(xy_head[:, 0]) - pad, np.nanmax(xy_head[:, 0]) + pad)
        plt.ylim(np.nanmin(xy_head[:, 1]) - pad, np.nanmax(xy_head[:, 1]) + pad)

        plt.colorbar(im, label='occupancy')

        plt.savefig(os.path.join(repo, 'occupancy_fine.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

        half_pct = [np.sum(safe_half) / len(safe_half) * 100,
                    np.sum(shock_half) / len(shock_half) * 100,
                    np.sum(between) / len(between) * 100]
        half_names = ['safe_half', 'shock_half', 'neutral_between']
        half_colors = ['tab:blue', 'tab:green', 'tab:orange']

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(half_names, half_pct, color=half_colors, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(half_pct):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('occupancy (%)')
        ax.set_ylim(0, 100)
        ax.set_title(f'{rat} — {phase}: half-split occupancy')
        fig.tight_layout()
        fig.savefig(os.path.join(repo, 'occupancy_half_bar.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()


with h5py.File(os.path.join(folder, 'all_xy_head.hdf5'), 'w') as f:
    for r in all_xy_head:
        for p in all_xy_head[r]:
            f[f'{r}/{p}/xy_head'] = all_xy_head[r][p]

with h5py.File(os.path.join(folder, 'all_xy_back.hdf5'), 'w') as f:
    for r in all_xy_back:
        for p in all_xy_back[r]:
            f[f'{r}/{p}/xy_back'] = all_xy_back[r][p]


with h5py.File(os.path.join(folder, 'all_hd.hdf5'), 'w') as f:
    for r in all_hd:
        for p in all_hd[r]:
            f[f'{r}/{p}/all_hd'] = all_hd[r][p]



