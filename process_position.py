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
from math import sqrt
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import seaborn as sns
from Toolbox.signals.smooth.kernelsmoothing import smooth1d
from Toolbox.behavior import preprocessing as prep
import Toolbox.segments as seg
from skimage.transform import ProjectiveTransform
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
from scipy.stats import binned_statistic_2d

# testing velocity
from scipy.interpolate import interp1d

# Fixing dependency issues
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

if not hasattr(sc, 'interp'):
    sc.interp=np.interp



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
sf = 1/30 #sampling frequency, s
# pix2cm = 1 #0.08 #approximation
score_th = 0.0 #threshold score from sleap to keep prediction
pos_gap = 0.5 #s, interpolation gap for positions
interp_gap = 2#s, interpolation gap for between nodes interpolations
jump_size=5 #in cm
jump_duration = 0.2 #in s
smooth_speed=0.1
speed_th = 5#cm/s

bins = [np.linspace(-25, 50, 31),
        np.linspace(-25,100,51)]

folder = "/data07/Lina/6Chamber_LbO/Data_analysis"

rats = [
    'F_B1C1R1',
    # 'F_B1C1R3',
    # 'M_B1C1R1',
    # 'M_B1C1R3',
    # 'F_B2C1R1',
    # 'F_B2C1R3',
    # 'M_B2C1R1',
    # 'M_B2C1R3',
    # 'F_B3C1R1',
    # 'F_B3C1R3',
    # 'M_B3C1R1',
    # 'M_B3C1R3'
]

phases = [
    #'Baseline_Closed',
    #'Baseline_Open',
    #'Observation_Neutral',
    #'Observation_Shock',
    'Recall_Closed',
    #'Recall_Open'
]


# MAIN SCRIPT
# Load common parameters
overall_info = yaml.safe_load(open(os.path.join(folder, 'overall_info.yaml')))

for rat in rats:
    comp_phases = phases

    rat_folder = os.path.join(folder, '{}/Behavior'.format(rat))
    rat_info = yaml.safe_load(open(os.path.join(folder, rat, 'meta_data.yaml')))

    for phase in comp_phases:

        print('processing: {}, {}'.format(rat, phase))
        folder_analysis = os.path.join(rat_folder, phase)

        data = {}

        repo = os.path.join(rat_folder, phase, 'sleap_data')

        npz = np.load(os.path.join(repo, 'missinginstanceSLEAP_predictions_eye.npz'),
                      allow_pickle=True)
        nodes_loc = npz['xy'].swapaxes(1, 2)
        scores = npz['scores']

        time = np.arange(nodes_loc.shape[0]) * sf

        start_end = rat_info['phase'][phase]['start-end']

        # estimate calibration transform
        project_fcn = project2area(original_coords=rat_info['phase'][phase]['corners_coords'],
                                   projection_coords=overall_info['arena']['coords'])

        divisions = rat_info['phase'][phase].get('mid_compartment', {})
        top_chamber = _proj_line('left')
        bottom_chamber = _proj_line('right')
        line_midbtw = _proj_line('middle_between')

        # exclude low-confidence points (if threshold is set)
        idx_out = np.dstack([(scores < score_th), (scores < score_th)]).swapaxes(1, 2)
        nodes_loc[idx_out] = np.nan

        nodes_xy = nodes_loc.copy()
        nodes_xy = nodes_xy[start_end[0]:start_end[1]] # same length

        if len(nodes_xy) != len(time):
            idx_stop = np.min([len(nodes_xy), len(time)])
            time = time[:idx_stop]
            nodes_xy = nodes_xy[:idx_stop]

        # clean up data
        for n in range(nodes_loc.shape[-1]):
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
        except:
            print("No HD interpolation")

        # final smoothing
        for n in range(nodes_loc.shape[-1]):
            nodes_xy[:, :, n] = smooth1d(nodes_xy[:, :, n], axis=0, delta=sf, bandwidth=0.05, unbiased=True,
                                         nansaszero=True)

        xy_snout = nodes_xy[:, :, 0]
        xy_head = nodes_xy[:, :, 1]
        xy_back = nodes_xy[:, :, 2]

        plt.figure(figsize=(7, 7))
        plt.plot(xy_head[:, 0], xy_head[:, 1], color='k')
        #plt.axhline(37.5, color='crimson', ls='--', lw=1.5, label='shock/safe divider')
        for ln, name, col in [(top_chamber, 'right', 'tab:blue'),
                              (bottom_chamber, 'left', 'tab:green'),
                              (line_midbtw, 'mid', 'tab:orange')]:
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

        # align maps so that shock side is the same for everyone (now all hve shock side on the right!)
        if rat_info['shock_side'] == 'left':
            # arena_coords = np.array(overall_info['arena']['coords'])
            # center_arena = arena_coords.mean(axis=0)
            # flip_xy_head = xy_head.copy()
            # flip_xy_head[:, 1] = 2*center_arena[1]-xy_head[:,1]
            #
            # plt.figure(figsize=(7, 7))
            # plt.plot(flip_xy_head[:, 0], flip_xy_head[:, 1], color='k')
            # plt.title('Raw Traces (head)')
            # plt.savefig(os.path.join(repo, 'Traces_head_xy_flipped.png'),
            #             transparent=False)
            # plt.show()
            # plt.close()

            center_arena = arena_coords.mean(axis=0)

            for ln in (top_chamber, bottom_chamber, line_midbtw):
                if ln is not None:
                    ln[:, 1] = 2 * center_arena[1] - ln[:, 1]

            arena_coords = np.array(overall_info['arena']['coords'])
            center_arena = arena_coords.mean(axis=0)
            xy_head[:, 1] = 2 * center_arena[1] - xy_head[:, 1]

            plt.figure(figsize=(7, 7))
            plt.plot(xy_head[:, 0], xy_head[:, 1], color='k')
            #plt.axhline(37.5, color='crimson', ls='--', lw=1.5, label='shock/safe divider')
            for ln, name, col in [(top_chamber, 'right', 'tab:blue'),
                                  (bottom_chamber, 'left', 'tab:green'),
                                  (line_midbtw, 'mid', 'tab:orange')]:
                if ln is not None:
                    plt.plot(ln[:, 0], ln[:, 1], color=col, ls='--', lw=1.5, label=name)
            plt.title('Raw Traces (head)')
            plt.savefig(os.path.join(repo, 'Traces_head_xy_flip.png'),
                        transparent=False)
            plt.show()
            plt.close()

        # compute head direction
        head_direction = prep.compute_head_direction(xy_snout, xy_head, dx=sf, smooth=0.1)

        # compute veolcity
        raw_velocity = prep.compute_velocity(xy_head, dx=sf, smooth=None)
        velocity = prep.compute_velocity(xy_head, dx=sf, smooth=smooth_speed)

        max_speed = np.nanmax(np.vstack([np.abs(prep.compute_velocity(nodes_xy[:, :, n], dx=sf, smooth=smooth_speed))
                                         for n in range(nodes_loc.shape[-1])]), axis=0)

        xy_head_half = len(xy_head) // 2
        xy_head1 = xy_head[:xy_head_half]
        xy_head2 = xy_head[xy_head_half:]

        # Divide arena into two halves
        line2d = np.array([[0, 37.5], [25, 37.5]])
        top = _relative2line(xy_head, top_chamber, direction='above')
        bottom = _relative2line(xy_head, bottom_chamber, direction='below')

        center_between = _relative2line(xy_head, line_midbtw, direction='above')
        leftover = ~top & ~bottom & ~center_between

        cats = [('top', top, 'tab:blue'),
                ('bottom', bottom, 'tab:green'),
                ('center_between', center_between, 'tab:orange'),
                ('leftover', leftover, 'tab:red')]

        fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharex=True, sharey=True)
        for ax, (name, mask, col) in zip(axes, cats):
            ax.plot(xy_head[:, 0], xy_head[:, 1], color='0.85', lw=0.5, zorder=1)
            ax.plot(xy_head[mask, 0], xy_head[mask, 1],
                    'o', ms=2, color=col, alpha=0.4, zorder=2)
            for ln in (top_chamber, bottom_chamber, line_midbtw):
                if ln is not None:
                    ax.plot(ln[:, 0], ln[:, 1], color='k', ls='--', lw=1)
            ax.set_title(f'{name}  ({np.sum(mask) / len(mask) * 100:.1f}%)')
            ax.set_aspect('equal')
            ax.set(xlabel='X (cm)', ylabel='Y (cm)')

        fig.suptitle(f'{rat} — {phase}: head positions by category')
        fig.savefig(os.path.join(repo, 'traces_by_category.png'),
                    bbox_inches='tight', dpi=150)
        plt.tight_layout()
        plt.close()

        # count occupancy
        P_head_in_left = np.sum(top) / len(top) * 100
        P_head_in_right = np.sum(bottom) / len(bottom) * 100
        P_head_in_center_between = np.sum(center_between) / len(center_between) * 100
        P_head_center_top = np.sum(leftover) / len(leftover) * 100

        print(f'{rat} {phase}: % in Safe = {P_head_in_left:.1f} '
              f'% between Chambers ={P_head_in_center_between:.1f} % in Shock = {P_head_in_right:.1f} % in Center ={P_head_center_top:.1f} '
              f'(sum={P_head_in_left+ P_head_in_center_between + P_head_in_right + P_head_center_top:.1f})')

        # compute time in shock compartment
        # if rat_info['shock_side'] == 'left':
        #     head_in_shock = _relative2line(xy_head, line2d, direction='above')
        #     P_head_in_shock = np.sum(head_in_shock) / len(head_in_shock) * 100
        #     back_in_shock = _relative2line(xy_back, line2d, direction='above')
        #     P_back_in_shock = np.sum(back_in_shock) / len(back_in_shock) * 100
        #
        #     # For split: first half
        #     head_in_shock1 = _relative2line(xy_head1, line2d, direction='above')
        #     P_head_in_shock1 = np.sum(head_in_shock1) / len(head_in_shock1) * 100
        #
        #     # For split: second half
        #     head_in_shock2 = _relative2line(xy_head2, line2d, direction='above')
        #     P_head_in_shock2 = np.sum(head_in_shock2) / len(head_in_shock2) * 100

        # if rat_info['shock_side'] == 'right':
        #     head_in_shock = _relative2line(xy_head, line2d, direction='below')
        #     P_head_in_shock = np.sum(head_in_shock) / len(head_in_shock) * 100
        #     back_in_shock = _relative2line(xy_back, line2d, direction='below')
        #     P_back_in_shock = np.sum(back_in_shock) / len(back_in_shock) * 100
        #
        #     # For split: first half
        #     head_in_shock1 = _relative2line(xy_head1, line2d, direction='below')
        #     P_head_in_shock1 = np.sum(head_in_shock1) / len(head_in_shock1) * 100
        #
        #     # For split: second half
        #     head_in_shock2 = _relative2line(xy_head2, line2d, direction='below')
        #     P_head_in_shock2 = np.sum(head_in_shock2) / len(head_in_shock2) * 100

        shock_side = rat_info['shock_side']
        head_in_shock = _relative2line(xy_head, line2d, direction='below')
        P_head_in_shock = np.sum(head_in_shock) / len(head_in_shock) * 100

        back_in_shock = _relative2line(xy_back, line2d, direction='below')
        P_back_in_shock = np.sum(back_in_shock) / len(back_in_shock) * 100

        # For split: first half
        head_in_shock1 = _relative2line(xy_head1, line2d, direction='below')
        P_head_in_shock1 = np.sum(head_in_shock1) / len(head_in_shock1) * 100

        # For split: second half
        head_in_shock2 = _relative2line(xy_head2, line2d, direction='below')
        P_head_in_shock2 = np.sum(head_in_shock2) / len(head_in_shock2) * 100

        # compute occupancy head
        occupancy, occup_bins = np.histogramdd(xy_head, bins=bins)
        occupancy = occupancy * sf
        occupancy = nanGaussianSmooth(occupancy, sigma=1, truncate=2)

        # Occupancy xy_head
        occupancy, occup_bins = np.histogramdd(xy_head, bins=bins)
        occupancy = occupancy * sf
        occupancy = nanGaussianSmooth(occupancy, sigma=1, truncate=2)

        # Occupancy first half
        occupancy1, occup_bins1 = np.histogramdd(xy_head1, bins=bins)
        occupancy1 = occupancy1 * sf
        occupancy1 = nanGaussianSmooth(occupancy1, sigma=1, truncate=2)

        # Occupancy second half
        occupancy2, occup_bins2 = np.histogramdd(xy_head2, bins=bins)
        occupancy2 = occupancy2 * sf
        occupancy2 = nanGaussianSmooth(occupancy2, sigma=1, truncate=2)

        # compute occupancy back
        occupancy_back, occup_bins = np.histogramdd(xy_back, bins=bins)
        occupancy_back = occupancy_back * sf
        occupancy_back = nanGaussianSmooth(occupancy_back, sigma=1, truncate=2)

        # Compute average speed in each 2D bin
        speed_perbin, x_edges, y_edges, binnumber = binned_statistic_2d(
            xy_head[:, 0], xy_head[:, 1], np.nan_to_num(np.abs(velocity)), statistic='mean', bins=bins)
        speed_perbin = nanGaussianSmooth(speed_perbin, sigma=1, truncate=2)

        # Compute time spent freezing
        speed_th = cutoff_1d(np.log10(np.abs(velocity[~np.isnan(velocity)])), fac=0.5)
        speed_th = np.log10(0.05)

        low_speed = seg.Segment.fromlogical(np.log10(np.abs(velocity)) < speed_th,
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

        frozen_in_shock = frozen & head_in_shock
        P_freezing_in_shock = np.sum(frozen_in_shock) / len(frozen) * 100
        P_freezing_given_shock = (np.sum(frozen_in_shock) / np.sum(head_in_shock) * 100
                                  if np.sum(head_in_shock) else np.nan)

        head_in_safe = ~head_in_shock
        frozen_in_safe = frozen & head_in_safe

        P_head_in_safe = np.sum(head_in_safe) / len(head_in_safe) * 100
        P_freezing_in_safe = np.sum(frozen_in_safe) / len(frozen) * 100
        P_freezing_given_safe = (np.sum(frozen_in_safe) / np.sum(head_in_safe) * 100
                                 if np.sum(head_in_safe) else np.nan)


        order_first = rat_info['first_obs']
        # save to h5 file
        with h5py.File(os.path.join(repo, 'position.hdf5'), 'w') as f:
            f['behavior/xy_snout'] = xy_snout
            f['behavior/xy_head'] = xy_head
            f['behavior/xy_back'] = xy_back
            f['behavior/time'] = time
            f['behavior/head_direction'] = head_direction
            f['behavior/velocity'] = velocity
            f['behavior/speed'] = np.abs(velocity)
            f['behavior/max_speed'] = max_speed
            f['behavior/mouv_dir'] = np.angle(velocity)
            f['behavior/occupancy'] = occupancy
            f['behavior/occupancy_back'] = occupancy_back
            f['behavior/bin_speed'] = speed_perbin
            f['behavior/nodes_xy'] = nodes_xy
            f['behavior/head_in_shock'] = head_in_shock
            f['behavior/back_in_shock'] = back_in_shock
            f['behavior/P_head_in_shock'] = P_head_in_shock
            f['behavior/P_back_in_shock'] = P_back_in_shock
            #f['behavior/back_in_right'] = back_in_right
            f['behavior/shock_side'] = shock_side
            f['behavior/order_obs'] = order_first
            f['behavior/freezing'] = freezing # in %
            #f['behavior/divider_distance'] = dist2divider
            f['behavior/frozen'] = frozen # boolean array when freezing
            f['behavior/P_freezing_in_shock'] = P_freezing_in_shock
            f['behavior/P_freezing_given_shock'] = P_freezing_given_shock
            f['behavior/frozen_in_safe'] = frozen_in_safe
            f['behavior/P_freezing_in_safe'] = P_freezing_in_safe
            f['behavior/P_freezing_given_safe'] = P_freezing_given_safe
            f['behavior/top'] = top
            f['behavior/bottom'] = bottom
            f['behavior/center_between'] = center_between
            f['behavior/leftover'] = leftover
            f['behavior/P_head_in_safeCh'] = P_head_in_left
            f['behavior/P_head_in_shockCh'] = P_head_in_right
            f['behavior/P_head_inbetween'] = P_head_in_center_between
            f['behavior/P_head_center'] = P_head_center_top


        metrics = ['% on\nside', '% frozen on\nside (session)', '% frozen |\non side']
        shock_vals = [P_head_in_shock, P_freezing_in_shock, P_freezing_given_shock]
        safe_vals = [P_head_in_safe, P_freezing_in_safe, P_freezing_given_safe]

        x = np.arange(len(metrics))
        w = 0.38

        fig, ax = plt.subplots(figsize=(8, 4.5))
        b1 = ax.bar(x - w / 2, shock_vals, w, label='shock side', color='#b0413e')
        b2 = ax.bar(x + w / 2, safe_vals, w, label='safe side', color='#3b6978')

        ax.axhline(freezing, ls='--', lw=1, color='0.5',
                   label=f'total freezing ({freezing:.1f}%)')

        ax.set_ylim(0, 100)
        ax.set_ylabel('%')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_title(f'{rat} — {phase}\nfreezing by side (shock: {shock_side})', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

        for bars in (b1, b2):
            for b in bars:
                v = b.get_height()
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}',
                            ha='center', va='bottom', fontsize=8)

        fig.savefig(os.path.join(repo, 'freezing_by_side.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()


        # #plot occupancy with freezing in bins + total freezing in %
        # freez = sns.jointplot(x=x, y=y, kind='hist',
        #                    bins=(30, 30),
        #                    cmap='crest', cbar=False,
        #                    marginal_kws=dict(element='step'), color='#3b6978')
        # freez.set_axis_labels('x', 'y')
        # freez.ax_joint.axhline(37.5, color='grey', linestyle=':', linewidth=0.8)
        # freez.ax_joint.set_xlim(0, 25)
        # freez.ax_joint.set_ylim(0, 75)
        #
        # fig = freez.figure
        # fig.subplots_adjust(right=0.80)
        #
        # joint_pos = freez.ax_joint.get_position()
        # marg_pos = freez.ax_marg_y.get_position()
        #
        # bar_ax = fig.add_axes([marg_pos.x1 + 0.03, joint_pos.y0,
        #                        0.08, joint_pos.height])
        # bar_ax.bar(0, P_head_in_shock, width=0.6, color='k')
        # bar_ax.set_ylim(0, 100)
        # bar_ax.set_xticks([])
        # bar_ax.set_ylabel('Time Spent on Shock Side (in %)')
        # bar_ax.yaxis.set_label_position('right')
        # bar_ax.yaxis.tick_right()
        # freez.figure.suptitle(f'Occupancy for {rat} in {phase} (Shock side: {shock_side})')
        # fig.text(0.01, 0.01, f'n = {len(x)} frames',
        #          fontsize=7, color='grey', ha='left', va='bottom')
        #
        # freez.savefig(os.path.join(repo, 'occupancy_with_freezing.png'))
        # # g1.savefig(os.path.join(repo, 'occupancy_joint_quant.svg'))
        # plt.close(fig)



        #display data and save fig
        fig, ax = plt.subplots(1, 4, figsize=(20, 8), sharex=False, sharey=False)
        sns.histplot(np.degrees(head_direction), binwidth=10, ax=ax[0])
        ax[0].set(xlabel='head direction (degree)')
        coords = [c for c in overall_info['arena']['coords']]
        coords.append(coords[0])
        coords = np.array(coords)
        ax[1].plot(coords.T[0], coords.T[1], color='black')
        ax[1].plot(xy_head[:, 0], xy_head[:, 1],
                   'o', ms=2.0, color='blue', alpha=0.1, label='occupancy head')
        ax[1].plot(xy_snout[:, 0], xy_snout[:, 1],
                   'o', ms=2.0, color='green', alpha=0.1, label='occupancy snout')

        ax[1].plot(xy_head[np.abs(velocity) > 5, 0], xy_head[np.abs(velocity) > 5, 1],
                   ms=2.0, color='k', alpha=0.5, label='speed>10cm/s')
        ax[1].set(xlabel='X (cm)', ylabel='Y (cm)',
                  # ylim=(-10,85), xlim=(-20,45)
                  )
        ax[1].legend()

        ax[2].plot(np.linspace(-25, 100, 51)[:-1] - np.diff(np.linspace(-25, 100, 51))[0] / 2,
                   np.sum(occupancy, axis=0), label='head')
        ax[2].plot(np.linspace(-25, 100, 51)[:-1] - np.diff(np.linspace(-25, 100, 51))[0] / 2,
                   np.sum(occupancy_back, axis=0), label='back')

        ax[2].axvline(37.5, ls='--', lw=2.0, color='black')
        ax[2].legend()
        ax[2].set(xlabel='(cm)', ylabel='time(s)',
                  title="%time in shock, back: {:.2F}, head: {:.2F}".format(P_back_in_shock, P_head_in_shock),
                  ylim=(-10,85), xlim=(-20,45)
                  )

        sns.distplot( np.abs( velocity),
                     bins=np.linspace(0,50,201),
                     ax=ax[3] )
        ax[3].set(xlabel='speed',
                  xlim=(0,25), ylim=(0,0.5)
                 )

        sns.distplot(np.log10(np.abs(velocity[~np.isnan(velocity)])),
                     # bins=np.linspace(0,50,201),
                     ax=ax[3])
        ax[3].axvline(speed_th, lw=1.0, color='black')
        ax[3].set(xlabel='log(speed)',
                  title="% freezing = {:.2F}".format(freezing),
                  # xlim=(0,25), ylim=(0,0.5)
                  )
        fig.savefig(os.path.join(repo, 'position.png'),
                    transparent=False)
        plt.close()

        fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=False)
        ax[0].plot(time, np.abs(velocity), 'ro', markersize=1.0, color='grey')
        ax[0].plot(time, np.abs(raw_velocity), 'ro', markersize=1.0, color='grey')
        ax[0].axhline(0.05, color='red', alpha=0.5,
                      label='{:.2f}% below 0.05'.format(np.sum(np.abs(velocity) < 0.05) / len(velocity) * 100))
        ax[0].set(ylabel='speed (cm/s)', xlabel='')
        ax[0].legend()

        ax[1].plot(time, np.degrees(head_direction), 'ro', markersize=1.0, color='grey')
        ax[1].set(ylabel='head direction', xlabel='time(s)', ylim=(0, 360), xlim=(time[0], time[-1]))

        # ax[2].plot(time, np.min( dist2divider, axis=0), 'ro', markersize=1.0, color='grey' )
        # ax[2].set(ylabel='dist from divider', xlabel = 'time(s)', xlim=(time[0], time[-1]))

        fig.savefig(os.path.join(repo, 'speed_HeadDirection_dist.png'),
                    transparent=False)
        plt.close()

        # Occupancy Simple
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.heatmap(occupancy, cmap='crest', vmin=0, vmax=2)
        fig.savefig(os.path.join(repo, 'occupancy.png'),
                    transparent=False)
        fig.savefig(os.path.join(repo, 'occupancy.svg'),
                    transparent=False)
        plt.close()

        # Occupancy Simple: First half
        # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        # sns.heatmap(occupancy1, cmap='crest', vmin=0, vmax=2)
        # fig.savefig(os.path.join(repo, 'occupancy_firsthalf.png'),
        #             transparent=False)
        # plt.close()

        # Occupancy Simple: Second half
        # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        # sns.heatmap(occupancy2, cmap='crest', vmin=0, vmax=2)
        # fig.savefig(os.path.join(repo, 'occupancy_secondhalf.png'),
        #             transparent=False)
        # plt.close()

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

        # Occupancy with Histogram
        # valid = ~np.isnan(xy_head).any(axis=1)
        # x, y = xy_head[valid, 0], xy_head[valid, 1]

        x, y = xy_head[:, 0], xy_head[:, 1]

        sns.set_theme(style='ticks')
        g = sns.jointplot(x=x, y=y, kind='hist',
                          bins=(30, 30),
                          cmap='crest', cbar=True,
                          marginal_kws=dict(element='step'))
        g.set_axis_labels('x', 'y')
        g.ax_joint.set_xlim(0, 25)
        g.ax_joint.set_ylim(0, 75)
        fig.text(0.01, 0.01, f'n = {len(x)} frames',
                 fontsize=7, color='grey', ha='left', va='bottom')
        #g.ax_joint.plot(coords[:,0], coords[:, 1], 'k-', lw=1)
        #g.ax_joint.set_aspect('equal')
        #g.set_title(f'Occupancy for {phase} (Shock side: {shock_side})')
        g.savefig(os.path.join(repo, 'occupancy_joint.png'))
        g.savefig(os.path.join(repo, 'occupancy_joint.svg'))
        plt.close(g.figure)

        # Occupancy with Histogram + % in Shock Side barplot
        g1 = sns.jointplot(x=x, y=y, kind='hist',
                           bins=(30, 30),
                           cmap='crest', cbar=False,
                           marginal_kws=dict(element='step'), color='#3b6978')
        g1.set_axis_labels('x', 'y')
        g1.ax_joint.axhline(37.5, color='grey', linestyle=':', linewidth=0.8)
        g1.ax_joint.set_xlim(0, 25)
        g1.ax_joint.set_ylim(0, 75)

        fig = g1.figure
        fig.subplots_adjust(right=0.80)

        joint_pos = g1.ax_joint.get_position()
        marg_pos = g1.ax_marg_y.get_position()

        bar_ax = fig.add_axes([marg_pos.x1 + 0.03, joint_pos.y0,
                               0.08, joint_pos.height])
        bar_ax.bar(0, P_head_in_shock, width=0.6, color='k')
        bar_ax.set_ylim(0, 100)
        bar_ax.set_xticks([])
        bar_ax.set_ylabel('Time Spent on Shock Side (in %)')
        bar_ax.yaxis.set_label_position('right')
        bar_ax.yaxis.tick_right()
        g1.figure.suptitle(f'Occupancy for {rat} in {phase} (Shock side: {shock_side})')
        fig.text(0.01, 0.01, f'n = {len(x)} frames',
                 fontsize=7, color='grey', ha='left', va='bottom')

        g1.savefig(os.path.join(repo, 'occupancy_joint_quant.png'))
        #g1.savefig(os.path.join(repo, 'occupancy_joint_quant.svg'))
        plt.close(fig)


        # Occupancy with Histogram + % in Shock Side barplot: FIRST
        x_first, y_first = xy_head1[:, 0], xy_head1[:, 1]

        g2 = sns.jointplot(x=x_first, y=y_first, kind='hist',
                           bins=(30, 30),
                           cmap='flare', cbar=False,
                           marginal_kws=dict(element='step'), color='#edaf80')
        g2.set_axis_labels('x', 'y')
        g2.ax_joint.axhline(37.5, color='grey', linestyle=':', linewidth=0.8)
        g2.ax_joint.set_xlim(0, 25)
        g2.ax_joint.set_ylim(0, 75)

        fig = g2.figure
        fig.subplots_adjust(right=0.80)

        joint_pos = g2.ax_joint.get_position()
        marg_pos = g2.ax_marg_y.get_position()

        bar_ax = fig.add_axes([marg_pos.x1 + 0.03, joint_pos.y0,
                               0.08, joint_pos.height])
        bar_ax.bar(0, P_head_in_shock1, width=0.6, color='k')
        bar_ax.set_ylim(0, 100)
        bar_ax.set_xticks([])
        bar_ax.set_ylabel('Time Spent on Shock Side (in %)')
        bar_ax.yaxis.set_label_position('right')
        bar_ax.yaxis.tick_right()
        g2.figure.suptitle(f'Occupancy for {rat} in First Half of {phase} (Shock side: {shock_side})')
        fig.text(0.01, 0.01, f'n = {len(x_first)} frames',
                 fontsize=7, color='grey', ha='left', va='bottom')

        g2.savefig(os.path.join(repo, 'occupancy_joint_quant_first.png'))
        #g2.savefig(os.path.join(repo, 'occupancy_joint_quant_first.svg'))
        plt.close(fig)

        # Occupancy with Histogram + % in Shock Side barplot: SECOND
        x_second, y_second = xy_head2[:, 0], xy_head2[:, 1]

        g3 = sns.jointplot(x=x_second, y=y_second, kind='hist',
                           bins=(30, 30),
                           cmap='flare', cbar=False,
                           marginal_kws=dict(element='step'), color='#edaf80')
        g3.set_axis_labels('x', 'y')
        g3.ax_joint.axhline(37.5, color='grey', linestyle=':', linewidth=0.8)
        g3.ax_joint.set_xlim(0, 25)
        g3.ax_joint.set_ylim(0, 75)

        fig = g3.figure
        fig.subplots_adjust(right=0.80)

        joint_pos = g3.ax_joint.get_position()
        marg_pos = g3.ax_marg_y.get_position()

        bar_ax = fig.add_axes([marg_pos.x1 + 0.03, joint_pos.y0,
                               0.08, joint_pos.height])
        bar_ax.bar(0, P_head_in_shock2, width=0.6, color='k')
        bar_ax.set_ylim(0, 100)
        bar_ax.set_xticks([])
        bar_ax.set_ylabel('Time Spent on Shock Side (in %)')
        bar_ax.yaxis.set_label_position('right')
        bar_ax.yaxis.tick_right()
        g3.figure.suptitle(f'Occupancy for {rat} in Second Half of {phase} (Shock side: {shock_side})')
        fig.text(0.01, 0.01, f'n = {len(x_second)} frames',
                 fontsize=7, color='grey', ha='left', va='bottom')

        g3.savefig(os.path.join(repo, 'occupancy_joint_quant_second.png'))
        #g3.savefig(os.path.join(repo, 'occupancy_joint_quant_second.svg'))
        plt.close(fig)

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

        #smoothed kde
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
        plotts=ax2.scatter(xy_head[:, 0], xy_head[:, 1], c=kp, s=3, vmin=vmin, vmax=vmax)
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


        # plt.plot(thx_vel_fly0)
        # plt.savefig(os.path.join(repo, 'velocity_test.png'))
        # plt.show()







# EXPLORATORY STUFF
#         shock_side = rat_info['shock_side']
#
#         x0, x1 = coords[:, 0].min(), coords[:, 0].max()
#         y0, y1 = coords[:, 1].min(), coords[:, 1].max()
#
#         # --- occupancy histogram, arena extent only ---
#         valid = ~np.isnan(xy_head).any(axis=1)
#         xb = np.linspace(x0, x1, 26)  # ~1 cm bins along 25 cm
#         yb = np.linspace(y0, y1, 76)  # ~1 cm bins along 75 cm
#         counts, xe, ye = np.histogram2d(xy_head[valid, 0], xy_head[valid, 1], bins=[xb, yb])
#         occ = gaussian_filter(counts * (1 / 30), sigma=1.5)  # frames -> seconds, smoothed
#         occ = np.ma.masked_where(occ == 0, occ)
#
#
#         fig, ax = plt.subplots(figsize=(10, 3.8))  # landscape, ~3:1 like the picture
#         cmap = sns.color_palette("crest", as_cmap=True)  # light green -> dark blue
#
#         # plot UN-transposed with swapped extent: y horizontal, x vertical
#         im = ax.imshow(occ, origin='lower', extent=[y0, y1, x0, x1],
#                        cmap=cmap, aspect='equal', vmin=0, vmax=2)
#         plt.colorbar(im, ax=ax, label='Occupancy (s)', shrink=0.8)
#
#         # outline + 6-cell grid, with x and y swapped for the rotated frame
#         x_edges = np.linspace(x0, x1, 3)  # [0, 12.5, 25]  -> horizontal split lines
#         y_edges = np.linspace(y0, y1, 4)  # [0, 25, 50, 75] -> vertical split lines
#         for yd in y_edges:
#             ax.plot([yd, yd], [x0, x1], 'k-', lw=2)
#         for xd in x_edges:
#             ax.plot([y0, y1], [xd, xd], 'k-', lw=2)
#
#         # % per cell (text placed at swapped coords: horizontal=y, vertical=x)
#         counts_q, _, _ = np.histogram2d(xy_head[valid, 0], xy_head[valid, 1],
#                                         bins=[x_edges, y_edges])
#         pct_q = counts_q / counts_q.sum() * 100
#         for i in range(len(x_edges) - 1):
#             for j in range(len(y_edges) - 1):
#                 ax.text((y_edges[j] + y_edges[j + 1]) / 2,
#                         (x_edges[i] + x_edges[i + 1]) / 2,
#                         f'{pct_q[i, j]:.1f}%', ha='center', va='center',
#                         color='red', fontsize=11, fontweight='bold')
#         i, j = (1, 2) if shock_side == 'left' else (1, 0)  # left->top-right, right->bottom-right
#         ax.add_patch(mpatches.Rectangle(
#             (y_edges[j], x_edges[i]),  # lower-left corner (y horiz, x vert)
#             y_edges[j + 1] - y_edges[j],  # width  along y
#             x_edges[i + 1] - x_edges[i],  # height along x
#             fill=False, edgecolor='crimson', lw=4, zorder=5))
#
#         ax.set_title(f'Occupancy (in %) for {phase} ; shock side: {shock_side}', fontsize=11)
#
#         ax.set(xlabel='y (cm)', ylabel='x (cm)')  # axes are swapped now
#         fig.savefig(os.path.join(repo, 'occupancy_with%.png'), transparent=False)
#         plt.close()
#
#
#         arena = overall_info['arena']
#         coords = np.array(arena['coords'])
#
#         # arena bounds from YAML
#         x0, x1 = coords[:, 0].min(), coords[:, 0].max()  # 0, 25
#         y0, y1 = coords[:, 1].min(), coords[:, 1].max()  # 0, 75
#
#         # 6-cell grid edges: 2 across width (x), 3 along length (y)
#         x_edges = np.linspace(x0, x1, 3)  # [0, 12.5, 25]
#         y_edges = np.linspace(y0, y1, 4)  # [0, 25, 50, 75]
#
#         # drop NaN frames, then count
#         valid = ~np.isnan(xy_head).any(axis=1)
#         counts, _, _ = np.histogram2d(xy_head[valid, 0], xy_head[valid, 1],
#                                       bins=[x_edges, y_edges])
#
#         # counts is a (2, 3) array: counts[i, j] = frames in x-cell i, y-cell j
#         time_s = counts * sf  # frames -> seconds
#         pct = counts / counts.sum() * 100  # % of valid frames
#
#         labels = [['bottom-left', 'middle-left', 'top-left'],
#                   ['bottom-right', 'middle-right', 'top-right']]
#         for i in range(2):
#             for j in range(3):
#                 print(f'{labels[i][j]:>14}: {counts[i, j]:6.0f} frames  '
#                       f'{time_s[i, j]:6.1f} s  {pct[i, j]:5.1f}%')
#
#         print(f"total frames:        {len(xy_head)}")
#         print(f"valid (non-NaN):     {valid.sum()}")
#         print(f"counted in arena:    {int(counts.sum())}")
#         print(f"dropped (NaN/outside):{len(xy_head) - int(counts.sum())}")
#
#         # --- which cell is shock-paired ---
#         hi = (1, 2) if shock_side == 'left' else (1, 0)  # left->top-right, right->bottom-right
#
#         # --- plot (landscape: y horizontal, x vertical) ---
#         fig, ax = plt.subplots(figsize=(10, 3.6))
#         im = ax.imshow(pct, origin='lower', extent=[y0, y1, x0, x1],  # <- pct, NOT pct.T
#                        cmap='YlGnBu', aspect='equal', vmin=0, vmax=pct.max())
#
#         for yd in y_edges:
#             ax.plot([yd, yd], [x0, x1], color='0.35', lw=1)
#         for xd in x_edges:
#             ax.plot([y0, y1], [xd, xd], color='0.35', lw=1)
#
#         # per-cell labels — text color flips for readability on dark cells
#         thr = pct.max() * 0.55
#         for i in range(2):
#             for j in range(3):
#                 tc = 'white' if pct[i, j] > thr else '#08306b'
#                 ax.text((y_edges[j] + y_edges[j + 1]) / 2,
#                         (x_edges[i] + x_edges[i + 1]) / 2,
#                         f'{pct[i, j]:.1f}%\n{time_s[i, j]:.0f} s',
#                         ha='center', va='center', color=tc, fontsize=11, fontweight='bold')
#
#         # highlight shock-paired chamber with a thick outline
#         i, j = hi
#         ax.add_patch(mpatches.Rectangle(
#             (y_edges[j], x_edges[i]), y_edges[j + 1] - y_edges[j], x_edges[i + 1] - x_edges[i],
#             fill=False, edgecolor='crimson', lw=4, zorder=5))
#
#         cbar = fig.colorbar(im, ax=ax, label='% of frames', shrink=0.7, pad=0.02)
#         cbar.outline.set_visible(False)
#         ax.set(xlabel='y (cm)', ylabel='x (cm)')
#         ax.set_title(f'Occupancy for {phase} ; shock side: {shock_side}', fontsize=11)
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#
#         fig.savefig(os.path.join(repo, 'chamber_occupancy.png'),
#                     transparent=False, bbox_inches='tight', dpi=150)
#         plt.close()


import matplotlib.gridspec as gridspec

def _frz_in(mask):
    n = np.sum(mask)
    return (np.sum(frozen & mask) / n * 100) if n else np.nan

cats = [('top',            top,            'tab:blue'),
        ('center_between', center_between, 'tab:orange'),
        ('bottom',         bottom,         'tab:green'),
        ('leftover',       leftover,       'tab:red')]

arena_coords = np.array(overall_info['arena']['coords'])
ax0, ax1 = arena_coords[:, 0].min(), arena_coords[:, 0].max()
ay0, ay1 = arena_coords[:, 1].min(), arena_coords[:, 1].max()

fine_bins = [np.arange(bins[0][0], bins[0][-1] + 0.5  , 0.5  ),
             np.arange(bins[1][0], bins[1][-1] + 0.5  , 0.5  )]
occ_fine, _ = np.histogramdd(xy_head, bins=fine_bins)
occ_fine = occ_fine * sf
occ_fine = nanGaussianSmooth(occ_fine, sigma=0.8 / 0.5  , truncate=2)
vmax_fine = np.nanpercentile(occ_fine[occ_fine > 0], 99)   # or 95

extent = [fine_bins[0][0], fine_bins[0][-1], fine_bins[1][0], fine_bins[1][-1]]

fig = plt.figure(figsize=(9, 7))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 5, 0.4], wspace=0.1)
bax = fig.add_subplot(gs[0])
ax  = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])

im = ax.imshow(occ_fine.T, origin='lower', extent=extent,
               cmap='crest', aspect='equal', vmin=0, vmax=vmax_fine)

for ln in (top_chamber, bottom_chamber, line_midbtw):
    if ln is not None:
        ax.plot(ln[:, 0], ln[:, 1], color='k', ls='--', lw=1)

cx_col = (ax0 + ax1) / 2
y_mid  = (ay0 + ay1) / 2
dx     = 6

label_pos = {
    'top':            (cx_col-dx,      ay1 - 20),
    'bottom':         (cx_col-dx,      ay0 + 20),
    'leftover':       (cx_col - dx, y_mid),
    'center_between': (cx_col + dx, y_mid),
}

for name, mask, col in cats:
    if np.sum(mask) == 0:
        continue
    lx, ly = label_pos[name]
    ax.text(lx, ly, f'{_frz_in(mask):.1f}%', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

ax.set_xlim(ax0 - 2, ax1 + 2)
ax.set_ylim(ay0 - 2, ay1 + 2)
ax.set(xlabel='X (cm)', ylabel='Y (cm)', title=f'Occupancy & Freezing\n{rat} — {phase}')

fig.colorbar(im, cax=cax, orientation='vertical', label='occupancy (s)')

bax.bar(0, freezing, width=0.6, color='0.3')
bax.set_ylim(0, 100)
bax.set_xticks([])
bax.set_ylabel('Total % Freezing')
bax.text(0, freezing + 1.5, f'{freezing:.1f}%', ha='center', va='bottom', fontsize=9)

fig.savefig(os.path.join(repo, 'occupancy_freezing_by_quadrant.png'),
            bbox_inches='tight', dpi=150)
plt.close()