import numpy as np
from scipy.stats import norm
from nilearn.masking import apply_mask, unmask
# from nilearn.image import new_img_like, smooth_img
# import scipy
# from scipy.ndimage import label


def hoyer_sparseness(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculate sparseness of x over given axis.
    """
    n = np.invert(np.isnan(x)).sum(axis=axis)  # n of not nan in each voxel
    l1 = np.nansum(np.abs(x), axis=axis)
    l2 = np.sqrt(np.nansum(x**2, axis=axis))
    s = (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)
    return s


def sparseness_zp_noisepool(svol, r2vol, maskvol, r2thresh=0.0, return_data=False):
    """
    Calculate z- and p-values for voxel-wise sparseness values based on a noisepool of voxels
    determined based on the r-square of the encoding model.
    """
    # define noise/signal pool
    r2 = apply_mask(r2vol, maskvol)
    noise_mask = r2 < r2thresh
    signal_mask = r2 >= r2thresh
    # get sparseness in noise mask, log transform, calculate mean and sd
    s = apply_mask(svol, maskvol)
    s_noise_ = np.log(s[noise_mask])
    val_mask_ = np.invert(np.isinf(s_noise_))  # ignore infs
    s_noise = s_noise_[val_mask_]
    noise_m, noise_sd = s_noise.mean(), np.std(s_noise)
    # get sparseness in signal pool, log transform
    s_signal_ = np.log(s[signal_mask])
    val_mask = np.invert(np.isinf(s_signal_))
    s_signal = s_signal_[val_mask]
    # z value of signal voxels wrt noise distribution
    s_z_ = (s_signal - noise_m) / noise_sd
    # bring into original shape, save as nifti
    fillval = s_z_.min()
    s_z_sigmask = np.full(val_mask.shape, fillval)
    s_z_sigmask[val_mask] = s_z_
    s_z = np.full(s.shape, fillval)
    s_z[signal_mask] = s_z_sigmask
    s_z_img = unmask(s_z, maskvol)
    # transform to p value
    s_p = norm.cdf(s_z)
    s_p_img = unmask(s_p, maskvol)
    # return nilearn image objects
    if return_data:
        return s_z_img, s_p_img, s_signal, s_noise
    else:
        return s_z_img, s_p_img


# def filter_clusters(pimg, pthresh=0.95, min_size=10, smooth_pimg=0.0):
#     pimg_ = smooth_img(pimg, smooth_pimg) if smooth_img else pimg
#     parr = pimg_.get_fdata()
#     # find voxels above p threshold
#     sig = parr > pthresh
#     # Label each cluster of connected True values
#     labeled_array, num_features = label(sig)
#     # Count the size of each cluster
#     with scipy.ndimage.sum as sum:
#         cluster_sizes = sum(sig, labeled_array, range(1, num_features + 1))
#     # Create a boolean mask for clusters that are too small
#     too_small = np.isin(labeled_array, np.where(cluster_sizes < min_size))
#     # Apply the mask to the original array
#     sig[too_small] = False
#     sigimg = new_img_like(pimg, sig)
#     return sigimg
