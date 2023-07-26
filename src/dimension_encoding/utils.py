import numpy as np
import warnings
from os.path import join as pjoin
import pandas as pd
import scipy
import requests
import io
import glob
from sklearn.linear_model import LinearRegression


def df_from_url(url: str, sep: str, header):
    s = requests.get(url).content
    return pd.read_csv(io.StringIO(s.decode("utf-8")), sep=sep, header=header)


def regress_out(
    x: np.ndarray,
    y: np.ndarray,
    dtype=np.single,
    lr_kws: dict = dict(copy_X=True, fit_intercept=True, n_jobs=-1),
) -> np.ndarray:
    reg = LinearRegression(**lr_kws)
    reg.fit(x, y)
    resid = y - reg.predict(x)
    return resid.astype(dtype)


def pearsonr_nd(arr1: np.ndarray, arr2: np.ndarray, alongax: int = 0) -> np.ndarray:
    """
    Pearson correlation between respective variables in two arrays.
    arr1 and arr2 are 2d arrays. Rows correspond to observations, columns to variables.
    Returns:
        correlations: np.ndarray (shape nvariables)
    """
    # center each feature
    arr1_c = arr1 - arr1.mean(axis=alongax)
    arr2_c = arr2 - arr2.mean(axis=alongax)
    # get sum of products for each voxel (numerator)
    numerators = np.sum(arr1_c * arr2_c, axis=alongax)
    # denominator
    arr1_sds = np.sqrt(np.sum(arr1_c**2, axis=alongax))
    arr2_sds = np.sqrt(np.sum(arr2_c**2, axis=alongax))
    denominators = arr1_sds * arr2_sds
    # for many voxels, this division will raise RuntimeWarnings for divide by zero. Ignore these.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = numerators / denominators
    return r


def load_clip66_preds(
    dim66_dir,
    fnames_with_folder_structure=False,
    emb_f=None,  # only needed if you want a special file name for the embedding, e.g. when embedding is saved for different DNNs.
):
    """
    Load predicted image-wise embedding of the 66 dimensional SPOSE model.

    returns:
        embedding: np.array of float values, shape (26107, 66)
            Dimensional embedding of each THINGS image on each of the 66 dimensions.
        filenames: np.array of string values, shape (26107,)
            File name of each things image (e.g. 'aardvark_01b.jpg').
            If input argumenet fnames_with_folder_structure is set to True, filenames
            will include folder hierarchy of the THINGS image database .
            (e.g. './images/aardvark/aardvark_01b.jpg')
        labels: np.array of string values, shape (66,)
            Label given to each dimension manually for interpretation.
    """
    # paths to relevant directory content
    if not emb_f:
        emb_f = "predictions_66d_elastic_clip-ViT_visual_THINGS.txt"
    emb_f = pjoin(dim66_dir, emb_f)
    lab_f = pjoin(dim66_dir, "labels_short.txt")
    fnames_f = pjoin(dim66_dir, "file_names_THINGS.txt")
    # load embedding
    embedding = np.loadtxt(emb_f)
    # load labels
    with open(lab_f, "r") as f:
        labels = np.array(f.readlines())
    labels = np.array([l.replace("\n", "") for l in labels])
    # load to filenames in desired format
    filenames = np.loadtxt(fnames_f, dtype=str)
    if fnames_with_folder_structure:
        concepts = [fn[:-8] for fn in filenames]
        filenames = np.array(
            [
                f"./images/{concept}/{filename}"
                for concept, filename in zip(concepts, filenames)
            ]
        )
    return embedding, filenames, labels


def load_singletrial_data(bidsroot, sub, drop_voxelid=True):
    betas_csv_dir = pjoin(bidsroot, "derivatives", "betas_csv")
    print(f"loading responses")
    respdata = pd.read_hdf(pjoin(betas_csv_dir, f"sub-{sub}_ResponseData.h5"))
    if drop_voxelid:
        print('dropping "voxel_id" column from response data')
        respdata = respdata.drop(columns="voxel_id")
    print(f"loading meta data")
    voxdata = pd.read_csv(pjoin(betas_csv_dir, f"sub-{sub}_VoxelMetadata.csv"))
    stimdata = pd.read_csv(pjoin(betas_csv_dir, f"sub-{sub}_StimulusMetadata.csv"))
    return respdata, voxdata, stimdata


def load_category_df(
    cat_tsv: str = "/LOCAL/ocontier/thingsmri/thingsmri-metadata/Categories_final_20200131.tsv",
) -> pd.DataFrame:
    # redundant_column = 'Definition (from WordNet, Google, or Wikipedia)' # used to be in older version of the tsv file
    df = pd.read_csv(cat_tsv, sep="\t")
    # if redundant_column in df.columns:
    #     df = df.drop(columns=redundant_column)
    return df


def neg2zero(x: np.array):
    x[x < 0] = 0
    return x


def psc(a: np.array, timeaxis: int = 0) -> np.array:
    """rescale array with fmri data to percent signal change (relative to the mean of each voxel time series)"""
    return 100 * ((a / a.mean(axis=timeaxis)) - 1)


def ci_array(a, confidence=0.95, alongax=0):
    """Returns tuple of upper and lower CI for mean along some axis in multidimensional array"""
    m, se = np.mean(a), scipy.stats.sem(a, axis=alongax)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, a.shape[alongax] - 1)
    return m - h, m + h


def get_hrflib(
    hrflib_url: str,
    rescale_amplitude: bool = True,
    resample_to_tr: bool = False,
    tr: float = 1.5,
    dtype=np.single,
) -> np.ndarray:
    """
    Get HRF library from Kendrick Kay's github repository.
    optionally rescale amplitudes of all HRFs to 1 (recommended) and resample to a specific TR (not recommended).
    """
    hrflib = np.array(df_from_url(hrflib_url, sep="\t", header=None))
    if resample_to_tr:  # resample to our TR
        sampleinds = np.arange(0, hrflib.shape[0], tr * 10, dtype=np.int16)
        hrflib = hrflib[sampleinds, :]
    if rescale_amplitude:  # rescale all HRFs to a peak amplitude of 1
        hrflib = hrflib / np.max(hrflib, axis=0)
    return hrflib.astype(dtype)


def vcorrcoef(X, y):
    """
    Calculate correlation between a vector y (size 1 x k) and each row in a matrix X (size N x k).
    Returns a vector of correlation coefficients r (size N x 1).
    """
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
    r = r_num / r_den
    return r


def get_prf_rois(sub, bidsroot, prf_derivname) -> dict:
    """
    Get file names of early visual ROIs deliniated with neuropythy.
    """
    rois = {}
    prf_dir = pjoin(bidsroot, "derivatives", prf_derivname, f"sub-{sub}")
    for va in range(1, 4):
        rois[f"V{va}"] = pjoin(prf_dir, f"resampled_va-{va}_interp-nn.nii.gz")
    rois["hV4"] = pjoin(prf_dir, f"resampled_va-4_interp-nn.nii.gz")
    rois["VO1"] = pjoin(prf_dir, f"resampled_va-5_interp-nn.nii.gz")
    rois["VO2"] = pjoin(prf_dir, f"resampled_va-6_interp-nn.nii.gz")
    rois["LO1 (prf)"] = pjoin(prf_dir, f"resampled_va-7_interp-nn.nii.gz")
    rois["LO2 (prf)"] = pjoin(prf_dir, f"resampled_va-8_interp-nn.nii.gz")
    rois["TO1"] = pjoin(prf_dir, f"resampled_va-9_interp-nn.nii.gz")
    rois["TO2"] = pjoin(prf_dir, f"resampled_va-10_interp-nn.nii.gz")
    rois["V3b"] = pjoin(prf_dir, f"resampled_va-11_interp-nn.nii.gz")
    rois["V3a"] = pjoin(prf_dir, f"resampled_va-12_interp-nn.nii.gz")
    return rois


def get_category_rois(sub, bidsroot, julian_derivname) -> dict:
    """
    Get file names of category seletive ROIS determined with a GLM on the localizer data.
    """
    julian_dir = pjoin(bidsroot, "derivatives", julian_derivname, f"sub-{sub}")
    rois = {}
    roinames = ["EBA", "FFA", "OFA", "STS", "PPA", "RSC", "TOS", "VWFA"]
    hemispheric_rois = []
    for roiname in roinames:
        hemispheric_rois += [f"l{roiname}"]
        hemispheric_rois += [f"r{roiname}"]
    roinames += hemispheric_rois
    # LOC added manually
    for roiname in roinames:
        found = glob.glob(pjoin(julian_dir, "*", f"*{roiname}*"))
        if found:
            rois[roiname] = found[0]
    for hemi in ["l", "r"]:  # LOC is in different directory
        rois[f"{hemi}LOC"] = pjoin(
            julian_dir.replace("_edited", "_intersected"),
            "object_parcels",
            f"sub-{sub}_{hemi}LOC.nii.gz",
        )
    return rois


def get_all_roi_files(sub, bidsroot, prf_derivname, julian_derivname) -> dict:
    """
    Returns a dict with roinames as keys and file names as values.
    category ROIs are separate per hemisphere, PRF rois are bihemispheric.
    """
    prf_rois = get_prf_rois(sub, bidsroot, prf_derivname)
    cat_rois = get_category_rois(sub, bidsroot, julian_derivname)
    # combine two dictionaries
    rois = {**prf_rois, **cat_rois}
    return rois
