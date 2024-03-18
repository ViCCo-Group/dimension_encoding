import os
import numpy as np
from os.path import join as pjoin
from nilearn.masking import unmask
import argparse

from dimension_encoding.shapemodel import (
    load_shapecomps,
    make_shape_model,
    find_embedding_fmritrials,
)
from dimension_encoding.utils import (
    load_singletrial_data,
    load_clip66_preds,
    regress_out,
    get_bmask,
    neg2zero,
)
from dimension_encoding.glm import LinRegCVPermutation


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run variance partitioning based on a cross-validated regression to compare a shape-based and a behavioral dimension model"
    )
    parser.add_argument("--sub", type=str, help="subject id", required=True)
    parser.add_argument(
        "--bidsroot",
        type=str,
        help="path to bids root directory",
        default="../data/thingsfmri/bids",
    )
    parser.add_argument(
        "--adjust_y",
        type=bool,
        help="Whether to also regress confound model out of fMRI data, before determining variance explained by model of interest",
        default=True,
    )
    parser.add_argument(
        "--nocv",
        help="whether to use within-sample instead of cross-validated r-squared. Turns this into plain regression.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="path to output directory",
        default="../results/varpart_shape_vs_dim_CV",
    )
    parser.add_argument(
        "--clip66dir",
        type=str,
        help="path to clip-predicted behavioral embeddings (66d)",
        default="../data/66d",
    )
    parser.add_argument(
        "--shapecompdir",
        type=str,
        help="path to shape component data for segmented things images",
        default="../data/ShapeComp",
    )
    args = parser.parse_args()
    return args

def main(
    args,
    metric="pearsonr",
):
    os.makedirs(args.outdir, exist_ok=True)
    print("loading brain data")
    brainmask = get_bmask(args.sub, args.bidsroot)
    respdata, _, stimdata = load_singletrial_data(args.bidsroot, args.sub)

    print("making shape model")
    shapecomps, shapecomp_fnames = load_shapecomps(args.shapecompdir)
    shapecomp_model_info = make_shape_model(stimdata, shapecomps, shapecomp_fnames)
    X_shape = shapecomp_model_info['design_matrix']
    val_inds = shapecomp_model_info['val_trial_inds']
    # filter fMRI trials based on shape model
    y = respdata.to_numpy().T[val_inds]
    stimdata_val = stimdata.iloc[val_inds]
    print(f"{respdata.shape[1] - len(val_inds)} trials removed from fMRI data due to missing shape model")
    import ipdb;ipdb.set_trace()

    print("making dimension model")
    # embedding, things_filenames, dim_labels = load_clip66_preds(args.clip66dir)
    trials = find_embedding_fmritrials(stimdata_val, args.clip66dir)
    X_dims = trials.drop(columns=["imagename"], inplace=False).to_numpy()

    # combined model
    X_comb = np.concatenate([X_dims, X_shape], axis=1)
    print("orthogonalizing X")
    X_shape_r = regress_out(X_dims, X_shape, dtype=np.double)
    X_dims_r = regress_out(X_shape, X_dims, dtype=np.double)
    if args.adjust_y:
        print("orthogonalizing y")
        y_no_shape = regress_out(X_shape, y, dtype=np.double)
        y_no_dims = regress_out(X_dims, y, dtype=np.double)
    else:
        print("not orthogonalizing y")
        y_no_shape = y
        y_no_dims = y
    # fit models
    # 10-fold CV because not all fMRI trials have shape model and n is no longer divisible by 12
    nfolds = 1 if args.nocv else 12
    model = LinRegCVPermutation(nperm=0, metric=metric, nfolds=nfolds)
    print("Estimating unique variance for shapes")
    r2u_shape, _ = model.fit_and_permute(X_shape_r, y_no_dims)
    print("Estimating unique variance for dimensions")
    r2u_dim, _ = model.fit_and_permute(X_dims_r, y_no_shape)
    print("Fitting combined model")
    r2_comb, _ = model.fit_and_permute(X_comb, y)
    print("compute shared variance")
    # set negative r2 to 0 before subtracting
    r2_comb, r2u_shape, r2u_dim = (
        neg2zero(r2_comb),
        neg2zero(r2u_shape),
        neg2zero(r2u_dim),
    )
    if metric == "pearsonr":
        r2_comb, r2u_shape, r2u_dim = r2_comb**2, r2u_shape**2, r2u_dim**2
    r2_shared = r2_comb - (r2u_shape + r2u_dim)
    # save data
    for arr, nam in zip(
        [r2_comb, r2_shared, r2u_shape, r2u_dim],
        ["r2_comb", "r2_shared", "r2u_shape", "r2u_dim"],
    ):
        img = unmask(arr, brainmask)
        fn = pjoin(args.outdir, f"sub-{args.sub}_{nam}.nii.gz")
        print("saving ", fn)
        img.to_filename(fn)
    return None


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
