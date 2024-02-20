import os
import pandas as pd
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
from nilearn.masking import unmask
import argparse
from sklearn.metrics import roc_auc_score

from dimension_encoding.utils import (
    load_singletrial_data,
    load_category_df,
    load_clip66_preds,
    load_facesbodies,
    regress_out,
    get_bmask,
    neg2zero,
)
from dimension_encoding.glm import LinRegCVPermutation


def load_category_labels(
    cat_tsv,
    exclude=["fastener", "personal hygiene item", "arts and crafts supply"],
):
    # get category labels for THINGS concepts
    allcats = load_category_df(cat_tsv)
    allcats["uniqueID"] = allcats.uniqueID.str.replace(" ", "_")  # change white spaces
    # allcats = allcats.drop(columns='Definition (from WordNet, Google, or Wikipedia)')
    allcats = allcats.drop(columns="Word")
    # exclude odd categories
    allcats = allcats.drop(columns=exclude)
    return allcats


def load_secondary_object_labels(l2s_tsv):
    """As generated by human annotator"""
    l2s = pd.read_csv(l2s_tsv, sep="\t")
    l2s["basename"] = l2s.image.str.split("/").str[-1]
    return l2s


def make_category_model(stimdata, allcats, l2s, verbose=False):
    catvecs = []
    n_ims_with_two_objs = 0
    for stim in tqdm(stimdata.stimulus):
        conc = stim[:-8]
        # find categories for this concept
        match = allcats.loc[allcats.uniqueID == conc]
        assert len(match) == 1
        match = match.drop(columns=["uniqueID"])
        catvec = np.nan_to_num(match.to_numpy())
        # if there is a second object label for this image
        if stim in l2s.basename.unique():
            obj2 = l2s.loc[l2s.basename == stim, "label"]
            assert len(obj2) == 1
            obj2 = obj2.values[0]
            if obj2 != "INVALID":
                n_ims_with_two_objs += 1
                match2 = allcats.loc[allcats.uniqueID == obj2]
                assert len(match2) == 1
                catvec2 = np.nan_to_num(match2.drop(columns="uniqueID").to_numpy())
                # take union of both category vectors
                catvec = np.logical_or(catvec, catvec2)
        catvecs.append(catvec)
    X_cat = np.array(catvecs).squeeze()
    if verbose:
        print(f"found {n_ims_with_two_objs} images with two objects in fMRI stimuli")
    return X_cat


def make_facesbodies_model(facesbodies_df, stimdata):
    X_facesbodies = np.zeros((len(stimdata), 2))
    for trial_i, trial_img in tqdm(enumerate(stimdata.stimulus), total=len(stimdata)):
        rows = facesbodies_df.loc[facesbodies_df.image_name == trial_img]
        assert len(rows) > 0, f"Could not find image {trial_img} in faces/bodies labels"
        X_facesbodies[rows.index] = rows[["face", "body"]].to_numpy()
    return X_facesbodies


def make_dimensions_model(
    stimdata, embedding, things_filenames, dim_labels, allcats, X_cat, verbose=False
):
    # find embedding for fmri images
    X_dims_ = np.full(shape=(len(stimdata), embedding.shape[1]), fill_value=np.nan)
    for trial_i, trial_fn in tqdm(enumerate(stimdata.stimulus), total=len(stimdata)):
        things_i = np.where(things_filenames == trial_fn)
        assert len(things_i[0]) == 1
        X_dims_[trial_i] = embedding[things_i]
    # find most diagnostic dimensions for each category
    catnames = allcats.drop(columns="uniqueID").columns
    ncats, ndims = X_cat.shape[1], X_dims_.shape[1]
    scores = np.full(shape=(ncats, ndims), fill_value=np.nan)
    for cat_i, (y, catname) in tqdm(enumerate(zip(X_cat.T, catnames)), total=ncats):
        for dim_i, (x, dimname) in enumerate(zip(X_dims_.T, dim_labels)):
            scores[cat_i, dim_i] = roc_auc_score(y, x)
    # select best dimensions and print some info
    best_dims_is = np.argmax(scores, axis=1)
    best_dims = dim_labels[best_dims_is]
    unique_dim_is = np.unique(best_dims_is)
    best_dims_unique = dim_labels[unique_dim_is]
    if verbose:
        for catname, dimname in zip(catnames, best_dims):
            print(f"{catname}: {dimname}")
        print(f"{len(best_dims_unique)} dimensions in final model: ", best_dims_unique)
    # dimension model
    X_dims = X_dims_[:, unique_dim_is]
    return X_dims


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run variance partitioning based on a cross-validated regression to compare a categorical and dimension model"
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
        "--include_faces_bodies",
        help='Whether to include manual face and body labels into the category model. Will be loaded from command line argument "facesbodiesdir"',
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--only_faces_bodies",
        help="If true, use only manual face and body labels as category model.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--facesbodiesdir",
        help="Where to load the face and body labels from. Only applies if --include_faces_bodies is specified",
        default="../data/facesbodies",
        type=str,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="path to output directory",
        default="../results/varpart_cat_vs_dim_CV",
    )
    parser.add_argument(
        "--l2s_tsv",
        type=str,
        help="path to file containing secondary object labels",
        default="../data/matchlabels/results/output_final.tsv",
    )
    parser.add_argument(
        "--clip66dir",
        type=str,
        help="path to clip-predicted behavioral embeddings (66d)",
        default="../data/66d",
    )
    parser.add_argument(
        "--cat_tsv",
        type=str,
        help="path to tsv containing high-level category labels",
        default="../data/Categories_final_20200131_fixedUniqueID.tsv",
    )
    args = parser.parse_args()
    return args


def main(
    args,
    metric="pearsonr",
    exclude_cats=["fastener", "personal hygiene item", "arts and crafts supply"],
):
    os.makedirs(args.outdir, exist_ok=True)
    print("loading brain data")
    brainmask = get_bmask(args.sub, args.bidsroot)
    respdata, _, stimdata = load_singletrial_data(args.bidsroot, args.sub)
    y = respdata.to_numpy().T
    stimdata["concept"] = stimdata.stimulus.str.split("_").str[0]
    print("making category model")
    allcats = load_category_labels(args.cat_tsv, exclude_cats)
    l2s = load_secondary_object_labels(args.l2s_tsv)
    X_cat = make_category_model(stimdata, allcats, l2s)
    print("making dimension model")
    embedding, things_filenames, dim_labels = load_clip66_preds(args.clip66dir)
    X_dims = make_dimensions_model(
        stimdata, embedding, things_filenames, dim_labels, allcats, X_cat
    )
    if args.only_faces_bodies:
        print("Only using faces and bodies as category model")
        # in this case, we use the same dimension model as in all other circumstances, i.e.
        # selected dimensions based on diagnosticity for superordinare categories
        facesbodies = load_facesbodies(args.facesbodiesdir)
        X_cat = make_facesbodies_model(facesbodies, stimdata)
    if args.include_faces_bodies:
        print("Including manual face and body labels into superordinate category model")
        facesbodies = load_facesbodies(args.facesbodiesdir)
        X_facesbodies = make_facesbodies_model(facesbodies, stimdata)
        X_cat = np.hstack([X_cat, X_facesbodies])
    X_comb = np.concatenate([X_dims, X_cat], axis=1)  # combined model
    print("orthogonalizing X")
    X_cat_r = regress_out(X_dims, X_cat, dtype=np.double)
    X_dims_r = regress_out(X_cat, X_dims, dtype=np.double)
    if args.adjust_y:
        print("orthogonalizing y")
        y_no_cat = regress_out(X_cat, y, dtype=np.double)
        y_no_dims = regress_out(X_dims, y, dtype=np.double)
    else:
        print("not orthogonalizing y")
        y_no_cat = y
        y_no_dims = y
    # fit models
    nfolds = 1 if args.nocv else 12
    model = LinRegCVPermutation(nperm=0, metric=metric, nfolds=nfolds)
    print("Estimating unique variance for categories")
    r2u_cat, _ = model.fit_and_permute(X_cat_r, y_no_dims)
    print("Estimating unique variance for dimensions")
    r2u_dim, _ = model.fit_and_permute(X_dims_r, y_no_cat)
    print("Fitting combined model")
    r2_comb, _ = model.fit_and_permute(X_comb, y)
    print("compute shared variance")
    # set negative r2 to 0 before subtracting
    r2_comb, r2u_cat, r2u_dim = neg2zero(r2_comb), neg2zero(r2u_cat), neg2zero(r2u_dim)
    if metric == "pearsonr":
        r2_comb, r2u_cat, r2u_dim = r2_comb**2, r2u_cat**2, r2u_dim**2
    r2_shared = r2_comb - (r2u_cat + r2u_dim)
    # save data
    for arr, nam in zip(
        [r2_comb, r2_shared, r2u_cat, r2u_dim],
        ["r2_comb", "r2_shared", "r2u_cat", "r2u_dim"],
    ):
        img = unmask(arr, brainmask)
        fn = pjoin(args.outdir, f"sub-{args.sub}_{nam}.nii.gz")
        print("saving ", fn)
        img.to_filename(fn)
    return None


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

