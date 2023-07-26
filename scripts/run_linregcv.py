from tqdm import tqdm
from os.path import join as pjoin
import pandas as pd
import os
from scipy.stats import zscore
import argparse
from nilearn.masking import unmask
import logging

from dimension_encoding.utils import load_clip66_preds, load_singletrial_data
from dimension_encoding.glm import LinRegCVPermutation


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run cross-validated linear regression where p-value is determiend by permutation in each fold"
    )
    parser.add_argument("--sub", type=str, help="subject id", required=True)
    parser.add_argument(
        "--bidsroot",
        type=str,
        help="path to bids root directory",
    )
    parser.add_argument(
        "--clip66dir", type=str, help="path to object dimensions", default="./data/66d"
    )
    parser.add_argument(
        "--brainmask",
        type=str,
        help="path to brain mask volume",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="path to output directory",
        default="./clippreds_on_betas_linereg_cvperm",
    )
    args = parser.parse_args()
    return args


def find_embedding_fmritrials(
    stimdata,
    dim66_dir,
):
    emb, fnames, labels = load_clip66_preds(
        fnames_with_folder_structure=False, dim66_dir=dim66_dir
    )
    clip_df = pd.DataFrame(emb)
    clip_df.set_axis(labels, axis=1, inplace=True)
    clip_df["imagename"] = fnames
    trialrows = [
        clip_df.loc[clip_df.imagename == im]
        for im in tqdm(
            stimdata.stimulus, leave=True, desc="find dimensions for fMRI stimuli"
        )
    ]
    trials = pd.concat(trialrows)
    assert trials.shape[0] == stimdata.shape[0]
    return trials


def main(args):
    sub_outdir = pjoin(args.outdir, f"sub-{args.sub}")
    os.makedirs(sub_outdir, exist_ok=True)
    logging.basicConfig(level=logging.ERROR)  # else joblib prints too much
    # load data
    respdata, _, stimdata = load_singletrial_data(args.bidsroot, args.sub)
    y = respdata.to_numpy().T
    # load clip embedding
    trials = find_embedding_fmritrials(stimdata, args.clip66dir)
    # make design matrix
    X = trials.drop(columns=["imagename"])
    X = zscore(X, axis=0)
    X = X.to_numpy()
    # run analysis
    model = LinRegCVPermutation()
    r, pval = model.fit_and_permute(X, y)
    # save output
    rimg = unmask(r, args.brainmask)
    pimg = unmask(pval, args.brainmask)
    rimg.to_filename(pjoin(sub_outdir, f"sub-{args.sub}_r.nii.gz"))
    pimg.to_filename(pjoin(sub_outdir, f"sub-{args.sub}_p.nii.gz"))
    return None


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
