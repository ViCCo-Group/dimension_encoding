"""
Run an encoding model of our behavioral dimensions on the BOLD 5000 imagenet data.
This script uses fracrional ridge regression for obtaining regularized and more stable 
dimension tuning maps.
"""

from os.path import join as pjoin
import os
import argparse
import numpy as np
from scipy.stats import zscore
from dimension_encoding.nsd import NsdLoader
from dimension_encoding.glm import FracRidgeVoxelwise



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Replicate our behavioral dimensions brain fit on the NSD dataset"
    )
    parser.add_argument(
        "--subject", type=str, help="NSD subject id", required=True
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="path to output directory",
        default="../results/nsd_fracridge",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="path to b5k directory",
        default="../data",
    )
    parser.add_argument(
        "--zscore_X",
        action="store_true",
        default=True,
        help="zscore regressors",
    )
    parser.add_argument(
        "--zscore_y",
        action="store_true",
        default=True,
        help="zscore responses",
    )
    args = parser.parse_args()
    return args


def main(args):
    sub_outdir = pjoin(args.outdir, f"sub-{args.subject}")
    os.makedirs(sub_outdir, exist_ok=True)
    dl = NsdLoader(data_dir=args.data_dir)
    bmask = dl.get_bmask_file(args.subject)
    x_dims, y = dl.make_dimensions_model(args.subject)
    if args.zscore_X:
        print("zscoring X")
        x_dims = zscore(x_dims, axis=0)
    print("loading responses")
    if args.zscore_y:
        print("zscoring y")
        y = zscore(y, axis=0)
    if args.zscore_X:
        print("zscoring X")
        x_dims = zscore(x_dims, axis=0)
    print("Start ridge regression")
    fr = FracRidgeVoxelwise(
        n_splits=10,
        test_size=0.0, 
        fracs=np.arange(0.01, 1.01, 0.01),
        run_pcorr=False,
    )
    betas, _, _, best_fracs, _ = fr.tune_and_eval(x_dims, y)
    print("save betas")
    for dim_i, b_dim in enumerate(betas.T):
        betas_img = unmask(b_dim, bmask)
        betas_img.to_filename(pjoin(sub_outdir, f"betas_dim-{dim_i+1}.nii.gz"))
    print("save regularization parameters")
    fracs_img = unmask(best_fracs, bmask)
    fracs_img.to_filename(pjoin(sub_outdir, f"best_fracs.nii.gz"))
    print("Done")
    return None


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
