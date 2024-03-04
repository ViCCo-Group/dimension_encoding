from os.path import join as pjoin
import os
from dimension_encoding.b5k import B5kLoader
from scipy.stats import zscore
from dimension_encoding.glm import LinRegCVPermutation
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Replicate our behavioral dimensions brain fit on the Bold 5000 dataset"
    )
    parser.add_argument("--subject", type=str, help="BOLD5000 subject id", required=True)
    parser.add_argument(
        "--outdir",
        type=str,
        help="path to output directory",
        default="../results/b5k_linereg_cvperm",
    )
    parser.add_argument(
        "--b5k_dir",
        type=str,
        help="path to b5k directory",
        default="../data/b5k/",
    )
    parser.add_argument(
        "--image_sets",
        type=list,
        help="Which image sets to include",
        choices=['imagenet', 'coco'],
        default=['imagenet', 'coco']
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
    dl = B5kLoader(b5k_dir=args.b5k_dir)
    print("making dimensions model")
    X_dims, y = dl.make_dimensions_model(args.subject, args.image_sets)
    if args.zscore_X:
        print("zscoring X")
        X_dims = zscore(X_dims, axis=0)
    print("loading responses")
    if args.zscore_y:
        print("zscoring y")
        y = zscore(y, axis=0)
    print("Estimate prediction accuracy")
    model = LinRegCVPermutation(nfolds=7, nperm=0)
    r, pval = model.fit_and_permute(X_dims, y)
    rimg = dl.array_to_volume(r, args.subject)
    rimg.to_filename(pjoin(sub_outdir, f"sub-{args.subject}_r.nii.gz"))
    print("save prediction accuracy to disk")
    if pval:
        pimg = dl.array_to_volume(pval, args.subject)
        pimg.to_filename(pjoin(sub_outdir, f"sub-{args.subject}_p.nii.gz"))
    print("fit again to obtain voxel-wise dimension weights")
    model.lr.fit(X_dims, y)
    print("save betas")
    for dim_i, betas in enumerate(model.lr.coef_.T):
        betas_img = dl.array_to_volume(betas, args.subject)
        betas_img.to_filename(pjoin(sub_outdir, f"betas_dim-{dim_i+1}.nii.gz"))
    return None


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
