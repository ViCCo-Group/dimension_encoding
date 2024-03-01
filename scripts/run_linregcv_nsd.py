import os
import argparse
from scipy.stats import zscore
from nilearn.masking import unmask
from os.path import join as pjoin
from dimension_encoding.nsd import NsdLoader
from dimension_encoding.glm import LinRegCVPermutation


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Replicate our behavioral dimensions brain fit on the NSD dataset"
    )
    parser.add_argument(
        "--subject", type=str, help="NSD subject id (e.g. 01)", required=True
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="path to output directory",
        default="../results/nsd_linereg_cvperm",
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
    parser.add_argument(
        "--njobs",
        type=int,
        default=10,
        help="number of jobs to use for loading the session betas",
    )
    return parser.parse_args()


def main(args):
    sub_outdir = pjoin(args.outdir, f"sub-{args.subject}")
    os.makedirs(sub_outdir, exist_ok=True)
    dl = NsdLoader(data_dir=args.data_dir)
    print("loading dimensions")
    x_dims = dl.make_dimensions_model(args.subject)
    if args.zscore_X:
        print("zscoring X")
        x_dims = zscore(x_dims, axis=0)
    print("loading responses")
    y = dl.load_betas(args.subject)
    if args.zscore_y:
        print("zscoring y")
        y = zscore(y, axis=0)
    print("Estimate prediction accuracy")
    model = LinRegCVPermutation(nfolds=40, nperm=0)
    r, pval = model.fit_and_permute(x_dims, y)
    bmask = dl.get_bmask_file(args.subject)
    rimg = unmask(r, bmask)
    rimg.to_filename(pjoin(sub_outdir, f"sub-{args.subject}_r.nii.gz"))
    print("save prediction accuracy to disk")
    if pval:
        pimg = unmask(pval, bmask)
        pimg.to_filename(pjoin(sub_outdir, f"sub-{args.subject}_p.nii.gz"))
    print("fit again to obtain voxel-wise dimension weights")
    model.lr.fit(x_dims, y)
    print("save betas")
    for dim_i, betas in enumerate(model.lr.coef_.T):
        betas_img = unmask(betas, bmask)
        betas_img.to_filename(pjoin(sub_outdir, f"betas_dim-{dim_i+1}.nii.gz"))
    return None


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
