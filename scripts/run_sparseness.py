import numpy as np
from nilearn.image import smooth_img
from nilearn.masking import apply_mask, unmask
from os.path import join as pjoin
from argparse import ArgumentParser
from dimensions_encoding.sparseness import hoyer_sparseness


def load_data(betasdir, sub):
    ndims = 66
    betafiles = [
        pjoin(betasdir, f"sub-{sub}", f"beta_regi-{i+1}_regname-{i}.nii.gz")
        for i in range(ndims)
    ]
    return betafiles


def main(args):
    # smooth encoding model weight maps (betas)
    betafiles = load_data(args.betasdir, args.sub)
    betas_fs = smooth_img(betafiles, args.fwhm)
    betas = apply_mask(betas_fs, args.brainmask, dtype=np.single)
    if args.posbetas:
        # set netative weights to nan
        betas[betas < 0.0] = np.nan
    # calculate sparseness
    s = hoyer_sparseness(betas, axis=0)
    # save result
    s_img = unmask(s, args.brainmask)
    s_img.to_filename(
        pjoin(
            args.outdir, f"sub-{args.sub}_sparseness_smoothing-{args.fwhm:.2f}.nii.gz"
        )
    )
    return None


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--brainmask",
        type=str,
        required=True,
        help="path to brain mask file, available in THINGS-data",
    )
    parser.add_argument(
        "--betasdir",
        type=str,
        required=True,
        help="path to directory containing beta maps resulting from the parametric modulation analysis",
    )
    parser.add_argument(
        "--posbetas", action="store_true", help="ignore negative encoding weights"
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        required=True,
        help="full width at half maximum for smoothing",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="path to output directory"
    )
    parser.add_argument("--sub", type=str, required=True, help='subject id (e.g. "01")')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
