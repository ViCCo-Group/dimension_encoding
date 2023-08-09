import argparse
from nilearn.image import load_img
from nilearn.masking import apply_mask
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
from dimension_encoding.TopObjects import TopObjectsProfile
from dimension_encoding.utils import get_all_roi_files, load_clip66_preds


def load_pmod_betas_img(sub,pmod_basedir):
    fs = [
        pjoin(pmod_basedir, f"sub-{sub}", f"beta_regi-{i + 1}_regname-{i}.nii.gz")
        for i in range(66)
    ]
    betas_img = load_img(fs)
    return betas_img

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bidsroot",
        type=str,
        help="path to bids root directory",
        default="../data/bids",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="path to output directory",
        default="../results/topobjects_66d",
    )
    parser.add_argument(
        "--clip66dir",
        type=str,
        help="path to clip-predicted behavioral embeddings (66d)",
        default="../data/66d",
    )
    parser.add_argument(
        "--imagedir",
        type=str,
        help="path to things image database",
        default="../data/things_images",
    )
    parser.add_argument(
        "--prf_roidir",
        type=str,
        help="path to directory containing prf roi files",
        default="../data/prf",
    )
    parser.add_argument(
        "--floc_roidir",
        type=str,
        help="path to directory containign fLOC roi files",
        default="../data/category_localizer",
    )
    args = parser.parse_args()
    return args


def main(args):
    subs = ["01", "02", "03"]
    roi_betas_subs = {}
    pmod_basedir = pjoin(args.bidsroot, "derivatives", "pmod_cv_clip_66d")
    for sub in tqdm(subs, desc="Getting betas for each subject"):
        betas_img = load_pmod_betas_img(sub, pmod_basedir)
        rois_dict = get_all_roi_files(
            sub,
            args.bidsroot,
            args.prf_roidir,
            args.floc_roidir,
        )
        sub_roi_betas = {}
        for roiname, roifile in rois_dict.items():
            # ignore hemisphere specific rois
            if roiname[0] == "r" or roiname[0] == "l":
                continue
            sub_roi_betas[roiname] = apply_mask(betas_img, roifile).mean(axis=-1)
        roi_betas_subs[sub] = sub_roi_betas
    # only keep rois shared across subjects
    roinames = set(roi_betas_subs[subs[0]].keys())
    for sub in subs[1:]:
        roinames_ = set(roi_betas_subs[sub].keys())
        roinames = roinames & roinames_
    for sub in subs:
        roi_betas = {rn: roi_betas_subs[sub][rn] for rn in roinames}
        roi_betas_subs[sub] = roi_betas
    # average profiles across subjects
    avg_roibetas = {
        rn: np.mean([roi_betas_subs[sub][rn] for sub in subs], axis=0)
        for rn in roinames
    }
    # get clip embedding
    embedding, filenames, _ = load_clip66_preds(
        dim66_dir=args.clip66dir,
        fnames_with_folder_structure=True,
    )
    # get full path to images, only complete THINGS image set
    filenames = np.array([pjoin(args.imagedir, fn.replace("./", "")) for fn in filenames])
    # define out dirs for train and test
    for roiname, roibetas in avg_roibetas.items():
        topobjects = TopObjectsProfile(roiname, roibetas, embedding, filenames, args.outdir)
        topobjects.make_fig()