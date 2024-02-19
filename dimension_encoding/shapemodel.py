from os.path import join as pjoin, pardir
import pandas as pd
import numpy as np
from dimension_encoding.utils import load_clip66_preds
from tqdm import tqdm
import os


def load_shapecomps(shapecompdir=pjoin(pardir, "data", "ShapeComp")):
    # shapecompdir=pjoin(pardir, 'data', 'ShapeComp')
    # fnames_shapecomp = np.loadtxt(
    #     pjoin(shapecompdir, "THINGS_ShapeComp_fnames.txt"), dtype=str
    # )
    # shapecomps = pd.read_csv(
    #     pjoin(shapecompdir, "THINGS_ShapeComp.csv"), header=None
    # ).to_numpy()
    fnames_shapecomp = np.loadtxt(
        pjoin(shapecompdir, "THINGSfMRI_ShapeComp_fnames.txt"), dtype=str
    )
    shapecomps = pd.read_csv(
        pjoin(shapecompdir, "THINGSfMRI_ShapeComp.csv"), header=None
    ).to_numpy()
    return shapecomps, fnames_shapecomp


def make_shape_model(stimmeta, shapecomps, shapecomp_fnames):
    shapecomp_exemplars = np.array(
        [os.path.basename(f).split(".")[0] for f in shapecomp_fnames]
    )
    fmri_exemplars = stimmeta.stimulus.str.replace(".jpg", "").to_numpy()
    val_trial_inds = []
    excluded = []
    design_matrix = []
    for trial_i, fmri_exemplar in enumerate(fmri_exemplars):
        if fmri_exemplar not in shapecomp_exemplars:
            excluded.append(fmri_exemplar)
            continue
        else:
            hits = np.where(shapecomp_exemplars == fmri_exemplar)[0]
            assert len(hits) == 1
            shapecomp_i = hits[0]
            val_trial_inds.append(trial_i)
            design_matrix.append(shapecomps[shapecomp_i])
    val_trial_inds = np.array(val_trial_inds)
    design_matrix = np.array(design_matrix)
    shapecomp_model_info = dict(
        design_matrix=design_matrix, val_trial_inds=val_trial_inds, excluded=excluded
    )
    return shapecomp_model_info


def find_embedding_fmritrials(
    stimdata,
    dim66_dir,
):
    emb, fnames, labels = load_clip66_preds(
        fnames_with_folder_structure=False, dim66_dir=dim66_dir
    )
    clip_df = pd.DataFrame(emb)
    clip_df.set_axis(labels, axis=1)
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
