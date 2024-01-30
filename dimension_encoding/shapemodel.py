from os.path import join as pjoin, pardir
import pandas as pd
import numpy as np
from dimension_encoding.utils import load_clip66_preds
from tqdm import tqdm


def load_shapecomps(shapecompdir=pjoin(pardir, "data", "ShapeComp")):
    # shapecompdir=pjoin(pardir, 'data', 'ShapeComp')
    fnames_shapecomp = np.loadtxt(
        pjoin(shapecompdir, "THINGS_ShapeComp_fnames.txt"), dtype=str
    )
    shapecomps = pd.read_csv(
        pjoin(shapecompdir, "THINGS_ShapeComp.csv"), header=None
    ).to_numpy()
    return shapecomps, fnames_shapecomp


def make_shape_model(stimmeta, shapecomps, shapecomp_fnames):
    shapecomp_fnames_nosuff = np.array([f.split(".")[0] for f in shapecomp_fnames])
    fmri_stims_nosuff = stimmeta.stimulus.str.split(".").str[0].to_numpy()
    val_fmri_inds = []
    X = []
    for fmri_i, stim in enumerate(fmri_stims_nosuff):
        if stim not in shapecomp_fnames_nosuff:
            continue
        else:
            hits = np.where(shapecomp_fnames_nosuff == stim)[0]
            assert len(hits) == 1
            shapecomp_i = hits[0]
            val_fmri_inds.append(fmri_i)
            X.append(shapecomps[shapecomp_i])
    val_fmri_inds = np.array(val_fmri_inds)
    X = np.array(X)
    return val_fmri_inds, X


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