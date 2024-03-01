from os.path import join as pjoin
import os
import numpy as np
from nilearn.image import load_img, new_img_like
from nilearn.masking import apply_mask, unmask
from tqdm import tqdm
import pandas as pd
from dimension_encoding.utils import calc_nc
import glob

class B5kLoader():
    
    def __init__(self, b5k_dir="../data/b5k/"):
        self.b5k_dir = b5k_dir
        assert os.path.exists(b5k_dir), "b5k_dir does not exist"
        print("B5k data loading from: {}".format(b5k_dir))
        self.subjects = ("CSI1", 'CSI2', 'CSI3', "CSI4")
        self.n_sess_per_subject = {
            "CSI1":15, 'CSI2':15, 'CSI3':15, "CSI4":9
        }
        self.brain_inds_dir = self.b5k_dir.replace("b5k", "b5k_braininds")
        self.dimpreds_dir = self.b5k_dir.replace("b5k", "b5k_predicted_dimensions")
    
    def get_responses_filenames(self, subj):
        nses = self.n_sess_per_subject[subj]
        fs = [
            pjoin(self.b5k_dir, f'{subj}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-{ses_i+1:02d}.nii.gz')
            for ses_i in range(nses)
        ]
        return fs
    
    def get_braininds_file(self, subj):
        return pjoin(self.brain_inds_dir, f"{subj}_brain_inds.nii.gz")
    
    def load_predicted_spose_dimensions(self, image_sets=['imagenet', 'coco']):
        """Get predicted spose dimensions for stimuli used in b5k"""
        fn_list = []
        emb_list = []
        for image_set in image_sets:
            fn_list.append(np.loadtxt(glob.glob(pjoin(self.dimpreds_dir, image_set, 'file_names*.txt'))[0], dtype=str))
            emb_list.append(np.loadtxt(glob.glob(pjoin(self.dimpreds_dir, image_set, 'predictions*.txt'))[0]))
        fnames = np.hstack(fn_list)
        embedding = np.vstack(emb_list)
        return embedding, fnames
    
    def _make_braininds(self, subj):
        """
        the brain masks provided by b5k are misaligned with the data. hence, we create our own based on the volumetric responses.
        This function should only be called once per subject.
        """
        imgs = self.load_responses_as_imgs(subj)
        arrs = [img.get_fdata() for img in tqdm(imgs, total=len(imgs), desc='converting to array data')]
        responses_4d = np.concatenate(arrs, axis=3)
        nan_inds = np.any(np.isnan(responses_4d), axis=-1)  # 3d
        brain_inds = np.logical_not(nan_inds)  # 3d
        _, brain_inds = self.load_responses(subj)
        brain_inds_img = new_img_like(load_img(self.get_responses_filenames(subj)[0]), brain_inds)
        braininds_f = self.get_braininds_file(subj)
        brain_inds_img.to_filename(braininds_f)
        
    def load_responses_as_imgs(self, subj):
        responses_niftis = self.get_responses_filenames(subj)
        return [load_img(nii, dtype=np.single) for nii in tqdm(responses_niftis, 'loading niftis')]
    
    def load_responses(self, subj):
        responses_niftis = self.get_responses_filenames(subj)
        braininds_file = self.get_braininds_file(subj)
        return apply_mask(responses_niftis, braininds_file)
    
    def array_to_volume(self, arr, subj):
        braininds_f = self.get_braininds_file(subj)
        return unmask(arr, braininds_f)
    
    def load_stimdata(self, subj):
        return np.loadtxt(pjoin(self.b5k_dir, f"{subj}_imgnames.txt"), dtype='str')
    
    def make_dimensions_model(self, subj, image_sets=['imagenet', 'coco']):
        """
        Make a design matrix for the predicted spose dimensions, 
        also return trial indices for filtering the relevant trial responses 
        and the stimulus names
        """
        stimdata = self.load_stimdata(subj)
        embedding, fnames = self.load_predicted_spose_dimensions(image_sets=image_sets)
        trial_is = []
        X_dims = []
        for stim_i, stim in enumerate(stimdata):
            found_is = np.where(fnames == stim)[0]
            if len(found_is) == 0:
                continue
            trial_is.append(stim_i)
            X_dims.append(embedding[found_is[0]])
        X_dims = np.array(X_dims)
        trial_is = np.array(trial_is)
        stims = stimdata[trial_is]
        return X_dims, trial_is, stims
    
    
def compute_noise_ceiling_b5k(responses, stimdata, select_trials_with_nreps=4):
    stim_df = pd.DataFrame({"image":stimdata})
    betas_rep = []
    for _, rows in stim_df.groupby('image'):
        if len(rows)==select_trials_with_nreps:
            betas_rep.append(responses[rows.index].T)
    betas_rep = np.stack(betas_rep,axis=-1)
    nc = calc_nc(betas_rep, n=1)
    return nc