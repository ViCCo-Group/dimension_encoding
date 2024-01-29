from os.path import join as pjoin
import os
import numpy as np
from nilearn.image import load_img, new_img_like
from nilearn.masking import apply_mask, unmask
from tqdm import tqdm


class B5kLoader():
    
    def __init__(self, b5k_dir="../data/b5k/"):
        self.b5k_dir = b5k_dir
        assert os.path.exists(b5k_dir), "b5k_dir does not exist"
        print("B5k data loading from: {}".format(b5k_dir))
        self.nses_per_subject = {
            "CSI1":15, 'CSI2':15, 'CSI3':15, "CSI4":9
        }
        self.brain_inds_dir = self.b5k_dir.replace("b5k", "b5k_braininds")
    
    def get_responses_filenames(self, subj):
        nses = self.nses_per_subject[subj]
        fs = [
            pjoin(self.b5k_dir, f'{subj}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-{ses_i+1:02d}.nii.gz')
            for ses_i in range(nses)
        ]
        return fs
    
    def get_braininds_file(self, subj):
        return pjoin(self.brain_inds_dir, f"{subj}_brain_inds.nii.gz")
    
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
        _, brain_inds = self.load_responses(subj, return_brain_inds=True)
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