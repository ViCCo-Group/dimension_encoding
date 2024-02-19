import os
from os.path import join as pjoin
import nibabel as nib
import numpy as np
import pandas as pd
from fracridge import FracRidgeRegressor, fracridge
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import load_img, concat_imgs
from nilearn.masking import apply_mask, intersect_masks, unmask
from scipy.stats import zscore
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from warnings import warn
from tqdm import tqdm

from dimension_encoding.dataset import ThingsMRIdataset
from dimension_encoding.utils import psc, ci_array, get_hrflib, regress_out, pearsonr_nd


def get_nuisance_df(noiseregs, nuisance_tsv, include_all_aroma=False):
    """Make pd.DataFrame based on list of desired noise regressors and a nuisance_tsv file returned by fmriprep"""
    noiseregs_copy = noiseregs[:]
    nuisance_df = pd.read_csv(nuisance_tsv, sep="\t")
    if include_all_aroma:
        noiseregs_copy += [c for c in nuisance_df.columns if "aroma" in c]
    nuisance_df = nuisance_df[noiseregs_copy]
    if "framewise_displacement" in noiseregs_copy:
        nuisance_df["framewise_displacement"] = nuisance_df[
            "framewise_displacement"
        ].fillna(0)
    return nuisance_df


def df_to_boxcar_design(
    design_df: pd.DataFrame, frame_times: np.ndarray, add_constant: bool = False
) -> pd.DataFrame:
    """
    Make boxcar design matrix from data frame with one regressor for each trial_type (and no constant).
    CAVEAT: nilearn sorts the conditions alphabetically, not by onset.
    """
    dropcols = [] if add_constant else ["constant"]
    trialtypes = design_df["trial_type"].unique().tolist()
    designmat = make_first_level_design_matrix(
        frame_times=frame_times,
        events=design_df,
        hrf_model=None,
        drift_model=None,
        high_pass=None,
        drift_order=None,
        oversampling=1,
    ).drop(columns=dropcols)
    return designmat[trialtypes]


def load_masked(bold_file, mask, rescale="psc", dtype=np.single):
    if rescale == "psc":
        return np.nan_to_num(psc(apply_mask(bold_file, mask, dtype=dtype)))
    elif rescale == "z":
        return np.nan_to_num(
            zscore(apply_mask(bold_file, mask, dtype=dtype), nan_policy="omit", axis=0)
        )
    elif rescale == "center":
        data = np.nan_to_num(apply_mask(bold_file, mask, dtype=dtype))
        data -= data.mean(axis=0)
    else:
        return apply_mask(bold_file, mask, dtype=dtype)


class THINGSGLM(object):
    """
    Parent class for different GLMs to run on the things mri dataset,
    mostly handling IO.
    """

    def __init__(
        self,
        bidsroot: str,
        subject: str,
        out_deriv_name: str = "glm",
        noiseregs: list = [
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "framewise_displacement",
        ],
        acompcors: bool or int = 10,
        include_all_aroma: bool = False,
        # include_manual_ica: bool = False,
        hrf_model: str or None = "spm + derivative",
        noise_model: str = "ols",
        high_pass: float = 0.01,
        sigscale_nilearn: bool or int or tuple = False,
        standardize: bool = True,
        verbosity: int = 3,
        nruns_perses: int = 10,
        nprocs: int = 1,
        lowmem=False,
        ntrs: int = 284,
        tr: float = 1.5,
        drift_model: str = "cosine",
        drift_order: int = 4,
        fwhm: bool or None = None,
        overwrite: bool = False,
        stc_reftime: float = 0.75,
    ):
        self.bidsroot = os.path.abspath(bidsroot)
        self.include_all_aroma = include_all_aroma
        # self.include_manual_ica = include_manual_ica
        self.subject = subject
        self.out_deriv_name = out_deriv_name
        self.verbosity = verbosity
        self.lowmem = lowmem
        self.nprocs = nprocs
        self.acompcors = acompcors
        self.tr = tr
        self.ntrs = ntrs
        self.nruns_perses = nruns_perses
        self.high_pass = high_pass
        self.hrf_model = hrf_model
        self.noise_model = noise_model
        self.drift_model = drift_model
        self.drift_order = drift_order
        self.sigscale_nilearn = sigscale_nilearn
        self.standardize = standardize
        self.fwhm = fwhm
        self.stc_reftime = stc_reftime
        self.overwrite = overwrite
        self.ds = ThingsMRIdataset(self.bidsroot)
        self.n_sessions = len(self.ds.things_sessions)
        self.nruns_total = self.n_sessions * self.nruns_perses
        self.subj_prepdir = pjoin(
            bidsroot, "derivatives", "fmriprep", f"sub-{self.subject}"
        )
        self.subj_outdir = pjoin(
            bidsroot, "derivatives", out_deriv_name, f"sub-{self.subject}"
        )
        self.icalabelled_dir = pjoin(
            bidsroot, "derivatives", "ICAlabelled", f"sub-{self.subject}"
        )
        if not os.path.exists(self.subj_outdir):
            os.makedirs(self.subj_outdir)
        if acompcors:
            noiseregs += [f"a_comp_cor_{i:02}" for i in range(self.acompcors)]
        self.noiseregs = noiseregs
        self.frame_times_tr = (
            np.arange(0, self.ntrs * self.tr, self.tr) + self.stc_reftime
        )
        # get image dimensions
        example_img = load_img(
            self.ds.layout.get(
                session="things01", extension="nii.gz", suffix="bold", subject="01"
            )[0].path
        )
        self.nx, self.ny, self.nz, self.ntrs = example_img.shape
        self.n_samples_total, self.nvox_masked, self.union_mask = None, None, None

    def _get_events_files(self):
        return self.ds.layout.get(task="things", subject=self.subject, suffix="events")

    def _get_bold_files(self):
        bold_files = [
            pjoin(
                self.subj_prepdir,
                f"ses-{sesname}",
                "func",
                f"sub-{self.subject}_ses-{sesname}_task-things_run-{run_i + 1}_space-T1w_desc-preproc_bold.nii.gz",
            )
            for sesname in self.ds.things_sessions
            for run_i in range(10)
        ]
        for boldfile in bold_files:
            assert os.path.exists(boldfile), f"\nboldfile not found:\n{boldfile}\n"
        return bold_files

    def _get_masks(self):
        masks = [
            pjoin(
                self.subj_prepdir,
                f"ses-{sesname}",
                "func",
                f"sub-{self.subject}_ses-{sesname}_task-things_run-{run_i + 1}_space-T1w_desc-brain_mask.nii.gz",
            )
            for sesname in self.ds.things_sessions
            for run_i in range(10)
        ]
        for mask in masks:
            assert os.path.exists(mask), f"\nmask not found:\n{mask}\n"
        return masks

    def _get_nuisance_tsvs(self):
        nuisance_tsvs = [
            pjoin(
                self.subj_prepdir,
                f"ses-{sesname}",
                "func",
                f"sub-{self.subject}_ses-{sesname}_task-things_run-{run_i + 1}_desc-confounds_timeseries.tsv",
            )
            for sesname in self.ds.things_sessions
            for run_i in range(10)
        ]
        for tsv in nuisance_tsvs:
            assert os.path.exists(tsv), f"\nnuisance tsv not found:\n{tsv}\n"
        return nuisance_tsvs

    def _get_ica_txts(self):
        ica_txts = [
            pjoin(
                self.icalabelled_dir,
                f"ses-{sesname}",
                f"sub-{self.subject}_ses-{sesname}_task-things_run-{run_i + 1:02d}.txt",
            )
            for sesname in self.ds.things_sessions
            for run_i in range(10)
        ]
        for txt in ica_txts:
            assert os.path.exists(txt), f"\nica tsv not found:\n{txt}\n"
        return ica_txts

    def get_inputs(self):
        event_files = self._get_events_files()
        bold_files = self._get_bold_files()
        masks = self._get_masks()
        nuisance_tsvs = self._get_nuisance_tsvs()
        assert (
            len(event_files) == len(bold_files) == len(nuisance_tsvs) == len(masks)
        ), f"\ninputs have unequal length\n"
        return event_files, bold_files, nuisance_tsvs, masks

    def add_union_mask(self, masks):
        """Create a union mask based on the run-wise brain masks"""
        self.union_mask = intersect_masks(masks, threshold=0)

    def vstack_data_masked(
        self, bold_files, rescale_runwise="psc", rescale_global="off", dtype=np.single
    ):
        arrs = Parallel(n_jobs=20)(
            delayed(load_masked)(bf, self.union_mask, rescale_runwise, dtype)
            for bf in bold_files
        )
        if rescale_global == "psc":
            data = np.nan_to_num(psc(np.vstack(arrs)))
        elif rescale_global == "z":
            data = np.nan_to_num(zscore(np.vstack(arrs), nan_policy="omit", axis=0))
        elif rescale_global:
            data = np.vstack(arrs)
            data -= data.mean(axis=0)
        else:
            data = np.vstack(arrs)
        self.nvox_masked = data.shape[1]
        return data.astype(dtype)

    def load_data_concat_volumes(self, bold_files):
        print("concatinating bold files")
        bold_imgs = [
            nib.Nifti2Image.from_image(load_img(b))
            for b in tqdm(bold_files, "loading nifti files")
        ]
        return concat_imgs(bold_imgs, verbose=self.verbosity)

    def init_glm(self, mask):
        print(f"instantiating model with nprocs: {self.nprocs}")
        return FirstLevelModel(
            minimize_memory=self.lowmem,
            mask_img=mask,
            verbose=self.verbosity,
            noise_model=self.noise_model,
            t_r=self.tr,
            standardize=self.standardize,
            signal_scaling=self.sigscale_nilearn,
            n_jobs=self.nprocs,
            smoothing_fwhm=self.fwhm,
        )


class PMod(THINGSGLM):
    """
    Parametric modulation GLM.
    Besides one general onset regressor, each dimension from dims_arr is entered as a separate regressor with modulated
    amplitude. Each run is modelled separately and the effect of each regressor is averaged across runs according
    to fixed effects assumptions.
    """

    def __init__(
        self,
        bidsroot: str,
        subject: str,
        dims_arr: np.ndarray,
        dimnames_df: pd.DataFrame,
        out_deriv_name: str = "pmod",
        rescale_runwise_regressors: str = "demean",  # must be zscore, demean, or 'off'
        onset_only_tuning: bool = False,
        zscore_dimsarr: bool = True,
        noiseregs: list = [],
        include_all_aroma: bool = False,
        acompcors: bool or int = 0,
        hrf_model: str = "spm",
        noise_model: str = "ols",
        drift_model: str = "cosine",
        high_pass: float = 0.01,
        drift_order: int = 4,
        sigscale_nilearn: bool or int or tuple = False,
        standardize: bool = True,
        verbosity: int = 3,
        nruns_perses: int = 10,
        nprocs: int = 1,
        lowmem=False,
        ntrs: int = 284,
        tr: float = 1.5,
    ):
        super().__init__(
            bidsroot=bidsroot,
            subject=subject,
            out_deriv_name=out_deriv_name,
            noiseregs=noiseregs,
            acompcors=acompcors,
            hrf_model=hrf_model,
            noise_model=noise_model,
            high_pass=high_pass,
            sigscale_nilearn=sigscale_nilearn,
            standardize=standardize,
            verbosity=verbosity,
            nruns_perses=nruns_perses,
            nprocs=nprocs,
            lowmem=lowmem,
            ntrs=ntrs,
            tr=tr,
            include_all_aroma=include_all_aroma,
        )
        if type(dims_arr) == np.ndarray:  # omit if "None" is passed (for PModCVCLIP)
            self.dims_arr = zscore(dims_arr, axis=0) if zscore_dimsarr else dims_arr
            self.nobjects, self.ndims = dims_arr.shape
        else:
            self.nobjects, self.ndims = 1854, 49
        self.nstimregs = self.ndims + 1
        self.dim_reg_names = [str(_) for _ in range(self.ndims)]
        self.dims_range = list(range(self.ndims))
        self.dimnames_df = dimnames_df

        assert rescale_runwise_regressors in ("zscore", "demean", "off")
        self.rescale_runwise_regressors = rescale_runwise_regressors
        self.onset_only_tuning = onset_only_tuning
        # init some empty props to be filled with results
        self.model_ran = False
        self.design_matrices = []
        self.union_mask = None
        self.glm = None
        self.contrast_dicts = []
        self.rsquared_file = None
        self.drift_model = (drift_model,)
        self.drift_order = (drift_order,)

    def make_trials_df(self, names, onsets):
        """Create a dataframe with trialwise onsets, names, and modulations"""
        trial_dfs = []
        for i, (name, onset) in enumerate(zip(names, onsets)):
            matches = self.dimnames_df.index[self.dimnames_df.uniqueID == name].tolist()
            if len(matches) == 0:
                warn(
                    f"\nWARNING: Found {len(matches)} entries in spose dimensions table for event {name}, "
                    f"onset {onset}. Skipping\n"
                )
                continue
            name_index = matches[0]
            dims = self.dims_arr[name_index, :]
            trial_df = pd.DataFrame()
            trial_df["trial_type"] = self.dims_range
            trial_df["modulation"] = dims
            trial_df["onset"] = onset
            trial_dfs.append(trial_df)
        trials_df = pd.concat(trial_dfs)
        trials_df["duration"] = 0.5
        trials_df.trial_type = trials_df.trial_type.astype(str)
        trials_df = trials_df[["trial_type", "onset", "duration", "modulation"]]
        return trials_df

    def make_design_mat(self, eventfile, nuisance_tsv):
        """
        Make design matrix for one run which includes both regressors of interest and noise regressors.
        This is NOT to be used when convolution with HRFlibrary is performed, because it has a different time resolution
        self.onset_only_tuning == True will create only one regressor coding the onset of all stimuli.
        """
        onset_df = eventfile.get_df()
        onset_df["name"] = (
            onset_df.file_path.str.split("/")
            .str[1]
            .str.split("_")
            .str[:-1]
            .str.join("_")
        )
        onset_df["modulation"] = 1
        exp_df = onset_df[onset_df.trial_type.isin(["exp", "test"])]
        onset_df.trial_type = "onset"
        onset_df = onset_df[["trial_type", "duration", "onset", "name", "modulation"]]
        names = np.array(exp_df.name.array)
        onsets = np.array(exp_df.onset.array)
        if self.onset_only_tuning:
            design_df = onset_df
            self.nstimregs = 1
        else:
            trials_df = self.make_trials_df(names, onsets)
            if self.rescale_runwise_regressors == "demean":
                for dim_i in range(self.ndims):
                    trials_df.loc[trials_df.trial_type == f"{dim_i}", "modulation"] = (
                        trials_df.loc[trials_df.trial_type == f"{dim_i}", "modulation"]
                        - trials_df.loc[
                            trials_df.trial_type == f"{dim_i}", "modulation"
                        ].mean()
                    )
            elif self.rescale_runwise_regressors == "zscore":
                for dim_i in range(self.ndims):
                    trials_df.loc[trials_df.trial_type == f"{dim_i}", "modulation"] = (
                        zscore(
                            trials_df.loc[
                                trials_df.trial_type == f"{dim_i}", "modulation"
                            ]
                        )
                    )
            design_df = pd.concat([onset_df[trials_df.columns], trials_df])
        nuisance_df = get_nuisance_df(
            self.noiseregs, nuisance_tsv, include_all_aroma=self.include_all_aroma
        )
        return make_first_level_design_matrix(
            self.frame_times_tr,
            events=design_df,
            hrf_model=self.hrf_model,
            high_pass=self.high_pass,
            drift_order=self.drift_order,
            drift_model=self.drift_model,
            add_regs=nuisance_df,
        )

    def fit_runs(self):
        event_files, bold_files, nuisance_tsvs, masks = self.get_inputs()
        if not self.union_mask:
            self.add_union_mask(masks)
        self.glm = self.init_glm(self.union_mask)
        print("making design matrices")
        self.design_matrices = [
            self.make_design_mat(ef, ntsv)
            for ef, ntsv in zip(event_files, nuisance_tsvs)
        ]
        print("starting model fitting")
        self.glm = self.glm.fit(bold_files, design_matrices=self.design_matrices)
        self.model_ran = True

    def compute_contrasts(self):
        if not self.model_ran:
            print(
                "cannot compute contrasts before fitting the model, returning None. Call .fit_runs() first"
            )
            return
        else:
            self.trial_types = list(self.design_matrices[0].columns)
            for trialtype in self.trial_types:
                print(f"computing contrast for {trialtype}")
                self.contrast_dicts.append(
                    self.glm.compute_contrast(
                        trialtype, output_type="all", stat_type="t"
                    )
                )

    def save_outputs(self):
        if not self.contrast_dicts:
            self.compute_contrasts()
        else:
            if not os.path.exists(self.subj_outdir):
                print(f"creating output directory for subject {self.subject}")
                os.makedirs(self.subj_outdir)
            for contrastdict, trialtype in zip(self.contrast_dicts, self.trial_types):
                print(f"saving image for {trialtype}")
                for stattype in contrastdict.keys():
                    contrastdict[stattype].to_filename(
                        pjoin(
                            self.subj_outdir,
                            f"sub-{self.subject}_{trialtype}_{stattype}.nii.gz",
                        )
                    )
        print("saving r-squared image")
        self.rsquared_file = pjoin(
            self.subj_outdir, f"sub-{self.subject}_rsquared.nii.gz"
        )
        self.glm.r_square[0].to_filename(self.rsquared_file)


class PModCV(PMod):
    """
    Parametric modulation encoding model of our SpoSE dimensions.
    includes HRF fitting and fractional ridge regression, hyperparameters are determined via cross validation.
    """

    def __init__(
        self,
        bidsroot: str,
        subject: str,
        dims_arr: np.ndarray or None,
        dimnames_df: pd.DataFrame or None,
        out_deriv_name: str = "pmod_cv",
        hrflib_url: str = "https://raw.githubusercontent.com/kendrickkay/GLMdenoise/master/utilities"
        "/getcanonicalhrflibrary.tsv",
        rescale_runwise_regressors: str = "off",
        zscore_convolved_design_total: bool = True,
        zscore_stickfunc_total: bool = True,
        zscore_data_sessionwise: bool = False,  # applies to both data and unconvolved model features.
        standardize_noiseregs: bool = True,
        onset_only_tuning: bool = False,
        rescale_runwise_data: str = "psc",
        noiseregs: list = [],
        include_all_aroma: bool = False,
        manual_ica_regressors: bool = True,
        acompcors: bool or int = 0,
        drift_model: str = "polynomial",
        high_pass: float = None,
        drift_order: int = 4,
        nboots: int = 0,
        usecache: bool = True,
        nprocs: int = 1,
        tr: float = 1.5,
        hrflib_resolution: float = 0.1,
        n_parallel_noisereg: int = 10,
    ):
        super().__init__(
            bidsroot=bidsroot,
            subject=subject,
            dims_arr=dims_arr,
            dimnames_df=dimnames_df,
            out_deriv_name=out_deriv_name,
            rescale_runwise_regressors=rescale_runwise_regressors,
            noiseregs=noiseregs,
            acompcors=acompcors,
            noise_model="ols",
            high_pass=high_pass,
            sigscale_nilearn=False,
            standardize=False,
            tr=tr,
            nprocs=nprocs,
            include_all_aroma=include_all_aroma,
            onset_only_tuning=onset_only_tuning,
        )
        self.hrflib_url = hrflib_url
        self.drift_model = drift_model
        self.drift_order = drift_order
        self.usecache = usecache
        self.nboots = nboots
        self.zscore_data_sessionwise = zscore_data_sessionwise
        self.zscore_convolved_design_total = zscore_convolved_design_total
        self.zscore_stickfunc_total = zscore_stickfunc_total
        self.manual_ica_regressors = manual_ica_regressors
        self.standardize_noiseregs = standardize_noiseregs
        self.cv_results, self.best_param_inds, self.y, self.x_stim = (
            None,
            None,
            None,
            None,
        )
        self.kf = KFold(n_splits=self.n_sessions, shuffle=False)
        self.stimreg_names = None
        if not os.path.exists(self.subj_outdir):
            os.makedirs(self.subj_outdir)
        self.cv_results_tmpfile = pjoin(self.subj_outdir, "r2s_cv.npz")
        fracs_lower = np.arange(0.05, 0.9, 0.05)
        fracs_upper = np.arange(0.92, 1.01, 0.01)
        self.fracs = np.hstack([fracs_lower, fracs_upper])
        self.n_fracs = len(self.fracs)
        self.hrflib = get_hrflib(self.hrflib_url)
        self.nsamples_hrf, self.nhrfs = self.hrflib.shape
        self.frs, self.frf = self.init_fracridges()
        self.hrf_model = None
        self.rescale_runwise_data = rescale_runwise_data
        self.hrflib_resolution = hrflib_resolution
        self.microtime_factor = int(
            self.tr / self.hrflib_resolution
        )  # should be 15 in our case
        # frame_times_microtime are matched to HFFlibrary time resolution
        self.frame_times_microtime = (
            np.arange(0, self.ntrs * self.tr, self.hrflib_resolution) + self.stc_reftime
        )
        self.n_parallel_noisereg = n_parallel_noisereg

    def init_fracridges(self):
        frs = FracRidgeRegressor(
            fracs=self.fracs, fit_intercept=True, normalize=False
        )  # encoding model for cv
        frf = FracRidgeRegressor(
            fracs=self.fracs, fit_intercept=True, normalize=False
        )  # stimulus model for final fit
        return frs, frf

    def make_design_df(self, eventfile):
        onset_df = eventfile.get_df()
        onset_df["name"] = (
            onset_df.file_path.str.split("/")
            .str[1]
            .str.split("_")
            .str[:-1]
            .str.join("_")
        )
        onset_df["modulation"] = 1
        exp_df = onset_df[onset_df.trial_type.isin(["exp", "test"])]
        onset_df.trial_type = "onset"
        onset_df = onset_df[["trial_type", "duration", "onset", "name", "modulation"]]
        names = np.array(exp_df.name.array)
        onsets = np.array(exp_df.onset.array)
        if self.onset_only_tuning:
            design_df = onset_df
            self.nstimregs = 1
        else:
            trials_df = self.make_trials_df(names, onsets)
            if self.rescale_runwise_regressors == "demean":
                for dim_i in range(self.ndims):
                    dim_amps = trials_df.loc[
                        trials_df.trial_type == f"{dim_i}", "modulation"
                    ]
                    dim_amps_c = dim_amps - dim_amps.mean()
                    trials_df.loc[trials_df.trial_type == f"{dim_i}", "modulation"] = (
                        dim_amps_c
                    )
            elif self.rescale_runwise_regressors == "zscore":
                for dim_i in range(self.ndims):
                    dim_amps = trials_df.loc[
                        trials_df.trial_type == f"{dim_i}", "modulation"
                    ]
                    dim_amps_z = zscore(dim_amps)
                    trials_df.loc[trials_df.trial_type == f"{dim_i}", "modulation"] = (
                        dim_amps_z
                    )
            design_df = pd.concat([onset_df[trials_df.columns], trials_df])
        return design_df

    def make_noise_mat(self, nuisance_tsv, ica_txt=None, add_constant=False):
        """
        Make design matrix for noise regressors obtained from fmripreps nuisance tsv files
        and/or our manually classified ICs.
        """
        nuisance_df = get_nuisance_df(
            self.noiseregs, nuisance_tsv, include_all_aroma=self.include_all_aroma
        )
        if ica_txt:
            ica_arr = np.loadtxt(ica_txt)
            nuisance_df = pd.DataFrame(
                np.hstack([nuisance_df.to_numpy(), ica_arr]),
                columns=[
                    f"noisereg-{i}"
                    for i in range(nuisance_df.shape[1] + ica_arr.shape[1])
                ],
            )
        dropcols = [] if add_constant else ["constant"]
        return make_first_level_design_matrix(
            frame_times=self.frame_times_tr,
            add_regs=nuisance_df,
            hrf_model=None,
            drift_model=self.drift_model,
            drift_order=self.drift_order,
            high_pass=self.high_pass,
            events=None,
        ).drop(columns=dropcols)

    def zscore_unconvolved_features_total(self, design_dfs):
        """zscore features over entire dataset before turning into design matrix"""
        print("Z-scoring each dimension stick function across entire dataset")
        assert len(np.unique([len(df) for df in design_dfs])) == 1
        df_len = len(design_dfs[0])
        # concatenate all data frames
        df_total = pd.concat(design_dfs).copy()
        for dim_i in range(self.ndims):
            dim_amps = df_total.loc[df_total.trial_type == str(dim_i), "modulation"]
            assert len(dim_amps) > 0
            dim_amps_z = zscore(dim_amps)
            df_total.loc[df_total.trial_type == str(dim_i), "modulation"] = dim_amps_z
        # split up into run dfs again
        design_dfs_zscored = [
            df_total.iloc[start_trial : start_trial + df_len, :]
            for start_trial in range(0, df_len * self.nruns_total, df_len)
        ]
        return design_dfs_zscored

    def make_Xstim_Xnoise(self, event_files, nuisance_tsvs):
        print("making design and noise arrays")
        design_dfs = [
            self.make_design_df(ef)
            for ef in tqdm(event_files, desc="Making design data frames")
        ]
        if self.zscore_stickfunc_total:
            design_dfs = self.zscore_unconvolved_features_total(design_dfs)
        stim_mats_l = [
            df_to_boxcar_design(
                design_df, frame_times=self.frame_times_microtime, add_constant=False
            )
            for design_df in tqdm(
                design_dfs, desc="creating microtime boxcar design matrices"
            )
        ]
        stim_mats = np.stack(stim_mats_l)

        if self.manual_ica_regressors:
            ica_tsvs = self._get_ica_txts()
            noise_mats = [
                self.make_noise_mat(nuisance_tsv, ica_tsv, add_constant=False)
                for nuisance_tsv, ica_tsv in zip(nuisance_tsvs, ica_tsvs)
            ]
        else:
            noise_mats = [
                self.make_noise_mat(nuisance_tsv, add_constant=False)
                for nuisance_tsv in nuisance_tsvs
            ]
        if self.onset_only_tuning:
            self.stimreg_names = ["onset"]
        else:
            self.stimreg_names = np.array(stim_mats_l[0].columns)
            print(f"\nRegressors in design matrix: {self.stimreg_names}\n")
        if self.standardize_noiseregs:
            noise_mats = [zscore(nm, axis=0) for nm in noise_mats]
        return stim_mats, noise_mats

    def conv_stim_mats(self, stim_mats):
        """
        Convolve design matrix boxcar with HRF library.
        It is assumed that self has an attriute self.microtime_factor. microtime_factor is a scalar that should match
        the difference in temporal resolution between the HRF-library and our data (in our case, the data has a TR
        of 1.5 secs and the HRF a temporal resolution of .1 secs, which corresponds microtime_factor=15.
        returns:
            array of shape (nsamples_total, nstimregs, nhrfs).
        """
        convolved_ups = np.zeros(
            shape=(
                self.nruns_total,
                self.ntrs * self.microtime_factor,
                self.nstimregs,
                self.nhrfs,
            )
        )
        for hrf_i in tqdm(
            range(self.nhrfs), desc="convolving stimulus design with different HRFs"
        ):
            conv_thishrf_ups = np.apply_along_axis(
                lambda m: np.convolve(m, self.hrflib[:, hrf_i], mode="full"),
                arr=stim_mats,
                axis=1,
            )[:, : self.ntrs * self.microtime_factor, :]
            convolved_ups[:, :, :, hrf_i] = conv_thishrf_ups
        convolved = convolved_ups[:, :: self.microtime_factor, :, :]
        return convolved.reshape(
            (self.nruns_total * self.ntrs, self.nstimregs, self.nhrfs)
        )

    """
    convolved_ups = np.zeros(
        shape=(self.nhrfs, self.ntrs * self.microtime_factor, designmat.shape[1]))
    for hrf_i in range(self.nhrfs):
        conv_thishrf_ups = np.apply_along_axis(
            lambda m: np.convolve(m, self.hrflib[:, hrf_i], mode='full'),
            arr=designmat, axis=0
        )[:self.ntrs * self.microtime_factor, :]
        if rescale_hrflib_amplitude:
            conv_thishrf_ups = np.nan_to_num(conv_thishrf_ups / conv_thishrf_ups.max(axis=0))
        convolved_ups[hrf_i] = conv_thishrf_ups
    convolved_designmat = convolved_ups[:, ::self.microtime_factor, :]
    return convolved_designmat
        """

    def regress_out_noise_runwise(
        self, noise_mats, data, zscore_residuals=False, dtype=np.single
    ):
        """
        Regress the noise matrices out of our data separately for run.
        Original data is overwritten.
        Arguments:
            noise_mats: list
                each element should contain the design matrix with only noise regressors for a given run.
            data: np.ndarray
                stacked data of shape (nsamples_total, nvoxels_masked)
            zscore_residuals: bool
                whether to apply z-scoring to the residuals of the noise model
            dtype:
                numpy data type enforced upon the output.
        """
        # fit intercept only if data was not runwise demeaned
        fitint = True if self.rescale_runwise_data in ["off", "psc"] else False
        start_samples = [run_i * self.ntrs for run_i in range(self.nruns_total)]
        stop_samples = [start + self.ntrs for start in start_samples]
        with Parallel(n_jobs=self.n_parallel_noisereg) as parallel:
            filtered_runs = parallel(
                delayed(regress_out)(
                    noise_mats[run_i],
                    data[start:stop],
                    lr_kws=dict(
                        copy_X=False, fit_intercept=fitint, normalize=False, n_jobs=-1
                    ),
                )
                for run_i, (start, stop) in tqdm(
                    enumerate(zip(start_samples, stop_samples)),
                    total=len(start_samples),
                    desc="runs",
                )
            )
        if zscore_residuals:
            filtered_runs = [
                zscore(d, axis=0)
                for d in tqdm(
                    filtered_runs, desc="z-scoring residuals from noise regression"
                )
            ]
        data = np.vstack(filtered_runs).astype(dtype)
        return data

    def load_data(self):
        event_files, bold_files, nuisance_tsvs, masks = self.get_inputs()
        self.add_union_mask(masks)
        print("making design matrices")
        stim_mats, noise_mats = self.make_Xstim_Xnoise(event_files, nuisance_tsvs)
        print("convolving stimulus design matrices")
        x_stim = self.conv_stim_mats(stim_mats)
        if self.zscore_convolved_design_total:
            x_stim = zscore(x_stim, axis=0).astype(np.single)
        print("saving x_stim", pjoin(self.subj_outdir, "design_matrices.npz"))
        np.savez(pjoin(self.subj_outdir, "design_matrices.npz"), x_stim=x_stim)
        data = self.vstack_data_masked(
            bold_files, rescale_runwise=self.rescale_runwise_data
        )
        if self.zscore_data_sessionwise:
            volsperses = self.ntrs * self.nruns_perses
            for ses_i in tqdm(range(self.n_sessions), desc="session-wise z-scoring"):
                start_i, stop_i = ses_i * volsperses, ses_i * volsperses + volsperses
                data[start_i:stop_i] = zscore(data[start_i:stop_i], axis=0)
        print("regressing out noise")
        data = self.regress_out_noise_runwise(noise_mats, data, zscore_residuals=False)
        # After regressing out noise, y still has the same scale as before (e.g. psc) but zero mean
        print("unloading unfiltered data")
        return data, x_stim

    def run_cv(self, y, x_stim):
        """
        Run cross validation. This entails...
         3) iterating over cv-folds/hrfs/regularization params
         4) scoring prediction vs. test data
        """
        if os.path.exists(self.cv_results_tmpfile):
            print(
                f"found cached Rsquared results from previous cross validation\n{self.cv_results_tmpfile}\n"
            )
            if self.usecache:
                print(
                    f"usecache is True, omitting CV and loading cached Rsquared file instead"
                )
                cv_results = np.load(self.cv_results_tmpfile)["cv_results"]
                return cv_results
            else:
                print(
                    "usecache is False, existing cross validation results will be overwritten"
                )
        cv_results = np.zeros(
            shape=(self.nhrfs, self.n_fracs, self.nvox_masked), dtype=np.single
        )
        for fold_i, (train_inds, test_inds) in tqdm(
            enumerate(self.kf.split(y)),
            desc="CV fold",
            leave=True,
            total=self.n_sessions,
        ):
            y_train = y[train_inds]
            y_test = y[test_inds]
            for hrf_i in tqdm(range(self.nhrfs), desc="evaluating HRF", leave=False):
                x_train = x_stim[train_inds, :, hrf_i]
                x_test = x_stim[test_inds, :, hrf_i]
                self.frs.fit(x_train, y_train)
                pred = self.frs.predict(x_test)
                for aa_i in tqdm(
                    range(self.n_fracs), desc="r2-scoring fracs", leave=False
                ):
                    r2 = r2_score(y_test, pred[:, aa_i, :], multioutput="raw_values")
                    cv_results[hrf_i, aa_i, :] += r2
        cv_results /= self.n_sessions
        np.savez_compressed(self.cv_results_tmpfile, cv_results=cv_results)
        return cv_results

    def eval_cv(self, cv_results):
        """Find indices of best parameter combinations based on CV results"""
        print("Finding best parameter combinations")
        mean_r2s_re = cv_results.reshape(
            (self.nhrfs * self.n_fracs, self.nvox_masked)
        )  # reshape for convenience
        best_param_inds = np.argmax(
            mean_r2s_re, axis=0
        )  # voxel-wise flat indices for best param combinations
        print(
            "saving r-squared map for best parameter combinations, best performing hrf and regularization fraction"
        )
        self.save_r2s(best_param_inds, mean_r2s_re)
        self.save_best_fracs_hrfs(best_param_inds)
        return best_param_inds

    def save_r2s(self, best_param_inds, mean_r2s_re):
        """Save cross-validated r-squared values (for each voxel for the best parameter combination)"""
        r2_best = mean_r2s_re.T[np.arange(len(mean_r2s_re.T)), list(best_param_inds)]
        r2_best_img = unmask(r2_best, self.union_mask)
        r2_best_img.to_filename(pjoin(self.subj_outdir, f"best_cv_r2.nii.gz"))

    def save_best_fracs_hrfs(self, best_param_inds):
        """Save niftis containing the voxelwise best HRF indices and alpha fractions"""
        best_hrf_inds, best_frac_inds = np.unravel_index(
            best_param_inds, (self.nhrfs, self.n_fracs)
        )
        best_fracs = self.fracs[best_frac_inds]
        hrf_inds_img = unmask(best_hrf_inds, self.union_mask)
        hrf_inds_img.to_filename(pjoin(self.subj_outdir, "best_hrf_inds.nii.gz"))
        fracs_img = unmask(best_fracs, self.union_mask)
        fracs_img.to_filename(pjoin(self.subj_outdir, "best_fracs.nii.gz"))

    def cv_onsetonly_bestparams(self, y, x_stim, best_param_inds, onsetreg_i=-1):
        """
        Get cross-validated R-squared for the onset regressor based on the hyperparameters identified on
        the cross-validation of the full model.
        Works, but takes ~ 10 extra hours.
        """
        print("Calculating cross-validated R2 for the onset regressor.")
        x_stim_norm = zscore(x_stim, axis=0) if self.frf.normalize else x_stim
        r2s_onset = np.zeros(shape=(self.n_sessions, self.nvox_masked))
        for fold_i, (train_inds, test_inds) in tqdm(
            enumerate(self.kf.split(y)), desc="CV R-squared for onset only"
        ):
            for param_ind in tqdm(
                np.unique(best_param_inds),
                desc="iterating through best parameter combinations",
            ):
                hrf_ind, frac_ind = np.unravel_index(
                    param_ind, (self.nhrfs, self.n_fracs)
                )
                x_ = np.stack(
                    [
                        x_stim_norm[:, onsetreg_i, hrf_ind],
                        np.ones(shape=self.n_samples_total),
                    ],
                    axis=-1,
                )
                voxel_inds = np.where(best_param_inds == param_ind)
                x_train, x_test = x_[train_inds], x_[test_inds]
                y_train = y[train_inds][:, voxel_inds[0]]
                y_test = y[test_inds][:, voxel_inds[0]]
                coef, alpha = fracridge(x_train, y_train, fracs=self.fracs[frac_ind])
                r2s_onset[fold_i][voxel_inds[0]] = r2_score(
                    y_test, np.dot(x_test, coef), multioutput="raw_values"
                )
        meanr2_onset = r2s_onset.mean(axis=0)
        r2onset_img = unmask(meanr2_onset, self.union_mask)
        r2onset_img.to_filename(pjoin(self.subj_outdir, f"cv_r2_onset.nii.gz"))

    def final_fit(self, y, x_stim, best_param_inds):
        """
        Fit the dimensions on the entire time series and save the results for the models with best
        performing parameters respectively.
        """
        np.savetxt(
            pjoin(self.subj_outdir, "regressor_names.txt"), self.stimreg_names, fmt="%s"
        )
        coefs = np.zeros(shape=(self.nvox_masked, self.nstimregs))
        r2s = np.zeros(shape=self.nvox_masked)
        alphas = np.zeros(shape=self.nvox_masked)
        for param_ind in tqdm(
            np.unique(best_param_inds),
            desc="Final fit for different parameter combinations",
        ):
            hrf_i, frac_i = np.unravel_index(param_ind, (self.nhrfs, self.n_fracs))
            voxel_inds = np.where(best_param_inds == param_ind)
            x_ = x_stim[:, :, hrf_i]
            y_ = y[:, voxel_inds].squeeze()
            self.frf.fracs = self.fracs[frac_i]
            self.frf.fit(x_, y_)
            if len(voxel_inds[0]) == 1:
                # when one voxel is passed, frf.coef_ has redundant first dimension (nreg, nreg). (I double checked)
                # and .predict() method doesn't work anymore.
                coefs_ = self.frf.coef_[0]
                pred = np.dot(x_, coefs_)
            else:
                coefs_ = self.frf.coef_.T
                pred = self.frf.predict(x_)
            coefs[voxel_inds] = coefs_
            r2 = r2_score(y_, pred, multioutput="raw_values")  # shape nvox_selected
            r2s[voxel_inds] = r2
            alphas[voxel_inds] = self.frf.alpha_  # has shape (1, nvox_selected)
        # save results to nifti
        r2s_img = unmask(r2s, self.union_mask)
        r2s_img.to_filename(pjoin(self.subj_outdir, f"withinsample_r2.nii.gz"))
        alphas_img = unmask(alphas, self.union_mask)
        alphas_img.to_filename(pjoin(self.subj_outdir, f"best_alphas.nii.gz"))
        for reg_i, regname in enumerate(self.stimreg_names):
            reg_img = unmask(coefs[:, reg_i], self.union_mask)
            reg_img.to_filename(
                pjoin(self.subj_outdir, f"beta_regi-{reg_i}_regname-{regname}.nii.gz")
            )

    def bootstrap_betas(self, y, x_stim, best_param_inds):
        """
        Bootstrap by selecting N runs with replacement and estimate the betas. Lower and upper bounds of the
        confidence interval over all bootstraps are calculated. If these include 0, the beta is deemed nonsignificant.
        Significance is stored in a binary nifti.
        10 bootstraps takes ~ 1493.85 s -> 1000 bootstraps takes ~ 41.49 hours
        """
        #
        x_stim_z = zscore(x_stim, axis=0)
        cis = np.zeros(shape=(self.nvox_masked, self.nstimregs, 2))
        import time

        start = time.time()
        for param_ind in tqdm(np.unique(best_param_inds), desc="param combinations"):
            hrf_ind, frac_ind = np.unravel_index(param_ind, (self.nhrfs, self.n_fracs))
            voxel_inds = np.where(best_param_inds == param_ind)
            boot_results = np.zeros(
                shape=(self.nboots, len(voxel_inds[0]), self.nstimregs)
            )
            for boot_i in tqdm(range(self.nboots), desc="bootstraps"):
                run_is = np.random.choice(120, 120, replace=True)
                samp_is = np.array(
                    [
                        np.arange(run_i * self.ntrs, run_i * self.ntrs + self.ntrs)
                        for run_i in run_is
                    ]
                ).flatten()
                x_ = x_stim_z[samp_is, :, hrf_ind]
                x_ = np.hstack([x_, np.ones(shape=(self.n_samples_total, 1))])
                y_ = y[np.ix_(samp_is, voxel_inds[0])].squeeze()
                coef, _ = fracridge(x_, y_, fracs=self.fracs[frac_ind])
                boot_results[boot_i] = coef[:-1].T
            lci, uci = ci_array(boot_results)
            cis[voxel_inds] = np.stack([lci, uci], axis=-1)
        end = time.time()
        print(f"bootstrap runtime in hours: ", ((end - start) / 60) / 60)
        sigs = np.sign(np.prod(cis, axis=-1))
        for reg_i in range(self.nstimregs):
            ci_img = unmask(cis[:, reg_i, :].T, self.union_mask)
            ci_img.to_filename(pjoin(self.subj_outdir, f"ci_reg-{reg_i}.nii.gz"))
            sigs_img = unmask(sigs[:, reg_i], self.union_mask)
            sigs_img.to_filename(pjoin(self.subj_outdir, f"sig_reg-{reg_i}.nii.gz"))

    def run(self):
        y, x_stim = self.load_data()
        cv_results = self.run_cv(y, x_stim)
        best_param_inds = self.eval_cv(cv_results)
        self.final_fit(y, x_stim, best_param_inds)
        if self.nboots:
            self.bootstrap_betas(y, x_stim, best_param_inds)


class PModCVCLIP(PModCV):
    """
    Use the image-wise predictions of behavioral dimensions based on CLIP penultimate layer.
    """

    def __init__(
        self,
        bidsroot: str,
        subject: str,
        clip_preds,
        clip_fnames,
        out_deriv_name: str = "pmod_cv_clip",
        usecache: bool = True,
    ):
        super().__init__(
            bidsroot=bidsroot,
            subject=subject,
            dims_arr=None,
            dimnames_df=None,
            out_deriv_name=out_deriv_name,
            usecache=usecache,
        )
        self.clip_preds = clip_preds
        self.clip_fnames = clip_fnames
        self.ndims = clip_preds.shape[1]
        self.nstimregs = self.ndims + 1

    def make_design_df(self, eventfile):
        """Make design matrix for one run based on the CLIP predictions."""
        onset_df = eventfile.get_df()
        onset_df["modulation"] = 1
        exp_df = onset_df[onset_df.trial_type.isin(["exp", "test"])]
        onset_df["trial_type"] = "onset"
        if self.onset_only_tuning:
            design_df = onset_df
            self.nstimregs = 1
        else:
            # construct file names for our trials to find in the things database,
            basenames_trials = exp_df.file_path.str.split("/").str[-1].to_numpy()
            objnames_trials = (
                exp_df.file_path.str.split("/").str[-1].str[:-8].to_numpy()
            )
            fnames_trials = np.array(
                [
                    f"./images/{objname}/{basename}"
                    for objname, basename in zip(objnames_trials, basenames_trials)
                ]
            )
            # for each trial, find THINGS-index of that image
            img_is = np.vstack(
                [
                    np.argwhere(fname_trial == self.clip_fnames)
                    for fname_trial in fnames_trials
                ]
            ).squeeze()
            dimweights = self.clip_preds[img_is]  # weights belonging to each trial
            assert img_is.shape == fnames_trials.shape
            assert dimweights.shape[0] == len(exp_df)
            # data frame with 49 regressors for each trial
            dfs_ = []
            for trialweights, onset in zip(dimweights, exp_df["onset"]):
                trial_df = pd.DataFrame()
                trial_df["modulation"] = trialweights
                trial_df["onset"] = onset
                trial_df["trial_type"] = [str(i) for i in range(self.ndims)]
                dfs_.append(trial_df)
            trials_df = pd.concat(dfs_)
            trials_df["duration"] = 0.5
            if self.rescale_runwise_regressors == "demean":
                for dim_i in range(self.ndims):
                    trials_df.loc[trials_df.trial_type == f"{dim_i}", "modulation"] = (
                        trials_df.loc[trials_df.trial_type == f"{dim_i}", "modulation"]
                        - trials_df.loc[
                            trials_df.trial_type == f"{dim_i}", "modulation"
                        ].mean()
                    )
            elif self.rescale_runwise_regressors == "zscore":
                for dim_i in range(self.ndims):
                    trials_df.loc[trials_df.trial_type == f"{dim_i}", "modulation"] = (
                        zscore(
                            trials_df.loc[
                                trials_df.trial_type == f"{dim_i}", "modulation"
                            ]
                        )
                    )
            design_df = pd.concat([onset_df[trials_df.columns], trials_df])
        return design_df


def shuffle_and_correlate(perminds_fold, y_pred, y_true, metric="pearsonr"):
    assert metric in ("pearsonr", "r2")
    y_true_perm = y_true[perminds_fold]
    if metric == "pearsonr":
        score = pearsonr_nd(y_pred, y_true_perm)
    elif metric == "r2":
        score = r2_score(y_true_perm, y_pred, multioutput="raw_values")
    return score


class LinRegCVPermutation:
    def __init__(
        self,
        nperm=10_000,
        nfolds=12,
        random_folds=False,
        fit_intercept=True,
        n_jobs_permutation=50,
        n_jobs_fit=-1,
        metric="pearsonr",
    ):
        self.nperm = nperm
        self.nfolds = nfolds
        self.random_folds = random_folds
        self.kf = (
            None if self.nfolds == 1 else KFold(n_splits=nfolds, shuffle=random_folds)
        )
        self.n_jobs_permutation = n_jobs_permutation
        self.n_jobs_fit = n_jobs_fit
        self.lr = LinearRegression(fit_intercept=fit_intercept, n_jobs=self.n_jobs_fit)
        self.perminds = None
        allowed_metrics = ("pearsonr", "r2")
        assert (
            metric in allowed_metrics
        ), f"metric {metric} must be in {allowed_metrics}"
        self.metric = metric
        self.perminds = None

    def calc_nsamples_per_fold(self, X):
        nsamples_per_fold = X.shape[0] / self.nfolds
        if not nsamples_per_fold.is_integer():
            warn(
                f"number of samples {X.shape[0]} is not divisible by number of CV folds {self.nfolds}"
            )
        return int(nsamples_per_fold)

    def make_permutation_inds(self, nsamples_test):
        perminds = np.array(
            [
                np.random.choice(nsamples_test, nsamples_test, replace=False)
                for _ in range(self.nperm)
            ]
        )
        return perminds

    def calc_p(self, rtrue, rsperm):
        p = np.mean(rsperm >= rtrue, axis=0)
        return p

    def fit_predict_eval(self, x_train, x_test, y_train, y_test):
        self.lr.fit(x_train, y_train)
        pred = self.lr.predict(x_test)
        if self.metric == "pearsonr":
            r = pearsonr_nd(pred, y_test)
        elif self.metric == "r2":
            r = r2_score(y_test, pred, multioutput="raw_values")
        return r, pred

    def make_splits(self, X):
        if self.nfolds == 1:
            splits = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]
        else:
            splits = self.kf.split(X)
        return splits

    def fit_and_permute(self, X, y):
        nvox = y.shape[1]

        if self.nperm:
            rperm = np.zeros(shape=(self.nfolds, self.nperm, nvox))
        else:
            rperm = None
            print(f"nperm set to 0, not permuting, returned pval will be None")
        rtrue = np.zeros(shape=(self.nfolds, nvox))

        splits = self.make_splits(X)
        if self.nfolds > 1:
            nsamples_per_fold = self.calc_nsamples_per_fold(X)
            self.perminds = self.make_permutation_inds(nsamples_per_fold)

        for fold_i, (train_is, test_is) in tqdm(
            enumerate(splits), total=self.nfolds, desc="CV folds", leave=True
        ):
            # index train/test
            x_train, x_test = X[train_is], X[test_is]
            y_train, y_test = y[train_is], y[test_is]
            # determine true performance in this fold.
            rtrue[fold_i], pred_test = self.fit_predict_eval(
                x_train, x_test, y_train, y_test
            )
            # determine permuted performances in this fold
            if self.nperm:
                rs_this_perm = Parallel(n_jobs=self.n_jobs_permutation)(
                    delayed(shuffle_and_correlate)(
                        perminds_i, pred_test, y_test, self.metric
                    )
                    for perminds_i in tqdm(
                        self.perminds,
                        total=self.nperm,
                        desc="permutations",
                        leave=False,
                    )
                )
                rperm[fold_i] = np.array(rs_this_perm)
        # average over CV folds
        rtrue = rtrue.mean(axis=0)
        if self.nperm:
            rperm = rperm.mean(axis=0)
            # determine pvalue
            pval = self.calc_p(rtrue, rperm)
        else:
            pval = None
        return rtrue, pval


class FracRidgeVoxelwise:
    """
    A class for performing voxel-wise fractional ridge regression on fMRI data.

    This class encapsulates the functionality needed to tune, fit, and evaluate
    fractional ridge regression models on a per-voxel basis in fMRI datasets. It
    allows for tuning of the regularization parameter (alpha) using K-Fold
    cross-validation, fitting the final model, and evaluating its performance.
    Optionally, it can compute partial correlations between predicted and observed
    responses after accounting for other predictors.

    Parameters:
    - n_splits (int): Number of splits for K-Fold cross-validation.
    - test_size (float, optional): Proportion of the dataset to include in the test split.
      If set to 0, both training and testing are performed on the entire dataset.
      Defaults to 1/12.
    - fracs (np.ndarray, optional): Array of fractional values to be used for ridge
      regularization. Defaults to np.arange(0.01, 1.01, 0.01).
    - run_pcorr (bool, optional): If True, computes partial correlations after model fitting.
      Defaults to False.
    - fracridge_kws (dict, optional): Additional keyword arguments to be passed to the
      FracRidgeRegressor. Defaults to {"fit_intercept": True, "normalize": True}.

    Methods:
    - tune_alpha: Tunes the regularization parameter (alpha) for the model.
    - fit_evaulate: Fits the model and evaluates its performance on test data.
    - fit_pcorrs: Computes partial correlations between predicted and observed responses.
    - tune_and_eval: A high-level method that combines tuning, fitting, and evaluating.

    The class is designed to be used with fMRI data, where each voxel's response is predicted
    from a set of features (e.g., stimulus properties). The fractional ridge approach allows
    for fine-grained control over the amount of regularization applied to each voxel's model.
    """

    def __init__(
        self,
        n_splits,
        test_size=0.0833,  # = 1/12
        fracs=np.arange(0.01, 1.01, 0.01),
        run_pcorr=False,
        fracridge_kws={"fit_intercept": True, "normalize": True},
    ):
        self.n_splits = n_splits
        self.fracs = fracs
        self.nfracs = len(fracs)
        self.fr = FracRidgeRegressor(fracs=fracs, **fracridge_kws)
        self.test_size = test_size
        self.run_pcorr = run_pcorr

    def tune_alpha(self, X_train, y_train):
        nvox = y_train.shape[1]
        kfold = KFold(n_splits=self.n_splits, shuffle=False)
        r2s = np.zeros((self.n_splits, self.nfracs, nvox))
        for fold_i, (tune_inds, val_inds) in tqdm(
            enumerate(kfold.split(X_train)),
            desc="tuning regularization parameter",
            total=self.n_splits,
        ):
            X_tune, y_tune = X_train[tune_inds], y_train[tune_inds]
            X_val, y_val = X_train[val_inds], y_train[val_inds]
            self.fr.fit(X_tune, y_tune)
            pred = self.fr.predict(X_val)
            for frac_i in range(self.nfracs):
                # TODO: Input contains NaNs
                r2s[fold_i, frac_i] = r2_score(
                    np.nan_to_num(y_val),
                    np.nan_to_num(pred[:, frac_i, :]),
                    multioutput="raw_values",
                )
        r2_train = r2s.mean(0)
        best_fracinds = np.argmax(r2_train, axis=0)
        best_fracs = self.fracs[best_fracinds]
        return best_fracs, r2_train

    def fit_evaulate(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        best_fracs,
    ):
        nvox, ndims = y_train.shape[1], X_train.shape[1]
        betas = np.zeros((nvox, ndims))
        r2_test = np.zeros(nvox)
        unique_fracs = np.unique(best_fracs)
        for frac in tqdm(
            unique_fracs,
            desc="fitting final model (for each relevant frac parameter)",
            total=len(unique_fracs),
        ):
            vox_is = np.where(best_fracs == frac)[0]
            self.fr.fracs = frac
            self.fr.fit(X_train, y_train[:, vox_is])
            betas_ = self.fr.coef_.T
            betas[vox_is] = betas_
            pred = self.fr.predict(X_test)
            r2_test[vox_is] = r2_score(
                np.nan_to_num(y_test[:, vox_is]), 
                np.nan_to_num(pred), 
                multioutput="raw_values"
            )
        return betas, r2_test

    def fit_pcorrs(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        best_fracs,
    ):
        # TODO: integrate with fit_evaluate, we don't have to do the voxel indexing twice.
        print(
            "\nDetermining partial correlations. Make sure your X and y are standardized.\n"
        )
        nvox, ndims = y_train.shape[1], X_train.shape[1]
        pcorrs = np.zeros((nvox, ndims))
        unique_fracs = np.unique(best_fracs)
        for frac in tqdm(
            unique_fracs,
            desc="Voxel sets (depending on frac)",
            total=len(unique_fracs),
            leave=True,
        ):
            # select relevant voxels
            self.fr.fracs = frac
            vox_is = np.where(best_fracs == frac)[0]
            y_train_thisfrac, y_test_thisfrac = y_train[:, vox_is], y_test[:, vox_is]
            for pred_i in tqdm(
                range(ndims), total=ndims, desc="predictors", leave=False
            ):
                # partial out other predictors from both this predictor and the response
                X_train_nuisance, X_test_nuisance = np.delete(
                    X_train, pred_i, axis=1
                ), np.delete(X_test, pred_i, axis=1)
                X_train_ = regress_out(
                    X_train_nuisance, X_train[:, pred_i].reshape(-1, 1)
                )
                X_test_ = regress_out(X_test_nuisance, X_test[:, pred_i].reshape(-1, 1))
                y_train_ = regress_out(X_train_nuisance, y_train_thisfrac)
                y_test_ = regress_out(X_test_nuisance, y_test_thisfrac)
                # fit model
                self.fr.fit(X_train_, y_train_)
                # determine correlation
                pred = self.fr.predict(X_test_)
                pcorrs[vox_is, pred_i] = pearsonr_nd(y_test_, pred)
        return pcorrs

    def tune_and_eval(self, X, y):
        """Tune alpha and fit final model"""
        if self.test_size:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y
        best_fracs, r2_train = self.tune_alpha(X_train, y_train)
        betas, r2_test = self.fit_evaulate(X_train, X_test, y_train, y_test, best_fracs)
        if self.run_pcorr:
            pcorrs = self.fit_pcorrs(X_train, X_test, y_train, y_test, best_fracs)
        else:
            pcorrs = np.zeros(r2_test.shape)
        return betas, r2_train, r2_test, best_fracs, pcorrs
