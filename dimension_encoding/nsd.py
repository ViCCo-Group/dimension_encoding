from os.path import join as pjoin
import re
import numpy as np
from nilearn.masking import apply_mask
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def nsd_fnames_to_indices(fnames):
    """
    Extract numerical indices from a list of filenames that were created for dimension predictions,
    and thus map onto the predicted embedding rows.

    Parameters:
    - fnames (list of str): A list of filenames where each filename contains an index number ('index-{}').

    Returns:
    - numpy.ndarray: An array of integers representing the extracted indices.
    """
    pattern = re.compile(r"index-(\d+)")
    index_numbers = []
    for filename in fnames:
        match = pattern.search(filename)
        if match:
            index_numbers.append(int(match.group(1)))
    embedding_indices = np.array(index_numbers)
    assert len(embedding_indices) == len(fnames)
    return embedding_indices


class NsdLoader:
    """
    Loader class for handling and processing data from the Natural Scenes Dataset (NSD).

    Attributes:
    - data_dir (str): Base directory where the NSD data is stored.
    - dimpreds_dir (str): Directory containing predicted dimensionality data.
    - nsd_download_dir (str): Directory containing the downloaded NSD data.
    - allowed_spaces (str): Specifies the allowed brain image spaces.
    - n_sessions (int): Number of sessions to be considered for data loading.

    Methods:
    - load_predicted_spose_dimensions: Loads the predicted spatial pose dimensions.
    - get_bmask_file: Retrieves the file path for the brain mask file for a given subject and space.
    - load_betas: Loads beta values for a subject in a specified space across all sessions.
    - load_trialinfo: Loads trial information for a given subject.
    - make_dimensions_model: Constructs a model using dimensions data matched to trial information.
    """

    def __init__(self, data_dir="../data"):
        """
        Initializes the NsdLoader with a specified data directory.

        Parameters:
        - data_dir (str): The base directory where NSD data is stored. Defaults to "../data".
        """
        self.data_dir = data_dir
        self.dimpreds_dir = pjoin(data_dir, "nsd_predicted_dimensions")
        self.nsd_download_dir = pjoin(data_dir, "nsd", "natural-scenes-dataset")
        self.allowed_spaces = "func1pt8"
        self.n_sessions = 40

    def load_predicted_spose_dimensions(self):
        """
        Loads embeddings and filenames for predicted spatial pose dimensions from text files.

        Returns:
        - Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the embedding array and the array of filenames.
        """
        fnames_txt = pjoin(self.dimpreds_dir, "file_names_nsd.txt")
        emb_txt = pjoin(
            self.dimpreds_dir, "predictions_66d_ridge_clip-RN50_visual_nsd.txt"
        )
        fnames = np.loadtxt(fnames_txt, dtype=str)
        embedding = np.loadtxt(emb_txt)
        return embedding, fnames

    def get_bmask_file(self, subj, space="func1pt8"):
        """
        Gets the file path for the brain mask file for a given subject and space.

        Parameters:
        - subj (int): The subject number.
        - space (str): The brain image space. Defaults to "func1pt8".

        Returns:
        - str: The file path to the brain mask NIfTI image.

        Raises:
        - AssertionError: If the specified space is not allowed.
        """
        assert space in self.allowed_spaces
        bmask_nii = pjoin(
            self.nsd_download_dir,
            "nsddata",
            "ppdata",
            f"subj{subj}",
            space + "mm",
            "brainmask.nii.gz",
        )
        return bmask_nii

    def load_betas(self, subj, space="func1pt8", njobs=4):
        """
        Loads beta values for a given subject and space, using parallel processing.

        Parameters:
        - subj (int): The subject number.
        - space (str): The brain image space. Defaults to "func1pt8".
        - njobs (int): Number of jobs to run in parallel. Defaults to 4.

        Returns:
        - numpy.ndarray: An array of beta values for the subject.

        Raises:
        - AssertionError: If the specified space is not allowed.
        """
        assert space in self.allowed_spaces
        bmask_nii = self.get_bmask_file(subj, space)
        niftis = [
            pjoin(
                self.nsd_download_dir,
                "nsddata_betas",
                "ppdata",
                f"subj{subj}",
                space,
                "betas_fithrf_GLMdenoise_RR",
                f"betas_session{ses_i:02d}.nii.gz",
            )
            for ses_i in range(1, self.n_sessions + 1)
        ]
        parallel = Parallel(n_jobs=6)
        betas_ = parallel(
            delayed(apply_mask)(nifti, bmask_nii)
            for nifti in tqdm(niftis, desc="loading session betas")
        )
        betas = np.vstack(betas_)
        return betas

    def load_trialinfo(self, subj):
        """
        Loads trial information for a given subject from a TSV file.

        Parameters:
        - subj (int): The subject number.

        Returns:
        - pandas.DataFrame: A DataFrame containing trial information for the subject.
        """
        trialinfo_tsv = pjoin(
            self.nsd_download_dir,
            "nsddata",
            "ppdata",
            f"subj{subj}",
            "behav",
            "responses.tsv",
        )
        trialinfo = pd.read_csv(trialinfo_tsv, sep="\t")
        trialinfo["fname_index"] = trialinfo["73KID"] - 1
        return trialinfo

    def average_over_repeats(self, betas, trialinfo):
        avgbetas = []
        fname_index_avgdata = []
        for group_i, rows in tqdm(
            trialinfo.groupby("fname_index"),
            total=len(trialinfo["fname_index"].unique()),
            desc="averaging over repeats",
        ):
            trial_is = rows.index
            avgbetas.append(betas[trial_is].mean(0))
            fname_index_avgdata.append(rows["fname_index"].values[0])
        avgbetas = np.stack(avgbetas, axis=0)
        trialinfo_avgbetas = pd.DataFrame(
            {"fname_index": np.array(fname_index_avgdata)}
        )
        return avgbetas, trialinfo_avgbetas

    def make_dimensions_model(self, subject):
        # get embedding and trial info
        embedding, fnames = self.load_predicted_spose_dimensions()
        embedding_indices = nsd_fnames_to_indices(fnames)
        trialinfo = self.load_trialinfo(subject)
        # average over repeats
        betas = self.load_betas(subject)
        y, trialinfo = self.average_over_repeats(betas, trialinfo)
        X = np.zeros((len(trialinfo), embedding.shape[1]))
        for trial_i in tqdm(
            range(len(trialinfo)), total=len(trialinfo), desc="making dimensions model"
        ):
            trial_row = trialinfo.iloc[trial_i]
            match_inds = np.where(trial_row["fname_index"] == embedding_indices)[0]
            assert len(match_inds) >= 1
            X[trial_i] = embedding[match_inds]
        return X, y
