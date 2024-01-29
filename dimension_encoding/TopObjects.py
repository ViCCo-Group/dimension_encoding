import os
from os.path import join as pjoin

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.metrics.pairwise import cosine_similarity

from dimension_encoding.utils import vcorrcoef


class TopObjectsProfile:
    """
    Find and plot top objects for only one regional tuning profile.
    """

    def __init__(
        self,
        roiname,
        roibetas,
        img_emb,
        img_fnames,
        outdir,
        similarity_metric="rectcosine",
        use_fisher=True,
        plot_k=20,
        plot_dpi=300,
        plot_fileformat="pdf",
        usecache=True,
        simulate_random_profiles=0,
    ):
        self.roiname = roiname
        self.roibetas = roibetas
        self.img_emb = img_emb
        self.img_fnames = img_fnames
        self.similarity_metric = similarity_metric
        self.use_fisher = use_fisher
        self.outdir = outdir
        self.results_df = None
        self.plot_k = plot_k
        self.plot_dpi = plot_dpi
        self.plot_fileformat = plot_fileformat
        self.usecache = usecache
        self.results_tsv = pjoin(self.outdir, "results.tsv")
        self.simulate_random_profiles = simulate_random_profiles
        self._check_inputs()

    def _check_inputs(self):
        assert self.similarity_metric in ("correlation", "rectcosine", "cosine")
        assert self.img_emb.shape[-1] == self.roibetas.shape[0]
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def find_top_objects(self):
        # compute similarity of ROI profile with all image profiles
        if self.similarity_metric == "correlation":
            s = vcorrcoef(self.img_emb, self.roibetas)
        elif self.similarity_metric == "rectcosine":
            self.roibetas[self.roibetas < 0] = 0
            s = cosine_similarity(self.img_emb, self.roibetas.reshape(1, -1)).flatten()
        elif self.similarity_metric == "cosine":
            s = cosine_similarity(self.img_emb, self.roibetas.reshape(1, -1)).flatten()
        # use fisher correction for correlations
        if self.use_fisher:
            s = np.arctanh(s)
        # find indices that sort by similarity
        sort_inds = np.argsort(s)
        # return sorted file names and similarity values
        fnames_sorted = self.img_fnames[sort_inds]
        s_sorted = s[sort_inds]
        return fnames_sorted, s_sorted

    def make_fig(self, figwidth=20):
        fnames_sorted, s_sorted = self.find_top_objects()
        outfile = pjoin(
            self.outdir,
            f"roiname-{self.roiname}_metric-{self.similarity_metric}_usefisher-{str(self.use_fisher)}_k-{self.plot_k}.{self.plot_fileformat}",
        )
        # grab highest and lowest image files (and similarity values)
        lowest_k_fns, highest_k_fns = (
            fnames_sorted[: self.plot_k],
            fnames_sorted[-self.plot_k :],
        )
        lowest_k_s, highest_k_s = s_sorted[: self.plot_k], s_sorted[-self.plot_k :]
        # suppress showing plots
        matplotlib.use("Agg")
        # make plot
        imwidth = figwidth / self.plot_k
        imtitlepadding = 1  # make figure a bit taller to account for sub-titles showing similarity values
        fig, axs = plt.subplots(
            nrows=2, ncols=self.plot_k, figsize=(figwidth, imwidth * 2 + imtitlepadding)
        )
        for row, fpaths, sims in zip(
            axs, [highest_k_fns, lowest_k_fns], [highest_k_s, lowest_k_s]
        ):
            for a, f, s in zip(row, fpaths, sims):
                im = io.imread(f)
                a.imshow(im)
                a.set_xticks([]), a.set_yticks([])
                a.set_title(f"s = {s:.3f}", fontsize=16)
        plt.suptitle(f"{self.roiname}", fontsize=20)
        plt.tight_layout()
        # save to pdf
        fig.savefig(outfile, dpi=self.plot_dpi)
        plt.close()



