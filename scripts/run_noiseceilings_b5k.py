from dimension_encoding.b5k import B5kLoader, compute_noise_ceiling_b5k
from tqdm import tqdm


if __name__=="__main__":
    dl = B5kLoader()
    for subj_int in tqdm(range(1,4), total=3):
        subject = f"CSI{subj_int}"
        responses = dl.load_responses(subject)
        stimdata = dl.load_stimdata(subject)
        nc = compute_noise_ceiling_b5k(responses, stimdata)
        nc_img = dl.array_to_volume(nc, subject)
        nc_img.to_filename(f'sub-{subject}_noiseceiling.nii.gz')