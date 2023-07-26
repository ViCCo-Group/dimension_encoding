from dimension_encoding.glm import PModCVCLIP
from dimension_encoding.utils import load_clip66_preds
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run variance partitioning based on a cross-validated regression to compare a categorical and dimension model"
    )
    parser.add_argument("--sub", type=str, help="subject id", required=True)
    parser.add_argument(
        "--bidsroot",
        type=str,
        help="path to bids root directory",
        default="/LOCAL/ocontier/thingsmri/bids/",
    )
    parser.add_argument(
        "--clip66dir",
        type=str,
        help="path to clip-predicted behavioral embeddings (66d)",
        default="/LOCAL/ocontier/thingsmri/bids/code/external_libraries/66d",
    )
    parser.add_argument(
        "--emb_f",
        type=str,
        help="file name of clip-predicted behavioral embeddings within the --clip66dir",
        default='predictions_66d_elastic_clip-ViT-B-32_visual_THINGS.txt',
    )
    args = parser.parse_args()
    return args

def main(args):
    embedding, filenames, _ = load_clip66_preds(
        args.clip66dir, fnames_with_folder_structure=True, emb_f=args.emb_f,
    )
    pmod_clip = PModCVCLIP(
        bidsroot=args.bidsroot, subject=args.sub, clip_preds=embedding, clip_fnames=filenames, usecache=False,
        out_deriv_name='pmod_cv_clip_66d',
    )
    pmod_clip.run()
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)