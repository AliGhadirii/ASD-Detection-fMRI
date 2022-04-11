from nilearn.datasets import fetch_abide_pcp, fetch_atlas_basc_multiscale_2015
import argparse
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Data",
        help="Path to the data directory",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out",
        help="Path to the output file",
        required=True,
    )
    args = parser.parse_args()
    return args


def load_data(data_dir, output_dir, pipeline="cpac", quality_checked=True):
    # get dataset
    print("Loading dataset...")
    abide = fetch_abide_pcp(
        data_dir=data_dir,
        SITE_ID=["CMU"],
        pipeline=pipeline,
        quality_checked=quality_checked,
    )

    # make list of filenames
    fmri_filenames = abide.func_preproc

    # load atlas
    multiscale = fetch_atlas_basc_multiscale_2015()
    atlas_filename = multiscale.scale064
    print(f"atalas file names are: {atlas_filename}")

    # initialize masker object
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename, standardize=True, memory="nilearn_cache", verbose=0
    )

    # initialize correlation measure
    correlation_measure = ConnectivityMeasure(
        kind="correlation", vectorize=False, discard_diagonal=True
    )

    try:  # check if feature file already exists
        # load features
        feat_file = os.path.join(output_dir, "ABIDE_adjacency.npz")
        adj_mat = np.load(feat_file)["a"]
        print("Feature file found.")

    except:  # if not, extract features
        adj_mat = []  # To contain upper half of matrix as 1d array
        print("No feature file found. Extracting features...")

        for i, sub in enumerate(fmri_filenames):
            # extract the timeseries from the ROIs in the atlas
            time_series = masker.fit_transform(sub)
            # create a region x region correlation matrix
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            # add to our container
            adj_mat.append(correlation_matrix)
            # keep track of status
            print("finished extracting %s of %s" % (i + 1, len(fmri_filenames)))
        # Save features
        np.savez_compressed(os.path.join(output_dir, "ABIDE_adjacency"), a=adj_mat)

    abide_pheno = pd.DataFrame(abide.phenotypic)

    # Get the target vector
    y_target = abide_pheno["DX_GROUP"]

    return adj_mat, y_target


def run():
    args = parse_arguments()
    adj_mat, y_target = load_data(args.input_path, args.output_path)
    print(adj_mat.shape)
    print(type(adj_mat))
    print(adj_mat)
    print("***************************")
    print(y_target)


if __name__ == "__main__":
    run()
