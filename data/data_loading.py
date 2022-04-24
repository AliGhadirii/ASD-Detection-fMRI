from nilearn.datasets import (
    fetch_abide_pcp,
    fetch_atlas_basc_multiscale_2015,
    fetch_coords_power_2011,
)
import argparse
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker
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
    parser.add_argument(
        "--fc_matrix_kind",
        type=str,
        default="correlation",
        help="different kinds of functional connectivity matrices : covariance, correlation, partial correlation, tangent, precision",
        required=False,
    )
    args = parser.parse_args()
    return args


def load_data(
    data_dir, output_dir, fc_matrix_kind, pipeline="cpac", quality_checked=True
):
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

    """ Previous Atlas
    # load atlas
    multiscale = fetch_atlas_basc_multiscale_2015()
    # print(multiscale)
    atlas_filename = multiscale.scale064
    # print(f"atalas file names are: {atlas_filename}")
    
    initialize masker object
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename, standardize=True, memory="nilearn_cache", verbose=0
    )
    """

    # load power atlas
    power = fetch_coords_power_2011()
    coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T
    print("Stacked power coordinates in array of shape {0}.".format(coords.shape))

    # initialize masker object
    # NiftiSpheresMasker is useful when data from given seeds should be extracted.
    masker = NiftiSpheresMasker(
        seeds=coords,
        radius=5,  # Indicates, in millimeters, the radius for the sphere around the seed
        standardize=True,  # the signal is z-scored. Timeseries are shifted to zero mean and scaled to unit variance
        memory="nilearn_cache",
        verbose=2,
    )

    # initialize correlation measure
    correlation_measure = ConnectivityMeasure(
        kind=fc_matrix_kind, vectorize=False, discard_diagonal=True
    )

    try:  # check if feature file already exists
        # load features
        feat_file = os.path.join(output_dir, "ABIDE_adjacency.npz")
        correlation_matrices = np.load(feat_file)["a"]
        print("Feature file found.")

    except:  # if not, extract features
        print("No feature file found. Extracting features...")

        if fc_matrix_kind == "tangent":
            time_series_ls = []
            for i, sub in enumerate(fmri_filenames):
                # extract the timeseries from the ROIs in the atlas
                time_series = masker.fit_transform(sub)

                print(f"shape of time series{i}: {time_series.shape}")
                time_series_ls.append(time_series)
                print("finished extracting %s of %s" % (i + 1, len(fmri_filenames)))

            correlation_matrices = correlation_measure.fit_transform(time_series_ls)

        else:
            correlation_matrices = []
            for i, sub in enumerate(fmri_filenames):
                # extract the timeseries from the ROIs in the atlas
                time_series = masker.fit_transform(sub)
                # create a region x region correlation matrix
                correlation_matrix = correlation_measure.fit_transform([time_series])[0]
                # add to our container
                correlation_matrices.append(correlation_matrix)

                print("finished extracting %s of %s" % (i + 1, len(fmri_filenames)))

        np.savez_compressed(
            os.path.join(output_dir, "ABIDE_adjacency"), a=correlation_matrices
        )
        correlation_matrices = np.array(correlation_matrices)

    # Get the target vector
    abide_pheno = pd.DataFrame(abide.phenotypic)
    y_target = abide_pheno["DX_GROUP"]
    y_target = y_target.apply(lambda x: x - 1)
    np.savez_compressed(os.path.join(output_dir, "Y_target"), a=y_target)

    return correlation_matrices, y_target


def run():
    args = parse_arguments()
    adj_mat, y_target = load_data(
        args.input_path, args.output_path, args.fc_matrix_kind
    )
    print(adj_mat.shape)
    print("****************************")
    print(y_target.shape)


if __name__ == "__main__":
    run()
