from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import os
import pandas as pd


def prepare_data(data_dir, pipeline="cpac", quality_checked=True):
    # get dataset
    print("Loading dataset...")
    abide = datasets.fetch_abide_pcp(
        data_dir=data_dir,
        SITE_ID=["CMU"],
        pipeline=pipeline,
        quality_checked=quality_checked,
    )


def run():
    path = r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Data"
    prepare_data(path)


if __name__ == "__main__":
    run()
