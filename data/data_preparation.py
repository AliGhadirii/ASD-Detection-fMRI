import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.cluster import clustering
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import skew, kurtosis


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for training the Inception_v3 model"
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out\ABIDE_adjacency.npz",
        help="Path to the adjacancy matrix",
        required=True,
    )
    parser.add_argument(
        "--time_series_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\competition\out\time_series.npy",
        help="Path to the time series matrix",
        required=True,
    )
    parser.add_argument(
        "--y_path",
        type=str,
        default=r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out\Y_target.npz",
        help="Path to the y target",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Size of batch",
        required=False,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold",
        required=False,
    )
    parser.add_argument(
        "--data_scaler_type",
        type=str,
        default=None,
        help="Method used for scaling the data. options: ['MinMax', 'Standard']",
        required=False,
    )

    args = parser.parse_args()
    return args


def data_preparation(
    adj_path, time_series_path, y_path, batch_size=1, threshold=0.2, scaler_type=None
):
    """
    Creates Data object of pytorch_geometric using graph features and edge list

    Args:
        adj_path (str): path to the adjacancy matrix (.npz)
        time_series_path (str): path to the time_series matrix (.npy)
        y_path (str): path to the label matrix (.npz)
        batch_size (int, optional): batch_size used for dataloaders. Defaults to 1.
        threshold (float, optional): threshold used to remove noisy connections from adj_mat. Defaults to 0.2.

    Returns:
        tuple: train, validation, test dataloaders
    """

    adj_mat = np.load(adj_path)["a"]
    time_series_ls = np.load(time_series_path, allow_pickle=True)
    y_target = np.load(y_path)["a"]

    adj_mat[adj_mat <= threshold] = 0

    data_list = []
    ## Create a graph using networkx
    for i in range(adj_mat.shape[0]):

        G = nx.from_numpy_matrix(adj_mat[i], create_using=nx.Graph)

        ## Extract features

        ## dict(G.degree(weight="weight")).values()
        ## dict(betweenness_centrality(G, weight="weight")).values()

        features = pd.DataFrame(
            {
                "degree": dict(G.degree).values(),
                "betweenness": dict(nx.betweenness_centrality(G)).values(),
                # "eccentricity": dict(nx.eccentricity(G)).values(),
                "ts_mean": time_series_ls[i].mean(axis=0),
                "ts_variance": time_series_ls[i].var(axis=0),
                "ts_skewness": skew(time_series_ls[i], axis=0),
                "ts_kurtosis": kurtosis(time_series_ls[i], axis=0),
            }
        )

        # scale the data (optional)
        if scaler_type in ["MinMax", "Standard"]:
            if scaler_type == "MinMax":
                scaler = MinMaxScaler()
                features = scaler.fit_transform(features)
            else:
                scaler = StandardScaler()
                features = scaler.fit_transform(features)

            X = torch.from_numpy(features)
        else:
            X = torch.tensor(features.values)

        print(X.shape)
        edge_index = torch.tensor(list(G.edges()))
        data_list.append(Data(x=X, edge_index=edge_index.T, y=y_target[i].item()))

    ## Split Dataset
    train, test = train_test_split(
        data_list, test_size=0.25, shuffle=True, random_state=42
    )

    train_data_loader = DataLoader(train, batch_size=batch_size)
    test_data_loader = DataLoader(test, batch_size=batch_size)

    return train_data_loader, test_data_loader


def main():
    args = parse_arguments()
    train_data_loader, test_data_loader = data_preparation(
        adj_path=args.adj_path,
        time_series_path=args.time_series_path,
        y_path=args.y_path,
        batch_size=args.batch_size,
        threshold=args.threshold,
        scaler_type=args.data_scaler_type,
    )
    for data in train_data_loader:  # every batch
        print(data, data.y)


if __name__ == "__main__":
    main()
