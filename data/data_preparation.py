import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.centrality import (
    eigenvector,
    betweenness_centrality,
    closeness_centrality,
)
from networkx.algorithms.cluster import clustering
import torch
from torch_geometric.data import Data


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
    args = parser.parse_args()
    return args


def data_preparation(adj_path):
    """Creates Data object of pytorch_geometric using graph features and edge list

    Parameters
    ----------
    adj_path : str
        path to the adjacancy matrix (.npz)

    Returns
    -------
    Data Object [torch_geometric.loader.DataLoader]
    """

    data = np.load(args.adj_path)
    adj_mat = data["a"]
    adj_mat = np.greater_equal(adj_mat, 0.5).astype(int)

    ## Create a graph using networkx
    G = nx.from_numpy_matrix(adj_mat[0])

    ## Extract features
    features = pd.DataFrame(
        {
            "degree": dict(G.degree).values(),
            "eigen_vector_centrality": dict(nx.eigenvector_centrality(G)).values(),
            "betweenness": dict(betweenness_centrality(G)).values(),
            "closeness": dict(closeness_centrality(G)).values(),
            "clustring_coef": dict(clustering(G)).values(),
        }
    )

    X = torch.tensor(features.values)
    edge_index = torch.tensor(list(G.edges()))
    data_obj = Data(x=X, edge_index=edge_index)
    return data_obj


def main():
    args = parse_arguments()
    data = data_preparation(args.adj_path)
    print(data)


if __name__ == "__main__":
    main()
