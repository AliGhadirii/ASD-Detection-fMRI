import numpy as np
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

data = np.load(
    r"C:\Users\Afrooz Sheikholeslam\Education\8th semester\Project1\Code\Out\ABIDE_adjacency.npz"
)
adj_mat = data["a"]
adj_mat = np.greater_equal(adj_mat, 0.5).astype(int)
G = nx.from_numpy_matrix(adj_mat[0])
# print(G.number_of_nodes())
# print(G.nodes())
# print(torch.tensor(list(G.edges())))

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
data = Data(x=X, edge_index=edge_index)
print(data)
