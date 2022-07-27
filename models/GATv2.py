import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2(torch.nn.Module):
    def __init__(
        self,
        input_feat_dim,
        dim_shapes,
        heads,
        num_classes,
        dropout_rate=None,
        last_sigmoid=False,
    ):

        super(GATv2, self).__init__()
        self.num_layers = len(dim_shapes)
        self.last_sigmoid = last_sigmoid
        self.dropout_rate = dropout_rate
        self.linear = None

        assert (
            self.num_layers >= 1
        ), "Number of layers should be more than or equal to 1"

        if input_feat_dim != dim_shapes[0][0]:
            self.linear = nn.Linear(input_feat_dim, dim_shapes[0][0])

        self.convs = nn.ModuleList()
        for l in range(self.num_layers):
            if l == 0:
                self.convs.append(
                    GATv2Conv(dim_shapes[l][0], dim_shapes[l][1], heads=heads)
                )
            else:
                self.convs.append(
                    GATv2Conv(dim_shapes[l][0] * heads, dim_shapes[l][1], heads=heads)
                )

        self.pooling = global_mean_pool

        self.classifier = nn.Sequential(
            nn.Linear(heads * dim_shapes[-1][1], 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, batched_data):

        x, edge_index, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.batch,
        )

        if self.linear is not None:
            x = self.linear(x.float())

        for l in range(self.num_layers):
            x = F.relu(self.convs[l](x.float(), edge_index))

        x = self.pooling(x, batch)
        if self.dropout_rate is not None:
            x = F.dropout(x, p=self.dropout_rate)
        x = self.classifier(x)

        if self.last_sigmoid:
            return torch.sigmoid(x)

        return x

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
