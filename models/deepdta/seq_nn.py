"""
DeepDTA based models for drug-target interaction prediction problem.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool





class GCNEncoder(nn.Module):
    r"""
    The GraphDTA's GCN encoder module, which comprises three graph convolutional layers and one full connected layer.
    The model is a variant of DeepDTA and is applied to encoding drug molecule graph information. The original paper
    is  `"GraphDTA: Predicting drug–target binding affinity with graph neural networks"
    <https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btaa921/5942970>`_ .

    Args:
        in_channel (int): Dimension of each input node feature.
        out_channel (int): Dimension of each output node feature.
        dropout_rate (float): dropout rate during training.
    """

    def __init__(self, in_channel=78, out_channel=128, dropout_rate=0.2):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channel, in_channel)
        self.conv2 = GCNConv(in_channel, in_channel * 2)
        self.conv3 = GCNConv(in_channel * 2, in_channel * 4)
        self.fc = nn.Linear(in_channel * 4, out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = global_max_pool(x, batch)
        x = self.fc(x)
        x = self.dropout(x)
        return x